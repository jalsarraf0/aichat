import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:logging/logging.dart';
import 'package:path/path.dart' as p;
import 'package:shelf/shelf.dart'
    show Cascade, Handler, Middleware, Pipeline, Request, Response;
import 'package:shelf_router/shelf_router.dart' show Router;

import 'compaction.dart';
import 'config.dart';
import 'database.dart';
import 'llm_client.dart';
import 'mcp_client.dart';
import 'model_profiles.dart';
import 'models.dart';
import 'personalities.dart';
import 'tool_router.dart' as tool_router;

final _log = Logger('Router');

class AppRouter {
  final Config config;
  final AppDatabase db;
  final LlmClient llm;
  final McpClient mcp;
  final Compactor compactor;

  late final Router _router;

  AppRouter({
    required this.config,
    required this.db,
    required this.llm,
    required this.mcp,
    required this.compactor,
  }) {
    _router = Router()
      ..get('/health', _health)
      ..get('/api/conversations', _listConversations)
      ..post('/api/conversations', _createConversation)
      ..get('/api/conversations/<id>', _getConversation)
      ..delete('/api/conversations/<id>', _deleteConversation)
      ..patch('/api/conversations/<id>', _updateConversation)
      ..post('/api/conversations/<id>/messages', _sendMessage)
      ..get('/api/tools', _listTools)
      ..post('/api/tools/refresh', _refreshTools)
      ..get('/api/models', _listModels)
      ..get('/api/personalities', _listPersonalities)
      ..post('/api/warmup', _warmupModel)
      ..post('/api/unload', _unloadModel);
  }

  Handler get handler {
    final cascade = Cascade().add(_router.call).add(_staticHandler);
    return const Pipeline().addMiddleware(_cors()).addHandler(cascade.handler);
  }

  // ── Static file handler ────────────────────────────────────────────

  FutureOr<Response> _staticHandler(Request request) {
    var filePath = request.url.path;
    if (filePath.isEmpty || filePath == '/') filePath = 'index.html';

    // Resolve the absolute path and verify it stays within webDir
    // to prevent path traversal attacks (e.g. ../../etc/passwd).
    final webRoot = p.canonicalize(config.webDir);
    final resolved = p.canonicalize(p.join(config.webDir, filePath));
    if (!p.isWithin(webRoot, resolved) && resolved != webRoot) {
      return Response.forbidden('Forbidden');
    }

    final file = File(resolved);
    if (!file.existsSync()) {
      // SPA fallback — exclude API paths so they get proper 404 JSON
      final index = File(p.join(config.webDir, 'index.html'));
      if (!filePath.startsWith('api/') && index.existsSync()) {
        return Response.ok(
          index.openRead(),
          headers: {'Content-Type': 'text/html; charset=utf-8'},
        );
      }
      return Response.notFound('Not found');
    }

    final ext = p.extension(filePath).toLowerCase();
    final contentType = _mimeType(ext);
    return Response.ok(
      file.openRead(),
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'no-cache, no-store, must-revalidate',
      },
    );
  }

  String _mimeType(String ext) {
    switch (ext) {
      case '.html':
        return 'text/html; charset=utf-8';
      case '.css':
        return 'text/css; charset=utf-8';
      case '.js':
        return 'application/javascript; charset=utf-8';
      case '.json':
        return 'application/json; charset=utf-8';
      case '.png':
        return 'image/png';
      case '.jpg':
      case '.jpeg':
        return 'image/jpeg';
      case '.svg':
        return 'image/svg+xml';
      case '.ico':
        return 'image/x-icon';
      default:
        return 'application/octet-stream';
    }
  }

  // ── CORS middleware ─────────────────────────────────────────────────

  Middleware _cors() {
    return (Handler innerHandler) {
      return (Request request) async {
        if (request.method == 'OPTIONS') {
          return Response.ok('', headers: _corsHeaders);
        }
        final response = await innerHandler(request);
        return response.change(headers: _corsHeaders);
      };
    };
  }

  static const _corsHeaders = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Allow-Methods': 'GET, POST, PATCH, DELETE, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };

  // ── User Identity ──────────────────────────────────────────────────

  /// Extract authenticated user ID from X-Auth-User header (set by auth proxy).
  String _getUserId(Request request) {
    return request.headers['x-auth-user'] ?? '';
  }

  // ── API Handlers ───────────────────────────────────────────────────

  Response _health(Request request) {
    return _json({
      'ok': true,
      'service': 'dartboard',
      'version': '1.0.0',
      'lm_studio': config.lmStudioUrl,
      'mcp': config.mcpUrl,
    });
  }

  Response _listConversations(Request request) {
    final userId = _getUserId(request);
    final limit =
        int.tryParse(request.url.queryParameters['limit'] ?? '') ?? 50;
    final offset =
        int.tryParse(request.url.queryParameters['offset'] ?? '') ?? 0;
    final convs = db.listConversations(
        userId: userId, limit: limit, offset: offset);
    return _json({'conversations': convs.map((c) => c.toJson()).toList()});
  }

  Future<Response> _createConversation(Request request) async {
    final body = await _readJson(request);
    final customPrompt = body?['system_prompt'] as String?;
    final personalityId = body?['personality_id'] as String?;

    // Determine model first (needed for prompt sizing)
    var requestedModel = body?['model'] as String?;
    if (requestedModel == null || requestedModel.isEmpty) {
      final loaded = await llm.listLoadedModels();
      if (loaded.isNotEmpty) {
        requestedModel = loaded.first;
        _log.info(
          'No model specified — defaulting to loaded model: $requestedModel',
        );
      } else {
        requestedModel = config.model;
      }
    }

    // Build system prompt — use condensed version for small-context models
    String systemPrompt;
    if (customPrompt != null && customPrompt.isNotEmpty) {
      systemPrompt = customPrompt;
    } else {
      final modelProfile = getProfile(requestedModel);
      systemPrompt = buildSystemPrompt(
        personalityId ?? 'general',
        condensed: modelProfile.promptSize == 'condensed',
      );
      // Optimize prompt on Arc A380 for condensed models
      if (modelProfile.promptSize == 'condensed' &&
          config.toolRouterUrl.isNotEmpty) {
        systemPrompt = await tool_router.optimizePrompt(
          systemPrompt,
          promptSize: modelProfile.promptSize,
          personalityId: personalityId ?? 'general',
          routerUrl: config.toolRouterUrl,
        );
      }
    }

    final userId = _getUserId(request);
    final conv = db.createConversation(
      userId: userId,
      title: body?['title'] as String? ?? 'New Chat',
      model: requestedModel,
      systemPrompt: systemPrompt,
    );
    // Add system prompt as first message
    db.addMessage(
      conversationId: conv.id,
      role: 'system',
      content: conv.systemPrompt.isNotEmpty
          ? conv.systemPrompt
          : config.systemPrompt,
    );
    return _json(conv.toJson(), status: 201);
  }

  Response _getConversation(Request request, String id) {
    final userId = _getUserId(request);
    final conv = db.getConversation(id, userId: userId);
    if (conv == null) return _json({'error': 'Not found'}, status: 404);
    final messages = db.getMessages(id);
    return _json({
      ...conv.toJson(),
      'messages': messages.map((m) => m.toJson()).toList(),
    });
  }

  Response _deleteConversation(Request request, String id) {
    final userId = _getUserId(request);
    db.deleteConversation(id, userId: userId);
    return _json({'status': 'deleted'});
  }

  Future<Response> _updateConversation(Request request, String id) async {
    final userId = _getUserId(request);
    // Verify ownership before updating
    final existing = db.getConversation(id, userId: userId);
    if (existing == null) return _json({'error': 'Not found'}, status: 404);
    final body = await _readJson(request);
    if (body == null) return _json({'error': 'Invalid JSON'}, status: 400);
    db.updateConversation(
      id,
      title: body['title'] as String?,
      model: body['model'] as String?,
      systemPrompt: body['system_prompt'] as String?,
    );
    final conv = db.getConversation(id, userId: userId);
    if (conv == null) return _json({'error': 'Not found'}, status: 404);
    return _json(conv.toJson());
  }

  Future<Response> _sendMessage(Request request, String id) async {
    final userId = _getUserId(request);
    final conv = db.getConversation(id, userId: userId);
    if (conv == null) return _json({'error': 'Not found'}, status: 404);

    final body = await _readJson(request);
    var userContent = body?['content'] as String? ?? '';

    // Process file attachments
    final rawFiles = body?['files'] as List?;
    List<Map<String, dynamic>>? imageAttachments;

    if (rawFiles != null && rawFiles.isNotEmpty) {
      imageAttachments = [];
      final textParts = StringBuffer();

      for (final f in rawFiles) {
        final file = Map<String, dynamic>.from(f as Map);
        final name = file['name'] as String? ?? 'file';
        final type = file['type'] as String? ?? '';
        final data = file['data'] as String? ?? '';

        if (type.startsWith('image/')) {
          imageAttachments.add({'type': 'image', 'data': data, 'name': name});
          textParts.writeln('[Attached image: $name]');
        } else {
          // Text file — inline content into message
          textParts.write('\n\n---\nFile: $name\n```\n$data\n```');
        }
      }

      if (textParts.isNotEmpty) {
        userContent = '$userContent\n${textParts.toString()}'.trim();
      }
      if (imageAttachments.isEmpty) imageAttachments = null;
    }

    if (userContent.isEmpty) {
      return _json({'error': 'content is required'}, status: 400);
    }

    // Store user message
    db.addMessage(conversationId: id, role: 'user', content: userContent);
    db.updateTokenCount(id);

    // Auto-generate title from first user message
    if (conv.title == 'New Chat') {
      final title = userContent.length > 50
          ? '${userContent.substring(0, 50)}...'
          : userContent;
      db.updateConversation(id, title: title);
    }

    // Determine which model to use for this conversation.
    // If the conversation has an explicit model (user picked it), use it.
    // LM Studio JIST will auto-load/swap as needed.
    var effectiveModel = conv.model.isNotEmpty ? conv.model : config.model;

    // Only fall back to the loaded model when NO model is specified at all.
    if (effectiveModel.isEmpty) {
      final loaded = await llm.listLoadedModels();
      if (loaded.isNotEmpty) {
        effectiveModel = loaded.first;
        _log.info('No model specified — using loaded model: $effectiveModel');
        db.updateConversation(id, model: effectiveModel);
      }
    }

    // Capacity guard: if the model is not loaded and slots are full, reject
    final busyMsg = await llm.ensureModelOrBusy(
      effectiveModel,
      maxLoaded: config.maxLoadedModels,
    );
    if (busyMsg != null) {
      final busyController = StreamController<List<int>>();
      _sseEvent(busyController, 'error', {'message': busyMsg});
      busyController.close();
      return Response.ok(
        busyController.stream,
        headers: {
          'Content-Type': 'text/event-stream',
          'Cache-Control': 'no-cache',
          'Connection': 'keep-alive',
          ..._corsHeaders,
        },
      );
    }

    // Check compaction
    await compactor.compactIfNeeded(id, model: effectiveModel);

    // Check if tools are enabled for this request
    final useTools = body?['tools_enabled'] as bool? ?? true;

    // Build SSE stream
    final controller = StreamController<List<int>>();

    // Let JIST handle model loading transparently.
    // No preflight — it races with JIST swaps and causes false rejections.
    if (!(await llm.isModelLoaded(effectiveModel))) {
      _sseEvent(controller, 'status', {
        'text': 'Loading $effectiveModel...',
      });
    }

    // Run the LLM loop asynchronously
    _runChatLoop(
      id,
      effectiveModel,
      controller,
      imageAttachments: imageAttachments,
      useTools: useTools,
    ).catchError((e) {
      _log.severe('Chat loop error: $e');
      _sseEvent(controller, 'error', {'message': '$e'});
      if (!controller.isClosed) controller.close();
    });

    return Response.ok(
      controller.stream,
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        ..._corsHeaders,
      },
    );
  }

  Future<void> _runChatLoop(
    String conversationId,
    String model,
    StreamController<List<int>> controller, {
    List<Map<String, dynamic>>? imageAttachments,
    bool useTools = true,
  }) async {
    // Apply per-model optimization profile
    final profile = getProfile(model);
    final effectiveTemp = profile.temperature;
    final effectiveMaxTokens = profile.maxTokens;

    // enforceTools overrides the client toggle — tools are always on
    final effectiveUseTools = profile.enforceTools || useTools;

    List<Map<String, dynamic>> openAiTools = [];
    if (effectiveUseTools && profile.supportsTools) {
      var tools = await mcp.getTools();
      // Step 1: filter to model's allowed tools
      if (profile.allowedTools != null) {
        tools = tools
            .where((t) => profile.allowedTools!.contains(t.name))
            .toList();
      }
      // Step 2: route to only the 1-3 tools this message needs
      final userMsgs = db.getMessages(conversationId)
          .where((m) => m.role == 'user');
      if (userMsgs.isNotEmpty) {
        tools = await tool_router.selectTools(
          userMsgs.last.content,
          tools,
          routerUrl: config.toolRouterUrl,
        );
      }
      openAiTools = tools.map((t) => t.toOpenAiFormat()).toList();
    }

    // Session-level image dedup — tracks ALL emitted image URLs
    final emittedImageUrls = <String>{}; // base URLs (no query params, lowercased)
    var totalImagesEmitted = 0;
    const maxImagesPerResponse = 4;

    for (var iteration = 0; iteration < config.maxToolIterations; iteration++) {
      final messages = db.getMessages(conversationId);
      final llmMessages = messages.map((m) => m.toLlmDict()).toList();

      // Truncate system prompt if model has a character limit
      if (profile.systemPromptMaxChars != null && llmMessages.isNotEmpty) {
        final first = llmMessages[0];
        if (first['role'] == 'system') {
          final content = first['content'] as String? ?? '';
          if (content.length > profile.systemPromptMaxChars!) {
            llmMessages[0] = {
              ...first,
              'content': content.substring(0, profile.systemPromptMaxChars!),
            };
          }
        }
      }

      // On first iteration, inject image attachments into the last user message
      if (iteration == 0 &&
          imageAttachments != null &&
          imageAttachments.isNotEmpty) {
        final lastUserIdx = llmMessages.lastIndexWhere(
          (m) => m['role'] == 'user',
        );
        if (lastUserIdx >= 0) {
          final userMsg = llmMessages[lastUserIdx];
          final contentParts = <Map<String, dynamic>>[
            {'type': 'text', 'text': userMsg['content'] as String? ?? ''},
          ];
          for (final att in imageAttachments) {
            contentParts.add({
              'type': 'image_url',
              'image_url': {'url': att['data'] as String},
            });
          }
          llmMessages[lastUserIdx] = {'role': 'user', 'content': contentParts};
        }
      }

      final fullContent = StringBuffer();
      final thinkingContent = StringBuffer();
      var pendingToolCalls = <ToolCallData>[];

      // Track whether this iteration will call tools — if so, suppress
      // intermediate narration tokens ("I'll search for...") from the UI.
      // The user only wants the final answer, not the play-by-play.
      var iterationHasToolCalls = false;
      final iterationTokens = StringBuffer();

      // enforceTools models use 'required' for first 3 iterations to ensure
      // tool usage, then switch to 'auto' so the model can synthesize text.
      final toolChoice = (profile.enforceTools && iteration < 3)
          ? 'required'
          : 'auto';

      await for (final event in llm.chatStream(
        model: model,
        messages: llmMessages,
        tools: openAiTools,
        toolChoice: toolChoice,
        maxTokens: effectiveMaxTokens,
        temperature: effectiveTemp,
      )) {
        switch (event) {
          case ReasoningTokenEvent(:final text):
            thinkingContent.write(text);
            _sseEvent(controller, 'thinking', {'text': text});
          case TokenEvent(:final text):
            iterationTokens.write(text);
            fullContent.write(text);
            // Don't emit token events yet — wait to see if tools are called
          case ToolCallsEvent(:final toolCalls):
            pendingToolCalls = toolCalls;
            iterationHasToolCalls = true;
          case DoneEvent(:final finishReason):
            if (finishReason == 'tool_calls' && pendingToolCalls.isNotEmpty) {
              // Store assistant message with tool calls
              db.addMessage(
                conversationId: conversationId,
                role: 'assistant',
                content: fullContent.toString(),
                toolCalls: pendingToolCalls,
              );

              // Execute each tool
              for (final tc in pendingToolCalls) {
                _sseEvent(controller, 'tool_start', {
                  'id': tc.id,
                  'name': tc.name,
                  'arguments': tc.arguments,
                });

                Map<String, dynamic> args;
                try {
                  args = jsonDecode(tc.arguments) as Map<String, dynamic>;
                } catch (_) {
                  args = {};
                }

                // Smart fallback: when LLM produces empty/incomplete arguments
                // (streaming gap, weak model), infer from tool name + user message.
                if (!args.containsKey('action') && tc.name != 'think') {
                  _log.warning(
                    'Tool "${tc.name}" called with empty/missing action: '
                    '${tc.arguments}',
                  );
                  final userMsgs = db.getMessages(conversationId)
                      .where((m) => m.role == 'user')
                      .toList();
                  final lastUserText = userMsgs.isNotEmpty
                      ? userMsgs.last.content
                          .replaceAll(RegExp(r'\[.*?\]'), '')
                          .trim()
                      : '';
                  if (lastUserText.isNotEmpty) {
                    args = _inferToolArgs(tc.name, lastUserText);
                    _log.info('Inferred args for "${tc.name}": $args');
                  }
                }

                final result = await _withKeepalive(
                  controller,
                  () => mcp.callTool(tc.name, args),
                );
                final resultText = McpClient.extractText(result);
                final images = McpClient.extractImages(result);

                // Extract + validate + dedup image URLs (image tool only)
                final isImageTool = tc.name == 'image';
                var validatedUrls = <String>[];
                if (isImageTool && totalImagesEmitted < maxImagesPerResponse) {
                  final raw = _extractImageUrls(resultText);
                  // Dedup against session
                  final fresh = <String>[];
                  for (final url in raw) {
                    final base = url.split('?').first.toLowerCase();
                    if (emittedImageUrls.add(base)) fresh.add(url);
                    if (fresh.length + totalImagesEmitted >= maxImagesPerResponse) break;
                  }
                  // Trust image tool URLs directly — the MCP image tool
                  // already validates them. HEAD requests fail on many CDNs
                  // that block HEAD or require specific referrer headers.
                  validatedUrls = fresh;
                  totalImagesEmitted += validatedUrls.length;
                }

                _sseEvent(controller, 'tool_result', {
                  'id': tc.id,
                  'name': tc.name,
                  'text': resultText,
                  'images': isImageTool ? images : <Map<String, dynamic>>[],
                  'imageUrls': validatedUrls,
                  'isError': result['isError'] ?? false,
                });

                // Sanitize before storing — LLM sees clean text, not raw data
                final cleanedResult = _sanitizeToolResult(resultText);
                db.addMessage(
                  conversationId: conversationId,
                  role: 'tool',
                  content: cleanedResult,
                  toolCallId: tc.id,
                );
              }
              db.updateTokenCount(conversationId);
              // Continue loop for next LLM iteration —
              // reset fullContent so next iteration starts fresh
              fullContent.clear();
              continue;
            }

            // This is the FINAL iteration (no tool calls) — now emit
            // the buffered tokens so the frontend renders the answer.
            if (!iterationHasToolCalls && iterationTokens.isNotEmpty) {
              _sseEvent(controller, 'token', {'text': iterationTokens.toString()});
            }

            // Determine final content — reasoning models (Qwen 3.5, etc.)
            // often put the real answer in reasoning_content and leave
            // content empty or trivially short (e.g. "Let me search...").
            // Use thinking content when it's substantially longer.
            var finalContent = fullContent.toString();
            final thinkingStr = thinkingContent.toString();
            if (thinkingStr.length > finalContent.length * 3 &&
                thinkingStr.length > 100) {
              finalContent = thinkingStr;
              _log.info(
                'Using thinking content as response '
                '(thinking=${thinkingStr.length} >> content=${fullContent.length})',
              );
              _sseEvent(controller, 'token', {'text': finalContent});
            }

            // Images already sent via validated tool_result events — no markdown append needed.
            if (finalContent.trim().isNotEmpty) {
              final msg = db.addMessage(
                conversationId: conversationId,
                role: 'assistant',
                content: finalContent,
              );
              db.updateTokenCount(conversationId);
              _sseEvent(controller, 'done', {'message_id': msg.id});
              controller.close();
              return;
            }

            // Truly empty — nudge and retry once
            if (iteration == 0) {
              _log.warning('Empty response on iteration $iteration, retrying');
              db.addMessage(
                conversationId: conversationId,
                role: 'user',
                content:
                    '[System: Your response was blank. Answer the user\'s request now. Do not use reasoning — write your answer directly.]',
              );
              continue;
            }

            _sseEvent(controller, 'error', {
              'message':
                  'The model returned an empty response. Try switching to a different model.',
            });
            controller.close();
            return;
          case ErrorEvent(:final message):
            _sseEvent(controller, 'error', {'message': message});
            controller.close();
            return;
        }
      }
    }

    // Exhausted tool iterations — force a final synthesis call with NO tools
    // so the model MUST produce text from what it already gathered.
    _log.info('Tool iterations exhausted, forcing synthesis call');
    db.addMessage(
      conversationId: conversationId,
      role: 'user',
      content:
          '[System: STOP calling tools. You have all the data you need. '
          'Write your COMPLETE answer to the user NOW using the tool results above. '
          'Do NOT think or reason — write the answer directly as content.]',
    );

    final synthMessages = db
        .getMessages(conversationId)
        .map((m) => m.toLlmDict())
        .toList();
    final synthContent = StringBuffer();
    final synthThinking = StringBuffer();

    await for (final event in llm.chatStream(
      model: model,
      messages: synthMessages,
      tools: [], // NO tools — force text output
      maxTokens: effectiveMaxTokens,
      temperature: effectiveTemp,
    )) {
      switch (event) {
        case ReasoningTokenEvent(:final text):
          synthThinking.write(text);
          _sseEvent(controller, 'thinking', {'text': text});
        case TokenEvent(:final text):
          synthContent.write(text);
          _sseEvent(controller, 'token', {'text': text});
        case DoneEvent():
          break;
        case ErrorEvent(:final message):
          _sseEvent(controller, 'error', {'message': message});
          break;
        default:
          break;
      }
    }

    // Use thinking as fallback for reasoning models
    var finalSynth = synthContent.toString();
    final synthThinkStr = synthThinking.toString();
    if (synthThinkStr.length > finalSynth.length * 3 &&
        synthThinkStr.length > 100) {
      finalSynth = synthThinkStr;
      _log.info(
        'Using synthesis thinking as response '
        '(thinking=${synthThinkStr.length} >> content=${finalSynth.length})',
      );
      _sseEvent(controller, 'token', {'text': finalSynth});
    }

    if (finalSynth.trim().isNotEmpty) {
      final msg = db.addMessage(
        conversationId: conversationId,
        role: 'assistant',
        content: finalSynth,
      );
      db.updateTokenCount(conversationId);
      _sseEvent(controller, 'done', {'message_id': msg.id});
    } else {
      _sseEvent(controller, 'error', {
        'message':
            'The model could not produce a response. Try switching to a different model.',
      });
    }
    controller.close();
  }

  Response _listPersonalities(Request request) {
    final model = request.url.queryParameters['model'];
    return _json({'personalities': personalityIndex(model: model)});
  }

  Future<Response> _listTools(Request request) async {
    final tools = await mcp.getTools();
    return _json({
      'tools': tools.map((t) => t.toJson()).toList(),
      'count': tools.length,
    });
  }

  /// Force re-initialize MCP connection and refresh all tools.
  /// Useful after MCP container restarts or stack changes.
  Future<Response> _refreshTools(Request request) async {
    _log.info('Manual tool refresh requested');
    final tools = await mcp.reinitialize();
    return _json({
      'status': tools.isNotEmpty ? 'ok' : 'error',
      'tools': tools.length,
      'initialized': mcp.isInitialized,
    });
  }

  // Model validation cache: model_id → capabilities
  final _modelCaps = <String, Map<String, dynamic>>{};

  Future<Response> _listModels(Request request) async {
    final models = await llm.listModels();

    // Fetch load-state info from v0 API and index by model ID
    final v0Models = await llm.listModelsV0();
    final v0ById = <String, Map<String, dynamic>>{};
    for (final m in v0Models) {
      final id = m['id'] as String? ?? '';
      if (id.isNotEmpty) v0ById[id] = m;
    }

    // Annotate each model with cached validation results, load state,
    // and per-model optimization profile.
    final annotated = models.map((m) {
      final id = m['id'] as String? ?? '';
      final caps = _modelCaps[id];
      final v0 = v0ById[id];
      final profile = getProfile(id);
      return {
        ...m,
        'validated': caps != null,
        if (caps != null) ...caps,
        if (v0 != null) 'state': v0['state'],
        if (v0 != null && v0.containsKey('type')) 'model_type': v0['type'],
        if (v0 != null && v0.containsKey('quantization'))
          'quantization': v0['quantization'],
        'profile': profile.toJson(),
      };
    }).toList();
    return _json({'models': annotated});
  }

  /// Warmup + validate a model by running e2e tests.
  /// Tests: (1) chat response, (2) tool calling ability.
  /// Results are cached so each model is only validated once per session.
  Future<Response> _warmupModel(Request request) async {
    final body = await _readJson(request);
    final model = body?['model'] as String?;
    if (model == null || model.isEmpty) {
      return _json({'error': 'model is required'}, status: 400);
    }

    // Skip if already validated
    if (_modelCaps.containsKey(model)) {
      _log.info('Model $model already validated');
      return _json({'status': 'ready', 'model': model, ..._modelCaps[model]!});
    }

    // Capacity guard: don't trigger warmup if loading this model would
    // evict another (and it's not already loaded).
    final warmupBusy = await llm.ensureModelOrBusy(
      model,
      maxLoaded: config.maxLoadedModels,
    );
    if (warmupBusy != null) {
      _log.info('Warmup skipped for $model — at capacity');
      return _json({
        'status': 'busy',
        'model': model,
        'message': warmupBusy,
      }, status: 503);
    }

    // Skip embedding models entirely
    if (model.toLowerCase().contains('embed')) {
      final caps = {
        'chat': false,
        'tools': false,
        'reasoning': false,
        'embedding': true,
        'limitation': 'Embedding model — not a chat model',
      };
      _modelCaps[model] = caps;
      return _json({'status': 'limited', 'model': model, ...caps});
    }

    _log.info('Validating model: $model');
    var chatOk = false;
    var toolsOk = false;
    var reasoning = false;
    String? limitation;

    // Test 1: Chat — can it produce a response?
    try {
      final result = await llm.chatOnce(
        model: model,
        messages: [
          {'role': 'user', 'content': 'What is 2+2? Answer in one word.'},
        ],
        maxTokens: 100,
        temperature: 0,
      );
      chatOk = result.isNotEmpty && !result.startsWith('[');
      if (result.contains('reasoning') || result.contains('think')) {
        reasoning = true;
      }
      _log.info(
        'Model $model chat test: ${chatOk ? "PASS" : "FAIL"} ($result)',
      );
    } catch (e) {
      _log.warning('Model $model chat test failed: $e');
      limitation = 'Cannot produce chat responses: $e';
    }

    // Test 2: Tools — can it make tool calls?
    if (chatOk) {
      try {
        final tools = await mcp.getTools();
        final openAiTools = tools.map((t) => t.toOpenAiFormat()).toList();

        var foundToolCall = false;
        await for (final event in llm.chatStream(
          model: model,
          messages: [
            {
              'role': 'system',
              'content': 'You have tools. Use the web tool to search.',
            },
            {'role': 'user', 'content': 'Search the web for "test query"'},
          ],
          tools: openAiTools,
          maxTokens: 200,
          temperature: 0,
        )) {
          if (event is ToolCallsEvent) {
            foundToolCall = true;
            break;
          }
          if (event is DoneEvent) break;
        }
        toolsOk = foundToolCall;
        if (!toolsOk) {
          limitation = 'This model cannot use tools — text-only responses';
        }
        _log.info('Model $model tool test: ${toolsOk ? "PASS" : "FAIL"}');
      } catch (e) {
        _log.warning('Model $model tool test failed: $e');
        limitation ??= 'Tool calling not supported';
      }
    }

    // Check if this is a reasoning model by name patterns
    final lm = model.toLowerCase();
    if (lm.contains('qwen3') ||
        lm.contains('magistral') ||
        lm.contains('reasoning') ||
        lm.contains('think') ||
        lm.contains('phi-4')) {
      reasoning = true;
    }

    final caps = <String, dynamic>{
      'chat': chatOk,
      'tools': toolsOk,
      'reasoning': reasoning,
      'embedding': false,
      if (limitation != null) 'limitation': limitation,
    };
    _modelCaps[model] = caps;

    // Store detected capabilities as a runtime profile override so future
    // requests use the auto-detected settings instead of heuristic defaults.
    final baseProfile = getProfile(model);
    setRuntimeProfile(
      model,
      ModelProfile(
        temperature: baseProfile.temperature,
        maxTokens: baseProfile.maxTokens,
        supportsTools: toolsOk,
        supportsReasoning: reasoning,
        systemPromptMaxChars: baseProfile.systemPromptMaxChars,
        notes: 'Auto-detected during warmup',
      ),
    );

    final status = chatOk ? 'ready' : 'error';
    _log.info('Model $model validated: $caps');
    return _json({'status': status, 'model': model, ...caps});
  }

  /// Best-effort model unload — LM Studio JIST handles this automatically,
  /// but we attempt to free resources when the user leaves the page.
  Future<Response> _unloadModel(Request request) async {
    final body = await _readJson(request);
    final model = body?['model'] as String?;
    if (model == null || model.isEmpty) {
      return _json({'status': 'skipped'});
    }
    _log.info('Unload requested for: $model (JIST auto-manages)');
    // LM Studio doesn't expose a public unload API —
    // JIST automatically unloads idle models. Log the intent.
    return _json({'status': 'acknowledged', 'model': model});
  }

  // ── Tool Result Sanitizer ─────────────────────────────────────────

  /// Clean tool results before feeding back to LLM to prevent
  /// raw data (base64, JSON dumps, binary) from leaking into responses.
  String _sanitizeToolResult(String text) {
    if (text.length < 200) return text;

    var cleaned = text;

    // Strip base64 data blocks (long alphanumeric strings 100+ chars)
    cleaned = cleaned.replaceAll(
      RegExp(r'[A-Za-z0-9+/=]{100,}'),
      '[binary data removed]',
    );

    // Strip data: URIs
    cleaned = cleaned.replaceAll(
      RegExp(r'data:[a-z/+]+;base64,[A-Za-z0-9+/=]+'),
      '[embedded data removed]',
    );

    // Strip raw byte strings like b'...'
    cleaned = cleaned.replaceAll(
      RegExp(r"b'[^']{50,}'"),
      '[binary data removed]',
    );

    // Strip very long repeated number sequences (coordinates, pixel data)
    cleaned = cleaned.replaceAll(
      RegExp(r'(\d{1,5}[-,. ]\s*){20,}'),
      '[numeric data removed] ',
    );

    // Strip raw hex dumps
    cleaned = cleaned.replaceAll(
      RegExp(r'(\\x[0-9a-fA-F]{2}){10,}'),
      '[hex data removed]',
    );

    // Strip very large JSON-looking blocks (nested braces with many fields)
    cleaned = cleaned.replaceAll(
      RegExp(r'\{[^{}]{5000,}\}'),
      '[data object removed]',
    );

    // Truncate if still too long (keep first 2000 chars)
    if (cleaned.length > 2000) {
      cleaned =
          '${cleaned.substring(0, 2000)}\n[... truncated ${cleaned.length - 2000} chars]';
    }

    return cleaned;
  }

  /// Extract HTTP image URLs from tool result text, filtering out junk.
  ///
  /// Skips: site logos, favicons, placeholder/default images, tracking pixels,
  /// tiny icons, and generic CDN chrome. Only returns URLs likely to be actual
  /// content images worth rendering.
  List<String> _extractImageUrls(String text) {
    final urls = <String>[];
    final pattern = RegExp(
      r'https?://[^\s"<>]+\.(?:png|jpg|jpeg|gif|webp)(?:\?[^\s"<>]*)?',
      caseSensitive: false,
    );
    for (final match in pattern.allMatches(text)) {
      final url = match.group(0)!;
      if (_isJunkImage(url)) continue;
      urls.add(url);
      if (urls.length >= 6) break;
    }
    return urls;
  }

  /// True if [url] looks like a logo, favicon, placeholder, or tracking pixel.
  /// Uses path-segment matching to avoid false positives from substring hits.
  bool _isJunkImage(String url) {
    final lower = url.toLowerCase();
    // Check path segments — more precise than substring
    final segments = Uri.tryParse(lower)?.pathSegments ?? lower.split('/');
    const junkSegments = {
      'logo', 'favicon', 'icon', 'avatar', 'placeholder',
      'pixel', 'tracking', 'beacon', 'spacer', 'blank', 'spinner',
      'loading', 'arrow', 'button', 'badge', 'sprite', 'emoji',
      'ads', 'ad', '1x1', '2x2',
    };
    for (final seg in segments) {
      if (junkSegments.contains(seg)) return true;
    }
    // Domain-level blocks
    const junkDomains = ['gravatar.com', 'googleusercontent.com/s/'];
    for (final d in junkDomains) {
      if (lower.contains(d)) return true;
    }
    // SVGs are almost always icons
    if (lower.endsWith('.svg')) return true;
    // Very short filenames (< 4 chars before extension) are usually icons
    final filename = url.split('/').last.split('?').first;
    if (filename.length < 4) return true;
    return false;
  }

  // ── Tool Argument Inference ──────────────────────────────────────

  /// Infer reasonable default arguments for a mega-tool when the LLM
  /// produced empty or incomplete arguments.
  Map<String, dynamic> _inferToolArgs(String toolName, String userText) {
    switch (toolName) {
      case 'web':
        return {'action': 'search', 'query': userText};
      case 'browser':
        return {'action': 'navigate', 'url': userText};
      case 'image':
        return {'action': 'search', 'query': userText, 'count': '6'};
      case 'research':
        return {'action': 'deep', 'question': userText};
      case 'data':
        return {'action': 'search', 'q': userText};
      case 'memory':
        return {'action': 'recall', 'pattern': userText};
      case 'knowledge':
        return {'action': 'search', 'query': userText};
      case 'vector':
        return {'action': 'search', 'query': userText};
      case 'code':
        return {'action': 'python', 'code': userText};
      case 'planner':
        return {'action': 'plan', 'task': userText};
      default:
        return {'action': 'search', 'query': userText};
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────

  /// Send an SSE comment to keep the connection alive through proxies.
  void _sseKeepalive(StreamController<List<int>> controller) {
    if (controller.isClosed) return;
    controller.add(utf8.encode(':keepalive\n\n'));
  }

  /// Run an async operation while sending SSE keepalives every 15 seconds.
  /// Prevents proxy timeouts (524) during long tool calls.
  Future<T> _withKeepalive<T>(
    StreamController<List<int>> controller,
    Future<T> Function() work,
  ) async {
    final timer = Timer.periodic(
      const Duration(seconds: 15),
      (_) => _sseKeepalive(controller),
    );
    try {
      return await work();
    } finally {
      timer.cancel();
    }
  }

  void _sseEvent(
    StreamController<List<int>> controller,
    String event,
    Map<String, dynamic> data,
  ) {
    if (controller.isClosed) return;
    final payload = 'event: $event\ndata: ${jsonEncode(data)}\n\n';
    controller.add(utf8.encode(payload));
  }

  Response _json(Map<String, dynamic> data, {int status = 200}) {
    return Response(
      status,
      body: jsonEncode(data),
      headers: {'Content-Type': 'application/json'},
    );
  }

  Future<Map<String, dynamic>?> _readJson(Request request) async {
    try {
      final body = await request.readAsString();
      if (body.isEmpty) return {};
      return jsonDecode(body) as Map<String, dynamic>;
    } catch (_) {
      return null;
    }
  }
}
