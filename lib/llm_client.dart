import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:logging/logging.dart';
import 'models.dart';

final _log = Logger('LlmClient');

/// Events yielded during a streaming chat call.
sealed class ChatEvent {}

class TokenEvent extends ChatEvent {
  final String text;
  TokenEvent(this.text);
}

class ReasoningTokenEvent extends ChatEvent {
  final String text;
  ReasoningTokenEvent(this.text);
}

class ToolCallsEvent extends ChatEvent {
  final List<ToolCallData> toolCalls;
  ToolCallsEvent(this.toolCalls);
}

class DoneEvent extends ChatEvent {
  final String? finishReason;
  DoneEvent(this.finishReason);
}

class ErrorEvent extends ChatEvent {
  final String message;
  ErrorEvent(this.message);
}

class LlmClient {
  String baseUrl;
  final String? fallbackUrl;
  final http.Client _client;

  LlmClient({required this.baseUrl, this.fallbackUrl, http.Client? client})
    : _client = client ?? http.Client();

  void close() => _client.close();

  /// HEAD request for URL validation (e.g., checking if an image exists).
  Future<http.Response> head(Uri url, {Duration? timeout}) =>
      _client.head(url).timeout(timeout ?? const Duration(seconds: 5));

  /// Try primary URL, switch to fallback on connection failure.
  Future<void> _tryFallback() async {
    if (fallbackUrl == null || fallbackUrl == baseUrl) return;
    try {
      final r = await _client
          .get(Uri.parse('$fallbackUrl/v1/models'))
          .timeout(const Duration(seconds: 3));
      if (r.statusCode == 200) {
        _log.info('Switching to fallback LM Studio URL: $fallbackUrl');
        baseUrl = fallbackUrl!;
      }
    } catch (_) {}
  }

  /// Check if a model is currently loaded in LM Studio (via v0 API).
  Future<bool> isModelLoaded(String model) async {
    final models = await listModelsV0();
    return models.any(
      (m) => m['id'] == model && m['state'] == 'loaded',
    );
  }

  /// Quick non-streaming check if a model responds.
  /// Covers both preflight (loaded model, short timeout) and JIST
  /// loading (not-loaded model, longer timeout).
  Future<bool> isModelResponsive(String model,
      {Duration timeout = const Duration(seconds: 10)}) async {
    try {
      final response = await _client
          .post(
            Uri.parse('$baseUrl/v1/chat/completions'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode({
              'model': model,
              'messages': [
                {'role': 'user', 'content': 'ok'},
              ],
              'max_tokens': 1,
              'temperature': 0,
            }),
          )
          .timeout(timeout);
      if (response.statusCode != 200) {
        _log.warning('Model $model check: status ${response.statusCode}');
        return false;
      }
      return true;
    } catch (e) {
      _log.warning('Model $model check failed: $e');
      return false;
    }
  }

  /// Stream chat completion events from LM Studio.
  Stream<ChatEvent> chatStream({
    required String model,
    required List<Map<String, dynamic>> messages,
    List<Map<String, dynamic>>? tools,
    String toolChoice = 'auto',
    int maxTokens = 4096,
    double temperature = 0.7,
  }) async* {
    final payload = <String, dynamic>{
      'model': model,
      'messages': messages,
      'stream': true,
      'max_tokens': maxTokens,
      'temperature': temperature,
    };
    if (tools != null && tools.isNotEmpty) {
      payload['tools'] = tools;
      payload['tool_choice'] = toolChoice;
    }

    final request = http.Request(
      'POST',
      Uri.parse('$baseUrl/v1/chat/completions'),
    );
    request.headers['Content-Type'] = 'application/json';
    request.body = jsonEncode(payload);

    http.StreamedResponse response;
    try {
      response = await _client
          .send(request)
          .timeout(const Duration(seconds: 120));
    } catch (e) {
      // Try fallback URL on connection failure
      if (fallbackUrl != null && fallbackUrl != baseUrl) {
        _log.info('Primary LM Studio failed, trying fallback...');
        await _tryFallback();
        final retryRequest = http.Request(
          'POST',
          Uri.parse('$baseUrl/v1/chat/completions'),
        );
        retryRequest.headers['Content-Type'] = 'application/json';
        retryRequest.body = request.body;
        try {
          response = await _client
              .send(retryRequest)
              .timeout(const Duration(seconds: 120));
        } catch (e2) {
          yield ErrorEvent('LM Studio connection failed (both URLs): $e2');
          return;
        }
      } else {
        yield ErrorEvent('LM Studio connection failed: $e');
        return;
      }
    }

    if (response.statusCode != 200) {
      final body = await response.stream.bytesToString();
      yield ErrorEvent('LM Studio error ${response.statusCode}: $body');
      return;
    }

    // Accumulate partial tool calls across SSE chunks
    final toolCallAccum = <int, _ToolCallAccum>{};
    var buffer = '';

    // Timeout if no SSE chunk arrives within 60 seconds.
    // JIST model loading takes 12-25s, plus first inference with large
    // system prompts and 16 tool definitions can take another 15-25s.
    final timedStream = response.stream
        .transform(utf8.decoder)
        .timeout(const Duration(seconds: 90), onTimeout: (sink) {
      _log.warning('SSE stream timeout — no data for 90s (model: $model)');
      sink.addError(TimeoutException('No data from LLM for 90 seconds'));
      sink.close();
    });

    try {
    await for (final chunk in timedStream) {
      buffer += chunk;
      final lines = buffer.split('\n');
      // Keep the last incomplete line in the buffer
      buffer = lines.removeLast();

      for (final line in lines) {
        if (line.isEmpty || line.startsWith(':')) continue;
        if (!line.startsWith('data: ')) continue;
        final data = line.substring(6).trim();
        if (data == '[DONE]') {
          // Flush accumulated tool calls
          if (toolCallAccum.isNotEmpty) {
            yield ToolCallsEvent(
              toolCallAccum.values.map((a) {
                var args = a.arguments.toString();
                if (args.isEmpty) {
                  _log.warning('Tool "${a.name}" has empty arguments (streaming gap)');
                  args = '{}';
                }
                return ToolCallData(id: a.id, name: a.name, arguments: args);
              }).toList(),
            );
            toolCallAccum.clear();
          }
          yield DoneEvent('stop');
          return;
        }

        Map<String, dynamic> parsed;
        try {
          parsed = jsonDecode(data) as Map<String, dynamic>;
        } catch (_) {
          continue;
        }

        final choices = parsed['choices'] as List?;
        if (choices == null || choices.isEmpty) continue;
        final choice = choices[0] as Map<String, dynamic>;
        final delta = choice['delta'] as Map<String, dynamic>?;
        final finishReason = choice['finish_reason'] as String?;

        if (delta != null) {
          // Reasoning content token (thinking models like Qwen 3.5, phi-4, etc.)
          final reasoning = delta['reasoning_content'] as String?;
          if (reasoning != null && reasoning.isNotEmpty) {
            yield ReasoningTokenEvent(reasoning);
          }

          // Content token
          final content = delta['content'] as String?;
          if (content != null && content.isNotEmpty) {
            yield TokenEvent(content);
          }

          // Tool call deltas
          final tcDeltas = delta['tool_calls'] as List?;
          if (tcDeltas != null) {
            for (final tc in tcDeltas) {
              final tcMap = Map<String, dynamic>.from(tc as Map);
              final idx = tcMap['index'] as int? ?? 0;
              final accum = toolCallAccum.putIfAbsent(
                idx,
                () => _ToolCallAccum(),
              );
              if (tcMap.containsKey('id')) {
                accum.id = tcMap['id'] as String;
              }
              final fn = tcMap['function'] as Map<String, dynamic>?;
              if (fn != null) {
                if (fn.containsKey('name')) {
                  accum.name = fn['name'] as String;
                }
                if (fn.containsKey('arguments')) {
                  accum.arguments.write(fn['arguments'] as String);
                }
              }
            }
          }
        }

        if (finishReason == 'tool_calls') {
          if (toolCallAccum.isNotEmpty) {
            yield ToolCallsEvent(
              toolCallAccum.values.map((a) {
                // Ensure arguments is valid JSON — LM Studio crashes
                // if the assistant message has empty-string arguments.
                var args = a.arguments.toString();
                if (args.isEmpty) {
                  _log.warning('Tool "${a.name}" has empty arguments (streaming gap)');
                  args = '{}';
                }
                return ToolCallData(id: a.id, name: a.name, arguments: args);
              }).toList(),
            );
            toolCallAccum.clear();
          }
          yield DoneEvent('tool_calls');
          return;
        } else if (finishReason == 'stop') {
          yield DoneEvent('stop');
          return;
        }
      }
    }

    // Process any remaining data left in the buffer after stream ends
    if (buffer.trim().isNotEmpty) {
      for (final line in buffer.split('\n')) {
        if (line.isEmpty || line.startsWith(':')) continue;
        if (!line.startsWith('data: ')) continue;
        final data = line.substring(6).trim();
        if (data == '[DONE]') {
          if (toolCallAccum.isNotEmpty) {
            yield ToolCallsEvent(
              toolCallAccum.values.map((a) {
                var args = a.arguments.toString();
                if (args.isEmpty) {
                  _log.warning('Tool "${a.name}" has empty arguments (streaming gap)');
                  args = '{}';
                }
                return ToolCallData(id: a.id, name: a.name, arguments: args);
              }).toList(),
            );
          }
          yield DoneEvent('stop');
          return;
        }
      }
    }
    } on TimeoutException {
      yield ErrorEvent(
        '$model is not responding — it may be loading or hung. '
        'Try switching to a different model.',
      );
    }
  }

  /// Non-streaming chat (used for compaction summarization).
  Future<String> chatOnce({
    required String model,
    required List<Map<String, dynamic>> messages,
    int maxTokens = 2048,
    double temperature = 0.3,
  }) async {
    final payload = {
      'model': model,
      'messages': messages,
      'stream': false,
      'max_tokens': maxTokens,
      'temperature': temperature,
    };
    try {
      final response = await _client
          .post(
            Uri.parse('$baseUrl/v1/chat/completions'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(payload),
          )
          .timeout(const Duration(seconds: 60));
      if (response.statusCode != 200) {
        return '[Summarization failed: HTTP ${response.statusCode}]';
      }
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final choices = data['choices'] as List?;
      if (choices == null || choices.isEmpty) return '[No response]';
      final msg = (choices[0] as Map)['message'] as Map?;
      // Reasoning models may return content in reasoning_content
      final content = msg?['content'] as String?;
      final reasoning = msg?['reasoning_content'] as String?;
      if (content != null && content.isNotEmpty) return content;
      if (reasoning != null && reasoning.isNotEmpty) return reasoning;
      return '[Empty response]';
    } catch (e) {
      return '[Summarization error: $e]';
    }
  }

  /// List available models from LM Studio (v1 API — no load state).
  Future<List<Map<String, dynamic>>> listModels() async {
    try {
      final response = await _client
          .get(Uri.parse('$baseUrl/v1/models'))
          .timeout(const Duration(seconds: 10));
      if (response.statusCode != 200) return [];
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return List<Map<String, dynamic>>.from(data['data'] as List? ?? []);
    } catch (e) {
      _log.warning('listModels failed: $e');
      return [];
    }
  }

  /// List models with load state from LM Studio v0 API.
  /// Returns the full model entries (with `state`, `type`, etc.).
  Future<List<Map<String, dynamic>>> listModelsV0() async {
    try {
      final response = await _client
          .get(Uri.parse('$baseUrl/api/v0/models'))
          .timeout(const Duration(seconds: 10));
      if (response.statusCode != 200) {
        _log.warning('listModelsV0 returned ${response.statusCode}');
        return [];
      }
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      return List<Map<String, dynamic>>.from(data['data'] as List? ?? []);
    } catch (e) {
      _log.warning('listModelsV0 failed: $e');
      return [];
    }
  }

  /// Return IDs of loaded non-embedding models from /api/v0/models.
  Future<List<String>> listLoadedModels() async {
    final models = await listModelsV0();
    return models
        .where((m) {
          final state = m['state'] as String? ?? '';
          final type = (m['type'] as String? ?? '').toLowerCase();
          return state == 'loaded' && !type.contains('embed');
        })
        .map((m) => m['id'] as String? ?? '')
        .where((id) => id.isNotEmpty)
        .toList();
  }

  /// True if the number of loaded models >= [maxLoaded].
  Future<bool> isAtCapacity({int maxLoaded = 2}) async {
    final loaded = await listLoadedModels();
    return loaded.length >= maxLoaded;
  }

  /// Check whether [model] can be used right now.
  ///
  /// Returns `null` if OK to proceed (model is loaded, or there is capacity
  /// for LM Studio to auto-load it). Returns an error message string if the
  /// model is not loaded and all slots are occupied.
  /// Check if adding this model would exceed capacity.
  /// Returns null (OK) in most cases — JIST handles model swapping.
  /// Only rejects if LM Studio is completely unreachable.
  Future<String?> ensureModelOrBusy(String model, {int maxLoaded = 2}) async {
    try {
      // Just verify LM Studio is reachable (longer timeout when JIST is busy)
      await _client
          .get(Uri.parse('$baseUrl/v1/models'))
          .timeout(const Duration(seconds: 15));
      return null; // LM Studio is up — JIST will handle loading/swapping
    } catch (e) {
      return 'LM Studio is not reachable: $e';
    }
  }
}

class _ToolCallAccum {
  String id = '';
  String name = '';
  final arguments = StringBuffer();
}
