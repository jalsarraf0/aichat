/// Tool router for the Dartboard AI Lab.
///
/// Two modes:
/// 1. **Model-based** (when [routerUrl] is set): Sends the user message to a
///    small local LLM (Qwen2.5-3B on Intel Arc A380) for classification.
///    ~500-1100ms latency, high accuracy.
/// 2. **Rule-based** (fallback): Keyword pattern matching, 0ms, good accuracy.
///
/// The model router falls back to rules on timeout or error.
library;

import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:logging/logging.dart';
import 'models.dart';

final _log = Logger('ToolRouter');

/// Maximum tools to send to a single LLM request.
const _maxTools = 3;

/// Valid tool names for filtering model output.
const _validTools = {
  'web', 'image', 'browser', 'research', 'code', 'document',
  'media', 'memory', 'knowledge', 'vector', 'data', 'planner',
  'think', 'system', 'jobs', 'custom_tools',
};

// ── Model-based routing ─────────────────────────────────────────────

const _routerSystemPrompt =
    'You are a tool router. Given the user message, pick 1-3 tools from: '
    'web, image, browser, research, code, document, media, memory, '
    'knowledge, vector, data, planner. '
    'Reply with ONLY comma-separated tool names. No explanation.';

/// Try model-based routing. Returns null on failure (caller falls back to rules).
Future<Set<String>?> _modelRoute(
  String routerUrl,
  String userMessage,
  http.Client client,
) async {
  try {
    final response = await client
        .post(
          Uri.parse('$routerUrl/v1/chat/completions'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'model': 'qwen2.5-3b-instruct',
            'messages': [
              {'role': 'system', 'content': _routerSystemPrompt},
              {'role': 'user', 'content': userMessage},
            ],
            'max_tokens': 30,
            'temperature': 0,
          }),
        )
        .timeout(const Duration(seconds: 3));

    if (response.statusCode != 200) return null;

    final data = jsonDecode(response.body) as Map<String, dynamic>;
    final content = ((data['choices'] as List?)?.first
        as Map<String, dynamic>?)?['message']?['content'] as String?;
    if (content == null || content.isEmpty) return null;

    // Parse comma-separated tool names, filter to valid ones only
    final tools = content
        .toLowerCase()
        .split(RegExp(r'[,\s]+'))
        .map((s) => s.trim())
        .where((s) => _validTools.contains(s))
        .toSet();

    if (tools.isEmpty) return null;
    _log.info('Model routed: ${tools.join(", ")} (raw: "$content")');
    return tools;
  } catch (e) {
    _log.warning('Model router failed, falling back to rules: $e');
    return null;
  }
}

// ── Rule-based routing ──────────────────────────────────────────────

const _rules = <String, List<String>>{
  'image': [
    'image', 'photo', 'picture', 'pic ', 'pics ', 'artwork', 'fanart',
    'wallpaper', 'screenshot', 'illustration', 'drawing',
    'show me', 'find images', 'search for images',
  ],
  'web': [
    'news', 'latest', 'today', 'current', 'headline', 'recent',
    'search', 'find', 'look up', 'what is', 'what are', 'who is',
    'when did', 'where is', 'how to', 'tell me about', 'explain',
    'update', 'stock', 'weather', 'score', 'result',
  ],
  'browser': [
    'browse', 'navigate', 'website', 'open the', 'go to',
    'visit', 'scrape', 'read the page',
  ],
  'research': [
    'research', 'deep dive', 'investigate', 'analyze', 'study',
    'comprehensive', 'in-depth', 'thorough',
  ],
  'code': [
    'code', 'python', 'script', 'program', 'javascript',
    'execute', 'run', 'compile', 'function', 'algorithm',
  ],
  'document': [
    'document', 'pdf', 'ocr', 'scan', 'read the file',
    'extract text', 'parse',
  ],
  'media': [
    'video', 'youtube', 'watch', 'clip', 'audio', 'tts',
    'speak', 'voice', 'detect object',
  ],
  'memory': [
    'remember', 'recall', 'memory', 'store this', 'save this',
    'don\'t forget', 'my name is', 'i prefer',
  ],
  'knowledge': [
    'knowledge', 'graph', 'relation', 'connected to', 'link between',
  ],
  'vector': [
    'vector', 'embed', 'similar to', 'semantic search',
  ],
  'data': [
    'database', 'stored data', 'cached', 'article', 'bookmark',
  ],
  'planner': [
    'plan', 'organize', 'schedule', 'steps to', 'break down',
    'task list', 'workflow',
  ],
};

Set<String> _ruleRoute(String userMessage, Set<String> availableNames) {
  final msg = userMessage.toLowerCase();
  final matched = <String>{};

  for (final entry in _rules.entries) {
    if (!availableNames.contains(entry.key)) continue;
    for (final keyword in entry.value) {
      if (msg.contains(keyword)) {
        matched.add(entry.key);
        break;
      }
    }
  }

  // Image-specific requests: send ONLY the image tool so the model
  // can't choose web (which returns page links, not actual images).
  // The image tool does its own SearXNG/DDG/Bing image search.
  if (matched.contains('image') && !matched.contains('research')) {
    return {'image'};
  }

  // Research requests get web as backup
  if (matched.contains('research') && availableNames.contains('web')) {
    matched.add('web');
  }

  // Fallback: if nothing matched, use web
  if (matched.isEmpty && availableNames.contains('web')) {
    matched.add('web');
  }

  return matched;
}

// ── Public API ──────────────────────────────────────────────────────

/// Select the most relevant tools for a user message.
///
/// Uses model-based routing when [routerUrl] is set, falls back to rules.
/// Returns a filtered list from [available] tools, capped at [_maxTools].
Future<List<McpTool>> selectTools(
  String userMessage,
  List<McpTool> available, {
  String? routerUrl,
  http.Client? client,
}) async {
  final availableNames = available.map((t) => t.name).toSet();
  Set<String> selected;

  // Try model-based routing first
  if (routerUrl != null && routerUrl.isNotEmpty) {
    final modelResult = await _modelRoute(
      routerUrl,
      userMessage,
      client ?? http.Client(),
    );
    if (modelResult != null) {
      // Intersect with available tools
      selected = modelResult.intersection(availableNames);
      if (selected.isNotEmpty) {
        // Ensure image gets web backup
        if (selected.contains('image') && availableNames.contains('web')) {
          selected.add('web');
        }
        final capped = selected.take(_maxTools).toSet();
        _log.info('GPU routed ${capped.length} tools: ${capped.join(", ")}');
        return available.where((t) => capped.contains(t.name)).toList();
      }
    }
    // Model failed or returned nothing valid — fall through to rules
  }

  // Rule-based routing
  selected = _ruleRoute(userMessage, availableNames);
  final capped = selected.take(_maxTools).toSet();
  _log.info(
    'Rule routed ${capped.length} tools: ${capped.join(", ")} '
    '(from ${available.length} available)',
  );
  // Sort so image tool comes first when present — models with
  // tool_choice=required tend to pick the first tool in the list.
  final result = available.where((t) => capped.contains(t.name)).toList();
  if (capped.contains('image')) {
    result.sort((a, b) => a.name == 'image' ? -1 : b.name == 'image' ? 1 : 0);
  }
  return result;
}

// ── Prompt Optimization ─────────────────────────────────────────────

/// Cache: personality_id + promptSize → optimized prompt
final _promptCache = <String, String>{};

const _condenserSystemPrompt =
    'You are a prompt optimizer. Condense the following AI personality description '
    'into a clear, direct system prompt under 500 characters. Keep the core '
    'personality traits and directives. Remove filler. Output ONLY the condensed prompt.';

/// Optimize a system prompt for a specific model's capabilities.
///
/// For condensed models: uses the Arc A380 LLM to generate a short version.
/// For full models: returns unchanged.
/// Results are cached per personality+size combo.
Future<String> optimizePrompt(
  String systemPrompt, {
  required String promptSize,
  required String personalityId,
  String? routerUrl,
  http.Client? client,
}) async {
  // Full-size models get the prompt unchanged
  if (promptSize != 'condensed') return systemPrompt;

  // Check cache
  final cacheKey = '${personalityId}_$promptSize';
  final cached = _promptCache[cacheKey];
  if (cached != null) return cached;

  // No router URL — return as-is (rule-based fallback)
  if (routerUrl == null || routerUrl.isEmpty) return systemPrompt;

  // Ask Arc A380 to condense the prompt
  try {
    final response = await (client ?? http.Client())
        .post(
          Uri.parse('$routerUrl/v1/chat/completions'),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'model': 'qwen2.5-3b-instruct',
            'messages': [
              {'role': 'system', 'content': _condenserSystemPrompt},
              {'role': 'user', 'content': systemPrompt},
            ],
            'max_tokens': 200,
            'temperature': 0.3,
          }),
        )
        .timeout(const Duration(seconds: 5));

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final content = ((data['choices'] as List?)?.first
          as Map<String, dynamic>?)?['message']?['content'] as String?;
      if (content != null && content.length > 50) {
        _log.info('Optimized prompt for $personalityId ($promptSize): '
            '${systemPrompt.length}→${content.length} chars');
        _promptCache[cacheKey] = content;
        return content;
      }
    }
  } catch (e) {
    _log.warning('Prompt optimization failed, using original: $e');
  }

  return systemPrompt;
}
