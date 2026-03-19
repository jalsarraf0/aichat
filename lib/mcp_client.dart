import 'dart:async';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:logging/logging.dart';
import 'models.dart';

final _log = Logger('McpClient');

class McpClient {
  final String baseUrl;
  final http.Client _client;
  List<McpTool>? _cachedTools;
  DateTime? _cacheTime;
  final Duration cacheTtl;

  /// True after a successful `initialize` handshake.
  bool _initialized = false;

  static int _nextId = 3;

  McpClient({
    required this.baseUrl,
    http.Client? client,
    this.cacheTtl = const Duration(seconds: 60),
  }) : _client = client ?? http.Client();

  void close() => _client.close();

  /// Whether the MCP session has been initialized.
  bool get isInitialized => _initialized;

  /// How many tools are currently cached.
  int get toolCount => _cachedTools?.length ?? 0;

  // ── Initialize ──────────────────────────────────────────────────────

  /// Single initialize attempt.  Returns true on success.
  Future<bool> _doInitialize() async {
    try {
      final body = jsonEncode({
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'initialize',
        'params': {
          'protocolVersion': '2024-11-05',
          'capabilities': {},
          'clientInfo': {'name': 'dartboard', 'version': '1.0.0'},
        },
      });
      final response = await _client
          .post(
            Uri.parse('$baseUrl/mcp'),
            headers: {'Content-Type': 'application/json'},
            body: body,
          )
          .timeout(const Duration(seconds: 10));
      if (response.statusCode == 200) {
        _initialized = true;
        _log.info('MCP initialized successfully');
        return true;
      }
      _log.warning('MCP initialize returned ${response.statusCode}');
      return false;
    } catch (e) {
      _log.warning('MCP initialize failed: $e');
      return false;
    }
  }

  /// Initialize with retries.  Tries [maxAttempts] times with exponential
  /// backoff (1s, 2s, 4s, 8s …).  Used at startup to wait for the MCP
  /// container to become healthy.
  Future<bool> initialize({int maxAttempts = 10}) async {
    for (var i = 1; i <= maxAttempts; i++) {
      if (await _doInitialize()) return true;
      if (i < maxAttempts) {
        final delay = Duration(seconds: 1 << (i - 1).clamp(0, 4)); // max 16s
        _log.info('MCP init retry $i/$maxAttempts in ${delay.inSeconds}s');
        await Future<void>.delayed(delay);
      }
    }
    _log.severe('MCP initialization failed after $maxAttempts attempts');
    return false;
  }

  /// Force a full re-initialization: re-handshake + refresh tool list.
  /// Returns the refreshed tool list.
  Future<List<McpTool>> reinitialize() async {
    _initialized = false;
    _cachedTools = null;
    _cacheTime = null;
    await initialize(maxAttempts: 5);
    return getTools(forceRefresh: true);
  }

  // ── Auto-recovery ───────────────────────────────────────────────────

  /// Attempt to recover the MCP connection.  Called automatically when
  /// getTools or callTool encounters a connection failure.
  Future<bool> _autoRecover() async {
    _log.info('MCP connection lost — attempting auto-recovery');
    _initialized = false;
    _cachedTools = null;
    _cacheTime = null;
    return _doInitialize();
  }

  // ── Get Tools ───────────────────────────────────────────────────────

  Future<List<McpTool>> getTools({bool forceRefresh = false}) async {
    if (!forceRefresh &&
        _cachedTools != null &&
        _cacheTime != null &&
        DateTime.now().difference(_cacheTime!) < cacheTtl) {
      return _cachedTools!;
    }

    // If not yet initialized (or lost connection), try to initialize first
    if (!_initialized) {
      await _doInitialize();
    }

    try {
      final tools = await _fetchTools();
      if (tools != null) return tools;

      // Fetch failed — try recovery once
      if (await _autoRecover()) {
        final retry = await _fetchTools();
        if (retry != null) return retry;
      }

      _log.warning('tools/list failed after recovery — returning cache');
      return _cachedTools ?? [];
    } catch (e) {
      _log.warning('getTools error: $e');
      return _cachedTools ?? [];
    }
  }

  /// Raw tools/list call.  Returns null on failure.
  Future<List<McpTool>?> _fetchTools() async {
    try {
      final body = jsonEncode({
        'jsonrpc': '2.0',
        'id': 2,
        'method': 'tools/list',
        'params': {},
      });
      final response = await _client
          .post(
            Uri.parse('$baseUrl/mcp'),
            headers: {'Content-Type': 'application/json'},
            body: body,
          )
          .timeout(const Duration(seconds: 15));
      if (response.statusCode != 200) {
        _log.warning('tools/list returned ${response.statusCode}');
        return null;
      }
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final result = data['result'] as Map<String, dynamic>?;
      if (result == null) return null;
      final tools = (result['tools'] as List)
          .map((t) => McpTool.fromJson(Map<String, dynamic>.from(t as Map)))
          .toList();
      _cachedTools = tools;
      _cacheTime = DateTime.now();
      _log.info('Loaded ${tools.length} MCP tools');
      return tools;
    } catch (e) {
      _log.warning('_fetchTools failed: $e');
      return null;
    }
  }

  // ── Call Tool ───────────────────────────────────────────────────────

  Future<Map<String, dynamic>> callTool(
    String name,
    Map<String, dynamic> arguments,
  ) async {
    var result = await _doCallTool(name, arguments);

    // If the call failed with a connection error, try recovery + retry once
    if (result['isError'] == true) {
      final msg = _textFromResult(result);
      if (msg.contains('MCP call failed:') ||
          msg.contains('Connection') ||
          msg.contains('timed out')) {
        _log.info('Tool call "$name" failed — attempting recovery');
        if (await _autoRecover()) {
          result = await _doCallTool(name, arguments);
        }
      }
    }

    return result;
  }

  Future<Map<String, dynamic>> _doCallTool(
    String name,
    Map<String, dynamic> arguments,
  ) async {
    final body = jsonEncode({
      'jsonrpc': '2.0',
      'id': _nextId++,
      'method': 'tools/call',
      'params': {'name': name, 'arguments': arguments},
    });
    try {
      final response = await _client
          .post(
            Uri.parse('$baseUrl/mcp'),
            headers: {'Content-Type': 'application/json'},
            body: body,
          )
          .timeout(const Duration(seconds: 120));
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final result = data['result'] as Map<String, dynamic>?;
      if (result != null) return result;
      final error = data['error'] as Map<String, dynamic>?;
      return {
        'content': [
          {'type': 'text', 'text': error?['message'] ?? 'Unknown MCP error'},
        ],
        'isError': true,
      };
    } catch (e) {
      return {
        'content': [
          {'type': 'text', 'text': 'MCP call failed: $e'},
        ],
        'isError': true,
      };
    }
  }

  String _textFromResult(Map<String, dynamic> result) {
    final content = result['content'] as List?;
    if (content == null || content.isEmpty) return '';
    final first = Map<String, dynamic>.from(content.first as Map);
    return first['text'] as String? ?? '';
  }

  // ── Static helpers ──────────────────────────────────────────────────

  /// Extract text content from an MCP tool result
  static String extractText(Map<String, dynamic> result) {
    final content = result['content'] as List?;
    if (content == null) return jsonEncode(result);
    final parts = <String>[];
    for (final block in content) {
      final m = Map<String, dynamic>.from(block as Map);
      if (m['type'] == 'text') {
        parts.add(m['text'] as String);
      } else if (m['type'] == 'image') {
        parts.add('[image: ${m['mimeType'] ?? 'unknown'}]');
      }
    }
    return parts.join('\n');
  }

  /// Extract image blocks from an MCP tool result
  static List<Map<String, String>> extractImages(Map<String, dynamic> result) {
    final content = result['content'] as List?;
    if (content == null) return [];
    final images = <Map<String, String>>[];
    for (final block in content) {
      final m = Map<String, dynamic>.from(block as Map);
      if (m['type'] == 'image' && m['data'] != null) {
        images.add({
          'data': m['data'] as String,
          'mimeType': m['mimeType'] as String? ?? 'image/png',
        });
      }
    }
    return images;
  }
}
