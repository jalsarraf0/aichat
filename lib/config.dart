import 'dart:io';

import 'personalities.dart';

class Config {
  final String lmStudioUrl;
  final String lmStudioFallbackUrl;
  final String mcpUrl;
  final int port;
  final String dbPath;
  final String model;
  final double temperature;
  final int maxTokens;
  final int maxToolIterations;
  final int compactionThreshold;
  final int compactionKeepRecent;
  final Duration toolCacheTtl;
  final String systemPrompt;
  final String webDir;
  final int maxLoadedModels;

  /// URL for an external tool-routing model. Empty = use built-in rules.
  final String toolRouterUrl;

  Config({
    required this.lmStudioUrl,
    required this.lmStudioFallbackUrl,
    required this.mcpUrl,
    required this.port,
    required this.dbPath,
    required this.model,
    required this.temperature,
    required this.maxTokens,
    required this.maxToolIterations,
    required this.compactionThreshold,
    required this.compactionKeepRecent,
    required this.toolCacheTtl,
    required this.systemPrompt,
    required this.webDir,
    required this.maxLoadedModels,
    required this.toolRouterUrl,
  });

  factory Config.fromEnv() {
    final env = Platform.environment;
    return Config(
      lmStudioUrl: env['LM_STUDIO_URL'] ?? 'http://192.168.50.2:1234',
      lmStudioFallbackUrl:
          env['LM_STUDIO_FALLBACK_URL'] ?? 'http://100.78.39.76:1234',
      mcpUrl: env['MCP_URL'] ?? 'http://localhost:8096',
      port: int.tryParse(env['PORT'] ?? '') ?? 8200,
      dbPath: env['DB_PATH'] ?? 'dartboard.db',
      model: env['MODEL'] ?? '',
      temperature: double.tryParse(env['TEMPERATURE'] ?? '') ?? 0.7,
      maxTokens: int.tryParse(env['MAX_TOKENS'] ?? '') ?? 4096,
      maxToolIterations: int.tryParse(env['MAX_TOOL_ITERATIONS'] ?? '') ?? 4,
      compactionThreshold:
          int.tryParse(env['COMPACTION_THRESHOLD'] ?? '') ?? 24000,
      compactionKeepRecent:
          int.tryParse(env['COMPACTION_KEEP_RECENT'] ?? '') ?? 6,
      toolCacheTtl: Duration(
        seconds: int.tryParse(env['TOOL_CACHE_TTL'] ?? '') ?? 60,
      ),
      systemPrompt: env['SYSTEM_PROMPT'] ?? buildSystemPrompt('general'),
      webDir: env['WEB_DIR'] ?? 'web',
      maxLoadedModels: int.tryParse(env['LM_STUDIO_MAX_LOADED'] ?? '') ?? 2,
      toolRouterUrl: env['TOOL_ROUTER_URL'] ?? '',
    );
  }
}
