import 'dart:io';

import 'package:logging/logging.dart';
import 'package:shelf/shelf.dart';
import 'package:shelf/shelf_io.dart' as io;

import 'package:dartboard/compaction.dart';
import 'package:dartboard/config.dart';
import 'package:dartboard/database.dart';
import 'package:dartboard/llm_client.dart';
import 'package:dartboard/mcp_client.dart';
import 'package:dartboard/router.dart';

void main(List<String> args) async {
  Logger.root.level = Level.INFO;
  Logger.root.onRecord.listen((record) {
    stderr.writeln(
      '${record.time.toIso8601String()} [${record.level.name}] '
      '${record.loggerName}: ${record.message}',
    );
  });

  final log = Logger('main');
  final config = Config.fromEnv();

  // Ensure DB directory exists
  final dbDir = File(config.dbPath).parent;
  if (!dbDir.existsSync()) dbDir.createSync(recursive: true);

  log.info('Opening database: ${config.dbPath}');
  final db = AppDatabase(config.dbPath);

  final llm = LlmClient(
    baseUrl: config.lmStudioUrl,
    fallbackUrl: config.lmStudioFallbackUrl,
  );
  final mcpClient = McpClient(
    baseUrl: config.mcpUrl,
    cacheTtl: config.toolCacheTtl,
  );

  // Initialize MCP connection with retries (waits for container to be healthy)
  log.info('Connecting to MCP at ${config.mcpUrl}...');
  final mcpOk = await mcpClient.initialize(maxAttempts: 10);
  if (mcpOk) {
    final tools = await mcpClient.getTools(forceRefresh: true);
    log.info('Loaded ${tools.length} MCP tools');
  } else {
    log.warning(
      'MCP not available at startup — tools will load on first request',
    );
  }

  final compactor = Compactor(db: db, llm: llm, config: config);

  final appRouter = AppRouter(
    config: config,
    db: db,
    llm: llm,
    mcp: mcpClient,
    compactor: compactor,
  );

  final handler = const Pipeline()
      .addMiddleware(logRequests())
      .addHandler(appRouter.handler);

  final server = await io.serve(handler, '0.0.0.0', config.port);
  log.info(
    'dartboard listening on http://${server.address.address}:${server.port}\n'
    '  LM Studio: ${config.lmStudioUrl}\n'
    '  MCP:       ${config.mcpUrl}\n'
    '  Tools:     ${mcpClient.toolCount}\n'
    '  Database:  ${config.dbPath}',
  );

  // Graceful shutdown on SIGTERM/SIGINT (Docker sends SIGTERM on stop)
  void shutdown() {
    log.info('Shutting down...');
    server.close();
    llm.close();
    mcpClient.close();
    db.dispose();
    exit(0);
  }

  ProcessSignal.sigterm.watch().listen((_) => shutdown());
  ProcessSignal.sigint.watch().listen((_) => shutdown());
}
