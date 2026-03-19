import 'package:logging/logging.dart';
import 'package:uuid/uuid.dart';
import 'config.dart';
import 'database.dart';
import 'llm_client.dart';
import 'models.dart';

final _log = Logger('Compaction');
const _uuid = Uuid();

class Compactor {
  final AppDatabase db;
  final LlmClient llm;
  final Config config;

  Compactor({required this.db, required this.llm, required this.config});

  /// Check if a conversation needs compaction and perform it if so.
  /// Returns true if compaction was performed.
  Future<bool> compactIfNeeded(String conversationId, {String? model}) async {
    final conv = db.getConversation(conversationId);
    if (conv == null) return false;
    if (conv.tokenCount < config.compactionThreshold) return false;

    _log.info(
      'Compacting conversation $conversationId '
      '(${conv.tokenCount} tokens > ${config.compactionThreshold})',
    );

    final messages = db.getMessages(conversationId);
    if (messages.length <= config.compactionKeepRecent + 1) return false;

    // Separate system messages, old messages, and recent messages
    final systemMsgs = messages.where((m) => m.role == 'system').toList();
    final nonSystem = messages.where((m) => m.role != 'system').toList();

    if (nonSystem.length <= config.compactionKeepRecent) return false;

    final cutoff = nonSystem.length - config.compactionKeepRecent;
    final oldMessages = nonSystem.sublist(0, cutoff);
    final recentMessages = nonSystem.sublist(cutoff);

    // Build summarization prompt
    final oldText = oldMessages
        .map((m) => '${m.role}: ${m.content}')
        .join('\n');
    final beforeTokens = conv.tokenCount;

    final summary = await llm.chatOnce(
      model: model ?? config.model,
      messages: [
        {
          'role': 'system',
          'content':
              'Summarize this conversation history concisely, preserving key facts, '
              'decisions, tool results, and important context. Be thorough but brief.',
        },
        {'role': 'user', 'content': oldText},
      ],
    );

    // Build new message list
    final summaryMsg = Message(
      id: _uuid.v4(),
      conversationId: conversationId,
      role: 'system',
      content: '[Conversation Summary]\n$summary',
    );

    final newMessages = [...systemMsgs, summaryMsg, ...recentMessages];
    db.replaceMessages(conversationId, newMessages);
    db.updateTokenCount(conversationId);

    final afterConv = db.getConversation(conversationId);
    final afterTokens = afterConv?.tokenCount ?? 0;

    db.logCompaction(
      conversationId: conversationId,
      beforeTokens: beforeTokens,
      afterTokens: afterTokens,
      summary: summary,
    );

    _log.info('Compacted: $beforeTokens → $afterTokens tokens');
    return true;
  }
}
