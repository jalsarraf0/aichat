import 'dart:convert';
import 'package:sqlite3/sqlite3.dart';
import 'package:uuid/uuid.dart';
import 'models.dart';

const _uuid = Uuid();

class AppDatabase {
  final Database _db;

  AppDatabase(String path) : _db = sqlite3.open(path) {
    _db.execute('PRAGMA journal_mode=WAL');
    _db.execute('PRAGMA foreign_keys=ON');
    _db.execute('PRAGMA busy_timeout=5000');
    _db.execute('PRAGMA synchronous=NORMAL');
    _db.execute('PRAGMA cache_size=-64000');
    _createTables();
  }

  void _createTables() {
    _db.execute('''
      CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT 'New Chat',
        model TEXT NOT NULL DEFAULT '',
        system_prompt TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0
      )
    ''');
    // Migration: add user_id column for per-user chat isolation
    try {
      _db.execute(
          "ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT ''");
    } catch (_) {
      // Column already exists — safe to ignore
    }
    _db.execute(
        'CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)');
    _db.execute(
        'CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC)');
    _db.execute('''
      CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL
          REFERENCES conversations(id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        tool_calls TEXT,
        tool_call_id TEXT,
        created_at TEXT NOT NULL,
        token_count INTEGER NOT NULL DEFAULT 0
      )
    ''');
    _db.execute('''
      CREATE INDEX IF NOT EXISTS idx_messages_conv_time
        ON messages(conversation_id, created_at)
    ''');
    _db.execute('''
      CREATE TABLE IF NOT EXISTS compaction_log (
        id TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL
          REFERENCES conversations(id) ON DELETE CASCADE,
        before_tokens INTEGER NOT NULL,
        after_tokens INTEGER NOT NULL,
        summary TEXT NOT NULL,
        compacted_at TEXT NOT NULL
      )
    ''');
  }

  // ── Conversations ──────────────────────────────────────────────────

  Conversation createConversation({
    String? id,
    String userId = '',
    String title = 'New Chat',
    String model = '',
    String systemPrompt = '',
  }) {
    final cid = id ?? _uuid.v4();
    final now = DateTime.now().toIso8601String();
    _db.execute(
      'INSERT INTO conversations (id, user_id, title, model, system_prompt, created_at, updated_at) '
      'VALUES (?, ?, ?, ?, ?, ?, ?)',
      [cid, userId, title, model, systemPrompt, now, now],
    );
    return Conversation(
      id: cid,
      userId: userId,
      title: title,
      model: model,
      systemPrompt: systemPrompt,
    );
  }

  Conversation? getConversation(String id, {String userId = ''}) {
    if (userId.isNotEmpty) {
      final result = _db.select(
        'SELECT * FROM conversations WHERE id = ? AND user_id = ?',
        [id, userId],
      );
      if (result.isEmpty) return null;
      return _rowToConversation(result.first);
    }
    final result = _db.select('SELECT * FROM conversations WHERE id = ?', [id]);
    if (result.isEmpty) return null;
    return _rowToConversation(result.first);
  }

  List<Conversation> listConversations({
    String userId = '',
    int limit = 50,
    int offset = 0,
  }) {
    if (userId.isNotEmpty) {
      final result = _db.select(
        'SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC LIMIT ? OFFSET ?',
        [userId, limit, offset],
      );
      return result.map(_rowToConversation).toList();
    }
    final result = _db.select(
      'SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ? OFFSET ?',
      [limit, offset],
    );
    return result.map(_rowToConversation).toList();
  }

  void updateConversation(
    String id, {
    String? title,
    String? model,
    String? systemPrompt,
  }) {
    final sets = <String>[];
    final args = <Object?>[];
    if (title != null) {
      sets.add('title = ?');
      args.add(title);
    }
    if (model != null) {
      sets.add('model = ?');
      args.add(model);
    }
    if (systemPrompt != null) {
      sets.add('system_prompt = ?');
      args.add(systemPrompt);
    }
    if (sets.isEmpty) return;
    sets.add('updated_at = ?');
    args.add(DateTime.now().toIso8601String());
    args.add(id);
    _db.execute(
      'UPDATE conversations SET ${sets.join(', ')} WHERE id = ?',
      args,
    );
  }

  void deleteConversation(String id, {String userId = ''}) {
    if (userId.isNotEmpty) {
      _db.execute(
        'DELETE FROM conversations WHERE id = ? AND user_id = ?',
        [id, userId],
      );
    } else {
      _db.execute('DELETE FROM conversations WHERE id = ?', [id]);
    }
  }

  void updateTokenCount(String conversationId) {
    final result = _db.select(
      'SELECT COALESCE(SUM(token_count), 0) as total FROM messages WHERE conversation_id = ?',
      [conversationId],
    );
    final total = result.first['total'] as int;
    _db.execute(
      'UPDATE conversations SET token_count = ?, updated_at = ? WHERE id = ?',
      [total, DateTime.now().toIso8601String(), conversationId],
    );
  }

  // ── Messages ───────────────────────────────────────────────────────

  Message addMessage({
    required String conversationId,
    required String role,
    required String content,
    List<ToolCallData>? toolCalls,
    String? toolCallId,
  }) {
    final mid = _uuid.v4();
    final now = DateTime.now().toIso8601String();
    final tokens = estimateTokens(content);
    final tcJson = toolCalls != null
        ? jsonEncode(toolCalls.map((t) => t.toJson()).toList())
        : null;
    _db.execute(
      'INSERT INTO messages (id, conversation_id, role, content, tool_calls, tool_call_id, created_at, token_count) '
      'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
      [mid, conversationId, role, content, tcJson, toolCallId, now, tokens],
    );
    _db.execute('UPDATE conversations SET updated_at = ? WHERE id = ?', [
      now,
      conversationId,
    ]);
    return Message(
      id: mid,
      conversationId: conversationId,
      role: role,
      content: content,
      toolCalls: toolCalls,
      toolCallId: toolCallId,
    );
  }

  List<Message> getMessages(String conversationId) {
    final result = _db.select(
      'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC',
      [conversationId],
    );
    return result.map(_rowToMessage).toList();
  }

  void replaceMessages(String conversationId, List<Message> messages) {
    _db.execute('BEGIN');
    try {
      _db.execute('DELETE FROM messages WHERE conversation_id = ?', [
        conversationId,
      ]);
      for (final m in messages) {
        final tcJson = m.toolCalls != null
            ? jsonEncode(m.toolCalls!.map((t) => t.toJson()).toList())
            : null;
        _db.execute(
          'INSERT INTO messages (id, conversation_id, role, content, tool_calls, tool_call_id, created_at, token_count) '
          'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
          [
            m.id,
            m.conversationId,
            m.role,
            m.content,
            tcJson,
            m.toolCallId,
            m.createdAt.toIso8601String(),
            m.tokenCount,
          ],
        );
      }
      _db.execute('COMMIT');
    } catch (e) {
      _db.execute('ROLLBACK');
      rethrow;
    }
    updateTokenCount(conversationId);
  }

  // ── Compaction Log ─────────────────────────────────────────────────

  void logCompaction({
    required String conversationId,
    required int beforeTokens,
    required int afterTokens,
    required String summary,
  }) {
    final id = _uuid.v4();
    final now = DateTime.now().toIso8601String();
    _db.execute(
      'INSERT INTO compaction_log (id, conversation_id, before_tokens, after_tokens, summary, compacted_at) '
      'VALUES (?, ?, ?, ?, ?, ?)',
      [id, conversationId, beforeTokens, afterTokens, summary, now],
    );
  }

  // ── Helpers ────────────────────────────────────────────────────────

  Conversation _rowToConversation(Row row) => Conversation(
    id: row['id'] as String,
    userId: row['user_id'] as String? ?? '',
    title: row['title'] as String,
    model: row['model'] as String,
    systemPrompt: row['system_prompt'] as String,
    createdAt: DateTime.parse(row['created_at'] as String),
    updatedAt: DateTime.parse(row['updated_at'] as String),
    tokenCount: row['token_count'] as int,
  );

  Message _rowToMessage(Row row) {
    final tcRaw = row['tool_calls'] as String?;
    List<ToolCallData>? toolCalls;
    if (tcRaw != null && tcRaw.isNotEmpty) {
      final list = jsonDecode(tcRaw) as List;
      toolCalls = list
          .map(
            (e) => ToolCallData.fromJson(Map<String, dynamic>.from(e as Map)),
          )
          .toList();
    }
    return Message(
      id: row['id'] as String,
      conversationId: row['conversation_id'] as String,
      role: row['role'] as String,
      content: row['content'] as String,
      toolCalls: toolCalls,
      toolCallId: row['tool_call_id'] as String?,
      createdAt: DateTime.parse(row['created_at'] as String),
      tokenCount: row['token_count'] as int,
    );
  }

  void dispose() => _db.dispose();
}
