import 'dart:io';
import 'package:test/test.dart';
import 'package:dartboard/database.dart';
import 'package:dartboard/models.dart';

void main() {
  late AppDatabase db;
  late String dbPath;

  setUp(() {
    dbPath =
        '${Directory.systemTemp.path}/dartboard_test_${DateTime.now().millisecondsSinceEpoch}.db';
    db = AppDatabase(dbPath);
  });

  tearDown(() {
    db.dispose();
    final f = File(dbPath);
    if (f.existsSync()) f.deleteSync();
    // Clean up WAL files
    for (final ext in ['-wal', '-shm']) {
      final w = File('$dbPath$ext');
      if (w.existsSync()) w.deleteSync();
    }
  });

  group('Conversations', () {
    test('create and retrieve', () {
      final conv = db.createConversation(title: 'Test');
      expect(conv.id, isNotEmpty);
      expect(conv.title, equals('Test'));

      final fetched = db.getConversation(conv.id);
      expect(fetched, isNotNull);
      expect(fetched!.title, equals('Test'));
    });

    test('list ordered by updated_at desc', () {
      db.createConversation(title: 'First');
      db.createConversation(title: 'Second');
      final list = db.listConversations();
      expect(list.length, equals(2));
      expect(list.first.title, equals('Second'));
    });

    test('update title', () {
      final conv = db.createConversation(title: 'Old');
      db.updateConversation(conv.id, title: 'New');
      final fetched = db.getConversation(conv.id);
      expect(fetched!.title, equals('New'));
    });

    test('delete cascades messages', () {
      final conv = db.createConversation(title: 'Del');
      db.addMessage(conversationId: conv.id, role: 'user', content: 'hi');
      db.deleteConversation(conv.id);
      expect(db.getConversation(conv.id), isNull);
      expect(db.getMessages(conv.id), isEmpty);
    });
  });

  group('Messages', () {
    test('add and retrieve in order', () {
      final conv = db.createConversation();
      db.addMessage(conversationId: conv.id, role: 'user', content: 'hello');
      db.addMessage(conversationId: conv.id, role: 'assistant', content: 'hi');
      final msgs = db.getMessages(conv.id);
      expect(msgs.length, equals(2));
      expect(msgs[0].role, equals('user'));
      expect(msgs[1].role, equals('assistant'));
    });

    test('token count estimation', () {
      final conv = db.createConversation();
      db.addMessage(conversationId: conv.id, role: 'user', content: 'a' * 400);
      db.updateTokenCount(conv.id);
      final fetched = db.getConversation(conv.id);
      expect(fetched!.tokenCount, equals(100));
    });

    test('tool calls stored and retrieved', () {
      final conv = db.createConversation();
      db.addMessage(
        conversationId: conv.id,
        role: 'assistant',
        content: '',
        toolCalls: [
          ToolCallData(
            id: 'call_1',
            name: 'web_search',
            arguments: '{"query":"test"}',
          ),
        ],
      );
      final msgs = db.getMessages(conv.id);
      expect(msgs[0].toolCalls, isNotNull);
      expect(msgs[0].toolCalls!.first.name, equals('web_search'));
    });

    test('replaceMessages works', () {
      final conv = db.createConversation();
      db.addMessage(conversationId: conv.id, role: 'user', content: 'old1');
      db.addMessage(conversationId: conv.id, role: 'user', content: 'old2');
      expect(db.getMessages(conv.id).length, equals(2));

      db.replaceMessages(conv.id, [
        Message(
          id: 'new1',
          conversationId: conv.id,
          role: 'system',
          content: 'summary',
        ),
      ]);
      final msgs = db.getMessages(conv.id);
      expect(msgs.length, equals(1));
      expect(msgs[0].content, equals('summary'));
    });
  });

  group('User Isolation', () {
    test('conversations scoped by userId', () {
      db.createConversation(userId: 'alice', title: 'Alice Chat');
      db.createConversation(userId: 'bob', title: 'Bob Chat');
      db.createConversation(userId: 'alice', title: 'Alice Chat 2');

      final aliceConvs = db.listConversations(userId: 'alice');
      expect(aliceConvs.length, equals(2));
      expect(aliceConvs.every((c) => c.userId == 'alice'), isTrue);

      final bobConvs = db.listConversations(userId: 'bob');
      expect(bobConvs.length, equals(1));
      expect(bobConvs.first.title, equals('Bob Chat'));
    });

    test('getConversation enforces ownership', () {
      final conv = db.createConversation(userId: 'alice', title: 'Private');

      // Alice can access her own
      expect(db.getConversation(conv.id, userId: 'alice'), isNotNull);
      // Bob cannot access Alice's conversation
      expect(db.getConversation(conv.id, userId: 'bob'), isNull);
    });

    test('deleteConversation enforces ownership', () {
      final conv = db.createConversation(userId: 'alice', title: 'Protected');

      // Bob tries to delete Alice's conversation — should silently fail
      db.deleteConversation(conv.id, userId: 'bob');
      expect(db.getConversation(conv.id, userId: 'alice'), isNotNull);

      // Alice can delete her own
      db.deleteConversation(conv.id, userId: 'alice');
      expect(db.getConversation(conv.id, userId: 'alice'), isNull);
    });

    test('userId stored in conversation JSON', () {
      final conv = db.createConversation(userId: 'testuser', title: 'Test');
      final json = conv.toJson();
      expect(json['user_id'], equals('testuser'));
    });
  });

  group('Compaction Log', () {
    test('log compaction event', () {
      final conv = db.createConversation();
      db.logCompaction(
        conversationId: conv.id,
        beforeTokens: 25000,
        afterTokens: 5000,
        summary: 'test summary',
      );
      // No error means success
    });
  });
}
