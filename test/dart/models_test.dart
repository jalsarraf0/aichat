import 'package:test/test.dart';
import 'package:dartboard/models.dart';

void main() {
  group('McpTool', () {
    test('fromJson parses correctly', () {
      final tool = McpTool.fromJson({
        'name': 'web_search',
        'description': 'Search the web',
        'inputSchema': {
          'type': 'object',
          'properties': {
            'query': {'type': 'string'},
          },
          'required': ['query'],
        },
      });
      expect(tool.name, equals('web_search'));
      expect(tool.description, equals('Search the web'));
      expect(tool.inputSchema['type'], equals('object'));
    });

    test('toOpenAiFormat converts correctly', () {
      final tool = McpTool(
        name: 'memory_store',
        description: 'Store a value',
        inputSchema: {
          'type': 'object',
          'properties': {
            'key': {'type': 'string'},
            'value': {'type': 'string'},
          },
          'required': ['key', 'value'],
        },
      );
      final oai = tool.toOpenAiFormat();
      expect(oai['type'], equals('function'));
      expect(oai['function']['name'], equals('memory_store'));
      expect(oai['function']['parameters']['type'], equals('object'));
    });

    test('toOpenAiFormat handles empty schema', () {
      final tool = McpTool(
        name: 'noop',
        description: 'No params',
        inputSchema: {},
      );
      final oai = tool.toOpenAiFormat();
      expect(oai['function']['parameters']['type'], equals('object'));
    });
  });

  group('ToolCallData', () {
    test('fromJson with OpenAI format', () {
      final tc = ToolCallData.fromJson({
        'id': 'call_1',
        'function': {'name': 'test', 'arguments': '{"a":1}'},
      });
      expect(tc.id, equals('call_1'));
      expect(tc.name, equals('test'));
      expect(tc.arguments, equals('{"a":1}'));
    });

    test('toOpenAiFormat roundtrips', () {
      final tc = ToolCallData(id: 'c1', name: 'foo', arguments: '{}');
      final oai = tc.toOpenAiFormat();
      expect(oai['id'], equals('c1'));
      expect(oai['type'], equals('function'));
      expect(oai['function']['name'], equals('foo'));
    });
  });

  group('estimateTokens', () {
    test('estimates chars/4', () {
      expect(estimateTokens('hello world'), equals(2)); // 11 chars / 4 = 2
      expect(estimateTokens('a' * 100), equals(25));
      expect(estimateTokens(''), equals(0));
    });
  });

  group('Message', () {
    test('toLlmDict includes tool_calls', () {
      final msg = Message(
        id: '1',
        conversationId: 'c1',
        role: 'assistant',
        content: 'Searching...',
        toolCalls: [ToolCallData(id: 'tc1', name: 'search', arguments: '{}')],
      );
      final d = msg.toLlmDict();
      expect(d['role'], equals('assistant'));
      expect(d['tool_calls'], isNotNull);
      expect((d['tool_calls'] as List).length, equals(1));
    });

    test('toLlmDict omits null tool_calls', () {
      final msg = Message(
        id: '2',
        conversationId: 'c1',
        role: 'user',
        content: 'hello',
      );
      final d = msg.toLlmDict();
      expect(d.containsKey('tool_calls'), isFalse);
    });
  });
}
