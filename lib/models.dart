class Conversation {
  final String id;
  final String userId;
  String title;
  String model;
  String systemPrompt;
  final DateTime createdAt;
  DateTime updatedAt;
  int tokenCount;

  Conversation({
    required this.id,
    this.userId = '',
    this.title = 'New Chat',
    this.model = '',
    this.systemPrompt = '',
    DateTime? createdAt,
    DateTime? updatedAt,
    this.tokenCount = 0,
  }) : createdAt = createdAt ?? DateTime.now(),
       updatedAt = updatedAt ?? DateTime.now();

  Map<String, dynamic> toJson() => {
    'id': id,
    'user_id': userId,
    'title': title,
    'model': model,
    'system_prompt': systemPrompt,
    'created_at': createdAt.toIso8601String(),
    'updated_at': updatedAt.toIso8601String(),
    'token_count': tokenCount,
  };
}

class Message {
  final String id;
  final String conversationId;
  final String role;
  final String content;
  final List<ToolCallData>? toolCalls;
  final String? toolCallId;
  final DateTime createdAt;
  final int tokenCount;

  Message({
    required this.id,
    required this.conversationId,
    required this.role,
    required this.content,
    this.toolCalls,
    this.toolCallId,
    DateTime? createdAt,
    int? tokenCount,
  }) : createdAt = createdAt ?? DateTime.now(),
       tokenCount = tokenCount ?? (content.length ~/ 4);

  Map<String, dynamic> toJson() => {
    'id': id,
    'conversation_id': conversationId,
    'role': role,
    'content': content,
    if (toolCalls != null)
      'tool_calls': toolCalls!.map((t) => t.toJson()).toList(),
    if (toolCallId != null) 'tool_call_id': toolCallId,
    'created_at': createdAt.toIso8601String(),
    'token_count': tokenCount,
  };

  Map<String, dynamic> toLlmDict() {
    final m = <String, dynamic>{'role': role, 'content': content};
    if (toolCalls != null && toolCalls!.isNotEmpty) {
      m['tool_calls'] = toolCalls!.map((t) => t.toOpenAiFormat()).toList();
    }
    if (toolCallId != null) {
      m['tool_call_id'] = toolCallId;
    }
    return m;
  }
}

class ToolCallData {
  final String id;
  final String name;
  final String arguments;

  ToolCallData({required this.id, required this.name, required this.arguments});

  Map<String, dynamic> toJson() => {
    'id': id,
    'name': name,
    'arguments': arguments,
  };

  Map<String, dynamic> toOpenAiFormat() => {
    'id': id,
    'type': 'function',
    'function': {'name': name, 'arguments': arguments},
  };

  factory ToolCallData.fromJson(Map<String, dynamic> json) => ToolCallData(
    id: json['id'] as String? ?? '',
    name:
        json['name'] as String? ??
        (json['function'] as Map?)?['name'] as String? ??
        '',
    arguments:
        json['arguments'] as String? ??
        (json['function'] as Map?)?['arguments'] as String? ??
        '{}',
  );
}

class McpTool {
  final String name;
  final String description;
  final Map<String, dynamic> inputSchema;

  McpTool({
    required this.name,
    required this.description,
    required this.inputSchema,
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'description': description,
    'input_schema': inputSchema,
  };

  Map<String, dynamic> toOpenAiFormat() {
    // LM Studio requires parameters.type == "object" — MCP schemas may omit it.
    final params = inputSchema.isNotEmpty
        ? Map<String, dynamic>.from(inputSchema)
        : <String, dynamic>{'properties': {}};
    params['type'] = 'object';
    params.putIfAbsent('properties', () => <String, dynamic>{});
    return {
      'type': 'function',
      'function': {
        'name': name,
        'description': description,
        'parameters': params,
      },
    };
  }

  factory McpTool.fromJson(Map<String, dynamic> json) => McpTool(
    name: json['name'] as String,
    description: json['description'] as String? ?? '',
    inputSchema: Map<String, dynamic>.from(json['inputSchema'] as Map? ?? {}),
  );
}

class CompactionLog {
  final String id;
  final String conversationId;
  final int beforeTokens;
  final int afterTokens;
  final String summary;
  final DateTime compactedAt;

  CompactionLog({
    required this.id,
    required this.conversationId,
    required this.beforeTokens,
    required this.afterTokens,
    required this.summary,
    DateTime? compactedAt,
  }) : compactedAt = compactedAt ?? DateTime.now();
}

int estimateTokens(String text) => text.length ~/ 4;
