/// Per-model optimization profiles for the Dartboard AI Lab.
///
/// Each [ModelProfile] captures inference parameters, capability flags,
/// and the specific set of tools each model can use (probed empirically).
library;

class ModelProfile {
  final double temperature;
  final int maxTokens;
  final bool supportsTools;
  final bool supportsReasoning;

  /// When true, tools are ALWAYS sent regardless of the client toggle.
  final bool enforceTools;

  /// Which tools this model can use.  Null = all tools (unrestricted).
  /// Empty list = no tools.
  final List<String>? allowedTools;

  /// Prompt tier: 'full' (default), 'condensed' (small models).
  final String promptSize;

  final int? systemPromptMaxChars;
  final String notes;

  const ModelProfile({
    required this.temperature,
    required this.maxTokens,
    required this.supportsTools,
    required this.supportsReasoning,
    this.enforceTools = false,
    this.allowedTools,
    this.promptSize = 'full',
    this.systemPromptMaxChars,
    this.notes = '',
  });

  /// How many tools this model uses (null allowedTools = 9 = Default+Extended).
  int get toolCount => allowedTools?.length ?? 9;

  Map<String, dynamic> toJson() => {
    'temperature': temperature,
    'max_tokens': maxTokens,
    'supports_tools': supportsTools,
    'supports_reasoning': supportsReasoning,
    'enforce_tools': enforceTools,
    'tool_count': toolCount,
    'prompt_size': promptSize,
    if (systemPromptMaxChars != null)
      'system_prompt_max_chars': systemPromptMaxChars,
    if (notes.isNotEmpty) 'notes': notes,
  };
}

// ── Built-in profiles keyed by exact model ID ──────────────────────
//
// allowedTools populated from empirical probing (Phase 1 audit).
// null = all 16 tools (unrestricted).

const _builtinProfiles = <String, ModelProfile>{
  // ── Strong models: Default tier (7) + Extended (media, data) = 9 tools ──
  'openai/gpt-oss-20b': ModelProfile(
    temperature: 0.7,
    maxTokens: 4096,
    supportsTools: true,
    supportsReasoning: false,
    allowedTools: [
      'web', 'image', 'browser', 'research', 'code', 'document', 'memory',
      'media', 'data',
    ], // 9 — Default tier + extended
    notes: 'LLM, MXFP4, 131K ctx, strong general model',
  ),
  'dolphin-mistral-glm-4.7-flash-24b-venice-edition-thinking-uncensored-i1':
      ModelProfile(
    temperature: 0.7,
    maxTokens: 4096,
    supportsTools: true,
    supportsReasoning: true, // new model has thinking capability
    enforceTools: true,
    allowedTools: [
      'web', 'image', 'browser', 'research', 'code', 'document', 'memory',
      'media', 'data',
    ], // 9 — Default tier + extended, enforced
    notes: 'LLM, Q4_K_S, 32K ctx, UNRESTRICTED, thinking, tool_choice=required',
  ),

  // ── Reasoning VLMs: Default tier (7) tools ──
  'qwen/qwen3.5-9b': ModelProfile(
    temperature: 0.5,
    maxTokens: 4096,
    supportsTools: true,
    supportsReasoning: true,
    allowedTools: [
      'web', 'image', 'browser', 'research', 'code', 'document', 'memory',
    ], // 7 — Default tier only
    notes: 'VLM, Q4_K_M, 262K ctx, reasoning',
  ),
  'zai-org/glm-4.6v-flash': ModelProfile(
    temperature: 0.5,
    maxTokens: 8192, // needs headroom for reasoning_content tokens
    supportsTools: true,
    supportsReasoning: true,
    allowedTools: [
      'web', 'image', 'browser', 'research', 'document', 'data', 'media',
    ], // 7 — probed subset
    notes: 'VLM, Q8_0, 131K ctx, reasoning, needs high maxTokens',
  ),

  // ── Small / weak models: reduced tool sets ──
  'ibm/granite-4-h-tiny': ModelProfile(
    temperature: 0.7,
    maxTokens: 2048,
    supportsTools: true,
    supportsReasoning: false,
    allowedTools: [
      'web', 'image', 'browser', 'code', 'memory',
    ], // 5 — minimal reliable set
    promptSize: 'condensed',
    systemPromptMaxChars: 4000,
    notes: 'LLM, Q8_0, 1M ctx, tiny, condensed prompt',
  ),
  'microsoft/phi-4-mini-reasoning': ModelProfile(
    temperature: 0.3,
    maxTokens: 4096,
    supportsTools: true,
    supportsReasoning: true,
    allowedTools: ['web', 'browser'], // 2 — very weak tools
    promptSize: 'condensed',
    notes: 'LLM, Q8_0, 131K ctx, reasoning, very weak tools',
  ),

  // ── Stale models (not in current LM Studio inventory, kept for reference) ──
  // 'mistralai/ministral-3-14b-reasoning': VLM, 14B reasoning, needs probing
  // 'deepseek/deepseek-r1-0528-qwen3-8b': LLM, 8B reasoning, tool calling broken
};

const defaultProfile = ModelProfile(
  temperature: 0.7,
  maxTokens: 4096,
  supportsTools: true,
  supportsReasoning: false,
  notes: 'Default profile for unknown models',
);

// ── Heuristic profiles for name-based matching ─────────────────────

const _reasoningProfile = ModelProfile(
  temperature: 0.5,
  maxTokens: 4096,
  supportsTools: true,
  supportsReasoning: true,
  notes: 'Heuristic: detected as reasoning model',
);

const _smallModelProfile = ModelProfile(
  temperature: 0.7,
  maxTokens: 2048,
  supportsTools: true,
  supportsReasoning: false,
  promptSize: 'condensed',
  systemPromptMaxChars: 4000,
  notes: 'Heuristic: detected as small model',
);

// ── Runtime-detected overrides (populated during warmup) ───────────

final _runtimeProfiles = <String, ModelProfile>{};

/// Store a runtime-detected profile that overrides built-in defaults.
void setRuntimeProfile(String modelId, ModelProfile profile) {
  _runtimeProfiles[modelId] = profile;
}

/// Retrieve the best [ModelProfile] for [modelId].
///
/// Priority: runtime-detected > exact built-in > heuristic > default.
ModelProfile getProfile(String modelId) {
  // 1. Runtime-detected override (from warmup auto-detection)
  final runtime = _runtimeProfiles[modelId];
  if (runtime != null) return runtime;

  // 2. Exact built-in match
  final builtin = _builtinProfiles[modelId];
  if (builtin != null) return builtin;

  // 3. Heuristic match by name patterns
  final lm = modelId.toLowerCase();

  if (lm.contains('reasoning') ||
      lm.contains('think') ||
      lm.contains('qwen3') ||
      lm.contains('phi-4') ||
      lm.contains('magistral') || lm.contains('ministral') ||
      lm.contains('glm')) {
    return _reasoningProfile;
  }

  if (lm.contains('tiny') || lm.contains('mini') || lm.contains('nano')) {
    return _smallModelProfile;
  }

  // 4. Default
  return defaultProfile;
}
