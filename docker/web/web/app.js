'use strict';

// ── Global Error Handler — makes silent failures visible ──────────
window.onerror = function(msg, src, line, col, err) {
  // Only show banner for errors from our own scripts
  const s = String(src || '');
  const m = String(msg || '');
  const isOwn = s.includes('/app.js') || s.includes('/style.css');
  const isExtension = s.startsWith('chrome-extension://') || s.startsWith('moz-extension://') || s.startsWith('safari-extension://');
  const isThirdParty = m === 'Script error.' || isExtension || m.includes('ethereum') || m.includes('web3') || (!s && !line);
  if (isThirdParty && !isOwn) return;
  console.error('JS Error:', msg, 'at', src, line);
  const el = document.getElementById('error-banner');
  if (el) { el.textContent = 'JS Error: ' + msg; el.classList.remove('hidden'); el.onclick = () => el.classList.add('hidden'); }
};
window.addEventListener('unhandledrejection', function(e) {
  const msg = String(e.reason?.message || e.reason || '');
  // Filter out browser extension noise (crypto wallets, ad blockers, etc.)
  if (msg.includes('ethereum') || msg.includes('web3') || msg.includes('extension')) return;
  console.error('Unhandled promise:', e.reason);
  const el = document.getElementById('error-banner');
  if (el) { el.textContent = 'Error: ' + msg; el.classList.remove('hidden'); el.onclick = () => el.classList.add('hidden'); }
});

// ── Auth ──────────────────────────────────────────────────────────
function authFetch(url, opts = {}) {
  const token = localStorage.getItem('dartboard-jwt');
  if (!token || token.length < 10) {
    // No valid token — don't even try the request
    showAuthScreen();
    return Promise.reject(new Error('Not authenticated'));
  }
  if (!opts.headers) opts.headers = {};
  opts.headers['Authorization'] = 'Bearer ' + token;
  return fetch(url, opts).then(res => {
    if (res.status === 401 && url.startsWith('/api/')) {
      // Only log out if we're sure the token is bad (not a transient error)
      const currentToken = localStorage.getItem('dartboard-jwt');
      if (currentToken === token) {
        localStorage.removeItem('dartboard-jwt');
        showAuthScreen();
      }
      throw new Error('Session expired');
    }
    return res;
  });
}

function showAuthScreen() {
  document.getElementById('auth-screen').classList.remove('hidden');
  document.getElementById('app').classList.add('hidden');
}
function hideAuthScreen() {
  document.getElementById('auth-screen').classList.add('hidden');
  document.getElementById('app').classList.remove('hidden');
}
function showLogin() {
  document.getElementById('auth-login').classList.remove('hidden');
  document.getElementById('auth-register').classList.add('hidden');
  document.getElementById('auth-sub').textContent = 'Sign in to continue';
  clearAuthMsg();
}
function showRegister() {
  document.getElementById('auth-login').classList.add('hidden');
  document.getElementById('auth-register').classList.remove('hidden');
  document.getElementById('auth-sub').textContent = 'Request access';
  clearAuthMsg();
}
function showAuthMsg(text, isError) {
  const el = document.getElementById('auth-msg');
  el.textContent = text;
  el.className = isError ? 'error' : 'success';
}
function clearAuthMsg() {
  const el = document.getElementById('auth-msg');
  el.textContent = '';
  el.className = 'hidden';
}

async function checkAuth() {
  const token = localStorage.getItem('dartboard-jwt');
  if (!token) { showAuthScreen(); return false; }
  try {
    const res = await fetch('/auth/verify', { headers: { 'Authorization': 'Bearer ' + token } });
    if (res.ok) { hideAuthScreen(); return true; }
  } catch {}
  localStorage.removeItem('dartboard-jwt');
  showAuthScreen();
  return false;
}

async function authLogin() {
  const user = document.getElementById('login-user').value.trim();
  const pass = document.getElementById('login-pass').value;
  if (!user || !pass) { showAuthMsg('Enter username and password', true); return; }
  const btn = document.getElementById('login-btn');
  btn.disabled = true;
  let banned = false;
  try {
    const res = await fetch('/auth/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, password: pass }),
    });
    const data = await res.json();
    if (!res.ok) {
      showAuthMsg(data.error || 'Login failed', true);
      // If IP is banned (403), disable the form permanently
      if (res.status === 403 && (data.error || '').includes('banned')) {
        banned = true;
        document.getElementById('login-user').disabled = true;
        document.getElementById('login-pass').disabled = true;
      }
      return;
    }
    if (!data.token || data.token.length < 10) { showAuthMsg('Server returned invalid token', true); return; }
    localStorage.setItem('dartboard-jwt', data.token);
    hideAuthScreen();
    await Promise.allSettled([loadModels(), loadConversations(), loadToolCount(), loadPersonalities()]);
  } catch (e) { showAuthMsg('Connection error', true); }
  finally { if (!banned) btn.disabled = false; }
}

async function authRegister() {
  const user = document.getElementById('reg-user').value.trim();
  const pass = document.getElementById('reg-pass').value;
  const pass2 = document.getElementById('reg-pass2').value;
  if (!user || !pass) { showAuthMsg('Fill in all fields', true); return; }
  if (pass !== pass2) { showAuthMsg('Passwords do not match', true); return; }
  const btn = document.getElementById('reg-btn');
  btn.disabled = true;
  try {
    const res = await fetch('/auth/register', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: user, password: pass }),
    });
    const data = await res.json();
    if (!res.ok) { showAuthMsg(data.error || 'Registration failed', true); return; }
    showAuthMsg(data.message, false);
    document.getElementById('reg-user').value = '';
    document.getElementById('reg-pass').value = '';
    document.getElementById('reg-pass2').value = '';
  } catch (e) { showAuthMsg('Connection error', true); }
  finally { btn.disabled = false; }
}

function authLogout() {
  // Stop any active generation
  if (isStreaming) stopGeneration();
  // Clear auth token
  localStorage.removeItem('dartboard-jwt');
  // Reset client state so the next login starts clean
  currentConvId = null;
  allConversations = [];
  availableModels = [];
  selectedModel = null;
  selectedModelReady = false;
  pendingFiles = [];
  document.getElementById('messages').innerHTML = '';
  document.getElementById('conv-list').innerHTML = '';
  showView('welcome');
  // Clear and re-enable login form fields
  document.getElementById('login-user').value = '';
  document.getElementById('login-pass').value = '';
  document.getElementById('login-user').disabled = false;
  document.getElementById('login-pass').disabled = false;
  document.getElementById('login-btn').disabled = false;
  clearAuthMsg();
  showAuthScreen();
}

// Handle Enter key on auth forms
document.addEventListener('keydown', (e) => {
  if (e.key !== 'Enter') return;
  if (document.getElementById('auth-screen').classList.contains('hidden')) return;
  if (!document.getElementById('auth-register').classList.contains('hidden')) authRegister();
  else authLogin();
});

// ── State ──────────────────────────────────────────────────────────
let currentConvId = null;
let isStreaming = false;
let selectedModel = null;
let selectedModelReady = false; // Blocks send until model loaded
let availableModels = [];
let allConversations = [];
let abortController = null;
let pendingFiles = [];
let toolsEnabled = true; // ON by default
let allPersonalities = [];
let selectedPersonality = { id: 'general', name: 'General Assistant', icon: '\u{1F9E0}' };
let customSystemPrompt = null;

// ── Marked (defensive — CDN may fail) ─────────────────────────────
try { if (typeof marked !== 'undefined') marked.use({ breaks: true, gfm: true }); }
catch (e) { console.warn('marked.use failed:', e); }

// ── Model Capabilities ────────────────────────────────────────────
// Validated by server e2e tests. Falls back to name inference until validated.
const _validatedCaps = {};

function getModelCaps(id) {
  if (!id) return { vision: false, tools: false, reasoning: false, embedding: false };
  if (_validatedCaps[id]) return _validatedCaps[id];
  const l = id.toLowerCase();
  if (/embed|embedding/.test(l)) return { vision: false, tools: false, reasoning: false, embedding: true };
  const caps = { vision: false, tools: true, reasoning: false, embedding: false };
  if (/\bvl\b|-vl-|vision|4v|4\.6v/.test(l)) caps.vision = true;
  // Also check v0 API model_type for VLM
  const m = availableModels.find(m => m.id === id);
  if (m?.model_type === 'vlm') caps.vision = true;
  if (/reasoning|qwen3\.5|magistral|think|phi-4/.test(l)) caps.reasoning = true;
  return caps;
}

function getModelMeta(id) {
  const m = availableModels.find(m => m.id === id);
  return {
    state: m?.state || 'unknown',
    quant: m?.quantization || '',
    ctx: m?.max_context_length || m?.profile?.max_tokens || 0,
    type: m?.model_type || (m?.type === 'vlm' ? 'vlm' : 'llm'),
    toolCount: m?.profile?.tool_count ?? 9,
  };
}

function formatCtx(n) {
  if (!n) return '';
  if (n >= 1048576) return (n / 1048576).toFixed(0) + 'M';
  if (n >= 1024) return (n / 1024).toFixed(0) + 'K';
  return String(n);
}

function capsHTML(id) {
  const c = getModelCaps(id);
  if (c.embedding) return '<span class="cap-badge cap-embed" title="Embedding model">&#128202;</span>';
  let h = '';
  if (c.vision) h += '<span class="cap-badge cap-vision" title="Vision (VLM)">&#128065;&#65039;</span>';
  if (c.reasoning) h += '<span class="cap-badge cap-reason" title="Reasoning">&#129504;</span>';
  if (c.tools) h += '<span class="cap-badge cap-tools" title="Tool use">&#128296;</span>';
  else h += '<span class="cap-badge cap-notools" title="No tool support">&#128683;</span>';
  return h;
}

// ── Init ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  toolsEnabled = localStorage.getItem('dartboard-tools') !== 'false';
  updateToolsToggle();
  setupDragDrop();
  const authed = await checkAuth();
  if (authed) {
    await Promise.all([loadModels(), loadConversations(), loadToolCount(), loadPersonalities()]);
  }
});

// ── Models ─────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const res = await authFetch('/api/models');
    const data = await res.json();
    availableModels = (data.models || []).filter(m => m.id);
    // Store server-validated capabilities
    for (const m of availableModels) {
      if (m.validated) {
        _validatedCaps[m.id] = {
          chat: m.chat ?? true,
          tools: m.tools ?? true,
          reasoning: m.reasoning ?? false,
          embedding: m.embedding ?? false,
          vision: m.vision ?? false,
          limitation: m.limitation || null,
        };
      }
    }
    const stored = localStorage.getItem('dartboard-model');
    if (stored && availableModels.some(m => m.id === stored)) selectedModel = stored;
    // Don't auto-select — user must pick a model explicitly
    updateModelDisplay();
    updateConnStatus(true);
  } catch (e) {
    console.error('loadModels:', e);
    updateConnStatus(false);
  }
}

function updateModelDisplay() {
  const label = document.getElementById('model-label');
  const badge = document.getElementById('model-badge');
  const caps = document.getElementById('model-caps');
  label.textContent = selectedModel || 'Select a model';
  badge.textContent = selectedModel || '';
  if (caps) caps.innerHTML = capsHTML(selectedModel);

  // Always update tools toggle when model changes
  updateToolsToggle();
}

function updateConnStatus(ok) {
  const dot = document.getElementById('conn-status');
  dot.className = 'status-dot ' + (ok ? 'connected' : 'disconnected');
  dot.title = ok ? 'Connected to LM Studio' : 'Cannot reach LM Studio';
}

function toggleModelMenu() {
  const menu = document.getElementById('model-menu');
  if (menu.classList.contains('hidden')) { renderModelMenu(); menu.classList.remove('hidden'); }
  else menu.classList.add('hidden');
}

function renderModelMenu() {
  const menu = document.getElementById('model-menu');
  menu.innerHTML = '';
  if (availableModels.length === 0) {
    menu.innerHTML = '<div class="dropdown-empty">No models found.<br>Is LM Studio running?</div>';
    return;
  }
  // Sort: loaded first, then alphabetical
  const sorted = [...availableModels].sort((a, b) => {
    const aLoaded = a.state === 'loaded' ? 0 : 1;
    const bLoaded = b.state === 'loaded' ? 0 : 1;
    if (aLoaded !== bLoaded) return aLoaded - bLoaded;
    return (a.id || '').localeCompare(b.id || '');
  });
  for (const m of sorted) {
    // Skip embedding models from selector
    if ((m.model_type || m.type) === 'embeddings') continue;
    const meta = getModelMeta(m.id);
    const btn = document.createElement('button');
    btn.className = 'dropdown-item' + (m.id === selectedModel ? ' active' : '');

    // Top row: state dot + model name + caps + check
    const row = document.createElement('div');
    row.className = 'model-row';

    const dot = document.createElement('span');
    dot.className = 'state-dot ' + (meta.state === 'loaded' ? 'loaded' : 'unloaded');
    dot.title = meta.state === 'loaded' ? 'Loaded' : 'Not loaded';

    const nameSpan = document.createElement('span');
    nameSpan.className = 'model-id';
    nameSpan.textContent = m.id;

    const capsSpan = document.createElement('span');
    capsSpan.className = 'model-caps-row';
    capsSpan.innerHTML = capsHTML(m.id);

    const checkSpan = document.createElement('span');
    checkSpan.className = 'check';
    checkSpan.textContent = m.id === selectedModel ? '\u2713' : '';

    row.appendChild(dot);
    row.appendChild(nameSpan);
    row.appendChild(capsSpan);
    row.appendChild(checkSpan);
    btn.appendChild(row);

    // Meta row: quantization | context | type | tool count
    const metaRow = document.createElement('div');
    metaRow.className = 'model-meta';
    const tags = [];
    if (meta.quant) tags.push(meta.quant);
    if (meta.ctx) tags.push(formatCtx(meta.ctx) + ' ctx');
    tags.push(meta.type.toUpperCase());
    tags.push(meta.toolCount + ' tools');
    metaRow.innerHTML = tags.map(t => '<span class="meta-tag">' + esc(t) + '</span>').join('');
    btn.appendChild(metaRow);

    btn.onclick = (e) => { e.stopPropagation(); pickModel(m.id); menu.classList.add('hidden'); };
    menu.appendChild(btn);
  }
}

async function pickModel(modelId) {
  if (modelId === selectedModel) return;
  selectedModel = modelId;
  selectedModelReady = false; // Block send until loaded
  localStorage.setItem('dartboard-model', modelId);
  updateModelDisplay();
  updateActionBtn(); // Disable send immediately

  // Refresh personality list — model-restricted personalities may appear/disappear
  loadPersonalities();

  showModelLoading(`Loading ${modelId}...`);
  try {
    const res = await authFetch('/api/warmup', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model: modelId }),
    });
    const data = await res.json();
    _validatedCaps[modelId] = {
      chat: data.chat ?? true,
      tools: data.tools ?? true,
      reasoning: data.reasoning ?? false,
      embedding: data.embedding ?? false,
      vision: false,
      limitation: data.limitation || null,
    };
    updateModelDisplay();

    if (data.status === 'ready') {
      selectedModelReady = true;
      const capList = [];
      if (data.chat) capList.push('Chat');
      if (data.tools) capList.push('Tools');
      if (data.reasoning) capList.push('Reasoning');
      showModelLoading(`Ready: ${capList.join(', ')}`);
      setTimeout(hideModelLoading, 2000);
    } else if (data.status === 'limited') {
      selectedModelReady = true; // Allow with warning
      showModelLoading(`${data.limitation || 'Limited capabilities'}`);
      setTimeout(hideModelLoading, 3000);
    } else if (data.status === 'busy') {
      showModelLoading(`${data.message || 'Model busy — try again'}`);
      setTimeout(hideModelLoading, 4000);
    } else {
      showModelLoading(`Error: ${data.limitation || data.message || 'could not load model'}`);
      setTimeout(hideModelLoading, 4000);
    }
  } catch (e) {
    showModelLoading('Connection failed: ' + e.message);
    setTimeout(hideModelLoading, 3000);
  }
  updateActionBtn(); // Re-evaluate send button state

  if (currentConvId) {
    try {
      await authFetch(`/api/conversations/${currentConvId}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId }),
      });
    } catch { /* ignore */ }
  }
}

function showModelLoading(text) {
  document.getElementById('loading-text').textContent = text || 'Loading model...';
  document.getElementById('model-loading').classList.remove('hidden');
}
function hideModelLoading() {
  document.getElementById('model-loading').classList.add('hidden');
}

document.addEventListener('click', (e) => {
  const sel = document.getElementById('model-selector');
  const menu = document.getElementById('model-menu');
  if (sel && menu && !sel.contains(e.target) && !menu.contains(e.target)) menu.classList.add('hidden');
});

// ── Tools Toggle ───────────────────────────────────────────────────
function toggleTools() {
  const c = getModelCaps(selectedModel);
  if (c.embedding || !c.tools) return; // Can't enable for incapable models
  toolsEnabled = !toolsEnabled;
  localStorage.setItem('dartboard-tools', toolsEnabled);
  updateToolsToggle();
}
function updateToolsToggle() {
  const btn = document.getElementById('tools-toggle');
  const label = document.getElementById('tools-label');
  const c = getModelCaps(selectedModel);
  const toolCount = _getModelToolCount(selectedModel);
  toolsEnabled = true;
  if (c.embedding) {
    btn.classList.remove('active');
    btn.classList.add('disabled');
    label.textContent = 'Embed Only';
    btn.title = 'This is an embedding model — not for chat';
  } else if (!c.tools || toolCount === 0) {
    btn.classList.remove('active');
    btn.classList.add('disabled');
    label.textContent = 'No Tools';
    btn.title = c.limitation || 'This model cannot use tools — text-only responses';
  } else {
    btn.classList.add('active');
    btn.classList.remove('disabled');
    label.textContent = `${toolCount} Tools`;
    btn.title = `${toolCount} MCP tools active for this model`;
  }
}

function _getModelToolCount(modelId) {
  if (!modelId) return 0;
  const m = availableModels.find(m => m.id === modelId);
  if (!m || !m.profile) return 9; // default: Default+Extended tiers
  return m.profile.tool_count ?? 9;
}

// ── Tool Count ─────────────────────────────────────────────────────
async function loadToolCount() {
  try {
    const res = await authFetch('/api/tools');
    const data = await res.json();
    document.getElementById('tool-count').textContent = `${data.count} tools`;
  } catch { /* ignore */ }
}

// ── Conversations ──────────────────────────────────────────────────
async function loadConversations() {
  try {
    const res = await authFetch('/api/conversations?limit=50');
    const data = await res.json();
    allConversations = data.conversations || [];
    renderConvList(allConversations);
  } catch (e) { console.error('loadConversations:', e); }
}

function renderConvList(convs) {
  const list = document.getElementById('conv-list');
  list.innerHTML = '';
  for (const c of convs) {
    const item = document.createElement('div');
    item.className = 'conv-item' + (c.id === currentConvId ? ' active' : '');
    item.onclick = () => openConversation(c.id);
    const title = document.createElement('span');
    title.className = 'conv-title';
    title.textContent = c.title || 'New Chat';
    const tag = document.createElement('span');
    tag.className = 'conv-model-tag';
    tag.innerHTML = shortModel(c.model) + ' ' + capsHTML(c.model);
    tag.title = c.model || '';
    const del = document.createElement('button');
    del.className = 'del-btn'; del.textContent = '\u00d7'; del.title = 'Delete';
    del.onclick = (e) => { e.stopPropagation(); deleteConv(c.id); };
    item.appendChild(title); item.appendChild(tag); item.appendChild(del);
    list.appendChild(item);
  }
}

function shortModel(id) {
  if (!id) return '';
  let s = id.replace(/-instruct$/i, '').replace(/-chat$/i, '');
  return s.length > 14 ? s.substring(0, 14) + '\u2026' : s;
}

function filterConversations() {
  const q = document.getElementById('conv-search').value.toLowerCase();
  document.querySelectorAll('#conv-list .conv-item').forEach(el => {
    el.style.display = (el.querySelector('.conv-title')?.textContent?.toLowerCase() || '').includes(q) ? '' : 'none';
  });
}

async function newChat() {
  if (isStreaming) stopGeneration();
  currentConvId = null;
  showView('welcome');
  document.getElementById('messages').innerHTML = '';
  renderConvList(allConversations);
  closeMobileSidebar();
}

async function openConversation(id) {
  if (isStreaming) stopGeneration();
  currentConvId = id;
  closeMobileSidebar();
  try {
    const res = await authFetch(`/api/conversations/${id}`);
    const data = await res.json();
    if (data.model && availableModels.some(m => m.id === data.model)) {
      selectedModel = data.model;
      localStorage.setItem('dartboard-model', data.model);
      updateModelDisplay();
    }
    renderMessages(data.messages || []);
    showView('chat');
    renderConvList(allConversations);
  } catch (e) { console.error('openConversation:', e); }
}

async function deleteConv(id) {
  try {
    await authFetch(`/api/conversations/${id}`, { method: 'DELETE' });
    if (id === currentConvId) { currentConvId = null; showView('welcome'); document.getElementById('messages').innerHTML = ''; }
    await loadConversations();
  } catch (e) { console.error('deleteConv:', e); }
}

function isMobile() { return window.matchMedia('(max-width: 768px)').matches; }
function closeMobileSidebar() { if (isMobile()) document.getElementById('sidebar').classList.remove('open'); }

// ── View ───────────────────────────────────────────────────────────
function showView(v) {
  document.getElementById('welcome').classList.toggle('hidden', v !== 'welcome');
  document.getElementById('chat-view').classList.toggle('hidden', v !== 'chat');
}

// ── Rendering ──────────────────────────────────────────────────────
function renderMessages(msgs) {
  const c = document.getElementById('messages');
  c.innerHTML = '';
  for (const m of msgs) {
    if (m.role === 'system' || m.role === 'tool') continue;
    // Skip empty assistant messages (tool-call-only iterations)
    if (m.role === 'assistant' && !m.content?.trim() && m.tool_calls?.length > 0) continue;
    appendMessage(m.role, m.content, m.tool_calls);
  }
  scrollToBottom();
}

function appendMessage(role, content, toolCalls, files) {
  const container = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;

  if (role === 'user') {
    const bubble = document.createElement('div');
    bubble.className = 'msg-bubble';
    if (files?.length > 0) {
      const a = document.createElement('div'); a.className = 'msg-attachments';
      for (const f of files) {
        if (f.preview) { const img = document.createElement('img'); img.src = f.preview; img.alt = f.name; a.appendChild(img); }
        else { const c = document.createElement('span'); c.className = 'msg-attach-file'; c.textContent = '\uD83D\uDCC4 ' + f.name; a.appendChild(c); }
      }
      bubble.appendChild(a);
    }
    const el = document.createElement('div'); el.className = 'msg-content'; el.innerHTML = renderMd(content);
    bubble.appendChild(el); div.appendChild(bubble); container.appendChild(div);
    return { div, contentEl: el, bodyEl: bubble };
  }

  if (role === 'assistant') {
    const av = document.createElement('div'); av.className = 'msg-avatar'; av.textContent = '\uD83E\uDD16';
    const body = document.createElement('div'); body.className = 'msg-body';
    // Tool calls are hidden — they ran silently behind the thinking indicator
    const el = document.createElement('div'); el.className = 'msg-content';
    body.appendChild(el);
    if (content) { el.innerHTML = renderMd(content); postProcess(el); }
    div.appendChild(av); div.appendChild(body); container.appendChild(div);
    return { div, contentEl: el, bodyEl: body };
  }
  return { div, contentEl: null, bodyEl: null };
}

function renderMd(t) {
  if (!t) return '';
  try {
    const html = typeof marked !== 'undefined' ? marked.parse(t) : t.replace(/</g,'&lt;').replace(/\n/g,'<br>');
    return typeof DOMPurify !== 'undefined' ? DOMPurify.sanitize(html, { ADD_TAGS: ['img'], ADD_ATTR: ['src','alt','class','loading','decoding'] }) : html;
  } catch (e) { console.error('renderMd:', e); return esc(t); }
}

function postProcess(el) {
  if (!el) return;
  // Syntax highlighting
  try { if (typeof hljs !== 'undefined') el.querySelectorAll('pre code').forEach(b => hljs.highlightElement(b)); } catch {};
  // Code block wrappers with copy button
  el.querySelectorAll('pre').forEach(pre => {
    if (pre.closest('.code-block')) return;
    const w = document.createElement('div'); w.className = 'code-block';
    pre.parentNode.insertBefore(w, pre); w.appendChild(pre);
    const h = document.createElement('div'); h.className = 'code-header';
    const code = pre.querySelector('code');
    const lm = code?.className?.match(/language-(\S+)/);
    const ls = document.createElement('span'); ls.className = 'code-lang'; ls.textContent = lm ? lm[1] : '';
    const cb = document.createElement('button'); cb.className = 'copy-btn'; cb.textContent = 'Copy';
    cb.onclick = () => navigator.clipboard.writeText(code?.textContent||'').then(() => { cb.textContent='Copied!'; cb.classList.add('copied'); setTimeout(()=>{cb.textContent='Copy';cb.classList.remove('copied');},2000); });
    h.appendChild(ls); h.appendChild(cb); w.insertBefore(h, pre);
  });
  // Image carousel: group consecutive <img> or <p><img></p> into a carousel
  _buildImageCarousels(el);
}

function _buildImageCarousels(el) {
  // Collect runs of consecutive image elements (img or p>img with no other text)
  const children = Array.from(el.childNodes);
  let run = [];
  const runs = [];
  for (const node of children) {
    const isImg = node.nodeName === 'IMG' ||
      (node.nodeName === 'P' && node.querySelector('img') && (node.textContent||'').trim().length < 3);
    if (isImg) { run.push(node); }
    else { if (run.length >= 2) runs.push([...run]); run = []; }
  }
  if (run.length >= 2) runs.push([...run]);

  for (const imgNodes of runs) {
    const imgs = imgNodes.map(n => n.nodeName === 'IMG' ? n : n.querySelector('img')).filter(Boolean);
    if (imgs.length < 2) continue;

    // Build carousel
    const carousel = document.createElement('div');
    carousel.className = 'img-carousel';
    const track = document.createElement('div');
    track.className = 'carousel-track';
    const counter = document.createElement('div');
    counter.className = 'carousel-counter';
    let idx = 0;

    // Dedup by src
    const seen = new Set();
    const uniqueImgs = [];
    for (const img of imgs) {
      const src = img.src || img.getAttribute('src') || '';
      if (seen.has(src)) continue;
      seen.add(src);
      uniqueImgs.push(img);
    }

    for (const img of uniqueImgs) {
      const slide = document.createElement('div');
      slide.className = 'carousel-slide';
      const clone = img.cloneNode(true);
      clone.loading = 'lazy';
      clone.onerror = () => { slide.remove(); updateCounter(); };
      // Click to open full size
      clone.onclick = () => window.open(clone.src, '_blank');
      slide.appendChild(clone);
      track.appendChild(slide);
    }

    const btnL = document.createElement('button');
    btnL.className = 'carousel-btn carousel-prev';
    btnL.innerHTML = '&#8249;';
    const btnR = document.createElement('button');
    btnR.className = 'carousel-btn carousel-next';
    btnR.innerHTML = '&#8250;';

    function updateCounter() {
      const total = track.querySelectorAll('.carousel-slide').length;
      counter.textContent = `${idx + 1} / ${total}`;
      btnL.style.visibility = idx <= 0 ? 'hidden' : 'visible';
      btnR.style.visibility = idx >= total - 1 ? 'hidden' : 'visible';
    }
    function goTo(i) {
      const total = track.querySelectorAll('.carousel-slide').length;
      idx = Math.max(0, Math.min(i, total - 1));
      track.style.transform = `translateX(-${idx * 100}%)`;
      updateCounter();
    }
    btnL.onclick = () => goTo(idx - 1);
    btnR.onclick = () => goTo(idx + 1);

    carousel.appendChild(btnL);
    carousel.appendChild(track);
    carousel.appendChild(btnR);
    carousel.appendChild(counter);

    // Replace original nodes with carousel
    imgNodes[0].parentNode.insertBefore(carousel, imgNodes[0]);
    for (const n of imgNodes) n.remove();
    updateCounter();
  }
}

// ── Lightbox ──────────────────────────────────────────────────────
function openLightbox(src, allSrcs, startIndex) {
  if (document.querySelector('.lightbox-overlay')) return;
  let idx = startIndex || 0;
  const srcs = allSrcs || [src];

  const overlay = document.createElement('div');
  overlay.className = 'lightbox-overlay';
  overlay.onclick = (e) => { if (e.target === overlay) closeLightbox(); };

  const img = document.createElement('img');
  img.src = srcs[idx];
  img.alt = 'Full size';

  const closeBtn = document.createElement('button');
  closeBtn.className = 'lightbox-close';
  closeBtn.innerHTML = '&times;';
  closeBtn.onclick = closeLightbox;

  overlay.appendChild(img);
  overlay.appendChild(closeBtn);

  if (srcs.length > 1) {
    const prev = document.createElement('button');
    prev.className = 'lightbox-nav prev';
    prev.innerHTML = '&#8249;';
    prev.onclick = (e) => { e.stopPropagation(); idx = (idx - 1 + srcs.length) % srcs.length; img.src = srcs[idx]; updateCounter(); };
    const next = document.createElement('button');
    next.className = 'lightbox-nav next';
    next.innerHTML = '&#8250;';
    next.onclick = (e) => { e.stopPropagation(); idx = (idx + 1) % srcs.length; img.src = srcs[idx]; updateCounter(); };
    const counter = document.createElement('div');
    counter.className = 'lightbox-counter';
    function updateCounter() { counter.textContent = (idx + 1) + ' / ' + srcs.length; }
    updateCounter();
    overlay.appendChild(prev);
    overlay.appendChild(next);
    overlay.appendChild(counter);
  }

  document.body.appendChild(overlay);
  const onKey = (e) => {
    if (e.key === 'Escape') closeLightbox();
    else if (e.key === 'ArrowLeft' && srcs.length > 1) { idx = (idx - 1 + srcs.length) % srcs.length; img.src = srcs[idx]; }
    else if (e.key === 'ArrowRight' && srcs.length > 1) { idx = (idx + 1) % srcs.length; img.src = srcs[idx]; }
  };
  document.addEventListener('keydown', onKey);
  overlay._onKey = onKey;
}

function closeLightbox() {
  const overlay = document.querySelector('.lightbox-overlay');
  if (!overlay) return;
  if (overlay._onKey) document.removeEventListener('keydown', overlay._onKey);
  overlay.remove();
}

// ── Image URL Normalization (for dedup) ───────────────────────────
function normalizeImageUrl(url) {
  if (!url) return '';
  let u = url.split('?')[0].toLowerCase();
  // Strip common CDN size suffixes
  u = u.replace(/_\d+x\d+/g, '');
  u = u.replace(/-\d+x\d+/g, '');
  u = u.replace(/_thumb|_small|_medium|_large|_preview/g, '');
  return u;
}

function makeToolCard(tc, status) {
  const card = document.createElement('div'); card.className = 'tool-card'; card.dataset.toolName = tc.name; card.dataset.toolId = tc.id || '';
  const hdr = document.createElement('div'); hdr.className = 'tool-header';
  hdr.innerHTML = `<span class="tool-icon">\u2699</span><span class="tool-name">${esc(tc.name)}</span><span class="tool-status ${status}">${status==='running'?'running\u2026':status}</span>`;
  hdr.onclick = () => { const b=card.querySelector('.tool-body'); if(b) b.classList.toggle('open'); };
  const body = document.createElement('div'); body.className = 'tool-body';
  let args = tc.arguments||'{}'; try{args=JSON.stringify(JSON.parse(args),null,2)}catch{}
  body.innerHTML = `<pre><code>${esc(args)}</code></pre>`;
  card.appendChild(hdr); card.appendChild(body); return card;
}

function createThinkingCard() {
  const c = document.createElement('div'); c.className = 'thinking-card';
  const h = document.createElement('div'); h.className = 'thinking-header';
  h.innerHTML = '<span class="thinking-icon">\uD83D\uDCAD</span><span class="thinking-label">Thinking<span class="thinking-dots"></span></span>';
  h.onclick = () => { const b=c.querySelector('.thinking-body'); if(b) b.classList.toggle('open'); };
  const b = document.createElement('div'); b.className = 'thinking-body';
  c.appendChild(h); c.appendChild(b); return c;
}

// ── File Uploads ───────────────────────────────────────────────────
function handleFiles(input) {
  for (const file of Array.from(input.files)) {
    if (file.size > 10*1024*1024) { alert(`"${file.name}" too large (max 10MB)`); continue; }
    const r = new FileReader();
    if (file.type.startsWith('image/')) { r.onload=e=>{pendingFiles.push({name:file.name,type:file.type,data:e.target.result,preview:e.target.result});renderAttachments();}; r.readAsDataURL(file); }
    else { r.onload=e=>{pendingFiles.push({name:file.name,type:file.type||'text/plain',data:e.target.result,preview:null});renderAttachments();}; r.readAsText(file); }
  }
  input.value = '';
}

function renderAttachments() {
  const c = document.getElementById('attachments'); c.innerHTML = '';
  if (!pendingFiles.length) { c.classList.add('hidden'); updateActionBtn(); return; }
  c.classList.remove('hidden');
  pendingFiles.forEach((f,i) => {
    const ch = document.createElement('div'); ch.className = 'attach-chip';
    if (f.preview) { const img=document.createElement('img');img.src=f.preview;img.className='attach-thumb';ch.appendChild(img); }
    else { const ic=document.createElement('span');ic.className='attach-icon';ic.textContent='\uD83D\uDCC4';ch.appendChild(ic); }
    const n=document.createElement('span');n.className='attach-name';n.textContent=f.name;ch.appendChild(n);
    const rm=document.createElement('button');rm.className='attach-remove';rm.textContent='\u00d7';rm.onclick=()=>{pendingFiles.splice(i,1);renderAttachments();};ch.appendChild(rm);
    c.appendChild(ch);
  });
  updateActionBtn();
}

function setupDragDrop() {
  const m=document.getElementById('main'), r=document.getElementById('input-row');
  m.addEventListener('dragover',e=>{e.preventDefault();r.classList.add('drag-over');});
  m.addEventListener('dragleave',e=>{if(!m.contains(e.relatedTarget))r.classList.remove('drag-over');});
  m.addEventListener('drop',e=>{e.preventDefault();r.classList.remove('drag-over');if(e.dataTransfer?.files?.length)handleFiles({files:e.dataTransfer.files,value:''});});
}

// ── Send / Stream ──────────────────────────────────────────────────
async function send() {
  const input = document.getElementById('input');
  if (!input) { console.error('send: input element not found'); return; }
  const text = input.value.trim();
  if ((!text && !pendingFiles.length) || isStreaming) return;

  // Require a model to be selected AND loaded before sending
  if (!selectedModel || !selectedModelReady) {
    const modelBtn = document.getElementById('model-label');
    if (modelBtn) { modelBtn.style.animation = 'pulse 0.5s ease 2'; setTimeout(() => modelBtn.style.animation = '', 1200); }
    return;
  }

  if (!currentConvId) {
    try {
      const convBody = { model: selectedModel || '' };
      if (customSystemPrompt) {
        convBody.system_prompt = customSystemPrompt;
      } else if (selectedPersonality.id) {
        convBody.personality_id = selectedPersonality.id;
      }
      const r = await authFetch('/api/conversations', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(convBody) });
      if (!r.ok) { console.error('Create conv:', r.status); return; }
      currentConvId = (await r.json()).id;
    } catch (e) { console.error('Create conv:', e); return; }
  }

  const files = [...pendingFiles]; pendingFiles = []; renderAttachments();
  input.value = ''; input.style.height = 'auto';
  showView('chat');
  isStreaming = true; abortController = new AbortController(); updateActionBtn();

  let contentEl, bodyEl, aDiv, spinner;
  try {
    const displayText = text || (files.length ? 'What is this?' : '');
    appendMessage('user', displayText, null, files);
    scrollToBottom();

    const result = appendMessage('assistant', '');
    aDiv = result.div; contentEl = result.contentEl; bodyEl = result.bodyEl;
    aDiv.classList.add('streaming');
    // Images are now collected during streaming and rendered in the final pass

    spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    spinner.innerHTML = '<div class="spinner"></div><span class="spinner-text">Waiting for model...</span>';
    bodyEl.insertBefore(spinner, contentEl);
    scrollToBottom();
  } catch (domErr) {
    console.error('send DOM setup error:', domErr);
    isStreaming = false; abortController = null; updateActionBtn();
    return;
  }

  // === BUFFER-THEN-RENDER ARCHITECTURE ===
  // Buffer everything during streaming. Render ONCE when done.
  // No innerHTML updates during stream = no DOM thrashing.
  let fullContent = '';
  let thinkingCard = null, thinkingText = '', thinkingStart = 0, gotFirst = false;
  const collectedImages = []; // URLs collected from tool_results
  const collectedBase64 = []; // Base64 images from tool_results
  const seenImageBases = new Set(); // Dedup (normalized URLs)
  let hasError = false;
  let isDone = false; // Double-done guard
  const streamConvId = currentConvId; // Conversation switch guard
  const streamModel = selectedModel; // Model switch guard

  // Streaming stats
  let tokenCount = 0, streamStartTime = 0;
  let statsEl = null, statsInterval = null, toolTimerInterval = null;
  let activeToolCards = new Map(); // toolId → card element
  let toolStartTime = 0;

  try {
    const payload = { content: text || 'Describe the attached file(s).', tools_enabled: toolsEnabled };
    if (files.length) payload.files = files.map(f => ({ name: f.name, type: f.type, data: f.data }));

    const res = await authFetch(`/api/conversations/${streamConvId}/messages`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload), signal: abortController.signal,
    });

    if (!res.ok) {
      spinner.remove();
      contentEl.innerHTML = `<p style="color:var(--error)">Server error ${res.status}</p>`;
      aDiv.classList.remove('streaming'); isStreaming=false; abortController=null; updateActionBtn();
      return;
    }

    const reader = res.body.getReader(), dec = new TextDecoder();
    let buf = '';

    while (true) {
      // Guard: abort if conversation or model changed mid-stream
      if (currentConvId !== streamConvId || selectedModel !== streamModel) {
        try { reader.cancel(); } catch {}
        break;
      }
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split('\n'); buf = lines.pop() || '';
      let ev = null;
      for (const line of lines) {
        if (isDone) continue; // Ignore events after done
        if (line.startsWith('event: ')) { ev = line.substring(7).trim(); continue; }
        if (line.startsWith(':')) continue; // SSE comments (keepalives)
        if (!line.startsWith('data: ') || !ev) continue;
        let d; try { d = JSON.parse(line.substring(6)); } catch { ev=null; continue; }

        if (!gotFirst) {
          gotFirst = true;
          spinner.remove();
          streamStartTime = Date.now();
        }

        if (ev === 'thinking') {
          // Accumulate thinking in a collapsible card (hidden by default)
          if (!thinkingCard) {
            thinkingCard = createThinkingCard(); thinkingStart = Date.now();
            bodyEl.insertBefore(thinkingCard, contentEl);
          }
          thinkingText += d.text || '';
          const tb = thinkingCard.querySelector('.thinking-body');
          if (tb) tb.textContent = thinkingText;

        } else if (ev === 'token') {
          // Buffer content — show plain text streaming preview
          const chunk = d.text || '';
          fullContent += chunk;
          tokenCount += chunk.split(/\s+/).length;

          // Show streaming preview: line count + tail of content
          const lineCount = fullContent.split('\n').length;
          const preview = fullContent.length > 400 ? fullContent.slice(-400) : fullContent;
          contentEl.textContent = preview;

          // Show streaming stats
          if (!statsEl) {
            statsEl = document.createElement('div');
            statsEl.className = 'stream-stats';
            bodyEl.insertBefore(statsEl, contentEl);
            statsInterval = setInterval(() => {
              if (!statsEl) return;
              const elapsed = ((Date.now() - streamStartTime) / 1000).toFixed(0);
              const tps = streamStartTime ? (tokenCount / ((Date.now() - streamStartTime) / 1000)).toFixed(0) : '0';
              statsEl.innerHTML =
                `<span class="stat-tok">${tps} tok/s</span>` +
                `<span class="stat-lines">${lineCount} lines</span>` +
                `<span class="stat-time">${elapsed}s</span>`;
            }, 500);
          }
          scrollToBottom();

        } else if (ev === 'tool_start') {
          // Create a dedicated tool card (separate from thinking)
          // Suppress cards for intermediate tools the user doesn't need to see
          const suppressedTools = new Set(['browser', 'image', 'web', 'memory', 'vector']);
          toolStartTime = Date.now();
          if (!suppressedTools.has(d.name)) {
            const toolCard = makeToolCard({ id: d.id || '', name: d.name || 'tool', arguments: d.arguments || '{}' }, 'running');
            bodyEl.insertBefore(toolCard, contentEl);
            activeToolCards.set(d.id || d.name, toolCard);
          }

          // Live elapsed timer for tool execution
          if (toolTimerInterval) clearInterval(toolTimerInterval);
          toolTimerInterval = setInterval(() => {
            for (const [, card] of activeToolCards) {
              const statusEl = card.querySelector('.tool-status');
              if (statusEl && statusEl.classList.contains('running')) {
                const s = Math.round((Date.now() - toolStartTime) / 1000);
                statusEl.textContent = `running ${s}s\u2026`;
              }
            }
          }, 1000);

        } else if (ev === 'tool_result') {
          // Update the matching tool card
          const cardKey = d.id || d.name;
          const card = activeToolCards.get(cardKey);
          if (card) {
            const statusEl = card.querySelector('.tool-status');
            if (statusEl) {
              const elapsed = Math.round((Date.now() - toolStartTime) / 1000);
              if (d.isError) {
                statusEl.textContent = `error (${elapsed}s)`;
                statusEl.className = 'tool-status error';
              } else {
                statusEl.textContent = `done (${elapsed}s)`;
                statusEl.className = 'tool-status done';
              }
            }
            // Add brief result preview to tool body
            if (d.text) {
              const body = card.querySelector('.tool-body');
              if (body) {
                const preview = document.createElement('div');
                preview.style.cssText = 'margin-top:4px;font-size:11px;color:var(--text-muted);max-height:60px;overflow:hidden;';
                preview.textContent = (d.text || '').substring(0, 200);
                body.appendChild(preview);
              }
            }
            activeToolCards.delete(cardKey);
          }
          if (toolTimerInterval && activeToolCards.size === 0) {
            clearInterval(toolTimerInterval); toolTimerInterval = null;
          }

          // COLLECT images — don't render yet (dedup with normalized URLs)
          if (d.images?.length > 0) {
            for (const img of d.images) {
              collectedBase64.push(img);
            }
          }
          if (d.imageUrls?.length > 0) {
            for (const url of d.imageUrls) {
              const norm = normalizeImageUrl(url);
              if (!seenImageBases.has(norm)) {
                seenImageBases.add(norm);
                collectedImages.push(url);
              }
            }
          }

        } else if (ev === 'error') {
          hasError = true;
          contentEl.innerHTML += `<p style="color:var(--error)">${esc(d.message)}</p>`;

        } else if (ev === 'status') {
          contentEl.textContent = d.text || 'Loading...';
        }
        ev = null;
      }
    }
  } catch (e) {
    if (e.name !== 'AbortError') contentEl.innerHTML += `<p style="color:var(--error)">Error: ${esc(e.message)}</p>`;
  }

  // Clean up intervals
  if (statsInterval) { clearInterval(statsInterval); statsInterval = null; }
  if (toolTimerInterval) { clearInterval(toolTimerInterval); toolTimerInterval = null; }
  isDone = true;

  // === SINGLE RENDER PASS — everything at once ===

  // 1. Close thinking card
  if (thinkingCard) {
    const s = Math.round((Date.now()-thinkingStart)/1000);
    const l = thinkingCard.querySelector('.thinking-label');
    if (l) l.innerHTML = `Thought for ${s}s`;
    thinkingCard.classList.add('done');
  }
  if (!gotFirst) spinner.remove();

  // Remove streaming stats
  if (statsEl) { statsEl.remove(); statsEl = null; }

  // 2. Render images in responsive grid (collected from ALL tool_results)
  const totalImages = collectedBase64.length + collectedImages.length;
  if (totalImages > 0) {
    const imgGrid = document.createElement('div');
    imgGrid.className = 'tool-images' + (totalImages === 1 ? ' single-image' : '');
    const allSrcs = [];

    for (const img of collectedBase64) {
      const src = `data:${img.mimeType};base64,${img.data}`;
      allSrcs.push(src);
      const el = document.createElement('img');
      el.src = src;
      el.loading = 'lazy';
      el.decoding = 'async';
      el.alt = 'Generated image';
      el.onerror = () => { el.style.display = 'none'; };
      imgGrid.appendChild(el);
    }
    for (const url of collectedImages) {
      allSrcs.push(url);
      const el = document.createElement('img');
      el.src = url;
      el.loading = 'lazy';
      el.decoding = 'async';
      el.alt = 'Search result';
      el.onerror = () => { el.style.display = 'none'; };
      imgGrid.appendChild(el);
    }
    // Wire lightbox on all images
    imgGrid.querySelectorAll('img').forEach((img, i) => {
      img.onclick = () => openLightbox(img.src, allSrcs, i);
    });
    bodyEl.insertBefore(imgGrid, contentEl);
  }

  // 3. Render final content (one markdown parse, one DOM write)
  if (fullContent && !hasError) {
    // Only strip markdown images if we got images from tool_result events;
    // otherwise keep them since they're the model's only way to show images
    const hasToolImages = collectedImages.length > 0 || collectedBase64.length > 0;
    contentEl.innerHTML = renderMd(cleanResponse(fullContent, hasToolImages));
  } else if (!fullContent && !hasError && !contentEl.innerHTML.trim()) {
    contentEl.innerHTML = '<p style="color:var(--text-muted);font-style:italic">Model returned no response. Try rephrasing or switching models.</p>';
  }

  // 4. Post-process ENTIRE bodyEl (code highlighting, etc.)
  aDiv.classList.remove('streaming');
  // Clean up any orphan spinners
  document.querySelectorAll('.loading-spinner').forEach(el => el.remove());
  postProcess(bodyEl);
  scrollToBottom();
  isStreaming = false; abortController = null; updateActionBtn();
  await loadConversations();
}

function stopGeneration() {
  if (abortController) {
    try { abortController.abort(); } catch {}
    // Force reset state in case abort doesn't trigger catch
    setTimeout(() => {
      if (isStreaming) {
        isStreaming = false;
        abortController = null;
        updateActionBtn();
        document.querySelectorAll('.streaming').forEach(el => el.classList.remove('streaming'));
        document.querySelectorAll('.loading-spinner').forEach(el => el.remove());
      }
    }, 500);
  }
}
function handleAction() { isStreaming ? stopGeneration() : send(); }

function updateActionBtn() {
  const btn=document.getElementById('action-btn'), si=document.getElementById('icon-send'), sti=document.getElementById('icon-stop');
  if (isStreaming) { si.classList.add('hidden'); sti.classList.remove('hidden'); btn.classList.add('stop-mode'); btn.disabled=false; btn.title='Stop'; }
  else { si.classList.remove('hidden'); sti.classList.add('hidden'); btn.classList.remove('stop-mode'); btn.title=selectedModelReady?'Send':'Select and load a model first'; btn.disabled=!selectedModelReady||(!document.getElementById('input').value.trim()&&!pendingFiles.length); }
}

function handleKey(e) { if (e.key==='Enter'&&!e.shiftKey) { e.preventDefault(); if(!isStreaming) send(); } }
function handleInput() { const i=document.getElementById('input'); i.style.height='auto'; i.style.height=Math.min(i.scrollHeight,200)+'px'; updateActionBtn(); }
function toggleSidebar() { const s=document.getElementById('sidebar'); isMobile()?s.classList.toggle('open'):s.classList.toggle('collapsed'); }
function useSuggestion(t) { const i=document.getElementById('input'); i.value=t; i.focus(); handleInput(); }
function scrollToBottom() { const v=document.getElementById('chat-view'); if(v) v.scrollTop=v.scrollHeight; }
function esc(t) { if(!t) return ''; const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }

// ── Response Sanitizer ─────────────────────────────────────────────
// Strip raw data artifacts the LLM may echo from tool results.
// IMPORTANT: Only strip actual binary/base64 artifacts, not legitimate content.
function cleanResponse(text, stripMarkdownImages) {
  if (!text) return '';
  let c = text;
  // Strip data: URIs (base64-encoded inline data)
  c = c.replace(/data:[a-z/+]+;base64,[A-Za-z0-9+/=]+/gi, '');
  // Strip base64 blobs only when preceded by ;base64, marker
  c = c.replace(/;base64,[A-Za-z0-9+/=]{100,}/g, ';base64,[data removed]');
  // Strip raw byte strings b'...' (Python binary literals)
  c = c.replace(/b'[^']{50,}'/g, '');
  // Strip hex dumps \xff\xd8...
  c = c.replace(/(\\x[0-9a-fA-F]{2}){10,}/g, '');
  // Only strip markdown images if tool_result already provided them (prevents dupes).
  // If no tool_result images came through, keep markdown images — they're the only source.
  if (stripMarkdownImages) {
    c = c.replace(/!\[[^\]]*\]\([^)]+\)/g, '');
  }
  // Clean up resulting empty lines
  c = c.replace(/\n{3,}/g, '\n\n').trim();
  return c;
}

// ── Personalities ─────────────────────────────────────────────────
async function loadPersonalities() {
  try {
    const url = selectedModel
      ? '/api/personalities?model=' + encodeURIComponent(selectedModel)
      : '/api/personalities';
    const res = await authFetch(url);
    const data = await res.json();
    allPersonalities = data.personalities || [];
  } catch (e) { console.warn('loadPersonalities:', e); }

  // Restore from localStorage
  const stored = localStorage.getItem('dartboard-personality');
  if (stored) {
    try {
      const p = JSON.parse(stored);
      if (p.id) selectedPersonality = p;
    } catch { /* ignore */ }
  }
  const storedCustom = localStorage.getItem('dartboard-custom-prompt');
  if (storedCustom) customSystemPrompt = storedCustom;

  // If the selected personality is no longer in the available list
  // (model-restricted and current model changed), fall back to General.
  if (selectedPersonality.id &&
      !allPersonalities.some(p => p.id === selectedPersonality.id)) {
    const general = allPersonalities.find(p => p.id === 'general');
    if (general) {
      selectedPersonality = { id: general.id, name: general.name, icon: general.icon };
      localStorage.setItem('dartboard-personality', JSON.stringify(selectedPersonality));
    }
  }

  updatePersonalityDisplay();
}

function updatePersonalityDisplay() {
  const iconEl = document.getElementById('personality-icon');
  const nameEl = document.getElementById('personality-name');
  if (iconEl && nameEl) {
    if (customSystemPrompt) {
      iconEl.textContent = '\u270D';
      nameEl.textContent = 'Custom Prompt';
    } else {
      iconEl.textContent = selectedPersonality.icon || '\u{1F9E0}';
      nameEl.textContent = selectedPersonality.name || 'General Assistant';
    }
  }
}

function togglePersonalityModal() {
  const modal = document.getElementById('personality-modal');
  if (modal.classList.contains('hidden')) {
    renderPersonalityGrid();
    modal.classList.remove('hidden');
  } else {
    modal.classList.add('hidden');
  }
}

function closePersonalityModal() {
  document.getElementById('personality-modal').classList.add('hidden');
}

function renderPersonalityGrid() {
  const container = document.getElementById('personality-categories');
  container.innerHTML = '';

  // Group by category
  const groups = {};
  for (const p of allPersonalities) {
    if (!groups[p.category]) groups[p.category] = [];
    groups[p.category].push(p);
  }

  for (const [category, items] of Object.entries(groups)) {
    const section = document.createElement('div');
    section.className = 'personality-section';
    const heading = document.createElement('h3');
    heading.className = 'personality-category';
    heading.textContent = category;
    section.appendChild(heading);

    const grid = document.createElement('div');
    grid.className = 'personality-grid';
    for (const p of items) {
      const card = document.createElement('button');
      card.className = 'personality-card' + (!customSystemPrompt && p.id === selectedPersonality.id ? ' active' : '');
      card.dataset.id = p.id;
      card.onclick = () => pickPersonality(p);
      card.innerHTML =
        '<span class="p-icon">' + p.icon + '</span>' +
        '<span class="p-name">' + esc(p.name) + '</span>' +
        '<span class="p-desc">' + esc(p.description) + '</span>';
      grid.appendChild(card);
    }
    section.appendChild(grid);
    container.appendChild(section);
  }
}

function pickPersonality(p) {
  selectedPersonality = { id: p.id, name: p.name, icon: p.icon };
  customSystemPrompt = null;
  localStorage.setItem('dartboard-personality', JSON.stringify(selectedPersonality));
  localStorage.removeItem('dartboard-custom-prompt');
  updatePersonalityDisplay();
  closePersonalityModal();
}

function filterPersonalities() {
  const q = document.getElementById('personality-search').value.toLowerCase();
  document.querySelectorAll('.personality-card').forEach(card => {
    const text = (card.querySelector('.p-name')?.textContent || '') + ' ' + (card.querySelector('.p-desc')?.textContent || '');
    card.style.display = text.toLowerCase().includes(q) ? '' : 'none';
  });
  document.querySelectorAll('.personality-section').forEach(sec => {
    const hasVisible = sec.querySelector('.personality-card:not([style*="display: none"])');
    sec.style.display = hasVisible ? '' : 'none';
  });
}

function toggleCustomPrompt() {
  const area = document.getElementById('custom-prompt-area');
  area.classList.toggle('hidden');
  if (!area.classList.contains('hidden')) {
    const input = document.getElementById('custom-prompt-input');
    if (customSystemPrompt) input.value = customSystemPrompt;
    input.focus();
  }
}

function applyCustomPrompt() {
  const input = document.getElementById('custom-prompt-input');
  const prompt = input.value.trim();
  if (!prompt) return;
  customSystemPrompt = prompt;
  localStorage.setItem('dartboard-custom-prompt', prompt);
  updatePersonalityDisplay();
  closePersonalityModal();
}

// ── Model Unload on Page Exit ──────────────────────────────────────
window.addEventListener('beforeunload', () => {
  if (selectedModel && localStorage.getItem('dartboard-jwt')) {
    // sendBeacon can't carry auth headers — include token in body
    navigator.sendBeacon('/api/unload', new Blob([JSON.stringify({ model: selectedModel })], { type: 'application/json' }));
  }
});
