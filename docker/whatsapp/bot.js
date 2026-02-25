/**
 * aichat WhatsApp bot — Baileys + LM Studio with full MCP tool support.
 *
 * Pairing: visit http://localhost:8097 and scan the QR code once.
 * Auth credentials persist in the Docker volume /data/auth.
 *
 * Environment variables (all optional, shown with defaults):
 *   LM_STUDIO_URL   http://host.docker.internal:1234
 *   MCP_URL         http://aichat-mcp:8096
 *   MEMORY_URL      http://aichat-memory:8094
 *   LM_MODEL        local-model
 *   MAX_HISTORY     20        max conversation turns stored per contact
 *   MAX_TOOL_ITER   5         max tool-calling rounds per message
 *   MAX_TOKENS      1024
 *   TEMPERATURE     0.7
 *   ALLOW_GROUPS    false     set to "true" to respond in group chats
 *   GROUP_PREFIX    !ai       prefix required for group messages
 *   BOT_NAME        AI Assistant
 *   SYSTEM_PROMPT   (auto)
 */

import makeWASocket, {
  useMultiFileAuthState,
  DisconnectReason,
  makeCacheableSignalKeyStore,
  Browsers,
  isJidBroadcast,
  isJidGroup,
  fetchLatestBaileysVersion,
} from '@whiskeysockets/baileys'
import { Boom } from '@hapi/boom'
import axios from 'axios'
import QRCode from 'qrcode'
import pino from 'pino'
import { createServer } from 'node:http'

// ─── Configuration ───────────────────────────────────────────────────────────

const LM_STUDIO_URL  = process.env.LM_STUDIO_URL  ?? 'http://host.docker.internal:1234'
const MCP_URL        = process.env.MCP_URL         ?? 'http://aichat-mcp:8096'
const MEMORY_URL     = process.env.MEMORY_URL      ?? 'http://aichat-memory:8094'
const LM_MODEL       = process.env.LM_MODEL        ?? 'local-model'
const MAX_HISTORY    = Number(process.env.MAX_HISTORY    ?? '20')
const MAX_TOOL_ITER  = Number(process.env.MAX_TOOL_ITER  ?? '5')
const MAX_TOKENS     = Number(process.env.MAX_TOKENS     ?? '1024')
const TEMPERATURE    = Number(process.env.TEMPERATURE    ?? '0.7')
const ALLOW_GROUPS   = process.env.ALLOW_GROUPS === 'true'
const GROUP_PREFIX   = (process.env.GROUP_PREFIX ?? '!ai').toLowerCase()
const BOT_NAME       = process.env.BOT_NAME ?? 'AI Assistant'
const SYSTEM_PROMPT  = process.env.SYSTEM_PROMPT ??
  `You are ${BOT_NAME}, a helpful AI assistant available via WhatsApp. ` +
  `Be concise and conversational — your responses will be read on a mobile phone. ` +
  `You have access to tools: use them proactively when they would help answer the question. ` +
  `When you take a screenshot or fetch an image, it will be sent directly in this chat. ` +
  `Today's date is ${new Date().toDateString()}.`

const logger = pino({ level: 'warn' })

// ─── QR / Status HTTP server (port 8097) ─────────────────────────────────────

let _qrDataURL  = null   // base64 PNG data URL of current QR code
let _status     = 'starting'
let _myJid      = null

function buildHTML () {
  const statusBadge = _status === 'connected'
    ? `<div class="badge connected">✅ Connected${_myJid ? ' · ' + _myJid.split(':')[0] : ''}</div>`
    : _qrDataURL
      ? `<div class="badge waiting">⏳ Scan QR in WhatsApp to pair</div>
         <img src="${_qrDataURL}" alt="WhatsApp QR Code">
         <p class="hint">WhatsApp → Linked Devices → Link a Device → scan above<br>
         This page refreshes automatically every 5 seconds.</p>`
      : `<div class="badge waiting">⏳ Generating QR code…</div>`

  return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>aichat WhatsApp</title>
  <meta http-equiv="refresh" content="5">
  <style>
    *{box-sizing:border-box;margin:0;padding:0}
    body{font-family:sans-serif;background:#0d1117;color:#e6edf3;display:flex;
         flex-direction:column;align-items:center;justify-content:center;min-height:100vh;gap:1.5rem;padding:2rem}
    h1{color:#25d366;font-size:1.8rem}
    img{max-width:280px;border:4px solid #25d366;border-radius:12px;background:#fff;padding:8px}
    .badge{padding:.5rem 1.2rem;border-radius:8px;font-size:1.1rem;font-weight:600}
    .connected{background:#0d2818;color:#25d366}
    .waiting{background:#1c1c00;color:#ffd60a}
    .hint{color:#8b949e;font-size:.85rem;line-height:1.6;text-align:center}
    footer{color:#484f58;font-size:.75rem}
  </style>
</head>
<body>
  <h1>aichat WhatsApp Bot</h1>
  ${statusBadge}
  <footer>status: ${_status}</footer>
</body>
</html>`
}

const httpServer = createServer((req, res) => {
  if (req.url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' })
    res.end(JSON.stringify({ status: _status, jid: _myJid }))
    return
  }
  res.writeHead(200, { 'Content-Type': 'text/html' })
  res.end(buildHTML())
})

httpServer.listen(8097, () =>
  console.log('[whatsapp] QR/status page → http://localhost:8097'),
)

// ─── MCP tool schema cache ────────────────────────────────────────────────────

let _toolCache     = []       // OpenAI-format tool definitions
let _toolCacheTime = 0        // monotonic ms
const TOOL_TTL_MS  = 60_000   // re-fetch every 60 s

/** Convert one MCP tool schema to OpenAI function-calling format. */
function mcpToOpenAI (mcpTool) {
  return {
    type: 'function',
    function: {
      name:        mcpTool.name,
      description: mcpTool.description,
      parameters:  mcpTool.inputSchema ?? { type: 'object', properties: {} },
    },
  }
}

/** Return cached (or freshly fetched) OpenAI tool list from the MCP server. */
async function getTools () {
  const now = Date.now()
  if (_toolCache.length > 0 && now - _toolCacheTime < TOOL_TTL_MS) return _toolCache

  try {
    const { data } = await axios.post(
      `${MCP_URL}/mcp`,
      { jsonrpc: '2.0', id: 1, method: 'tools/list', params: {} },
      { timeout: 10_000 },
    )
    const tools = data?.result?.tools ?? []
    _toolCache     = tools.map(mcpToOpenAI)
    _toolCacheTime = now
    console.log(`[whatsapp] MCP tools refreshed: ${_toolCache.length} tools available`)
  } catch (err) {
    console.warn('[whatsapp] Could not fetch MCP tools:', err.message)
  }
  return _toolCache
}

// ─── Conversation history (aichat-memory) ────────────────────────────────────

/** Sanitise a JID into a memory key. */
const memKey = jid => `whatsapp:${jid.replace(/[:@.]/g, '_')}`

/** Fetch conversation history for a contact. Returns [] on any error. */
async function getHistory (jid) {
  try {
    const { data } = await axios.get(
      `${MEMORY_URL}/recall`,
      { params: { key: memKey(jid) }, timeout: 5_000 },
    )
    if (data.found && data.entries.length > 0) {
      return JSON.parse(data.entries[0].value)
    }
  } catch { /* memory unavailable — start fresh */ }
  return []
}

/** Save conversation history for a contact (keeps last MAX_HISTORY entries). */
async function saveHistory (jid, messages) {
  try {
    await axios.post(
      `${MEMORY_URL}/store`,
      { key: memKey(jid), value: JSON.stringify(messages.slice(-MAX_HISTORY)) },
      { timeout: 5_000 },
    )
  } catch { /* best-effort */ }
}

/** Delete conversation history for a contact. */
async function clearHistory (jid) {
  try {
    await axios.delete(`${MEMORY_URL}/delete`, { params: { key: memKey(jid) }, timeout: 5_000 })
  } catch { /* best-effort */ }
}

// ─── LM Studio tool-calling loop ─────────────────────────────────────────────

/**
 * Call the MCP server with a single tool invocation.
 * Returns { text, images } where:
 *   text   — string to use as the tool result in the conversation
 *   images — array of { data: base64String, mimeType } ready to send to WhatsApp
 */
async function callMCPTool (toolName, toolArgs) {
  const { data } = await axios.post(
    `${MCP_URL}/mcp`,
    {
      jsonrpc: '2.0',
      id:      Math.floor(Math.random() * 1e9),
      method:  'tools/call',
      params:  { name: toolName, arguments: toolArgs },
    },
    { timeout: 60_000 },
  )

  const content = data?.result?.content ?? []
  const textParts  = []
  const images     = []

  for (const block of content) {
    if (block.type === 'text')  textParts.push(block.text)
    if (block.type === 'image') images.push({ data: block.data, mimeType: block.mimeType ?? 'image/png' })
  }

  return { text: textParts.join('\n') || '(tool returned no text)', images }
}

/**
 * Run the full LM Studio conversation + tool-calling loop.
 * Returns the final assistant text and any images produced by tools.
 */
async function runLLM (jid, userMessage, sock) {
  const history = await getHistory(jid)
  const tools   = await getTools()

  // Build the messages array: system + prior history + current user message
  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...history,
    { role: 'user', content: userMessage },
  ]

  let allImages = []   // images emitted by tools during this turn

  for (let iter = 0; iter < MAX_TOOL_ITER; iter++) {
    const payload = {
      model:       LM_MODEL,
      messages,
      stream:      false,
      max_tokens:  MAX_TOKENS,
      temperature: TEMPERATURE,
    }
    if (tools.length > 0) {
      payload.tools       = tools
      payload.tool_choice = 'auto'
    }

    const { data } = await axios.post(
      `${LM_STUDIO_URL}/v1/chat/completions`,
      payload,
      { timeout: 120_000 },
    )

    const choice   = data.choices?.[0]
    const msg      = choice?.message
    const toolCalls = msg?.tool_calls

    if (!toolCalls || toolCalls.length === 0) {
      // Final text answer
      const finalText = (msg?.content ?? '').trim() || '(no response)'

      // Persist only the clean user/assistant turn (not tool details)
      await saveHistory(jid, [
        ...history,
        { role: 'user',      content: userMessage },
        { role: 'assistant', content: finalText   },
      ])

      return { text: finalText, images: allImages }
    }

    // Append the assistant's tool-call message to the conversation
    messages.push(msg)

    // Execute all tool calls in parallel
    const toolResults = await Promise.all(
      toolCalls.map(async tc => {
        const toolName = tc.function.name
        let   toolArgs = {}
        try { toolArgs = JSON.parse(tc.function.arguments ?? '{}') } catch { /* ignore */ }

        console.log(`[whatsapp] → tool: ${toolName}(${JSON.stringify(toolArgs).slice(0, 120)})`)

        let result
        try {
          result = await callMCPTool(toolName, toolArgs)
        } catch (err) {
          result = { text: `Tool error: ${err.message}`, images: [] }
        }

        // Send any images from this tool to WhatsApp immediately (fire-and-forget)
        for (const img of result.images) {
          try {
            await sock.sendMessage(jid, {
              image:   Buffer.from(img.data, 'base64'),
              caption: `📸 ${toolName}`,
              mimetype: img.mimeType,
            })
          } catch (err) {
            console.warn('[whatsapp] Failed to send tool image:', err.message)
          }
        }
        allImages = allImages.concat(result.images)

        return { id: tc.id, content: result.text }
      }),
    )

    // Append tool results back into the messages array
    for (const tr of toolResults) {
      messages.push({ role: 'tool', tool_call_id: tr.id, content: tr.content })
    }

    // Continue loop — LM Studio will now see the tool results
  }

  // Exhausted iterations — return whatever the last text content was
  const lastText = messages.findLast(m => m.role === 'assistant' && typeof m.content === 'string')
  const fallback = lastText?.content?.trim() || '(max tool iterations reached)'
  await saveHistory(jid, [
    ...history,
    { role: 'user',      content: userMessage },
    { role: 'assistant', content: fallback    },
  ])
  return { text: fallback, images: allImages }
}

// ─── Text extraction from WhatsApp message ───────────────────────────────────

function extractText (msg) {
  const m = msg?.message
  if (!m) return null
  return (
    m.conversation                    ??
    m.extendedTextMessage?.text       ??
    m.imageMessage?.caption           ??
    m.videoMessage?.caption           ??
    m.documentMessage?.caption        ??
    null
  )
}

// ─── Core message handler ─────────────────────────────────────────────────────

async function handleMessage (sock, msg) {
  // Ignore own messages, broadcasts, and non-text
  if (!msg.message)         return
  if (msg.key.fromMe)       return
  if (isJidBroadcast(msg.key.remoteJid)) return

  const isGroup = isJidGroup(msg.key.remoteJid)
  const jid     = msg.key.remoteJid

  if (isGroup && !ALLOW_GROUPS) return

  let text = extractText(msg)
  if (!text) return

  // In groups, require the configured prefix and strip it
  if (isGroup) {
    const lower = text.toLowerCase().trim()
    if (!lower.startsWith(GROUP_PREFIX)) return
    text = text.slice(GROUP_PREFIX.length).trim()
    if (!text) return
  }

  const senderName = msg.pushName ?? jid.split('@')[0]
  console.log(`[whatsapp] ${senderName} (${jid}): ${text.slice(0, 100)}`)

  // ── Built-in commands ────────────────────────────────────────────────────
  const cmd = text.trim().toLowerCase()

  if (cmd === '!clear' || cmd === '/clear') {
    await clearHistory(jid)
    await sock.sendMessage(jid, { text: '🗑️ Conversation history cleared.' }, { quoted: msg })
    return
  }

  if (cmd === '!help' || cmd === '/help') {
    const helpText =
      `*${BOT_NAME}* — aichat WhatsApp Bot\n\n` +
      `I'm an AI assistant with access to the following tools:\n` +
      `• 🌐 Web search & screenshots\n` +
      `• 💾 Memory & database storage\n` +
      `• 🔎 RSS feed research\n` +
      `• 🛠️ Custom tools\n\n` +
      `*Commands:*\n` +
      `!help — show this message\n` +
      `!clear — wipe conversation history\n\n` +
      (ALLOW_GROUPS ? `*Groups:* prefix messages with \`${GROUP_PREFIX}\`\n` : '') +
      `_Powered by LM Studio + aichat MCP_`
    await sock.sendMessage(jid, { text: helpText }, { quoted: msg })
    return
  }

  // ── AI response ──────────────────────────────────────────────────────────
  try {
    await sock.sendPresenceUpdate('composing', jid)

    let result
    try {
      result = await runLLM(jid, text, sock)
    } catch (err) {
      console.error('[whatsapp] LM Studio error:', err.message)
      await sock.sendPresenceUpdate('paused', jid)
      await sock.sendMessage(jid, {
        text: '⚠️ The AI is unavailable right now.\nMake sure LM Studio is running with a model loaded.',
      }, { quoted: msg })
      return
    }

    await sock.sendPresenceUpdate('paused', jid)

    // Only send a text reply if the response isn't empty or the entire answer
    // was conveyed via images (rare — e.g. screenshot-only answer)
    if (result.text && result.text !== '(no response)') {
      await sock.sendMessage(jid, { text: result.text }, { quoted: msg })
    }
  } catch (err) {
    console.error('[whatsapp] Unhandled error in handleMessage:', err)
    try { await sock.sendPresenceUpdate('paused', jid) } catch { /* ignore */ }
  }
}

// ─── WhatsApp connection ──────────────────────────────────────────────────────

async function connectToWhatsApp () {
  const { state, saveCreds } = await useMultiFileAuthState('/data/auth')
  const { version } = await fetchLatestBaileysVersion()

  const sock = makeWASocket({
    version,
    auth: {
      creds: state.creds,
      keys:  makeCacheableSignalKeyStore(state.keys, logger),
    },
    logger,
    printQRInTerminal:          false,   // we handle QR ourselves
    browser:                    Browsers.appropriate('Chrome'),
    syncFullHistory:            false,   // don't download message history
    generateHighQualityLinkPreview: false,
    markOnlineOnConnect:        true,
  })

  // ── Connection lifecycle ─────────────────────────────────────────────────
  sock.ev.on('connection.update', async update => {
    const { connection, lastDisconnect, qr } = update

    if (qr) {
      _status    = 'waiting_for_qr'
      _qrDataURL = await QRCode.toDataURL(qr)
      console.log('[whatsapp] QR ready — scan at http://localhost:8097')
    }

    if (connection === 'close') {
      const statusCode = new Boom(lastDisconnect?.error)?.output?.statusCode
      _status    = 'disconnected'
      _qrDataURL = null

      if (statusCode === DisconnectReason.loggedOut) {
        console.log(
          '[whatsapp] Logged out of WhatsApp.\n' +
          '[whatsapp] To re-pair: delete the auth volume data and restart.\n' +
          '[whatsapp]   docker exec aichat-aichat-whatsapp-1 rm -rf /data/auth/*\n' +
          '[whatsapp]   docker restart aichat-aichat-whatsapp-1',
        )
        // Do NOT reconnect — needs a fresh QR scan
      } else {
        console.log(`[whatsapp] Disconnected (code ${statusCode}), reconnecting in 5 s…`)
        setTimeout(connectToWhatsApp, 5_000)
      }
    }

    if (connection === 'open') {
      _status    = 'connected'
      _myJid     = sock.user?.id ?? null
      _qrDataURL = null
      console.log(`[whatsapp] ✅ Connected as ${_myJid}`)
      // Prime the tool cache on connection
      getTools().catch(() => {})
    }
  })

  sock.ev.on('creds.update', saveCreds)

  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return
    for (const msg of messages) {
      await handleMessage(sock, msg)
    }
  })
}

// ─── Entry point ─────────────────────────────────────────────────────────────

console.log('[whatsapp] Starting aichat WhatsApp bot…')
console.log(`[whatsapp] LM Studio : ${LM_STUDIO_URL}`)
console.log(`[whatsapp] MCP server: ${MCP_URL}`)
console.log(`[whatsapp] Memory    : ${MEMORY_URL}`)
console.log(`[whatsapp] Groups    : ${ALLOW_GROUPS ? `enabled (prefix: ${GROUP_PREFIX})` : 'disabled'}`)

connectToWhatsApp().catch(err => {
  console.error('[whatsapp] Fatal startup error:', err)
  process.exit(1)
})
