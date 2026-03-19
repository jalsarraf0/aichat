#!/usr/bin/env python3
"""Dartboard Auth System — reverse proxy with JWT auth (8200) + admin panel (8247).

Security features:
- IP ban after 3 failed login attempts (permanent until admin unban)
- Per-user chat isolation via X-Auth-User header forwarding
- Admin unban restricted to primary admin (ADMIN_USER) only
"""

import os
import sys
import time
import logging
import threading
from datetime import datetime, timedelta, timezone
from functools import wraps

import bcrypt
import jwt as pyjwt
import psycopg2
from psycopg2.extras import RealDictCursor
from flask import Flask, request, Response, jsonify, g, make_response
import requests as http_req

# ── Config ──────────────────────────────────────────────────────────
DARTBOARD_URL = os.environ.get('DARTBOARD_URL', 'http://app:8200')
DATABASE_URL = os.environ['DATABASE_URL']
JWT_SECRET = os.environ['JWT_SECRET']
ADMIN_USER = os.environ.get('ADMIN_USER', 'admin')
ADMIN_INITIAL_PASSWORD = os.environ.get('ADMIN_INITIAL_PASSWORD', '')
JWT_EXPIRY_HOURS = int(os.environ.get('JWT_EXPIRY_HOURS', '168'))  # 7 days
MAX_LOGIN_ATTEMPTS = 3  # Permanent IP ban after this many failures

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
log = logging.getLogger('auth')

# ── Database ────────────────────────────────────────────────────────
def get_conn():
    return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)


def db_retry(fn, retries=30, delay=2):
    """Retry DB operations during startup while postgres initializes."""
    for i in range(retries):
        try:
            return fn()
        except psycopg2.OperationalError:
            if i < retries - 1:
                log.info(f'Waiting for database... ({i+1}/{retries})')
                time.sleep(delay)
            else:
                raise


def init_db():
    """Create tables and seed admin user."""
    def _init():
        conn = get_conn()
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(64) UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                status VARCHAR(20) NOT NULL DEFAULT 'pending',
                role VARCHAR(20) NOT NULL DEFAULT 'user',
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        # IP ban tracking — permanent bans after MAX_LOGIN_ATTEMPTS failures
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ip_bans (
                id SERIAL PRIMARY KEY,
                ip_address VARCHAR(45) UNIQUE NOT NULL,
                reason TEXT NOT NULL DEFAULT 'Too many failed login attempts',
                banned_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                banned_by VARCHAR(64) NOT NULL DEFAULT 'system'
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS login_attempts (
                id SERIAL PRIMARY KEY,
                ip_address VARCHAR(45) NOT NULL,
                username VARCHAR(64) NOT NULL DEFAULT '',
                attempted_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_login_attempts_ip
                ON login_attempts(ip_address)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_ip_bans_ip
                ON ip_bans(ip_address)
        """)
        # Seed admin user if not exists
        cur.execute("SELECT id FROM users WHERE username = %s", (ADMIN_USER,))
        if not cur.fetchone() and ADMIN_INITIAL_PASSWORD:
            pw_hash = bcrypt.hashpw(ADMIN_INITIAL_PASSWORD.encode(), bcrypt.gensalt(12)).decode()
            cur.execute(
                "INSERT INTO users (username, password_hash, status, role) VALUES (%s, %s, 'approved', 'admin')",
                (ADMIN_USER, pw_hash)
            )
            log.info(f'Created admin user: {ADMIN_USER}')
        conn.close()
    db_retry(_init)


# ── IP Utilities ────────────────────────────────────────────────────
def get_client_ip():
    """Get the real client IP, handling proxies and Docker networking."""
    forwarded = request.headers.get('X-Forwarded-For', '')
    if forwarded:
        return forwarded.split(',')[0].strip()
    real_ip = request.headers.get('X-Real-IP', '')
    if real_ip:
        return real_ip.strip()
    return request.remote_addr or '0.0.0.0'


def is_ip_banned(cur, ip):
    """Check if an IP is permanently banned."""
    cur.execute("SELECT id FROM ip_bans WHERE ip_address = %s", (ip,))
    return cur.fetchone() is not None


def record_failed_attempt(cur, ip, username):
    """Record a failed login attempt and return the total count for this IP."""
    cur.execute(
        "INSERT INTO login_attempts (ip_address, username) VALUES (%s, %s)",
        (ip, username)
    )
    cur.execute(
        "SELECT COUNT(*) as cnt FROM login_attempts WHERE ip_address = %s",
        (ip,)
    )
    return cur.fetchone()['cnt']


def ban_ip(cur, ip, reason):
    """Permanently ban an IP address."""
    cur.execute(
        "INSERT INTO ip_bans (ip_address, reason, banned_by) VALUES (%s, %s, 'system') "
        "ON CONFLICT (ip_address) DO NOTHING",
        (ip, reason)
    )
    log.warning(f'IP PERMANENTLY BANNED: {ip} — {reason}')


def clear_failed_attempts(cur, ip):
    """Clear failed login attempts for an IP after successful login."""
    cur.execute("DELETE FROM login_attempts WHERE ip_address = %s", (ip,))


# ── JWT Utilities ───────────────────────────────────────────────────
def create_jwt(user_id, username, role):
    return pyjwt.encode({
        'sub': str(user_id),
        'username': username,
        'role': role,
        'exp': datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        'iat': datetime.now(timezone.utc),
    }, JWT_SECRET, algorithm='HS256')


def decode_jwt(token):
    try:
        return pyjwt.decode(token, JWT_SECRET, algorithms=['HS256'])
    except (pyjwt.ExpiredSignatureError, pyjwt.InvalidTokenError):
        return None


def get_token():
    """Extract JWT from Authorization header or cookie."""
    auth = request.headers.get('Authorization', '')
    if auth.startswith('Bearer '):
        return auth[7:]
    return request.cookies.get('dartboard_token', '')


# ═══════════════════════════════════════════════════════════════════
#  USER-FACING APP (port 8200)
# ═══════════════════════════════════════════════════════════════════
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB


def get_db():
    if 'db' not in g:
        g.db = get_conn()
        g.db.autocommit = True
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db:
        db.close()


def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token()
        payload = decode_jwt(token)
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        g.user = payload
        return f(*args, **kwargs)
    return decorated


# ── Auth Endpoints ──────────────────────────────────────────────────
@app.route('/auth/register', methods=['POST'])
def register():
    ip = get_client_ip()
    db = get_db()
    cur = db.cursor()

    # Check IP ban before allowing registration too
    if is_ip_banned(cur, ip):
        log.warning(f'Banned IP {ip} attempted registration')
        return jsonify({'error': 'Your IP address has been permanently banned.'}), 403

    data = request.get_json() or {}
    username = (data.get('username') or '').strip().lower()
    password = data.get('password') or ''

    if not username or len(username) < 3 or len(username) > 64:
        return jsonify({'error': 'Username must be 3-64 characters'}), 400
    if not username.isalnum() and not all(c.isalnum() or c in '-_.' for c in username):
        return jsonify({'error': 'Username may only contain letters, numbers, hyphens, dots, underscores'}), 400
    if len(password) < 10:
        return jsonify({'error': 'Password must be at least 10 characters'}), 400
    if not any(c.isupper() for c in password):
        return jsonify({'error': 'Password must contain at least one uppercase letter'}), 400
    if not any(c.isdigit() for c in password):
        return jsonify({'error': 'Password must contain at least one number'}), 400
    if not any(c in '!@#$%^&*()-_=+[]{}|;:,.<>?/' for c in password):
        return jsonify({'error': 'Password must contain at least one special character'}), 400

    pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()

    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, status) VALUES (%s, %s, 'pending') RETURNING id",
            (username, pw_hash)
        )
        user_id = cur.fetchone()['id']
        log.info(f'New registration request: {username} (id={user_id}) from {ip}')
        return jsonify({'message': 'Access request submitted. An admin will review it shortly.'}), 201
    except psycopg2.errors.UniqueViolation:
        db.rollback()
        return jsonify({'error': 'Username already taken'}), 409


@app.route('/auth/login', methods=['POST'])
def login():
    ip = get_client_ip()
    db = get_db()
    cur = db.cursor()

    # ── IP Ban Check ───────────────────────────────────────────────
    if is_ip_banned(cur, ip):
        log.warning(f'Banned IP {ip} attempted login')
        return jsonify({'error': 'Your IP address has been permanently banned.'}), 403

    data = request.get_json() or {}
    username = (data.get('username') or '').strip().lower()
    password = data.get('password') or ''

    cur.execute("SELECT * FROM users WHERE username = %s", (username,))
    user = cur.fetchone()

    if not user or not bcrypt.checkpw(password.encode(), user['password_hash'].encode()):
        # ── Record Failed Attempt ──────────────────────────────────
        count = record_failed_attempt(cur, ip, username)
        remaining = MAX_LOGIN_ATTEMPTS - count
        log.warning(f'Failed login from {ip} (user={username}, attempt={count}/{MAX_LOGIN_ATTEMPTS})')

        if count >= MAX_LOGIN_ATTEMPTS:
            ban_ip(cur, ip, f'{count} failed login attempts (last username: {username})')
            return jsonify({
                'error': 'Your IP address has been permanently banned due to too many failed login attempts.'
            }), 403

        return jsonify({
            'error': f'Invalid username or password. {remaining} attempt(s) remaining before permanent IP ban.'
        }), 401

    # ── Successful credentials — check account status ──────────────
    if user['status'] == 'pending':
        return jsonify({'error': 'Your account is pending admin approval.'}), 403
    if user['status'] == 'rejected':
        return jsonify({'error': 'Your access request was denied.'}), 403

    # ── Success — clear failed attempts and issue token ────────────
    clear_failed_attempts(cur, ip)
    token = create_jwt(user['id'], user['username'], user['role'])
    log.info(f'Login: {username} from {ip}')
    return jsonify({
        'token': token,
        'username': user['username'],
        'role': user['role'],
    })


@app.route('/auth/verify', methods=['GET'])
@require_auth
def verify():
    return jsonify({
        'valid': True,
        'username': g.user['username'],
        'role': g.user['role'],
    })


# ── Reverse Proxy ───────────────────────────────────────────────────
SKIP_HEADERS = frozenset(['host', 'content-length', 'transfer-encoding', 'connection'])


def proxy_request(url, require_jwt=False):
    """Proxy a request to the dartboard backend."""
    user_info = None
    if require_jwt:
        token = get_token()
        payload = decode_jwt(token)
        if not payload:
            t20 = repr(token[:20]) if token else 'NONE'
            hdr = repr(request.headers.get('Authorization', 'MISSING')[:50])
            log.warning(f'AUTH FAIL {request.method} {request.path} token_len={len(token) if token else 0} first20={t20} auth_hdr={hdr}')
            return jsonify({'error': 'Unauthorized'}), 401
        user_info = payload

    headers = {k: v for k, v in request.headers if k.lower() not in SKIP_HEADERS}

    # Forward authenticated user identity to the Dart backend
    # for per-user chat isolation
    if user_info:
        headers['X-Auth-User'] = user_info.get('username', '')
        headers['X-Auth-User-Id'] = user_info.get('sub', '')

    try:
        resp = http_req.request(
            method=request.method,
            url=url,
            headers=headers,
            data=request.get_data(),
            params=request.args,
            stream=True,
            timeout=(10, 600),  # (connect, read) — 10s connect, 10min read for SSE streams
        )
    except http_req.exceptions.ConnectionError:
        return jsonify({'error': 'Backend unavailable'}), 502
    except http_req.exceptions.ReadTimeout:
        return jsonify({'error': 'Request timed out'}), 524

    # Forward response, stripping hop-by-hop headers
    excluded = {'content-encoding', 'transfer-encoding', 'content-length', 'connection'}
    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}

    # For SSE streams, inject keepalives to prevent client timeouts
    content_type = resp.headers.get('Content-Type', '')
    if 'text/event-stream' in content_type:
        def sse_with_keepalive():
            import select
            for chunk in resp.iter_content(chunk_size=None):
                if chunk:
                    yield chunk
        return Response(
            sse_with_keepalive(),
            status=resp.status_code,
            headers=resp_headers,
            direct_passthrough=True,
        )

    return Response(
        resp.iter_content(chunk_size=None),
        status=resp.status_code,
        headers=resp_headers,
        direct_passthrough=True,
    )


@app.route('/api/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH'])
def proxy_api(path):
    """API calls require a valid JWT."""
    return proxy_request(f'{DARTBOARD_URL}/api/{path}', require_jwt=True)


@app.route('/health')
def proxy_health():
    """Health check — pass through without auth."""
    return proxy_request(f'{DARTBOARD_URL}/health')


@app.route('/')
@app.route('/<path:path>')
def proxy_static(path=''):
    """Static files — no auth required so the login page can load."""
    return proxy_request(f'{DARTBOARD_URL}/{path}')


# ═══════════════════════════════════════════════════════════════════
#  ADMIN PANEL (port 8247 — LAN/Tailscale only)
# ═══════════════════════════════════════════════════════════════════
admin_app = Flask(__name__)

ADMIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dartboard Admin</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #1a1a1a; color: #e0e0e0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding: 24px; max-width: 900px; margin: 0 auto; }
h1 { font-size: 22px; font-weight: 700; margin-bottom: 8px; color: #7c6bf5; }
.subtitle { color: #888; font-size: 14px; margin-bottom: 24px; }
h2 { font-size: 16px; font-weight: 600; margin: 24px 0 12px; color: #b4b4b4; text-transform: uppercase; letter-spacing: 0.05em; }
table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid #333; font-size: 14px; }
th { color: #888; font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
tr:hover { background: rgba(255,255,255,0.03); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
.badge-pending { background: rgba(251,191,36,0.15); color: #fbbf24; }
.badge-approved { background: rgba(74,222,128,0.15); color: #4ade80; }
.badge-rejected { background: rgba(248,113,113,0.15); color: #f87171; }
.badge-admin { background: rgba(124,107,245,0.15); color: #7c6bf5; }
.badge-banned { background: rgba(248,113,113,0.2); color: #f87171; }
.btn { padding: 6px 14px; border: none; border-radius: 6px; font-size: 13px; font-weight: 500; cursor: pointer; font-family: inherit; transition: opacity 0.2s; }
.btn:hover { opacity: 0.85; }
.btn:disabled { opacity: 0.4; cursor: not-allowed; }
.btn-approve { background: #22c55e; color: #fff; }
.btn-reject { background: #ef4444; color: #fff; }
.btn-delete { background: #555; color: #fff; }
.btn-primary { background: #7c6bf5; color: #fff; }
.btn-unban { background: #f59e0b; color: #000; font-weight: 600; }
.actions { display: flex; gap: 6px; }
.card { background: #222; border: 1px solid #333; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
.form-group { margin-bottom: 14px; }
.form-group label { display: block; font-size: 13px; color: #888; margin-bottom: 4px; }
.form-group input { width: 100%; padding: 8px 12px; background: #2a2a2a; border: 1px solid #444; border-radius: 6px; color: #e0e0e0; font-size: 14px; font-family: inherit; }
.form-group input:focus { border-color: #7c6bf5; outline: none; }
.msg { padding: 10px 14px; border-radius: 6px; margin-bottom: 16px; font-size: 14px; }
.msg-ok { background: rgba(74,222,128,0.12); color: #4ade80; }
.msg-err { background: rgba(248,113,113,0.12); color: #f87171; }
.empty { color: #666; font-style: italic; padding: 16px 0; }
.ban-count { display: inline-block; background: rgba(248,113,113,0.15); color: #f87171; padding: 2px 8px; border-radius: 4px; font-size: 13px; font-weight: 600; margin-left: 8px; }
.mono { font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 13px; }
.topbar { display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }
.topbar h1 { margin-bottom: 0; }
.btn-logout { background: #333; color: #f87171; border: 1px solid #555; padding: 6px 16px; border-radius: 6px; font-size: 13px; font-weight: 600; cursor: pointer; font-family: inherit; transition: background 0.2s, color 0.2s; }
.btn-logout:hover { background: #f87171; color: #fff; }
</style>
</head>
<body>
<div class="topbar">
  <div><h1>Dartboard Admin</h1><p class="subtitle" style="margin-bottom:0">User management, access control &amp; IP ban management</p></div>
  <button class="btn-logout" onclick="adminLogout()">Sign Out</button>
</div>
<div id="msg"></div>

<h2>Banned IPs <span id="ban-count" class="ban-count"></span></h2>
<div id="banned"></div>

<h2>Pending Requests</h2>
<div id="pending"></div>

<h2>Approved Users</h2>
<div id="approved"></div>

<h2>Rejected Users</h2>
<div id="rejected"></div>

<h2>Recent Failed Login Attempts</h2>
<div id="attempts"></div>

<h2>Change Admin Password</h2>
<div class="card">
  <div class="form-group"><label>New Password (min 10 chars)</label><input type="password" id="new-pw" placeholder="Enter new password"></div>
  <div class="form-group"><label>Confirm Password</label><input type="password" id="confirm-pw" placeholder="Confirm new password"></div>
  <button class="btn btn-primary" onclick="changePw()">Update Password</button>
</div>

<script>
async function api(path, opts={}) {
  const r = await fetch(path, { ...opts, headers: { 'Content-Type': 'application/json', ...opts.headers } });
  return r.json();
}

function renderTable(users, container, actions) {
  if (!users.length) { document.getElementById(container).innerHTML = '<p class="empty">None</p>'; return; }
  let h = '<table><tr><th>ID</th><th>Username</th><th>Status</th><th>Role</th><th>Created</th><th>Actions</th></tr>';
  for (const u of users) {
    const badge = u.status === 'pending' ? 'badge-pending' : u.status === 'approved' ? 'badge-approved' : 'badge-rejected';
    const roleBadge = u.role === 'admin' ? ' <span class="badge badge-admin">admin</span>' : '';
    const created = new Date(u.created_at).toLocaleDateString();
    h += '<tr><td>' + u.id + '</td><td>' + u.username + roleBadge + '</td><td><span class="badge ' + badge + '">' + u.status + '</span></td><td>' + u.role + '</td><td>' + created + '</td><td class="actions">';
    if (actions.includes('approve')) h += '<button class="btn btn-approve" onclick="act(\'approve\',' + u.id + ')">Approve</button>';
    if (actions.includes('reject')) h += '<button class="btn btn-reject" onclick="act(\'reject\',' + u.id + ')">Reject</button>';
    if (u.role !== 'admin') h += '<button class="btn btn-primary" onclick="act(\'make-admin\',' + u.id + ')" title="Promote to admin">Make Admin</button>';
    if (u.role === 'admin' && !actions.includes('approve')) h += '<button class="btn btn-delete" onclick="act(\'make-user\',' + u.id + ')" title="Demote to user">Make User</button>';
    if (u.role !== 'admin') h += '<button class="btn btn-delete" onclick="if(confirm(\'Delete ' + u.username + '?\'))act(\'delete\',' + u.id + ')">Delete</button>';
    h += '</td></tr>';
  }
  document.getElementById(container).innerHTML = h + '</table>';
}

function renderBannedIps(bans) {
  document.getElementById('ban-count').textContent = bans.length > 0 ? bans.length + ' banned' : '';
  if (!bans.length) { document.getElementById('banned').innerHTML = '<p class="empty">No banned IPs</p>'; return; }
  let h = '<table><tr><th>IP Address</th><th>Reason</th><th>Banned At</th><th>Banned By</th><th>Actions</th></tr>';
  for (const b of bans) {
    const when = new Date(b.banned_at).toLocaleString();
    h += '<tr><td class="mono">' + b.ip_address + '</td><td>' + b.reason + '</td><td>' + when + '</td><td>' + b.banned_by + '</td>';
    h += '<td class="actions"><button class="btn btn-unban" onclick="unbanIp(\'' + b.ip_address + '\')">Unban</button></td></tr>';
  }
  document.getElementById('banned').innerHTML = h + '</table>';
}

function renderAttempts(attempts) {
  if (!attempts.length) { document.getElementById('attempts').innerHTML = '<p class="empty">No recent failed attempts</p>'; return; }
  let h = '<table><tr><th>IP Address</th><th>Username</th><th>When</th><th>Count</th></tr>';
  // Group by IP
  const byIp = {};
  for (const a of attempts) {
    if (!byIp[a.ip_address]) byIp[a.ip_address] = [];
    byIp[a.ip_address].push(a);
  }
  for (const [ip, items] of Object.entries(byIp)) {
    const latest = new Date(items[items.length - 1].attempted_at).toLocaleString();
    const usernames = [...new Set(items.map(i => i.username))].join(', ');
    h += '<tr><td class="mono">' + ip + '</td><td>' + usernames + '</td><td>' + latest + '</td><td><span class="badge badge-rejected">' + items.length + '</span></td></tr>';
  }
  document.getElementById('attempts').innerHTML = h + '</table>';
}

async function load() {
  const data = await api('/admin/users');
  const users = data.users || [];
  renderTable(users.filter(u => u.status === 'pending'), 'pending', ['approve', 'reject']);
  renderTable(users.filter(u => u.status === 'approved'), 'approved', ['delete']);
  renderTable(users.filter(u => u.status === 'rejected'), 'rejected', ['approve', 'delete']);

  const banData = await api('/admin/banned-ips');
  renderBannedIps(banData.bans || []);

  const attemptData = await api('/admin/login-attempts');
  renderAttempts(attemptData.attempts || []);
}

async function act(action, id) {
  await api('/admin/' + action + '/' + id, { method: 'POST' });
  load();
}

async function unbanIp(ip) {
  if (!confirm('Unban IP ' + ip + '? This will allow them to attempt login again.')) return;
  const r = await api('/admin/unban', { method: 'POST', body: JSON.stringify({ ip: ip }) });
  if (r.error) showMsg(r.error, false);
  else showMsg('Unbanned ' + ip, true);
  load();
}

function showMsg(text, ok) {
  const el = document.getElementById('msg');
  el.className = 'msg ' + (ok ? 'msg-ok' : 'msg-err');
  el.textContent = text;
  setTimeout(() => el.textContent = '', 5000);
}

async function changePw() {
  const pw = document.getElementById('new-pw').value;
  const pw2 = document.getElementById('confirm-pw').value;
  if (pw !== pw2) { showMsg('Passwords do not match', false); return; }
  if (pw.length < 10) { showMsg('Password must be at least 10 characters', false); return; }
  const r = await api('/admin/change-password', { method: 'POST', body: JSON.stringify({ password: pw }) });
  if (r.error) showMsg(r.error, false);
  else { showMsg('Password updated. Refresh page to re-authenticate.', true); document.getElementById('new-pw').value = ''; document.getElementById('confirm-pw').value = ''; }
}

function adminLogout() {
  // Send a request with deliberately wrong credentials to clear Basic Auth cache
  const xhr = new XMLHttpRequest();
  xhr.open('GET', '/admin/logout', true);
  xhr.setRequestHeader('Authorization', 'Basic ' + btoa('logout:logout'));
  xhr.onreadystatechange = function() {
    if (xhr.readyState === 4) {
      // Redirect to root which will re-prompt for credentials
      window.location.href = '/';
    }
  };
  xhr.send();
}

load();
</script>
</body>
</html>"""


def get_admin_db():
    if 'db' not in g:
        g.db = get_conn()
        g.db.autocommit = True
    return g.db


@admin_app.teardown_appcontext
def close_admin_db(exc):
    db = g.pop('db', None)
    if db:
        db.close()


def require_admin(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not auth.username or not auth.password:
            return Response('Login required', 401, {'WWW-Authenticate': 'Basic realm="Dartboard Admin"'})
        db = get_admin_db()
        cur = db.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s AND role = 'admin'", (auth.username,))
        admin = cur.fetchone()
        if not admin or not bcrypt.checkpw(auth.password.encode(), admin['password_hash'].encode()):
            return Response('Invalid credentials', 401, {'WWW-Authenticate': 'Basic realm="Dartboard Admin"'})
        g.admin = admin
        return f(*args, **kwargs)
    return decorated


def require_primary_admin(f):
    """Restrict to the primary admin (ADMIN_USER / jalsarraf) only.
    Other admins cannot unban IPs — this is a security directive."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not auth.username or not auth.password:
            return Response('Login required', 401, {'WWW-Authenticate': 'Basic realm="Dartboard Admin"'})
        if auth.username != ADMIN_USER:
            return jsonify({'error': f'Only {ADMIN_USER} can perform this action'}), 403
        db = get_admin_db()
        cur = db.cursor()
        cur.execute("SELECT * FROM users WHERE username = %s AND role = 'admin'", (auth.username,))
        admin = cur.fetchone()
        if not admin or not bcrypt.checkpw(auth.password.encode(), admin['password_hash'].encode()):
            return Response('Invalid credentials', 401, {'WWW-Authenticate': 'Basic realm="Dartboard Admin"'})
        g.admin = admin
        return f(*args, **kwargs)
    return decorated


@admin_app.route('/admin/logout')
def admin_logout():
    """Force browser to discard Basic Auth credentials by returning 401."""
    return Response(
        'Logged out. <a href="/">Sign in again</a>',
        401,
        {'WWW-Authenticate': 'Basic realm="Dartboard Admin"',
         'Content-Type': 'text/html'},
    )


@admin_app.route('/')
@require_admin
def admin_dashboard():
    return ADMIN_HTML


@admin_app.route('/admin/users')
@require_admin
def admin_list_users():
    db = get_admin_db()
    cur = db.cursor()
    cur.execute("SELECT id, username, status, role, created_at, updated_at FROM users ORDER BY created_at DESC")
    users = cur.fetchall()
    # Serialize datetimes
    for u in users:
        u['created_at'] = u['created_at'].isoformat() if u['created_at'] else None
        u['updated_at'] = u['updated_at'].isoformat() if u['updated_at'] else None
    return jsonify({'users': users})


# ── IP Ban Management (primary admin only) ──────────────────────────
@admin_app.route('/admin/banned-ips')
@require_admin
def admin_list_banned_ips():
    db = get_admin_db()
    cur = db.cursor()
    cur.execute("SELECT ip_address, reason, banned_at, banned_by FROM ip_bans ORDER BY banned_at DESC")
    bans = cur.fetchall()
    for b in bans:
        b['banned_at'] = b['banned_at'].isoformat() if b['banned_at'] else None
    return jsonify({'bans': bans})


@admin_app.route('/admin/login-attempts')
@require_admin
def admin_list_login_attempts():
    db = get_admin_db()
    cur = db.cursor()
    cur.execute(
        "SELECT ip_address, username, attempted_at FROM login_attempts "
        "ORDER BY attempted_at DESC LIMIT 100"
    )
    attempts = cur.fetchall()
    for a in attempts:
        a['attempted_at'] = a['attempted_at'].isoformat() if a['attempted_at'] else None
    return jsonify({'attempts': attempts})


@admin_app.route('/admin/unban', methods=['POST'])
@require_primary_admin
def admin_unban_ip():
    """Unban an IP address. RESTRICTED to primary admin (jalsarraf) only."""
    data = request.get_json() or {}
    ip = (data.get('ip') or '').strip()
    if not ip:
        return jsonify({'error': 'IP address is required'}), 400

    db = get_admin_db()
    cur = db.cursor()
    cur.execute("DELETE FROM ip_bans WHERE ip_address = %s", (ip,))
    cur.execute("DELETE FROM login_attempts WHERE ip_address = %s", (ip,))
    log.info(f'Admin {g.admin["username"]} UNBANNED IP: {ip}')
    return jsonify({'success': True, 'ip': ip})


@admin_app.route('/admin/approve/<int:user_id>', methods=['POST'])
@require_admin
def admin_approve(user_id):
    db = get_admin_db()
    cur = db.cursor()
    cur.execute("UPDATE users SET status = 'approved', updated_at = NOW() WHERE id = %s AND role != 'admin'", (user_id,))
    log.info(f'Admin {g.admin["username"]} approved user id={user_id}')
    return jsonify({'success': True})


@admin_app.route('/admin/reject/<int:user_id>', methods=['POST'])
@require_admin
def admin_reject(user_id):
    db = get_admin_db()
    cur = db.cursor()
    cur.execute("UPDATE users SET status = 'rejected', updated_at = NOW() WHERE id = %s AND role != 'admin'", (user_id,))
    log.info(f'Admin {g.admin["username"]} rejected user id={user_id}')
    return jsonify({'success': True})


@admin_app.route('/admin/delete/<int:user_id>', methods=['POST'])
@require_admin
def admin_delete(user_id):
    db = get_admin_db()
    cur = db.cursor()
    if user_id == g.admin['id']:
        return jsonify({'error': 'Cannot delete yourself'}), 400
    cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
    log.info(f'Admin {g.admin["username"]} deleted user id={user_id}')
    return jsonify({'success': True})


@admin_app.route('/admin/make-admin/<int:user_id>', methods=['POST'])
@require_admin
def admin_make_admin(user_id):
    db = get_admin_db()
    cur = db.cursor()
    cur.execute("UPDATE users SET role = 'admin', updated_at = NOW() WHERE id = %s", (user_id,))
    log.info(f'Admin {g.admin["username"]} promoted user id={user_id} to admin')
    return jsonify({'success': True})


@admin_app.route('/admin/make-user/<int:user_id>', methods=['POST'])
@require_admin
def admin_make_user(user_id):
    db = get_admin_db()
    cur = db.cursor()
    # Don't allow demoting yourself
    if user_id == g.admin['id']:
        return jsonify({'error': 'Cannot demote yourself'}), 400
    cur.execute("UPDATE users SET role = 'user', updated_at = NOW() WHERE id = %s", (user_id,))
    log.info(f'Admin {g.admin["username"]} demoted user id={user_id} to user')
    return jsonify({'success': True})


@admin_app.route('/admin/change-password', methods=['POST'])
@require_admin
def admin_change_password():
    data = request.get_json() or {}
    new_pw = data.get('password', '')
    if len(new_pw) < 10:
        return jsonify({'error': 'Password must be at least 10 characters'}), 400
    pw_hash = bcrypt.hashpw(new_pw.encode(), bcrypt.gensalt(12)).decode()
    db = get_admin_db()
    cur = db.cursor()
    cur.execute("UPDATE users SET password_hash = %s, updated_at = NOW() WHERE id = %s", (pw_hash, g.admin['id']))
    log.info(f'Admin {g.admin["username"]} changed their password')
    return jsonify({'success': True})


# ═══════════════════════════════════════════════════════════════════
#  TLS Setup
# ═══════════════════════════════════════════════════════════════════
def _generate_self_signed_cert():
    """Generate a self-signed TLS cert for local HTTPS. No external deps."""
    import subprocess
    cert_dir = '/tmp/tls'
    cert_file = f'{cert_dir}/cert.pem'
    key_file = f'{cert_dir}/key.pem'
    os.makedirs(cert_dir, exist_ok=True)
    if not os.path.exists(cert_file):
        subprocess.run([
            'openssl', 'req', '-x509', '-newkey', 'rsa:2048',
            '-keyout', key_file, '-out', cert_file,
            '-days', '365', '-nodes',
            '-subj', '/CN=aichat-auth/O=aichat/C=US',
        ], check=True, capture_output=True)
        log.info('Generated self-signed TLS certificate')
    return cert_file, key_file


#  STARTUP
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    log.info('Initializing database...')
    init_db()
    log.info('Database ready.')

    # TLS config — set ENABLE_TLS=1 to encrypt connections
    use_tls = os.environ.get('ENABLE_TLS', '0') == '1'
    ssl_ctx = None
    if use_tls:
        cert_file, key_file = _generate_self_signed_cert()
        import ssl
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cert_file, key_file)
        log.info('TLS enabled (self-signed certificate)')

    # Start admin panel in background thread
    def run_admin():
        log.info('Admin panel listening on :8247')
        admin_app.run(host='0.0.0.0', port=8247, threaded=True, ssl_context=ssl_ctx)

    admin_thread = threading.Thread(target=run_admin, daemon=True)
    admin_thread.start()

    # Start user-facing proxy on main thread
    proto = 'https' if use_tls else 'http'
    log.info(f'Auth proxy listening on {proto}://:8200')
    log.info(f'JWT expiry: {JWT_EXPIRY_HOURS}h | IP ban threshold: {MAX_LOGIN_ATTEMPTS} attempts')
    log.info(f'Primary admin (unban authority): {ADMIN_USER}')
    app.run(host='0.0.0.0', port=8200, threaded=True, ssl_context=ssl_ctx)
