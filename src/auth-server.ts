/**
 * Standalone Bun HTTP server for ApexTrade auth API.
 * Runs on port 3001. Uses Bun.password for bcrypt hashing,
 * in-memory token store, and the shared JSON user DB.
 *
 * Endpoints:
 *   /api/auth/signup      POST   — create account
 *   /api/auth/login       POST   — log in
 *   /api/auth/logout      POST   — log out
 *   /api/auth/me          GET    — current user (now includes role)
 *   /api/admin/users      GET    — list all users (admin only)
 *   /api/admin/users/:id/role  POST — change role (admin only)
 *   /api/user/alpaca/connect   POST — store Alpaca keys
 *   /api/user/alpaca/status    GET  — Alpaca connection status
 *   /api/user/alpaca/disconnect POST — clear Alpaca keys
 */
import { findUserByEmail, findUserById, getAllUsers, createUser, updateUser } from "./lib/db";
import { createCipheriv, createDecipheriv, randomBytes, pbkdf2Sync } from "node:crypto";

const PORT = 3001;
const HOST = "0.0.0.0";
const TOKEN_COOKIE = "apextrade_token";
const COOKIE_MAX_AGE = 60 * 60 * 24 * 7; // 7 days

// ── Encryption for Alpaca keys ──────────────────────────────────────────
// Derive a 256-bit key from a fixed passphrase so keys survive server
// restarts. In production this would be an env var.
const ENC_PASSPHRASE = "apextrade-alpaca-encryption-key-2026";
const ENC_SALT = "apextrade-salt";
const ENC_ALGO = "aes-256-gcm";
const encKey = pbkdf2Sync(ENC_PASSPHRASE, ENC_SALT, 100_000, 32, "sha256");

function encrypt(text: string): string {
  const iv = randomBytes(12);
  const cipher = createCipheriv(ENC_ALGO, encKey, iv);
  const encrypted = Buffer.concat([cipher.update(text, "utf8"), cipher.final()]);
  const tag = cipher.getAuthTag();
  // Format: iv:tag:ciphertext (all hex)
  return `${iv.toString("hex")}:${tag.toString("hex")}:${encrypted.toString("hex")}`;
}

function decrypt(encoded: string): string {
  const [ivHex, tagHex, dataHex] = encoded.split(":");
  const iv = Buffer.from(ivHex, "hex");
  const tag = Buffer.from(tagHex, "hex");
  const encrypted = Buffer.from(dataHex, "hex");
  const decipher = createDecipheriv(ENC_ALGO, encKey, iv);
  decipher.setAuthTag(tag);
  return Buffer.concat([decipher.update(encrypted), decipher.final()]).toString("utf8");
}

// ── Token store ─────────────────────────────────────────────────────────
const tokens = new Map<string, { userId: number; email: string; role: string }>();

function generateToken(): string {
  const bytes = new Uint8Array(32);
  crypto.getRandomValues(bytes);
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// ── Cookie helpers ──────────────────────────────────────────────────────
function setAuthCookie(
  headers: Headers,
  token: string,
  maxAge: number = COOKIE_MAX_AGE
) {
  headers.append(
    "Set-Cookie",
    `${TOKEN_COOKIE}=${token}; HttpOnly; SameSite=Lax; Path=/; Max-Age=${maxAge}`
  );
}

function clearAuthCookie(headers: Headers) {
  headers.append(
    "Set-Cookie",
    `${TOKEN_COOKIE}=; HttpOnly; SameSite=Lax; Path=/; Max-Age=0`
  );
}

function getAuthToken(req: Request): string | null {
  const cookie = req.headers.get("cookie");
  if (!cookie) return null;
  const match = cookie.match(
    new RegExp(`(?:^|;\\s*)${TOKEN_COOKIE}=([^;]*)`)
  );
  return match ? match[1] : null;
}

// ── Response helpers ────────────────────────────────────────────────────
function jsonResponse(data: unknown, status = 200, extraHeaders?: Headers) {
  const headers = extraHeaders ?? new Headers();
  headers.set("Content-Type", "application/json");
  return new Response(JSON.stringify(data), { status, headers });
}

// ── Auth middleware ─────────────────────────────────────────────────────
// Returns the authenticated user record or null.
function authenticate(req: Request): import("./lib/db").User | null {
  const token = getAuthToken(req);
  if (!token || !tokens.has(token)) return null;
  const session = tokens.get(token)!;
  return findUserById(session.userId) ?? null;
}

// Returns the authenticated user or sends a 401 response.
function requireAuth(req: Request): import("./lib/db").User | Response {
  const user = authenticate(req);
  if (!user) return jsonResponse({ error: "Authentication required" }, 401);
  return user;
}

// Returns the authenticated admin or sends a 403 response.
function requireAdmin(req: Request): import("./lib/db").User | Response {
  const result = requireAuth(req);
  if (result instanceof Response) return result;
  if (result.role !== "admin")
    return jsonResponse({ error: "Admin access required" }, 403);
  return result;
}

// ── Handlers: Auth ──────────────────────────────────────────────────────
async function handleSignup(req: Request): Promise<Response> {
  let body: { email?: string; password?: string };
  try {
    body = await req.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON" }, 400);
  }

  const { email, password } = body;
  if (!email || !password) {
    return jsonResponse({ error: "Missing required fields" }, 400);
  }
  if (password.length < 6) {
    return jsonResponse(
      { error: "Password must be at least 6 characters" },
      400
    );
  }
  if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
    return jsonResponse({ error: "Invalid email address" }, 400);
  }

  const normalizedEmail = email.toLowerCase().trim();
  if (findUserByEmail(normalizedEmail)) {
    return jsonResponse(
      { error: "An account with this email already exists" },
      409
    );
  }

  const passwordHash = await Bun.password.hash(password);
  const user = createUser(normalizedEmail, passwordHash);

  const token = generateToken();
  tokens.set(token, { userId: user.id, email: user.email, role: user.role });

  const headers = new Headers();
  setAuthCookie(headers, token);

  return jsonResponse({ success: true, email: user.email, role: user.role }, 200, headers);
}

async function handleLogin(req: Request): Promise<Response> {
  let body: { email?: string; password?: string };
  try {
    body = await req.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON" }, 400);
  }

  const { email, password } = body;
  if (!email || !password) {
    return jsonResponse({ error: "Missing required fields" }, 400);
  }

  const normalizedEmail = email.toLowerCase().trim();
  const user = findUserByEmail(normalizedEmail);
  if (!user) {
    return jsonResponse({ error: "Invalid email or password" }, 401);
  }

  const valid = await Bun.password.verify(password, user.password_hash);
  if (!valid) {
    return jsonResponse({ error: "Invalid email or password" }, 401);
  }

  const token = generateToken();
  tokens.set(token, { userId: user.id, email: user.email, role: user.role });

  const headers = new Headers();
  setAuthCookie(headers, token);

  return jsonResponse({ success: true, email: user.email, role: user.role }, 200, headers);
}

async function handleLogout(req: Request): Promise<Response> {
  const token = getAuthToken(req);
  if (token) tokens.delete(token);

  const headers = new Headers();
  clearAuthCookie(headers);
  return jsonResponse({ success: true }, 200, headers);
}

async function handleMe(req: Request): Promise<Response> {
  const user = authenticate(req);
  if (!user) return jsonResponse(null);
  return jsonResponse({
    userId: user.id,
    email: user.email,
    role: user.role,
  });
}

// ── Handlers: Admin ─────────────────────────────────────────────────────
async function handleAdminUsers(req: Request): Promise<Response> {
  const admin = requireAdmin(req);
  if (admin instanceof Response) return admin;

  const users = getAllUsers();
  const result = users.map((u) => ({
    id: u.id,
    email: u.email,
    role: u.role,
    createdAt: u.created_at,
    alpacaConnected: u.alpacaConnected,
    alpacaPaperTrading: u.alpacaPaperTrading,
  }));
  return jsonResponse(result);
}

async function handleAdminSetRole(
  req: Request,
  userId: number,
): Promise<Response> {
  const admin = requireAdmin(req);
  if (admin instanceof Response) return admin;

  let body: { role?: string };
  try {
    body = await req.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON" }, 400);
  }

  if (!body.role || !["user", "admin"].includes(body.role)) {
    return jsonResponse({ error: "role must be 'user' or 'admin'" }, 400);
  }

  const target = findUserById(userId);
  if (!target) {
    return jsonResponse({ error: "User not found" }, 404);
  }

  // Prevent self-demotion (keep at least one admin)
  if (admin.id === userId && body.role !== "admin") {
    return jsonResponse({ error: "Cannot change your own admin role" }, 400);
  }

  const updated = updateUser(userId, { role: body.role as "user" | "admin" });
  if (!updated) {
    return jsonResponse({ error: "User not found" }, 404);
  }

  // Update any live token sessions for this user
  for (const [token, session] of tokens) {
    if (session.userId === userId) {
      session.role = body.role;
    }
  }

  return jsonResponse({
    id: updated.id,
    email: updated.email,
    role: updated.role,
  });
}

// ── Handlers: Alpaca ────────────────────────────────────────────────────
async function handleAlpacaConnect(req: Request): Promise<Response> {
  const user = requireAuth(req);
  if (user instanceof Response) return user;

  let body: { apiKey?: string; apiSecret?: string; paperTrading?: boolean };
  try {
    body = await req.json();
  } catch {
    return jsonResponse({ error: "Invalid JSON" }, 400);
  }

  if (!body.apiKey || !body.apiSecret) {
    return jsonResponse({ error: "apiKey and apiSecret are required" }, 400);
  }

  const paperTrading = body.paperTrading !== false; // default true

  const encryptedKey = encrypt(body.apiKey);
  const encryptedSecret = encrypt(body.apiSecret);

  updateUser(user.id, {
    alpacaKey: encryptedKey,
    alpacaSecret: encryptedSecret,
    alpacaConnected: true,
    alpacaPaperTrading: paperTrading,
  });

  return jsonResponse({ connected: true, paperTrading });
}

async function handleAlpacaStatus(req: Request): Promise<Response> {
  const user = requireAuth(req);
  if (user instanceof Response) return user;

  // Re-read from DB to get latest
  const fresh = findUserById(user.id);
  if (!fresh) return jsonResponse({ error: "User not found" }, 404);

  return jsonResponse({
    connected: fresh.alpacaConnected,
    paperTrading: fresh.alpacaPaperTrading,
  });
}

async function handleAlpacaDisconnect(req: Request): Promise<Response> {
  const user = requireAuth(req);
  if (user instanceof Response) return user;

  updateUser(user.id, {
    alpacaKey: null,
    alpacaSecret: null,
    alpacaConnected: false,
    alpacaPaperTrading: true,
  });

  return jsonResponse({ connected: false });
}

// ── Port management ─────────────────────────────────────────────────────
function freePort(port: number): string {
  return (
    `for _ in $(seq 1 25); do ` +
    `pids=$(lsof -t -iTCP:${port} -sTCP:LISTEN 2>/dev/null || true); ` +
    `if [ -z "$pids" ]; then exit 0; fi; ` +
    `kill $pids 2>/dev/null || true; sleep 0.2; ` +
    `done`
  );
}

// ── Startup: seed admin user ────────────────────────────────────────────
async function seedAdmin() {
  const adminEmail = "admin@apextrade.com";
  const existing = findUserByEmail(adminEmail);
  if (existing) {
    // Ensure existing admin has the admin role
    if (existing.role !== "admin") {
      updateUser(existing.id, { role: "admin" });
      console.log(`Updated ${adminEmail} to admin role`);
    }
    return;
  }
  const passwordHash = await Bun.password.hash("admin123");
  createUser(adminEmail, passwordHash, "admin");
  console.log(`Seeded admin user: ${adminEmail} / admin123`);
}

await seedAdmin();

// ── Main ────────────────────────────────────────────────────────────────
await Bun.$`sudo sh -c ${freePort(PORT)}`.quiet().nothrow();

Bun.serve({
  port: PORT,
  hostname: HOST,
  async fetch(req) {
    const url = new URL(req.url);
    const path = url.pathname;
    const method = req.method;

    // ── Auth routes ──────────────────────────────────────────────────
    if (path === "/api/auth/signup" && method === "POST") {
      return handleSignup(req);
    }
    if (path === "/api/auth/login" && method === "POST") {
      return handleLogin(req);
    }
    if (path === "/api/auth/logout" && method === "POST") {
      return handleLogout(req);
    }
    if (path === "/api/auth/me" && method === "GET") {
      return handleMe(req);
    }

    // ── Admin routes ─────────────────────────────────────────────────
    if (path === "/api/admin/users" && method === "GET") {
      return handleAdminUsers(req);
    }
    // POST /api/admin/users/:id/role
    const adminRoleMatch = path.match(/^\/api\/admin\/users\/(\d+)\/role$/);
    if (adminRoleMatch && method === "POST") {
      return handleAdminSetRole(req, parseInt(adminRoleMatch[1], 10));
    }

    // ── Alpaca routes ────────────────────────────────────────────────
    if (path === "/api/user/alpaca/connect" && method === "POST") {
      return handleAlpacaConnect(req);
    }
    if (path === "/api/user/alpaca/status" && method === "GET") {
      return handleAlpacaStatus(req);
    }
    if (path === "/api/user/alpaca/disconnect" && method === "POST") {
      return handleAlpacaDisconnect(req);
    }

    return jsonResponse({ error: "Not found" }, 404);
  },
});

console.log(`auth-server listening on http://${HOST}:${PORT}`);
