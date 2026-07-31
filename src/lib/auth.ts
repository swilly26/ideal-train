import { createServerFn } from "@tanstack/react-start";
import { getCookie, setCookie, deleteCookie } from "@tanstack/react-start/server";
import { findUserByEmail, createUser } from "./db";

const TOKEN_COOKIE = "apextrade_token";
const COOKIE_OPTS = {
  httpOnly: true,
  secure: false,
  sameSite: "lax" as const,
  path: "/",
  maxAge: 60 * 60 * 24 * 7,
};

// In-memory token store (resets on server restart; fine for Phase 1)
const tokens = new Map<string, { userId: number; email: string }>();

function generateToken(): string {
  const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  let token = "";
  for (let i = 0; i < 48; i++) {
    token += chars[Math.floor(Math.random() * chars.length)];
  }
  return token;
}

async function hashPassword(password: string): Promise<string> {
  const encoder = new TextEncoder();
  const data = encoder.encode(password + "apextrade-salt");
  const hash = await crypto.subtle.digest("SHA-256", data);
  return Array.from(new Uint8Array(hash))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

async function verifyPassword(
  password: string,
  hash: string,
): Promise<boolean> {
  return (await hashPassword(password)) === hash;
}

export const signup = createServerFn({ method: "POST" }).handler(
  async ({ data }: { data: unknown }) => {
    const d = data as { email: string; password: string };
    if (!d.email || !d.password) return { error: "Missing required fields" };
    if (d.password.length < 6)
      return { error: "Password must be at least 6 characters" };
    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(d.email))
      return { error: "Invalid email address" };

    const email = d.email.toLowerCase().trim();
    if (findUserByEmail(email))
      return { error: "An account with this email already exists" };

    const passwordHash = await hashPassword(d.password);
    const user = createUser(email, passwordHash);

    const token = generateToken();
    tokens.set(token, { userId: user.id, email: user.email });
    setCookie(TOKEN_COOKIE, token, COOKIE_OPTS);

    return { success: true, email: user.email };
  },
);

export const login = createServerFn({ method: "POST" }).handler(
  async ({ data }: { data: unknown }) => {
    const d = data as { email: string; password: string };
    if (!d.email || !d.password) return { error: "Missing required fields" };

    const email = d.email.toLowerCase().trim();
    const user = findUserByEmail(email);
    if (!user) return { error: "Invalid email or password" };

    const valid = await verifyPassword(d.password, user.password_hash);
    if (!valid) return { error: "Invalid email or password" };

    const token = generateToken();
    tokens.set(token, { userId: user.id, email: user.email });
    setCookie(TOKEN_COOKIE, token, COOKIE_OPTS);

    return { success: true, email: user.email };
  },
);

export const logout = createServerFn({ method: "POST" }).handler(async () => {
  const token = getCookie(TOKEN_COOKIE);
  if (token) tokens.delete(token);
  deleteCookie(TOKEN_COOKIE, { path: "/" });
  return { success: true };
});

export const getCurrentUser = createServerFn({ method: "GET" }).handler(
  async () => {
    const token = getCookie(TOKEN_COOKIE);
    if (!token || !tokens.has(token)) return null;
    const user = tokens.get(token)!;
    return { userId: user.userId, email: user.email };
  },
);
