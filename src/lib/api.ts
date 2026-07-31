/**
 * Client-side API wrappers for ApexTrade.
 * All requests go to same-origin /api/* (proxied to auth-server on port 3001).
 */

interface AuthResponse {
  success?: boolean;
  error?: string;
  email?: string;
  userId?: number;
  role?: string;
}

async function authFetch(
  path: string,
  options: RequestInit = {},
): Promise<AuthResponse | null> {
  const res = await fetch(path, {
    ...options,
    credentials: "include",
  });
  const data = await res.json();
  return data;
}

export const api = {
  signup(email: string, password: string): Promise<AuthResponse> {
    return authFetch("/api/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }) as Promise<AuthResponse>;
  },

  login(email: string, password: string): Promise<AuthResponse> {
    return authFetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    }) as Promise<AuthResponse>;
  },

  logout(): Promise<AuthResponse> {
    return authFetch("/api/auth/logout", {
      method: "POST",
    }) as Promise<AuthResponse>;
  },

  getMe(): Promise<{ userId: number; email: string; role: string } | null> {
    return authFetch("/api/auth/me") as Promise<{
      userId: number;
      email: string;
      role: string;
    } | null>;
  },
};
