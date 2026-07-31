import { createFileRoute, useRouter } from "@tanstack/react-router";
import { useState } from "react";
import { api } from "~/lib/api";

export const Route = createFileRoute("/login")({
  head: () => ({
    meta: [{ title: "Log In — ApexTrade" }],
  }),
  component: LoginPage,
});

function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const result = await api.login(email, password);
      if (result.error) {
        setError(result.error);
        return;
      }
      router.navigate({ to: "/dashboard" });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="auth-page">
      <nav className="site-nav container">
        <a className="brand" href="/">
          <span className="brand-mark">◈</span>Apex<span>Trade</span>
        </a>
      </nav>

      <div className="auth-container">
        <div className="auth-card">
          <h1>Welcome back</h1>
          <p className="auth-sub">Log in to your ApexTrade dashboard.</p>

          {error && <div className="auth-error">{error}</div>}

          <form onSubmit={handleSubmit} className="auth-form">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              required
              autoComplete="email"
            />

            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Your password"
              required
              autoComplete="current-password"
            />

            <button type="submit" className="button button-primary" disabled={loading}>
              {loading ? "Logging in..." : "Log in"} <span>↗</span>
            </button>
          </form>

          <p className="auth-footer">
            Don&apos;t have an account?{" "}
            <a href="/signup">Sign up</a>
          </p>
        </div>
      </div>
    </main>
  );
}
