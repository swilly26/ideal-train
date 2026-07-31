import { createFileRoute, useRouter } from "@tanstack/react-router";
import { useState } from "react";
import { api } from "~/lib/api";

export const Route = createFileRoute("/signup")({
  head: () => ({
    meta: [{ title: "Sign Up — ApexTrade" }],
  }),
  component: SignupPage,
});

function SignupPage() {
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
      const result = await api.signup(email, password);
      if (result.error) {
        setError(result.error);
        return;
      }
      router.navigate({ to: "/dashboard" });
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Signup failed");
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
          <h1>Create your account</h1>
          <p className="auth-sub">Start trading with AI-powered strategies.</p>

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
              placeholder="At least 6 characters"
              required
              minLength={6}
              autoComplete="new-password"
            />

            <button type="submit" className="button button-primary" disabled={loading}>
              {loading ? "Creating account..." : "Create account"} <span>↗</span>
            </button>
          </form>

          <p className="auth-footer">
            Already have an account?{" "}
            <a href="/login">Log in</a>
          </p>
        </div>
      </div>
    </main>
  );
}
