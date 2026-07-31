import { createFileRoute, useRouter } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { api } from "~/lib/api";

interface AlpacaStatus {
  connected: boolean;
  paperTrading: boolean;
}

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [{ title: "Dashboard — ApexTrade" }],
  }),
  component: DashboardPage,
});

function DashboardPage() {
  const router = useRouter();
  const [user, setUser] = useState<{
    email: string;
    userId: number;
    role: string;
  } | null>(null);
  const [checking, setChecking] = useState(true);
  const [alpacaStatus, setAlpacaStatus] = useState<AlpacaStatus | null>(null);

  // Alpaca connect form state
  const [showConnectForm, setShowConnectForm] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [apiSecret, setApiSecret] = useState("");
  const [paperTrading, setPaperTrading] = useState(true);
  const [connecting, setConnecting] = useState(false);
  const [connectError, setConnectError] = useState("");

  useEffect(() => {
    api.getMe().then((u) => {
      if (!u) {
        router.navigate({ to: "/login" });
      } else {
        setUser(u as any);
        setChecking(false);
        loadAlpacaStatus();
      }
    });
  }, [router]);

  async function loadAlpacaStatus() {
    try {
      const res = await fetch("/api/user/alpaca/status", {
        credentials: "include",
      });
      if (res.ok) {
        setAlpacaStatus(await res.json());
      }
    } catch {
      // ignore — user may not have keys yet
    }
  }

  async function handleLogout() {
    await api.logout();
    router.navigate({ to: "/" });
  }

  async function handleConnect(e: React.FormEvent) {
    e.preventDefault();
    setConnectError("");
    setConnecting(true);
    try {
      const res = await fetch("/api/user/alpaca/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ apiKey, apiSecret, paperTrading }),
      });
      const data = await res.json();
      if (res.ok) {
        setAlpacaStatus({ connected: true, paperTrading: data.paperTrading });
        setShowConnectForm(false);
        setApiKey("");
        setApiSecret("");
      } else {
        setConnectError(data.error || "Connection failed");
      }
    } catch {
      setConnectError("Network error — please try again");
    } finally {
      setConnecting(false);
    }
  }

  async function handleDisconnect() {
    try {
      const res = await fetch("/api/user/alpaca/disconnect", {
        method: "POST",
        credentials: "include",
      });
      if (res.ok) {
        setAlpacaStatus({ connected: false, paperTrading: true });
      }
    } catch {
      // ignore
    }
  }

  if (checking) {
    return (
      <main className="dashboard-page">
        <div
          className="container"
          style={{
            padding: "60px 0",
            textAlign: "center",
            color: "var(--muted)",
          }}
        >
          Loading...
        </div>
      </main>
    );
  }

  if (!user) return null;

  const isAdmin = user.role === "admin";

  return (
    <main className="dashboard-page">
      <header className="dashboard-header">
        <div className="container dash-header-inner">
          <a className="brand" href="/dashboard">
            <span className="brand-mark">◈</span>Apex<span>Trade</span>
          </a>
          <div className="dash-header-right">
            {isAdmin && (
              <a href="/admin" className="admin-nav-link">
                Admin
              </a>
            )}
            <span className="user-email">{user.email}</span>
            <button onClick={handleLogout} className="logout-btn">
              Log out
            </button>
          </div>
        </div>
      </header>

      <div className="container dash-content">
        <div className="dash-welcome">
          <h1>Dashboard</h1>
          <p>
            Welcome back, <strong>{user.email}</strong>
          </p>
        </div>

        <div className="dash-stats">
          <div className="stat-card">
            <label>Account Value</label>
            <strong>—</strong>
          </div>
          <div className="stat-card">
            <label>Today&apos;s P&amp;L</label>
            <strong>—</strong>
          </div>
          <div className="stat-card">
            <label>Win Rate</label>
            <strong>—</strong>
          </div>
        </div>

        {/* ── Alpaca Brokerage Card ──────────────────────────────── */}
        {alpacaStatus?.connected ? (
          <div className="connect-card connected-card">
            <div className="connect-icon">✅</div>
            <div className="connect-body">
              <h2>
                Connected to Alpaca
                {alpacaStatus.paperTrading ? " (Paper Trading)" : " (Live)"}
              </h2>
              <p>
                Your Alpaca account is linked and ready for AI-powered trading.
              </p>
            </div>
            <button
              onClick={handleDisconnect}
              className="button button-outline disconnect-btn"
            >
              Disconnect
            </button>
          </div>
        ) : (
          <div className="connect-card">
            <div className="connect-icon">⚡</div>
            <div className="connect-body">
              <h2>Connect your brokerage</h2>
              <p>
                Link your Alpaca account to start trading with AI-powered
                strategies.
              </p>
            </div>
            <button
              onClick={() => setShowConnectForm(!showConnectForm)}
              className="button button-primary connect-btn-active"
            >
              {showConnectForm ? "Cancel" : "Connect Alpaca"}
            </button>
          </div>
        )}

        {/* ── Alpaca Connect Form ────────────────────────────────── */}
        {showConnectForm && !alpacaStatus?.connected && (
          <div className="alpaca-form-card">
            <h3>Connect your Alpaca account</h3>
            <p className="alpaca-form-sub">
              You can find your API keys in your{" "}
              <a
                href="https://app.alpaca.markets/"
                target="_blank"
                rel="noopener noreferrer"
              >
                Alpaca dashboard
              </a>
              . We recommend starting with paper trading.
            </p>

            {connectError && (
              <div className="auth-error">{connectError}</div>
            )}

            <form onSubmit={handleConnect} className="alpaca-form">
              <label htmlFor="apiKey">API Key</label>
              <input
                id="apiKey"
                type="text"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="PK..."
                required
                autoComplete="off"
              />

              <label htmlFor="apiSecret">API Secret</label>
              <input
                id="apiSecret"
                type="password"
                value={apiSecret}
                onChange={(e) => setApiSecret(e.target.value)}
                placeholder="••••••••"
                required
                autoComplete="off"
              />

              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={paperTrading}
                  onChange={(e) => setPaperTrading(e.target.checked)}
                />
                Paper trading (recommended)
              </label>

              <button
                type="submit"
                className="button button-primary"
                disabled={connecting}
              >
                {connecting ? "Connecting..." : "Connect"}
              </button>
            </form>
          </div>
        )}

        <div className="empty-state">
          <div className="empty-icon">📊</div>
          <h3>No trades yet</h3>
          <p>Connect your brokerage to get started with automated trading.</p>
        </div>
      </div>
    </main>
  );
}
