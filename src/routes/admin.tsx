import { createFileRoute, useRouter } from "@tanstack/react-router";
import { useEffect, useState } from "react";
import { api } from "~/lib/api";

interface AdminUser {
  id: number;
  email: string;
  role: string;
  createdAt: string;
  alpacaConnected: boolean;
  alpacaPaperTrading: boolean;
}

export const Route = createFileRoute("/admin")({
  head: () => ({
    meta: [{ title: "Admin — ApexTrade" }],
  }),
  component: AdminPage,
});

function AdminPage() {
  const router = useRouter();
  const [currentUser, setCurrentUser] = useState<{
    email: string;
    userId: number;
    role: string;
  } | null>(null);
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api.getMe().then((u) => {
      if (!u) {
        router.navigate({ to: "/login" });
        return;
      }
      setCurrentUser(u as any);
      if ((u as any).role !== "admin") {
        router.navigate({ to: "/dashboard" });
        return;
      }
      loadUsers();
    });
  }, [router]);

  async function loadUsers() {
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/api/admin/users", { credentials: "include" });
      if (res.status === 403) {
        router.navigate({ to: "/dashboard" });
        return;
      }
      const data = await res.json();
      setUsers(data);
    } catch {
      setError("Failed to load users");
    } finally {
      setLoading(false);
    }
  }

  async function toggleRole(userId: number, currentRole: string) {
    const newRole = currentRole === "admin" ? "user" : "admin";
    try {
      const res = await fetch(`/api/admin/users/${userId}/role`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ role: newRole }),
      });
      const data = await res.json();
      if (res.ok) {
        setUsers((prev) =>
          prev.map((u) => (u.id === userId ? { ...u, role: data.role } : u)),
        );
      } else {
        setError(data.error || "Failed to update role");
      }
    } catch {
      setError("Failed to update role");
    }
  }

  async function handleLogout() {
    await api.logout();
    router.navigate({ to: "/" });
  }

  if (!currentUser || currentUser.role !== "admin") return null;

  return (
    <main className="admin-page">
      <header className="admin-header">
        <div className="container admin-header-inner">
          <a className="brand" href="/dashboard">
            <span className="brand-mark">◈</span>Apex<span>Trade</span>
          </a>
          <div className="admin-header-right">
            <a href="/dashboard" className="nav-link">Dashboard</a>
            <span className="user-email">{currentUser.email}</span>
            <span className="admin-badge">Admin</span>
            <button onClick={handleLogout} className="logout-btn">
              Log out
            </button>
          </div>
        </div>
      </header>

      <div className="container admin-content">
        <div className="admin-top">
          <h1>Admin Panel</h1>
          <p>Manage users and roles</p>
        </div>

        {error && <div className="admin-error">{error}</div>}

        {loading ? (
          <div className="admin-loading">Loading users...</div>
        ) : (
          <div className="admin-table-wrap">
            <table className="admin-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Email</th>
                  <th>Role</th>
                  <th>Created</th>
                  <th>Alpaca</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map((u) => (
                  <tr key={u.id}>
                    <td className="td-id">{u.id}</td>
                    <td>{u.email}</td>
                    <td>
                      <span
                        className={`role-badge ${u.role === "admin" ? "role-admin" : "role-user"}`}
                      >
                        {u.role}
                      </span>
                    </td>
                    <td className="td-date">
                      {new Date(u.createdAt).toLocaleDateString()}
                    </td>
                    <td>
                      {u.alpacaConnected ? (
                        <span className="alpaca-status connected">
                          ✅ {u.alpacaPaperTrading ? "Paper" : "Live"}
                        </span>
                      ) : (
                        <span className="alpaca-status disconnected">
                          Not connected
                        </span>
                      )}
                    </td>
                    <td>
                      <button
                        className="role-toggle-btn"
                        onClick={() => toggleRole(u.id, u.role)}
                        disabled={u.id === currentUser.userId}
                        title={
                          u.id === currentUser.userId
                            ? "Cannot change your own role"
                            : `Change role to ${u.role === "admin" ? "user" : "admin"}`
                        }
                      >
                        {u.role === "admin" ? "Make User" : "Make Admin"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </main>
  );
}
