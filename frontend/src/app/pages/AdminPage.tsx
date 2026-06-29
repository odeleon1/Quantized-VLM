import { useEffect, useState } from "react";
import { useAuth } from "../context/AuthContext";
import { api } from "../services/api";
import type { AdminUser } from "../services/api";

export function AdminPage() {
  const { user: self } = useAuth();
  const [users, setUsers]           = useState<AdminUser[]>([]);
  const [loading, setLoading]       = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  async function fetchUsers() {
    try {
      const data = await api.adminGetUsers();
      setUsers(data.users);
    } catch (err: unknown) {
      setFetchError(err instanceof Error ? err.message : "Failed to load users.");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { fetchUsers(); }, []);

  async function handlePromote(id: number) {
    setActionError(null);
    try {
      await api.adminPromote(id);
      await fetchUsers();
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : "Action failed.");
    }
  }

  async function handleDemote(id: number) {
    setActionError(null);
    try {
      await api.adminDemote(id);
      await fetchUsers();
    } catch (err: unknown) {
      setActionError(err instanceof Error ? err.message : "Action failed.");
    }
  }

  if (loading)    return <div className="admin-page"><div className="result-empty">Loading users…</div></div>;
  if (fetchError) return <div className="admin-page"><div className="result-error">{fetchError}</div></div>;

  return (
    <div className="admin-page">
      <div className="section-label blue">USER MANAGEMENT</div>
      {actionError && <div className="result-error" style={{ marginBottom: 12 }}>{actionError}</div>}
      <table className="admin-table">
        <thead>
          <tr>
            <th>Username</th>
            <th>Email</th>
            <th>Role</th>
            <th>Joined</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {users.map(u => (
            <tr key={u.id}>
              <td>{u.username}</td>
              <td>{u.email}</td>
              <td>
                <span className={`admin-badge ${u.is_admin ? "admin-badge-admin" : "admin-badge-user"}`}>
                  {u.is_admin ? "Admin" : "User"}
                </span>
              </td>
              <td>{u.created_at.slice(0, 10)}</td>
              <td>
                {u.id === self?.id ? (
                  <span className="admin-you">(you)</span>
                ) : u.is_admin ? (
                  <button className="btn btn-sm" onClick={() => handleDemote(u.id)}>Demote</button>
                ) : (
                  <button className="btn btn-sm btn-inspect" onClick={() => handlePromote(u.id)}>
                    Promote to Admin
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
