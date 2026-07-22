import { useEffect, useState } from "react";
import { X } from "lucide-react";
import { api } from "../services/api";
import { PasswordStrengthBar } from "./PasswordStrengthBar";

/**
 * Self-service password change. Posts to /auth/change-password, which verifies
 * the current password server-side and enforces the strength rules. The form
 * only does the cheap client checks (fields present, new pair matches); the
 * server is the authority on everything else.
 */
export function ChangePasswordModal({ onClose }: { onClose: () => void }) {
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (next !== confirm) {
      setError("The new passwords do not match.");
      return;
    }
    if (next === current) {
      setError("The new password must differ from the current one.");
      return;
    }
    setBusy(true);
    try {
      await api.authChangePassword(current, next);
      setDone(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not change password.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="pw-overlay" onClick={onClose}>
      <div className="auth-card pw-card" onClick={e => e.stopPropagation()}>
        <button className="pw-close" onClick={onClose} aria-label="Close">
          <X size={18} />
        </button>

        {done ? (
          <>
            <div className="auth-title">Password changed</div>
            <p className="auth-success">Your password has been updated. Use it next time you sign in.</p>
            <button className="auth-btn" onClick={onClose}>Done</button>
          </>
        ) : (
          <form className="auth-form" onSubmit={submit}>
            <div className="auth-title">Change password</div>

            <label className="auth-label">Current password</label>
            <input
              className="auth-input"
              type="password"
              autoComplete="current-password"
              value={current}
              onChange={e => setCurrent(e.target.value)}
              required
            />

            <label className="auth-label">New password</label>
            <input
              className="auth-input"
              type="password"
              autoComplete="new-password"
              value={next}
              onChange={e => setNext(e.target.value)}
              required
            />
            <PasswordStrengthBar password={next} />

            <label className="auth-label">Confirm new password</label>
            <input
              className="auth-input"
              type="password"
              autoComplete="new-password"
              value={confirm}
              onChange={e => setConfirm(e.target.value)}
              required
            />

            {error && <p className="auth-error">{error}</p>}

            <button className="auth-btn" type="submit" disabled={busy}>
              {busy ? "Updating…" : "Update password"}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
