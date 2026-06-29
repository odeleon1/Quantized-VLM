import { useState } from "react";
import { useAuth } from "../context/AuthContext";
import { api } from "../services/api";

interface Props {
  onSignup: () => void;
}

export function LoginPage({ onSignup }: Props) {
  const { login } = useAuth();
  const [identifier, setIdentifier] = useState("");
  const [password, setPassword]     = useState("");
  const [error, setError]           = useState<string | null>(null);
  const [loading, setLoading]       = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const res = await api.authLogin(identifier, password);
      login(res.access_token, res.user);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Login failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-brand">VLM EDGE</div>
        <h2 className="auth-title">Sign in</h2>
        <form className="auth-form" onSubmit={handleSubmit}>
          <label className="auth-label">Username or email</label>
          <input
            className="auth-input"
            type="text"
            value={identifier}
            onChange={e => setIdentifier(e.target.value)}
            autoComplete="username"
            autoFocus
          />
          <label className="auth-label">Password</label>
          <input
            className="auth-input"
            type="password"
            value={password}
            onChange={e => setPassword(e.target.value)}
            autoComplete="current-password"
          />
          {error && <div className="auth-error">{error}</div>}
          <button className="auth-btn" type="submit" disabled={loading}>
            {loading ? "Signing in…" : "Sign in"}
          </button>
        </form>
        <p className="auth-footer">
          Don't have an account?{" "}
          <button className="auth-link" type="button" onClick={onSignup}>
            Sign up
          </button>
        </p>
      </div>
    </div>
  );
}
