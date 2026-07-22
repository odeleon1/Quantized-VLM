import { useState } from "react";
import { AuthProvider, useAuth } from "./app/context/AuthContext";
import { AdminPage } from "./app/pages/AdminPage";
import { Dashboard } from "./app/pages/Dashboard";
import { EvalPage } from "./app/pages/EvalPage";
import { LibraryPage } from "./app/pages/LibraryPage";
import { LoginPage } from "./app/pages/LoginPage";
import { SignupPage } from "./app/pages/SignupPage";
import { ThemeToggle } from "./app/components/ThemeToggle";
import { ChangePasswordModal } from "./app/components/ChangePasswordModal";
import { KeyRound } from "lucide-react";

type AuthRoute = "login" | "signup";
type Page = "live" | "eval" | "library" | "admin";

function AppContent() {
  const { user, loading, logout } = useAuth();
  const [authRoute, setAuthRoute] = useState<AuthRoute>("login");
  const [page, setPage] = useState<Page>("live");
  const [showChangePw, setShowChangePw] = useState(false);

  if (loading) {
    return <div className="auth-loading">Loading…</div>;
  }

  if (!user) {
    return authRoute === "signup"
      ? <SignupPage onBack={() => setAuthRoute("login")} />
      : <LoginPage onSignup={() => setAuthRoute("signup")} />;
  }

  // Non-admins cannot access eval or admin pages
  const activePage: Page =
    ((page === "admin" || page === "eval") && !user.is_admin) ? "live" : page;

  return (
    <div className="app-root">
      <nav className="app-nav">
        <span className="app-nav-brand">VLM EDGE</span>
        <button
          className={`nav-tab ${activePage === "live" ? "nav-tab-active" : ""}`}
          onClick={() => setPage("live")}
        >
          Live
        </button>
        {user.is_admin && (
          <button
            className={`nav-tab ${activePage === "eval" ? "nav-tab-active" : ""}`}
            onClick={() => setPage("eval")}
          >
            Evaluation
          </button>
        )}
        <button
          className={`nav-tab ${activePage === "library" ? "nav-tab-active" : ""}`}
          onClick={() => setPage("library")}
        >
          Library
        </button>
        {user.is_admin && (
          <button
            className={`nav-tab ${activePage === "admin" ? "nav-tab-active" : ""}`}
            onClick={() => setPage("admin")}
          >
            Admin
          </button>
        )}
        <div className="nav-spacer" />
        <button
          className="nav-theme"
          onClick={() => setShowChangePw(true)}
          aria-label="Change password"
          title="Change password"
        >
          <KeyRound size={16} />
        </button>
        <ThemeToggle />
        <span className="nav-user">{user.username}</span>
        <button className="nav-logout" onClick={logout}>Log out</button>
      </nav>

      {showChangePw && <ChangePasswordModal onClose={() => setShowChangePw(false)} />}

      {activePage === "live"    && <Dashboard />}
      {activePage === "eval"    && <EvalPage />}
      {activePage === "library" && <LibraryPage />}
      {activePage === "admin"   && <AdminPage />}
    </div>
  );
}

export function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}
