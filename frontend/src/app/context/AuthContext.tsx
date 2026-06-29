import { createContext, useContext, useEffect, useState } from "react";

export interface AuthUser {
  id: number;
  username: string;
  email: string;
  is_admin: boolean;
}

interface AuthContextValue {
  user: AuthUser | null;
  loading: boolean;
  login: (token: string, user: AuthUser) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextValue>({
  user: null,
  loading: true,
  login: () => {},
  logout: () => {},
});

export function useAuth() {
  return useContext(AuthContext);
}

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [loading, setLoading] = useState(true);

  function doLogout() {
    // Clear session token and result history — sessionStorage survives page
    // refreshes but is wiped when the browser window/tab closes.
    sessionStorage.removeItem("vlmedge_token");
    sessionStorage.removeItem("vlmedge_results");
    setUser(null);
  }

  // Always register the 401 listener unconditionally so fresh-session users
  // still get auto-logout if the server rejects their token.
  useEffect(() => {
    const handle401 = () => doLogout();
    window.addEventListener("vlmedge:unauthorized", handle401);
    return () => window.removeEventListener("vlmedge:unauthorized", handle401);
  }, []);

  // Restore session from sessionStorage on page load / refresh.
  // sessionStorage is NOT cleared on refresh, so the user stays logged in.
  // It IS cleared when the browser window closes, forcing re-login next visit.
  useEffect(() => {
    const storedToken = sessionStorage.getItem("vlmedge_token");
    if (!storedToken) {
      setLoading(false);
      return;
    }

    fetch("/auth/me", { headers: { Authorization: `Bearer ${storedToken}` } })
      .then(r => (r.ok ? r.json() : Promise.reject()))
      .then((u: AuthUser) => setUser(u))
      .catch(() => sessionStorage.removeItem("vlmedge_token"))
      .finally(() => setLoading(false));
  }, []);

  function login(storedToken: string, u: AuthUser) {
    sessionStorage.setItem("vlmedge_token", storedToken);
    setUser(u);
  }

  return (
    <AuthContext.Provider value={{ user, loading, login, logout: doLogout }}>
      {children}
    </AuthContext.Provider>
  );
}
