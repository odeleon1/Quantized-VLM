import { useState } from "react";
import { Moon, Sun } from "lucide-react";

type Theme = "dark" | "light";

function current(): Theme {
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

/**
 * Flips the interface between dark (default) and light. The choice is written to
 * the root data-theme attribute, which drives the CSS token overrides, and
 * persisted to localStorage so it survives restarts. Theme is a device
 * preference, not session state, so it is intentionally not in sessionStorage.
 */
export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(current);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("vlmedge_theme", next);
    setTheme(next);
  }

  const goingTo = theme === "dark" ? "light" : "dark";
  return (
    <button
      className="nav-theme"
      onClick={toggle}
      aria-label={`Switch to ${goingTo} mode`}
      title={`Switch to ${goingTo} mode`}
    >
      {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
    </button>
  );
}
