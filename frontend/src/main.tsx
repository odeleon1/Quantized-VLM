import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
// Self-hosted fonts (bundled into dist) so LAN devices need no internet.
// IBM Plex Mono carries the instrument readout: labels, status, numbers.
// IBM Plex Sans carries running body copy. Same superfamily, so they cohere.
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/ibm-plex-mono/600.css";
import "@fontsource/ibm-plex-mono/700.css";
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "./index.css";
import { App } from "./App";

// Apply the saved theme before first paint so there is no light/dark flash.
// Dark is the default; only an explicit "light" choice overrides it.
const savedTheme = localStorage.getItem("vlmedge_theme") === "light" ? "light" : "dark";
document.documentElement.dataset.theme = savedTheme;

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
