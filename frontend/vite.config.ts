import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/stream":   "http://localhost:8000",
      "/analyze":  "http://localhost:8000",
      "/inspect":  "http://localhost:8000",
      "/snapshot": "http://localhost:8000",
      "/record":   "http://localhost:8000",
      "/autoscan": "http://localhost:8000",
      "/flag":     "http://localhost:8000",
      "/status":   "http://localhost:8000",
      "/eval":     "http://localhost:8000",
      "/auth":     "http://localhost:8000",
      "/admin":    "http://localhost:8000",
    },
  },
});
