import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { StatusResult } from "../services/api";

export function useStatus(intervalMs = 3000) {
  const [status, setStatus] = useState<StatusResult | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function poll() {
      try {
        const s = await api.getStatus();
        if (!cancelled) setStatus(s);
      } catch {
        // server not yet up — stay null
      }
    }

    poll();
    const id = setInterval(poll, intervalMs);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [intervalMs]);

  return status;
}
