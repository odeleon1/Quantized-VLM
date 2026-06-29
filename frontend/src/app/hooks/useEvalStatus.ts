import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { EvalStatus } from "../services/api";

// Self-adjusting poll: 1.5s while a run is active, 5s when idle.
export function useEvalStatus(): EvalStatus | null {
  const [status, setStatus] = useState<EvalStatus | null>(null);

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>;
    let cancelled = false;

    async function poll() {
      try {
        const s = await api.evalStatus();
        if (!cancelled) {
          setStatus(s);
          timer = setTimeout(poll, s.running ? 1500 : 5000);
        }
      } catch {
        if (!cancelled) timer = setTimeout(poll, 5000);
      }
    }

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  return status;
}
