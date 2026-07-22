import { Circle, ScanLine, TriangleAlert, Video } from "lucide-react";
import type { StatusResult } from "../services/api";

interface Props {
  status: StatusResult | null;
}

export function StatusBar({ status }: Props) {
  const modelText = !status
    ? "Connecting…"
    : status.model_ready
    ? "Moondream2 Q4_K_M · 25 layers GPU · Ready"
    : "Loading model… (~10s)";

  const modelColor = !status
    ? "var(--text3)"
    : status.model_ready
    ? "var(--green)"
    : "var(--amber)";

  const memText =
    status && status.memory_available_mb > 0
      ? `${status.memory_available_mb} MB free`
      : "";

  return (
    <div className="status-bar">
      <Circle size={9} fill="currentColor" style={{ color: modelColor }} />
      <span className="status-model" style={{ color: modelColor }}>{modelText}</span>

      {status?.recording && (
        <span className="status-badge red"><Video size={12} /> RECORDING</span>
      )}
      {status?.autoscan && (
        <span className="status-badge amber"><ScanLine size={12} /> AUTO-SCAN</span>
      )}
      {status && status.frame_age_s != null && status.frame_age_s > 1 && (
        <span className="status-badge red"><TriangleAlert size={12} /> STALE FRAME {status.frame_age_s.toFixed(1)}s</span>
      )}

      {memText && (
        <span className="status-mem">{memText}</span>
      )}
    </div>
  );
}
