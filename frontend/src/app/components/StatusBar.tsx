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
    ? "#44455a"
    : status.model_ready
    ? "#1DB954"
    : "#F7A928";

  const memText =
    status && status.memory_available_mb > 0
      ? `${status.memory_available_mb} MB free`
      : "";

  return (
    <div className="status-bar">
      <span className="status-dot" style={{ color: modelColor }}>●</span>
      <span className="status-model" style={{ color: modelColor }}>{modelText}</span>

      {status?.recording && (
        <span className="status-badge red">⏺ RECORDING</span>
      )}
      {status?.autoscan && (
        <span className="status-badge amber">⟳ AUTO-SCAN</span>
      )}
      {status && status.frame_age_s != null && status.frame_age_s > 1 && (
        <span className="status-badge red">⚠ STALE FRAME {status.frame_age_s.toFixed(1)}s</span>
      )}

      {memText && (
        <span className="status-mem">{memText}</span>
      )}
    </div>
  );
}
