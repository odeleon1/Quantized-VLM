import { useEffect, useRef } from "react";
import type { InferenceResult } from "../services/api";

export type ResultSource = "Analyze" | "Inspect" | "Auto-Scan";

export interface ResultEntry extends InferenceResult {
  source: ResultSource;
}

interface Props {
  results: ResultEntry[];
  busy: boolean;
  error: string | null;
}

const SOURCE_STYLE: Record<ResultSource, string> = {
  "Analyze":   "source-blue",
  "Inspect":   "source-amber",
  "Auto-Scan": "source-green",
};

export function ResultPanel({ results, busy, error }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [results.length, busy]);

  return (
    <div className="result-panel">
      <div className="section-label blue">RESULT</div>

      <div className="result-history">
        {results.length === 0 && !busy && !error && (
          <div className="result-empty">
            Press Analyze, Inspect, or enable Auto-Scan to start the model.
          </div>
        )}

        {results.map((entry, i) => (
          <div key={i} className="result-entry">
            <div className="result-entry-header">
              <span className={`result-source ${SOURCE_STYLE[entry.source]}`}>
                {entry.source}
              </span>
              <span className="result-entry-meta">
                {new Date(entry.timestamp).toLocaleTimeString()}
                <span className="dot">·</span>
                {entry.elapsed_s}s
                <span className="dot">·</span>
                {entry.tokens} tokens
              </span>
            </div>
            <p className="result-entry-text">{entry.text}</p>
          </div>
        ))}

        {busy && (
          <div className="result-thinking">
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-dot" />
            <span className="thinking-text">Analyzing…</span>
          </div>
        )}

        {error && !busy && (
          <div className="result-error">{error}</div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
