import { useState, useEffect, useRef } from "react";
import { Check } from "lucide-react";
import { api } from "../services/api";
import type { EvalStatus } from "../services/api";

interface Props {
  evalStatus: EvalStatus | null;
  autoscanActive: boolean;
  modelReady: boolean;
  onRunComplete: (reportId: string) => void;
}

export function EvalRunner({ evalStatus, autoscanActive, modelReady, onRunComplete }: Props) {
  const [startError, setStartError] = useState<string | null>(null);
  const [settingBaseline, setSettingBaseline] = useState(false);
  const [baselineSet, setBaselineSet] = useState(false);
  const wasRunning = useRef(false);

  useEffect(() => {
    if (!evalStatus) return;
    const justFinished = wasRunning.current && !evalStatus.running;
    wasRunning.current = evalStatus.running;
    if (justFinished && evalStatus.report_id && !evalStatus.error) {
      setBaselineSet(false);
      onRunComplete(evalStatus.report_id);
    }
  }, [evalStatus?.running, evalStatus?.report_id, evalStatus?.error, onRunComplete]);

  async function handleRun() {
    setStartError(null);
    setBaselineSet(false);
    try {
      await api.evalRun();
    } catch (e: unknown) {
      setStartError(e instanceof Error ? e.message : "Failed to start evaluation.");
    }
  }

  async function handleSetBaseline() {
    if (!evalStatus?.report_id) return;
    setSettingBaseline(true);
    try {
      await api.evalSetBaseline(evalStatus.report_id);
      setBaselineSet(true);
    } catch (e: unknown) {
      setStartError(e instanceof Error ? e.message : "Failed to set baseline.");
    } finally {
      setSettingBaseline(false);
    }
  }

  const running = evalStatus?.running ?? false;
  const progress = evalStatus?.progress ?? 0;
  const total = evalStatus?.total ?? 5;
  const results = evalStatus?.results ?? [];
  const done = !running && results.length > 0 && !evalStatus?.error;

  // Summary stats computed client-side from the status results
  const avgLatency = results.length > 0
    ? (results.reduce((s, r) => s + r.latency_s, 0) / results.length).toFixed(2)
    : null;
  const totalTokens = results.reduce((s, r) => s + r.tokens, 0);

  let disabledReason = "";
  if (!modelReady) disabledReason = "Model loading…";
  else if (autoscanActive) disabledReason = "Stop Auto-Scan first";
  else if (running) disabledReason = "Running…";

  return (
    <div className="eval-runner">
      <div className="section-label blue">EVALUATION</div>

      {/* Run button */}
      <div className="eval-controls">
        <button
          className="btn btn-primary"
          onClick={handleRun}
          disabled={!!disabledReason}
          title={disabledReason || "Run all 5 prompts against the current camera frame"}
        >
          {running ? "Running…" : "Run Evaluation"}
        </button>
        {disabledReason && !running && (
          <span className="eval-disabled-reason">{disabledReason}</span>
        )}
      </div>

      {/* Progress bar */}
      {(running || done) && (
        <div className="eval-progress">
          <div className="eval-progress-header">
            <span className="eval-step-label">
              {running
                ? (evalStatus?.current_label ?? "Starting…")
                : "Complete"}
            </span>
            <span className="eval-step-counter">{progress} / {total}</span>
          </div>
          <div className="eval-progress-track">
            <div
              className="eval-progress-fill"
              style={{ width: `${(progress / total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Errors */}
      {(startError || evalStatus?.error) && (
        <div className="result-error">{startError ?? evalStatus?.error}</div>
      )}

      {/* Completion summary */}
      {done && (
        <div className="eval-summary">
          <div className="eval-summary-stats">
            <div className="eval-stat-chip">
              <span className="eval-stat-label">avg latency</span>
              <span className="eval-stat-value">{avgLatency}s</span>
            </div>
            <div className="eval-stat-chip">
              <span className="eval-stat-label">total tokens</span>
              <span className="eval-stat-value">{totalTokens}</span>
            </div>
          </div>
          <button
            className={`btn ${baselineSet ? "btn-secondary" : "btn-inspect"}`}
            onClick={handleSetBaseline}
            disabled={settingBaseline || baselineSet}
          >
            {baselineSet ? <><Check size={15} /> Baseline Set</> : settingBaseline ? "Setting…" : "Set as Baseline"}
          </button>
        </div>
      )}

      {/* Idle empty state */}
      {!running && !done && !evalStatus?.error && !startError && (
        <div className="result-empty">
          Run the 5-prompt evaluation suite against the current camera frame.
          Select a past report on the right to compare runs.
        </div>
      )}
    </div>
  );
}
