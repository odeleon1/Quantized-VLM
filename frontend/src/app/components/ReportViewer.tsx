import { useState, useEffect } from "react";
import { api } from "../services/api";
import type { EvalReportDetail, EvalComparison } from "../services/api";

interface Props {
  reportId: string;
  onBaselineSet: () => void;
}

function DeltaBadge({ delta }: { delta: number }) {
  if (delta === 0) return <span className="delta-badge delta-same">—</span>;
  const faster = delta < 0;
  return (
    <span className={`delta-badge ${faster ? "delta-faster" : "delta-slower"}`}>
      {faster ? "▼" : "▲"} {Math.abs(delta).toFixed(2)}s
    </span>
  );
}

function LatencyChart({
  results,
  comparison,
}: {
  results: EvalReportDetail["results"];
  comparison: EvalComparison | null;
}) {
  const maxLatency = Math.max(...results.map(r => r.latency_s), 1);

  return (
    <div className="latency-chart">
      <div className="section-label-small">LATENCY</div>
      {results.map(r => {
        const cmp = comparison?.per_prompt[r.label];
        const baselineLatency = cmp ? r.latency_s - cmp.latency_delta : null;

        return (
          <div key={r.label} className="latency-row">
            <span className="latency-row-label">{r.label}</span>
            <div className="latency-bar-track">
              {/* Baseline ghost bar */}
              {baselineLatency != null && (
                <div
                  className="latency-bar-baseline"
                  style={{ width: `${(baselineLatency / maxLatency) * 100}%` }}
                />
              )}
              {/* Current run bar */}
              <div
                className={`latency-bar-fill ${cmp ? (cmp.latency_delta > 0 ? "bar-slower" : cmp.latency_delta < 0 ? "bar-faster" : "") : ""}`}
                style={{ width: `${(r.latency_s / maxLatency) * 100}%` }}
              />
            </div>
            <span className="latency-row-value">{r.latency_s}s</span>
            {cmp && <DeltaBadge delta={cmp.latency_delta} />}
          </div>
        );
      })}
    </div>
  );
}

export function ReportViewer({ reportId, onBaselineSet }: Props) {
  const [report, setReport] = useState<EvalReportDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [fetchError, setFetchError] = useState<string | null>(null);
  const [settingBaseline, setSettingBaseline] = useState(false);
  const [baselineSet, setBaselineSet] = useState(false);

  useEffect(() => {
    setLoading(true);
    setFetchError(null);
    setBaselineSet(false);
    api.evalReport(reportId)
      .then(data => { setReport(data); setLoading(false); })
      .catch(e => { setFetchError(e instanceof Error ? e.message : "Failed to load report."); setLoading(false); });
  }, [reportId]);

  async function handleSetBaseline() {
    setSettingBaseline(true);
    try {
      await api.evalSetBaseline(reportId);
      setBaselineSet(true);
      setReport(prev => prev ? { ...prev, is_baseline: true } : prev);
      onBaselineSet();
    } catch (e: unknown) {
      setFetchError(e instanceof Error ? e.message : "Failed to set baseline.");
    } finally {
      setSettingBaseline(false);
    }
  }

  if (loading) {
    return (
      <div className="report-viewer">
        <div className="report-viewer-loading">Loading report…</div>
      </div>
    );
  }
  if (fetchError) {
    return (
      <div className="report-viewer">
        <div className="result-error">{fetchError}</div>
      </div>
    );
  }
  if (!report) return null;

  const isBaseline = report.is_baseline || baselineSet;
  const { stats, comparison } = report;

  return (
    <div className="report-viewer">
      {/* Header: timestamp + action */}
      <div className="report-viewer-header">
        <div>
          <div className="section-label blue">REPORT</div>
          <span className="report-viewer-ts">{report.timestamp}</span>
        </div>
        <div className="report-viewer-actions">
          {isBaseline ? (
            <span className="report-badge report-badge-green">Baseline</span>
          ) : (
            <button
              className="btn btn-inspect"
              onClick={handleSetBaseline}
              disabled={settingBaseline}
            >
              {settingBaseline ? "Setting…" : "Set as Baseline"}
            </button>
          )}
        </div>
      </div>

      <div className="report-viewer-results">
        {/* ── Stats summary ──────────────────────────────────────────────── */}
        {stats && (
          <div className="stats-panel">
            <div className="stats-chips">
              <div className="eval-stat-chip">
                <span className="eval-stat-label">avg latency</span>
                <span className="eval-stat-value">{stats.avg_latency_s}s</span>
              </div>
              <div className="eval-stat-chip">
                <span className="eval-stat-label">total tokens</span>
                <span className="eval-stat-value">{stats.total_tokens}</span>
              </div>
              <div className="eval-stat-chip">
                <span className="eval-stat-label">fastest</span>
                <span className="eval-stat-value">{stats.fastest.label.split(" ")[0]} ({stats.fastest.latency_s}s)</span>
              </div>
              <div className="eval-stat-chip">
                <span className="eval-stat-label">slowest</span>
                <span className="eval-stat-value">{stats.slowest.label.split(" ")[0]} ({stats.slowest.latency_s}s)</span>
              </div>
              {comparison && (
                <div className={`eval-stat-chip stat-chip-${comparison.direction}`}>
                  <span className="eval-stat-label">vs baseline</span>
                  <span className="eval-stat-value">
                    {comparison.direction === "faster" ? "▼" : comparison.direction === "slower" ? "▲" : "—"}
                    {" "}{Math.abs(comparison.avg_latency_delta).toFixed(2)}s {comparison.direction}
                  </span>
                </div>
              )}
            </div>

            {/* Latency breakdown chart */}
            <LatencyChart results={report.results} comparison={comparison} />
          </div>
        )}

        {/* ── Per-prompt result cards ─────────────────────────────────────── */}
        {report.results.map((r, i) => {
          const cmp = comparison?.per_prompt[r.label];
          return (
            <div key={i} className="report-result-card">
              <div className="report-result-header">
                <span className="result-source source-blue">{r.label}</span>
                <span className="result-entry-meta">
                  {r.latency_s}s
                  <span className="dot">·</span>
                  {r.tokens} tokens
                  {cmp && (
                    <>
                      <span className="dot">·</span>
                      <DeltaBadge delta={cmp.latency_delta} />
                    </>
                  )}
                </span>
              </div>
              <p className="report-result-prompt">{r.prompt}</p>
              <p className="result-entry-text">{r.response}</p>
              <img
                className="report-result-img"
                src={api.evalFrameUrl(r.frame_url)}
                alt={r.label}
                loading="lazy"
                onError={(e) => { (e.currentTarget as HTMLImageElement).style.display = "none"; }}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}
