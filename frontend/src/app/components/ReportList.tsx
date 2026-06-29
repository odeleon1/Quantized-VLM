import { useEffect, useState } from "react";
import { api } from "../services/api";
import type { EvalReportMeta } from "../services/api";

interface Props {
  selected: string | null;
  onSelect: (id: string) => void;
}

function formatTimestamp(ts: string): string {
  try {
    return new Date(ts).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return ts;
  }
}

export function ReportList({ selected, onSelect }: Props) {
  const [reports, setReports] = useState<EvalReportMeta[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    setLoading(true);
    setError(false);
    api.evalReports()
      .then(data => { setReports(data.reports); setLoading(false); })
      .catch(() => { setError(true); setLoading(false); });
  }, []);

  return (
    <div className="report-list">
      <div className="section-label amber">PAST REPORTS</div>

      <div className="report-list-items">
        {loading && (
          <div className="report-list-empty">Loading…</div>
        )}
        {!loading && error && (
          <div className="report-list-empty">Failed to load reports.</div>
        )}
        {!loading && !error && reports.length === 0 && (
          <div className="report-list-empty">No evaluation reports yet.</div>
        )}
        {!loading && !error && reports.map(r => (
          <button
            key={r.id}
            className={`report-list-item ${selected === r.id ? "report-list-item-selected" : ""} ${r.legacy ? "report-list-item-legacy" : ""}`}
            onClick={() => !r.legacy && onSelect(r.id)}
            disabled={!!r.legacy}
            title={r.legacy ? "Legacy CLI report — no structured data available" : (r.timestamp ?? r.id)}
          >
            <div className="report-list-item-top">
              <span className="report-list-item-ts">
                {r.timestamp ? formatTimestamp(r.timestamp) : r.id.replace("report_", "")}
              </span>
              {r.is_baseline && (
                <span className="report-badge report-badge-green">Baseline</span>
              )}
              {r.legacy && (
                <span className="report-badge report-badge-gray">CLI</span>
              )}
            </div>
            {r.avg_latency_s != null && (
              <div className="report-list-item-meta">
                avg {r.avg_latency_s}s · {r.result_count} prompts
              </div>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
