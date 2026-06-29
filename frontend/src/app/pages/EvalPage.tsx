import { useState, useEffect, useRef } from "react";
import { EvalRunner } from "../components/EvalRunner";
import { ReportList } from "../components/ReportList";
import { ReportViewer } from "../components/ReportViewer";
import { useStatus } from "../hooks/useStatus";
import { useEvalStatus } from "../hooks/useEvalStatus";

export function EvalPage() {
  const [selectedReport, setSelectedReport] = useState<string | null>(null);
  // Increment to force ReportList to remount (re-fetch) after a new run completes
  const [reportListKey, setReportListKey] = useState(0);

  const status = useStatus(3000);
  const evalStatus = useEvalStatus();

  // Detect run completion: auto-select the newly completed report
  const wasRunning = useRef(false);
  useEffect(() => {
    if (!evalStatus) return;
    const justFinished = wasRunning.current && !evalStatus.running;
    wasRunning.current = evalStatus.running;
    if (justFinished && evalStatus.report_id && !evalStatus.error) {
      setSelectedReport(evalStatus.report_id);
      setReportListKey(k => k + 1);
    }
  }, [evalStatus?.running]);

  function handleRunComplete(reportId: string) {
    // EvalRunner also fires this; we use it as a second signal to refresh the list
    // when the run finishes (the useEffect above handles auto-select).
    setReportListKey(k => k + 1);
    setSelectedReport(reportId);
  }

  function handleBaselineSet() {
    // Refresh the report list so the Baseline badge updates
    setReportListKey(k => k + 1);
  }

  return (
    <div className="eval-page">
      {/* Left column: controls + live results */}
      <div className="eval-left-col">
        <EvalRunner
          evalStatus={evalStatus}
          autoscanActive={status?.autoscan ?? false}
          modelReady={status?.model_ready ?? false}
          onRunComplete={handleRunComplete}
        />
      </div>

      {/* Right column: report history + viewer */}
      <div className="eval-right-col">
        <ReportList
          key={reportListKey}
          selected={selectedReport}
          onSelect={setSelectedReport}
        />
        {selectedReport && (
          <ReportViewer
            reportId={selectedReport}
            onBaselineSet={handleBaselineSet}
          />
        )}
      </div>
    </div>
  );
}
