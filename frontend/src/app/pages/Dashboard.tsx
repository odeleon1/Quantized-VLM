import { useState, useCallback, useEffect, useRef } from "react";
import { CameraFeed } from "../components/CameraFeed";
import { StatusBar } from "../components/StatusBar";
import { ResultPanel } from "../components/ResultPanel";
import type { ResultEntry } from "../components/ResultPanel";
import { ButtonPanel } from "../components/ButtonPanel";
import { useStatus } from "../hooks/useStatus";
import type { InferenceResult } from "../services/api";
import type { ResultSource } from "../components/ResultPanel";

const RESULTS_KEY = "vlmedge_results";

function loadResults(): ResultEntry[] {
  try {
    const raw = sessionStorage.getItem(RESULTS_KEY);
    return raw ? (JSON.parse(raw) as ResultEntry[]) : [];
  } catch {
    return [];
  }
}

function saveResults(entries: ResultEntry[]) {
  try {
    sessionStorage.setItem(RESULTS_KEY, JSON.stringify(entries));
  } catch {
    // sessionStorage full or unavailable — skip silently
  }
}

export function Dashboard() {
  // Initialize results from sessionStorage so history survives tab switches.
  const [results, setResults] = useState<ResultEntry[]>(loadResults);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [snapshotToast, setSnapshotToast] = useState(false);
  const [flagToast, setFlagToast] = useState(false);

  // Persist result history to sessionStorage whenever it changes.
  // sessionStorage survives page refreshes but is cleared when the browser
  // window closes, matching the session lifetime the user expects.
  useEffect(() => {
    saveResults(results);
  }, [results]);

  // Derive recording and autoscan from server status so they survive tab switches.
  const status    = useStatus(1500);
  const autoscan  = status?.autoscan  ?? false;
  const recording = status?.recording ?? false;

  const showToast = (setter: (v: boolean) => void) => {
    setter(true);
    setTimeout(() => setter(false), 2500);
  };

  // Track the last result timestamp we've already added so we don't duplicate.
  const lastSeenTimestamp = useRef<string | null>(null);

  // Seed the ref from the most recent saved result so we don't re-add it on mount.
  useEffect(() => {
    const saved = loadResults();
    if (saved.length > 0) {
      lastSeenTimestamp.current = saved[saved.length - 1].timestamp;
    }
  }, []);

  const handleResult = useCallback((r: InferenceResult, source: ResultSource) => {
    lastSeenTimestamp.current = r.timestamp;
    setResults(prev => [...prev, { ...r, source }]);
    setError(null);
  }, []);

  // Pick up results that completed while the user was on another tab.
  // Uses server-supplied source so the label is always correct.
  useEffect(() => {
    const lr = status?.last_result;
    if (!lr || lr.timestamp === lastSeenTimestamp.current) return;
    lastSeenTimestamp.current = lr.timestamp;
    const source = (lr.source ?? "Auto-Scan") as ResultSource;
    setResults(prev => [...prev, {
      text: lr.text,
      tokens: lr.tokens,
      elapsed_s: lr.elapsed_s,
      timestamp: lr.timestamp,
      source,
    }]);
    setError(null);
  }, [status?.last_result?.timestamp]);

  const handleError = useCallback((msg: string) => setError(msg), []);
  const handleBusy  = useCallback((b: boolean)   => setBusy(b),   []);

  return (
    <div className="dashboard">
      <StatusBar status={status} />

      <main className="dashboard-main">
        {/* Left column: camera + buttons */}
        <div className="left-col">
          <CameraFeed cameraReady={status?.camera_ready ?? false} />

          <ButtonPanel
            modelReady={status?.model_ready ?? false}
            busy={busy}
            recording={recording}
            autoscan={autoscan}
            onResult={handleResult}
            onError={handleError}
            onBusy={handleBusy}
            onRecordingChange={() => {}}
            onAutoscanChange={() => {}}
            onSnapshot={() => showToast(setSnapshotToast)}
            onFlag={() => showToast(setFlagToast)}
          />
        </div>

        {/* Right column: result history */}
        <div className="right-col">
          <ResultPanel results={results} busy={busy} error={error} />
        </div>
      </main>

      {/* Toast notifications */}
      {snapshotToast && <div className="toast">Snapshot saved</div>}
      {flagToast && <div className="toast toast-amber">Frame flagged</div>}
    </div>
  );
}
