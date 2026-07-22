import { Camera, Eye, Flag, ScanLine, ShieldAlert, Square, Video } from "lucide-react";
import { api, type InferenceResult } from "../services/api";
import type { ResultSource } from "./ResultPanel";

interface Props {
  modelReady: boolean;
  busy: boolean;
  recording: boolean;
  autoscan: boolean;
  onResult: (r: InferenceResult, source: ResultSource) => void;
  onError: (msg: string) => void;
  onBusy: (b: boolean) => void;
  onRecordingChange: (r: boolean) => void;
  onAutoscanChange: (a: boolean) => void;
  onSnapshot: () => void;
  onFlag: () => void;
}

export function ButtonPanel({
  modelReady,
  busy,
  recording,
  autoscan,
  onResult,
  onError,
  onBusy,
  onRecordingChange,
  onAutoscanChange,
  onSnapshot,
  onFlag,
}: Props) {
  const disabled = !modelReady || busy;

  async function runInference(action: () => Promise<InferenceResult>, source: ResultSource) {
    onBusy(true);
    onError("");
    try {
      onResult(await action(), source);
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      onBusy(false);
    }
  }

  async function handleSnapshot() {
    try {
      await api.snapshot();
      onSnapshot();
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : "Snapshot failed");
    }
  }

  async function handleRecord() {
    try {
      if (recording) {
        await api.recordStop();
        onRecordingChange(false);
      } else {
        await api.recordStart();
        onRecordingChange(true);
      }
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : "Record error");
    }
  }

  async function handleAutoscan() {
    try {
      if (autoscan) {
        await api.autoscanStop();
        onAutoscanChange(false);
      } else {
        await api.autoscanStart(10);
        onAutoscanChange(true);
      }
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : "Auto-scan error");
    }
  }

  async function handleFlag() {
    try {
      await api.flag();
      onFlag();
    } catch (e: unknown) {
      onError(e instanceof Error ? e.message : "Flag error");
    }
  }

  return (
    <div className="button-panel">
      {/* Primary inference buttons */}
      <button
        className="btn btn-primary"
        onClick={() => runInference(api.analyze, "Analyze")}
        disabled={disabled}
        title="Describe what the camera sees"
      >
        <Eye size={15} /> Analyze
      </button>

      <button
        className="btn btn-inspect"
        onClick={() => runInference(api.inspect, "Inspect")}
        disabled={disabled}
        title="Detect defects, damage, or safety hazards"
      >
        <ShieldAlert size={15} /> Inspect
      </button>

      <div className="btn-divider" />

      {/* Session management */}
      <button
        className="btn btn-secondary"
        onClick={handleSnapshot}
        title="Save current frame as a JPEG snapshot"
      >
        <Camera size={15} /> Snapshot
      </button>

      <button
        className={`btn ${recording ? "btn-active-red" : "btn-secondary"}`}
        onClick={handleRecord}
        title={recording ? "Stop recording" : "Start recording session"}
      >
        {recording ? <><Square size={15} /> Stop Rec</> : <><Video size={15} /> Record</>}
      </button>

      <button
        className={`btn ${autoscan ? "btn-active-amber" : "btn-secondary"}`}
        onClick={handleAutoscan}
        disabled={!modelReady}
        title={autoscan ? "Stop auto-scan" : "Run Analyze every 10 seconds automatically"}
      >
        {autoscan ? <><Square size={15} /> Auto-Scan</> : <><ScanLine size={15} /> Auto-Scan</>}
      </button>

      <div className="btn-divider" />

      {/* Flag */}
      <button
        className="btn btn-flag"
        onClick={handleFlag}
        title="Flag current frame and last result for review"
      >
        <Flag size={15} /> Flag
      </button>
    </div>
  );
}
