import { api } from "../services/api";

interface Props {
  cameraReady: boolean;
}

export function CameraFeed({ cameraReady }: Props) {
  return (
    <div className="camera-card">
      <div className="section-label red">CAMERA FEED</div>
      {cameraReady ? (
        <img
          className="camera-img"
          src={api.streamUrl()}
          alt="Live camera feed"
        />
      ) : (
        <div className="camera-placeholder">
          <span>Waiting for camera…</span>
        </div>
      )}
    </div>
  );
}
