import threading
import time

import cv2

from app.services.capture import open_camera

# Number of consecutive failed reads (at ~0.1s each, so about 5 seconds) before
# the thread assumes the camera dropped off the bus and tries to reconnect.
_FAILURES_BEFORE_RECONNECT = 50


class CameraThread(threading.Thread):
    """
    Continuously reads frames from the camera in a background daemon thread and
    keeps the most recent JPEG available via get_latest_jpeg().

    A monotonic timestamp is recorded with every successful frame so callers can
    tell a live feed apart from a frozen one. get_latest_jpeg() returns None once
    the newest frame is older than max_age_s, which turns a silently stalled USB
    camera into a clean "camera not ready" signal instead of an endlessly repeated
    stale frame.
    """

    def __init__(self, cap: cv2.VideoCapture, *, device=0, width=640, height=480, fps=30):
        super().__init__(daemon=True)
        self.cap = cap
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.running = True
        self._jpeg: bytes | None = None
        self._ts: float | None = None  # time.monotonic() of the last good frame
        self._lock = threading.Lock()

    def run(self):
        consecutive_failures = 0
        reconnect_attempts = 0
        while self.running:
            ok, frame = self.cap.read()
            # cap.read() blocks until the camera delivers a frame, so on a healthy
            # camera it is the rate limiter and no sleep is needed. On failure it
            # tends to return immediately, so a short sleep keeps a dropped camera
            # from pegging a core on this 8GB shared-memory device.
            if ok:
                consecutive_failures = 0
                reconnect_attempts = 0
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                encoded = buf.tobytes()
                now = time.monotonic()
                with self._lock:
                    self._jpeg = encoded
                    self._ts = now
            else:
                consecutive_failures += 1
                time.sleep(0.1)
                if consecutive_failures >= _FAILURES_BEFORE_RECONNECT:
                    self._attempt_reconnect(reconnect_attempts)
                    reconnect_attempts += 1
                    consecutive_failures = 0

    def _attempt_reconnect(self, attempt: int):
        """Release and reopen the capture device. Backoff grows with each failed
        attempt so a permanently unplugged camera does not loop hot."""
        backoff = min(30.0, 2.0 * (attempt + 1))
        print(
            f"[camera] {_FAILURES_BEFORE_RECONNECT} consecutive read failures. "
            f"Reconnect attempt {attempt + 1} after {backoff:.0f}s backoff."
        )
        try:
            self.cap.release()
        except Exception:
            pass
        # Sleep in short slices so stop() stays responsive during the backoff.
        slept = 0.0
        while slept < backoff and self.running:
            time.sleep(0.5)
            slept += 0.5
        if not self.running:
            return
        try:
            self.cap = open_camera(self.device, self.width, self.height, self.fps)
            print("[camera] Reconnect succeeded.")
        except Exception as e:
            print(f"[camera] Reconnect failed: {e}")

    def get_latest_jpeg(self, max_age_s: float = 2.0) -> bytes | None:
        """Return the most recent JPEG, or None if there is no frame yet or the
        newest frame is older than max_age_s (a frozen or stalled camera)."""
        with self._lock:
            if self._jpeg is None or self._ts is None:
                return None
            if (time.monotonic() - self._ts) > max_age_s:
                return None
            return self._jpeg

    def frame_age_s(self) -> float | None:
        """Seconds since the last successful frame, or None if none captured yet."""
        with self._lock:
            if self._ts is None:
                return None
            return time.monotonic() - self._ts

    def stop(self):
        self.running = False
