import threading
import time

import cv2


class CameraThread(threading.Thread):
    """
    Continuously reads frames from the camera at 30 FPS in a background daemon thread.
    Latest JPEG is available at any time via get_latest_jpeg().
    Mirrors the VideoThread pattern from the PyQt5 app without any Qt dependency.
    """

    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.cap = cap
        self.running = True
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            # cap.read() blocks until the camera delivers a frame — it is the
            # rate limiter. Adding a sleep on top would halve the effective FPS.
            if ok:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self._lock:
                    self._jpeg = buf.tobytes()

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def stop(self):
        self.running = False
