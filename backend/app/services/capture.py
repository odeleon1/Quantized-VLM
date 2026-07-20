"""
Camera capture module.
Provides open/capture/release helpers used by the inference pipeline.
Run directly to verify the camera works independently of inference.
"""

import cv2
import sys
import os


def open_camera(device=0, width=1280, height=720, fps=45):
    """Open a USB/UVC camera and return the VideoCapture object."""
    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera at device index {device}. "
                           "Check that the camera is plugged in and not in use by another process.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep buffer minimal so reads stay fresh
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    actual_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: device {device}, requested {width}x{height}@{fps}fps, got {actual_w}x{actual_h}@{actual_fps:.0f}fps")
    return cap


def capture_frame(cap, flush=5):
    """
    Capture one frame and return it as JPEG bytes.
    Flushes stale buffered frames first so the result reflects the current scene.
    Raises RuntimeError if the frame cannot be read (camera disconnected, etc.).
    """
    # Drain any frames that accumulated while waiting (e.g. during input())
    for _ in range(flush):
        if not cap.grab():
            raise RuntimeError("Camera stopped responding during buffer flush.")
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Failed to read frame from camera. "
                           "Camera may have been disconnected.")
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode the captured frame.")
    return buf.tobytes()


def release_camera(cap):
    """Release the camera device."""
    cap.release()


# --- Standalone test ---
if __name__ == "__main__":
    device = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    num_frames = 5
    output_dir = "/tmp/vlm_camera_frames"
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== Camera Capture Test ===\n")
    print(f"Testing device {device}, capturing {num_frames} frames...\n")

    cap = open_camera(device)

    for i in range(num_frames):
        jpeg_bytes = capture_frame(cap)
        path = os.path.join(output_dir, f"frame_{i:03d}.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        print(f"Frame {i}: {len(jpeg_bytes):,} bytes → {path}")

    release_camera(cap)

    print(f"\n=== Camera Capture Test PASSED ===")
    print(f"Frames saved to {output_dir}/ — inspect them to confirm image quality.")
    print(f"To test with a different device index: python3 -m app.services.capture 1")
