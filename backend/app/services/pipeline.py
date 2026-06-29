"""
Phase 7 — End-to-End Integration Pipeline.
Camera capture → inference → text output, running continuously for N frames.
Annotated frames (with prompt and response overlaid) are saved to output/runs/.
Model is loaded once at startup.
"""

import os
import base64
import time
import sys
from datetime import datetime

from app.services.capture import open_camera, capture_frame, release_camera
from app.services.eval import annotate_frame
from app.core.config import MODEL_PATH, MMPROJ_PATH, RUNS_DIR, CAMERA_INDEX
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

N_FRAMES   = 10       # number of frames to process in the sustained test
FRAME_DELAY = 2.0     # seconds between frames
PROMPT = "Describe what you see in one or two sentences."


def memory_available_mb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return -1


def load_model():
    print("Loading model (this takes ~10s)...")
    t0 = time.time()
    chat_handler = MoondreamChatHandler(clip_model_path=MMPROJ_PATH)
    llm = Llama(
        model_path=MODEL_PATH,
        chat_handler=chat_handler,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s\n")
    return llm


def run_inference(llm, jpeg_bytes, prompt=None):
    if prompt is None:
        prompt = PROMPT
    img_b64 = base64.b64encode(jpeg_bytes).decode()
    t0 = time.time()
    response = llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }]
    )
    elapsed = time.time() - t0
    text = response["choices"][0]["message"]["content"].strip()
    tokens = response["usage"]["completion_tokens"]
    llm.reset()
    return text, tokens, elapsed


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)

    print("=== Phase 7 — End-to-End Integration Pipeline ===\n")

    llm = load_model()
    cap = open_camera(CAMERA_INDEX)
    print()

    mem_start = memory_available_mb()
    print(f"Memory available at start: {mem_start} MB\n")
    print(f"Running {N_FRAMES} frames (Ctrl+C to stop early)...\n")
    print("-" * 60)

    try:
        for i in range(N_FRAMES):
            frame_num = f"{i + 1:03d}"

            # Capture
            jpeg_bytes = capture_frame(cap)

            # Infer (llm.reset() called inside run_inference)
            text, tokens, elapsed = run_inference(llm, jpeg_bytes)

            # Save annotated frame
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_path = os.path.join(RUNS_DIR, f"frame_{ts}.jpg")
            with open(frame_path, "wb") as f:
                f.write(annotate_frame(jpeg_bytes, PROMPT, text))

            mem_now = memory_available_mb()
            print(f"[Frame {frame_num}] {elapsed:.1f}s | {tokens} tokens | mem avail: {mem_now} MB")
            print(f"  → {text}")
            print()

            if i < N_FRAMES - 1:
                time.sleep(FRAME_DELAY)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        release_camera(cap)

    mem_end = memory_available_mb()
    mem_delta = mem_start - mem_end
    print("-" * 60)
    print(f"\nMemory available at end:   {mem_end} MB")
    print(f"Memory delta:              {mem_delta:+d} MB", end="  ")
    if mem_delta < 1500:
        print("(stable — within normal range for CUDA buffer allocation)")
    else:
        print("(WARNING: memory consumption beyond expected model overhead — investigate)")

    print(f"\nAnnotated frames saved to {RUNS_DIR}/")
    print("\n=== Phase 7 complete ===")


if __name__ == "__main__":
    main()
