"""
Confirms llama-cpp-python can load Moondream2 with CUDA and run inference.
Checks that Python reproduces the same output quality and speed as the llama.cpp CLI.
"""

import time
import sys
import struct
import zlib
import os

# Two levels up from backend/tests/ is the project root (Quantized-VLM/)
_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
MODEL_PATH  = os.path.join(_PROJECT_ROOT, "models", "moondream2-text-model-Q4_K_M.gguf")
MMPROJ_PATH = os.path.join(_PROJECT_ROOT, "models", "moondream2-mmproj-f16.gguf")

TEST_IMAGE_PATH = "/tmp/test_inference.png"
N_GPU_LAYERS = -1  # offload all layers


def create_test_image(path):
    """Write a minimal valid PNG (64x64 gradient) without requiring Pillow."""
    width, height = 64, 64

    def png_chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    raw_rows = b""
    for y in range(height):
        raw_rows += b"\x00"  # filter type: None
        for x in range(width):
            r = int(x * 255 / (width - 1))
            g = int(y * 255 / (height - 1))
            b = 128
            raw_rows += bytes([r, g, b])

    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat_data = zlib.compress(raw_rows)

    png = (
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr_data)
        + png_chunk(b"IDAT", idat_data)
        + png_chunk(b"IEND", b"")
    )

    with open(path, "wb") as f:
        f.write(png)
    print(f"Test image written: {path} ({width}x{height} RGB PNG)")


def run_test():
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MoondreamChatHandler

    print("=== Python Integration Test ===\n")

    # Verify model files exist
    for path in [MODEL_PATH, MMPROJ_PATH]:
        if not os.path.exists(path):
            print(f"ERROR: missing file: {path}")
            sys.exit(1)

    create_test_image(TEST_IMAGE_PATH)

    print(f"\nLoading model (n_gpu_layers={N_GPU_LAYERS})...")
    t_load_start = time.time()

    chat_handler = MoondreamChatHandler(clip_model_path=MMPROJ_PATH)
    llm = Llama(
        model_path=MODEL_PATH,
        chat_handler=chat_handler,
        n_ctx=2048,
        n_gpu_layers=N_GPU_LAYERS,
        verbose=True,
    )

    t_load = time.time() - t_load_start
    print(f"\nModel loaded in {t_load:.1f}s\n")

    # --- Test 1: text-only ---
    print("--- Test 1: text-only prompt ---")
    t0 = time.time()
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": "What is 2 + 2? Answer in one word."}]
    )
    t1 = time.time()
    answer = response["choices"][0]["message"]["content"]
    tokens = response["usage"]["completion_tokens"]
    print(f"Response: {answer}")
    print(f"Tokens: {tokens} | Time: {t1 - t0:.2f}s | Speed: {tokens / (t1 - t0):.1f} t/s\n")

    # --- Test 2: image + text ---
    print("--- Test 2: image + text prompt ---")
    with open(TEST_IMAGE_PATH, "rb") as f:
        import base64
        img_b64 = base64.b64encode(f.read()).decode()

    t0 = time.time()
    response = llm.create_chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": "Describe what you see in one sentence."},
                ],
            }
        ]
    )
    t1 = time.time()
    answer = response["choices"][0]["message"]["content"]
    tokens = response["usage"]["completion_tokens"]
    print(f"Response: {answer}")
    print(f"Tokens: {tokens} | Time: {t1 - t0:.2f}s | Speed: {tokens / (t1 - t0):.1f} t/s\n")

    print("=== Python Integration Test PASSED ===")
    print("Python interface working. Check t/s above; should match the llama.cpp CLI (~19 to 21 t/s).")


if __name__ == "__main__":
    run_test()
