# LEARNING.md — Technical Reference & Concepts

**Project:** Edge VLM Integration on Jetson Orin Nano
**Last Updated:** Prompt 63 | July 20, 2026

> This document explains every tool, concept, model, and library used in this project.
> Written to be understood by someone encountering these topics for the first time,
> while still being precise enough to be useful as a reference.

---

## Table of Contents

1. [What Is a Large Language Model (LLM)?](#1-what-is-a-large-language-model-llm)
2. [What Is a Vision-Language Model (VLM)?](#2-what-is-a-vision-language-model-vlm)
3. [What Is Quantization?](#3-what-is-quantization)
4. [GGUF Format](#4-gguf-format)
5. [Models: Moondream2 & Phi-3.5-vision](#5-models-moondream2--phi-35-vision)
6. [llama.cpp](#6-llamacpp)
7. [llama-cpp-python](#7-llama-cpp-python)
8. [The NVIDIA Jetson Orin Nano](#8-the-nvidia-jetson-orin-nano)
9. [Unified Memory Architecture](#9-unified-memory-architecture)
10. [How the VLM Operates on the Jetson](#10-how-the-vlm-operates-on-the-jetson)
11. [Why Not TensorRT-LLM?](#11-why-not-tensorrt-llm)
12. [Building llama.cpp with CUDA on JetPack 7.2](#12-building-llamacpp-with-cuda-on-jetpack-72)
13. [Using llama-cpp-python on JetPack 7.2](#13-using-llama-cpp-python-on-jetpack-72)
14. [Camera Capture on Linux with OpenCV](#14-camera-capture-on-linux-with-opencv)
15. [Prompt Engineering for VLMs](#15-prompt-engineering-for-vlms)
16. [PyQt5 Desktop Applications on ARM/Jetson](#16-pyqt5-desktop-applications-on-armjetson)
17. [FastAPI + MJPEG Streaming on Jetson](#17-fastapi--mjpeg-streaming-on-jetson)
18. [React Frontend Architecture for Edge AI](#18-react-frontend-architecture-for-edge-ai)
19. [Web Evaluation: Background Threads, Stats, and Security](#19-web-evaluation-background-threads-stats-and-security)
20. [Web Authentication: JWT, SQLite, and React Auth Patterns](#20-web-authentication-jwt-sqlite-and-react-auth-patterns)
21. [Session Lifetime, Result History, and the Library](#21-session-lifetime-result-history-and-the-library)
22. [Latency, Camera Reliability, and Responsive Layout](#22-latency-camera-reliability-and-responsive-layout)
23. [Glossary](#23-glossary)

---

## 1. What Is a Large Language Model (LLM)?

An LLM is a neural network trained on massive amounts of text data. It learns statistical patterns in language — which words follow which other words, in what contexts, with what meaning — and uses those patterns to generate coherent, contextually appropriate text.

**How it works at a high level:**
- Input text is broken into tokens (roughly word-pieces)
- Each token gets converted to a vector (a list of numbers representing meaning)
- Layers of "attention" mechanisms let the model weigh relationships between all tokens
- The model predicts the most likely next token, then the next, and so on

**Parameters** are the numbers inside the network (weights and biases) that encode this learned knowledge. A "1.8B model" has 1.8 billion parameters. More parameters generally means more capability but also more memory and compute required.

---

## 2. What Is a Vision-Language Model (VLM)?

A VLM is an LLM extended to also understand images. It combines two components:

- **Vision encoder** — a model (often a variant of CLIP or ViT) that converts an image into a set of vector representations (embeddings)
- **Language model** — processes both the image embeddings and text tokens together to generate a response

**What VLMs can do:**
- Answer questions about an image ("What is in this image?")
- Describe scenes, objects, or text visible in a photo
- Read and interpret signage, gauges, or displays
- Caption images automatically

In this project, the VLM receives image input from a camera and produces text output — enabling the Jetson to "see and describe" its environment.

---

## 3. What Is Quantization?

Neural network weights are normally stored as 32-bit floating point numbers (FP32). Each parameter takes 4 bytes. A 1.8B parameter model in FP32 would need ~7.2GB just for weights — too large for most edge devices.

**Quantization** reduces the numerical precision of weights to use less memory:

| Format | Bits per weight | Memory for 1.8B model | Quality loss |
|--------|----------------|----------------------|-------------|
| FP32 | 32 | ~7.2 GB | None (baseline) |
| FP16 | 16 | ~3.6 GB | Negligible |
| INT8 | 8 | ~1.8 GB | Minor |
| INT4 | 4 | ~0.9 GB | Moderate |

**The tradeoff:** Lower precision = smaller model = less memory = faster inference, but also some degradation in output quality. The goal is to find the quantization level where quality loss is acceptable for the task.

**Q4_K_M** (the format used in this project) is a mixed-precision INT4 scheme. It uses 4-bit precision for most weights but applies higher precision to key layers (the "K" and "M" indicate which groups get special treatment). This produces better quality than straight INT4 quantization for a minimal size penalty.

---

## 4. GGUF Format

**GGUF** (GPT-Generated Unified Format) is a file format for storing quantized LLM weights. It is the standard format used by llama.cpp.

**Why GGUF matters:**
- Single-file format — the entire model (weights, tokenizer, metadata) is in one `.gguf` file
- Supports multiple quantization levels in the same ecosystem
- Optimized for CPU and GPU inference via llama.cpp
- Widely adopted — most open-source models have pre-quantized GGUF versions available on Hugging Face

Files are named with their quantization level, e.g.:
- `moondream2-Q4_K_M.gguf` — 4-bit mixed precision, good quality/size balance
- `moondream2-Q8_0.gguf` — 8-bit, higher quality, larger file

---

## 5. Models: Moondream2 & Phi-3.5-vision

### Moondream2 (Primary Choice)

- **Size:** 1.8 billion parameters
- **Developer:** Vikhyat Kopparapu (open source, community-driven)
- **Design goal:** Lightweight VLM specifically for edge and resource-constrained deployment
- **Capabilities:** Image captioning, visual question answering (VQA), object detection description
- **Why chosen:** Fits comfortably in Jetson Orin Nano memory even with quantization overhead; strong community; pre-quantized GGUF versions available

### Phi-3.5-vision-instruct (Fallback)

- **Size:** 4.2 billion parameters
- **Developer:** Microsoft
- **Design goal:** Small but capable multimodal model; part of Microsoft's "Phi" small model series
- **Capabilities:** More advanced reasoning than Moondream2, better instruction following
- **Why it's a fallback:** At INT4, it pushes the Jetson Orin Nano's available memory close to the limit, leaving little headroom for the OS and robotics stack

---

## 6. llama.cpp

**llama.cpp** is an open-source C/C++ inference engine for running LLMs and VLMs locally. Originally written to run Meta's LLaMA model on consumer hardware, it has grown into the most widely-used edge inference framework for quantized models.

**Key properties:**
- Runs on CPU and GPU (CUDA, Metal, OpenCL)
- First-class GGUF support
- Can offload model layers to GPU while keeping others on CPU — useful when the model is larger than GPU VRAM alone
- Actively maintained with growing VLM support (llava, moondream, etc.)
- On the Jetson Orin Nano, it is compiled with CUDA support to leverage the onboard GPU

**Why llama.cpp over TensorRT-LLM for this project:** See Section 11.

---

## 7. llama-cpp-python

**llama-cpp-python** is a Python wrapper around llama.cpp. It exposes llama.cpp's C++ inference engine as a Python library, making it easy to integrate into Python-based applications.

**What it enables:**
- Load a `.gguf` model file in Python
- Pass image + text prompt to a VLM
- Receive generated text output
- OpenAI-compatible API server mode (useful for integration with other tools)

**Installation note:** On the Jetson, it must be compiled with CUDA support enabled — a default install does not do this. See Section 13 for the exact build command and API usage patterns.

---

## 8. The NVIDIA Jetson Orin Nano

The **Jetson Orin Nano** is a small form-factor embedded computing module from NVIDIA designed for edge AI applications.

**Key hardware specs (8GB variant):**
- **CPU:** 6-core ARM Cortex-A78AE
- **GPU:** 1024-core NVIDIA Ampere GPU with 32 Tensor Cores
- **Memory:** 8GB LPDDR5 — shared between CPU and GPU (unified memory)
- **AI performance:** Up to 40 TOPS (Tera Operations Per Second)
- **Power envelope:** 7–15W (configurable)

**JetPack** is NVIDIA's SDK for Jetson devices. It bundles the OS (Ubuntu), CUDA, cuDNN, TensorRT, and other AI libraries. The version of JetPack determines which versions of CUDA and other libraries are available — this matters for compiling llama.cpp correctly.

**This project uses JetPack 7.2.** Relevant implications for the llama.cpp build:
- JetPack 7.2 ships with a specific CUDA toolkit version that must match the compute capability flags passed to llama.cpp's CMake build (`CMAKE_CUDA_ARCHITECTURES`)
- The Orin Nano's GPU has Ampere architecture compute capability 8.7 — this needs to be set explicitly since auto-detection can fail or default incorrectly on Jetson hardware
- JetPack 7.2 is a newer release; some pre-built binaries and community guides online target older JetPack versions (5.x/6.x), so build steps may need adaptation rather than direct copy-paste

---

## 9. Unified Memory Architecture

On a standard desktop or server, the CPU has its own RAM and the GPU has separate VRAM. The two pools don't overlap.

On the Jetson Orin Nano, **all 8GB of memory is shared** between the CPU and GPU. This is called unified memory or UMA (Unified Memory Architecture).

**Implications for this project:**
- There is no separate "GPU VRAM" limit — the GPU can use any of the 8GB
- But the OS, system processes, and any robotics stack also consume from this same 8GB pool
- Realistic available memory for the model at runtime: approximately 5–6GB
- This is why model size (and quantization level) is the most critical constraint

---

## 10. How the VLM Operates on the Jetson

At runtime, the pipeline looks like this:

```
Camera Frame
     │
     ▼
Vision Encoder (image → embeddings)
     │
     ▼
Language Model (embeddings + text prompt → token prediction)
     │
     ▼
Generated Text Output
```

**Step by step:**
1. A camera frame is captured (JPEG or raw)
2. The image is passed to the vision encoder component of the VLM, which produces a set of numerical embeddings representing the visual content
3. These embeddings are concatenated with the tokenized text prompt (e.g., "What do you see?")
4. The language model processes this combined input and generates tokens one by one
5. Tokens are decoded back into human-readable text

**GPU offloading:** llama.cpp allows specifying how many model layers to offload to the GPU (`--n-gpu-layers`). On the Jetson, setting this to a high value (all layers) significantly improves inference speed because the Ampere GPU's Tensor Cores can run matrix operations much faster than the ARM CPU.

---

## 11. Why Not TensorRT-LLM?

**TensorRT-LLM** is NVIDIA's optimized inference framework. On NVIDIA hardware it produces the fastest possible inference — it compiles models into highly optimized GPU kernels.

So why use llama.cpp instead?

| Factor | llama.cpp | TensorRT-LLM |
|--------|-----------|--------------|
| Build complexity on JetPack | Moderate | High |
| VLM support maturity | Good (Moondream native) | Limited/spotty |
| Debugging on-device | Straightforward | Difficult |
| Community / documentation | Extensive | Sparse for edge |
| Performance ceiling | Good | Best |

**The honest answer:** TensorRT-LLM is the right choice when you need to extract every last token/second from the hardware. For a 1.8B model on a Jetson, the raw inference speed with llama.cpp is already sufficient for most real-world applications. The engineering cost of getting TensorRT-LLM working with a VLM on JetPack is not justified at this stage. If performance becomes a bottleneck after the system is working, migrating to TensorRT-LLM is a valid optimization step.

---

## 12. Building llama.cpp with CUDA on JetPack 7.2

This section records what was learned during the actual build on JetPack 7.2 / CUDA 13.2.

### Key CMake Flags

```bash
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="87" \
    -DCMAKE_BUILD_TYPE=Release
```

- **`-DGGML_CUDA=ON`** — The correct flag for CUDA in modern llama.cpp (post mid-2024 backend refactor). Older guides say `-DLLAMA_CUBLAS=ON` — that flag is silently ignored on current llama.cpp and results in a CPU-only build with no warning.
- **`-DCMAKE_CUDA_ARCHITECTURES="87"`** — Must be set explicitly for Orin Nano (Ampere, compute capability 8.7). Without it, CMake tries to auto-detect by running a test binary on the host, which can fail or pick the wrong architecture on aarch64/Jetson.
- **`-DCMAKE_BUILD_TYPE=Release`** — Enables compiler optimizations. The Debug build runs noticeably slower on ARM.

### What LD_LIBRARY_PATH Does (and Doesn't Do)

`nvcc` is installed at `/usr/local/cuda-13.2/bin/nvcc` but not on PATH by default on JetPack 7.2. Fix:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

`PATH` lets the shell find `nvcc`. `LD_LIBRARY_PATH` lets compiled binaries find CUDA shared libraries at runtime. The `/usr/local/cuda` symlink points to `/usr/local/cuda-13.2` — using the symlink is better than hardcoding the version number.

### Validating GPU Offloading on Jetson

**`GR3D_FREQ` in `tegrastats` does NOT reflect CUDA compute activity.** It tracks the 3D graphics rasterizer (OpenGL/Vulkan), which stays at 0% during pure CUDA workloads. This is a JetPack 7.2 behavior, not a sign of failure.

To actually confirm GPU offloading, use `--verbose` when running llama-cli:
```
D load_tensors: layer 0 assigned to device CUDA0 ...
I load_tensors: offloaded 25/25 layers to GPU   ← this is the confirmation
```

### Unified Memory Buffer Sizes

On Jetson's unified memory architecture, `llama.cpp` reports:
```
CPU model buffer size  =  0.00 MiB
CUDA0 model buffer size = 0.00 MiB
```

This is **correct**. On a discrete GPU, llama.cpp allocates separate CPU and GPU buffers and copies data between them. On Jetson, the CPU and GPU share the same physical RAM — there's no copy, no separate allocation, so both buffer sizes report as 0. The model layers ARE on the GPU; they just live in unified memory that both can access directly.

### The mmproj File

For VLMs (Vision-Language Models) in llama.cpp, two files are always required:

1. **Main model GGUF** — the language model weights
2. **mmproj GGUF** — the multimodal projector, which maps image embeddings into the language model's embedding space

The mmproj is always kept at F16 (full precision), even if the main model is quantized. The projection between two different embedding spaces is sensitive to precision — quantizing it degrades image understanding quality noticeably.

When the mmproj is loaded, `llama-cli` reports `modalities: text, vision`. Without it: `modalities: text` only.

### Quantizing a Model Locally

When the official GGUF repo only publishes F16 (as was the case for Moondream2), you can quantize it yourself using `llama-quantize`:

```bash
./build/bin/llama-quantize \
    input-f16.gguf \
    output-Q4_K_M.gguf \
    Q4_K_M
```

This runs on CPU. For a 1.8B model it takes a few minutes on the Orin Nano's 6-core ARM CPU.

---

## 13. Using llama-cpp-python on JetPack 7.2

This section records what was learned during the actual Phase 5 installation and integration test on JetPack 7.2.

### Installation

A default install pulls a pre-built wheel with **no CUDA support**. On Jetson, always build from source with CUDA flags. The project uses [uv](https://docs.astral.sh/uv/) rather than pip to manage the venv and dependencies (see `pyproject.toml` at the repo root); uv still invokes the same CMake-based source build, so the same environment variables apply — they just need to be set before `uv sync` instead of `pip install`:

```bash
# From the repo root — uv creates/updates .venv itself, no manual venv step needed
PATH=/usr/local/cuda/bin:$PATH \
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87" \
uv sync
```

The resulting build is ~265MB (vs. ~4MB for the CPU-only build) because it contains compiled CUDA kernels. The large size is a useful sanity check that CUDA was actually compiled in.

Build time on the Orin Nano is similar to the llama.cpp build — expect 5–10 minutes. This is normal; it's compiling C++ and CUDA code.

### Project Environment Setup

uv manages a single dedicated venv (`.venv/` at the repo root) from `pyproject.toml` — never the system Python or another project's venv:

```bash
PATH=/usr/local/cuda/bin:$PATH CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87" uv sync
```

To run a script inside the managed environment: `uv run python your_script.py` — no need to `source activate`.

### Loading the Model in Python

```python
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

chat_handler = MoondreamChatHandler(clip_model_path="models/moondream2-mmproj-f16.gguf")

llm = Llama(
    model_path="models/moondream2-text-model-Q4_K_M.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,
    n_gpu_layers=-1,   # -1 = offload all layers to GPU
    verbose=False,     # True floods stdout with CUDA Graph messages
)
```

**`MoondreamChatHandler`** is the specific handler for Moondream2's multimodal format. Using the wrong handler (or no handler) with an image prompt will produce incorrect output or an error.

**`n_gpu_layers=-1`** means "offload everything." You can set a specific number (e.g., `25`) for Moondream2, but `-1` is simpler and future-proof if the layer count ever changes.

### Running Inference

**Text-only:**
```python
response = llm.create_chat_completion(
    messages=[{"role": "user", "content": "Describe what you see."}]
)
print(response["choices"][0]["message"]["content"])
```

**Image + text (for the camera pipeline):**
```python
import base64

with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = llm.create_chat_completion(
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            {"type": "text", "text": "What do you see?"},
        ],
    }]
)
print(response["choices"][0]["message"]["content"])
```

The image is passed as a **base64-encoded data URI**, not a file path. This is part of the OpenAI-compatible chat format that llama-cpp-python uses. The MIME type in the URL (`image/jpeg`, `image/png`) should match the actual image format.

### Measuring Speed Accurately

**Do not measure t/s by dividing tokens by wall-clock time on short responses.** For 1–14 tokens, CUDA graph compilation and first-token latency dominate the wall-clock time and make the number look far too low.

The accurate speed comes from llama's internal profiler in verbose mode:
```
llama_perf_context_print: eval time = 630.55 ms / 14 runs → 22.20 tokens/second
```

For production measurement without `verbose=True`, measure over a response of 50+ tokens so that per-token latency dominates rather than startup overhead.

**Confirmed Phase 5 speed: 22.20 t/s** — identical to Phase 4 CLI baseline (~19–21 t/s). Python is not adding measurable overhead.

### CUDA Graph Warmup

When `verbose=True`, you will see many lines like:
```
ggml_backend_cuda_graph_compute: CUDA graph warmup complete
CUDA Graph id 37 reused
```

This is normal and expected. CUDA graphs are a CUDA optimization where the GPU pre-records a sequence of operations and replays them without CPU intervention. The first time a computation graph runs, it goes through "warmup" (compilation/recording). Subsequent calls with the same graph structure reuse it directly, which is why generation gets faster after the first token. This is a good sign — it means the CUDA acceleration is working as intended.

---

## 14. Camera Capture on Linux with OpenCV

### V4L2 and /dev/video* Nodes

On Linux, cameras are exposed through the **V4L2** (Video4Linux2) kernel subsystem. Each camera device appears as one or more files under `/dev/video*`. The number of nodes depends on the camera:

- A basic USB webcam typically registers **one** node (`/dev/video0`)
- Higher-end cameras like the **Logitech BRIO** register **multiple** nodes (e.g., `/dev/video0` through `/dev/video3`) — each representing a different capture format or stream type the camera hardware supports (MJPEG, YUY2, H.264, metadata, etc.)

This is not a problem or a sign of duplicates. Device 0 is always the primary capture device. To identify what each node is, you can read `/sys/class/video4linux/video0/name` — all four BRIO nodes will show the same camera name.

### OpenCV VideoCapture

**OpenCV** is a widely-used computer vision library. For this project, it serves one purpose: reading frames from the USB camera. The relevant class is `cv2.VideoCapture`.

```python
import cv2

cap = cv2.VideoCapture(0)          # open device /dev/video0
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

ok, frame = cap.read()             # frame is a NumPy array: (H, W, 3) BGR
cap.release()
```

**Important:** OpenCV reads frames in **BGR** color order (Blue-Green-Red), not the more common RGB. For displaying with matplotlib or passing to most ML models you'd need to convert (`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`). In this project we encode to JPEG immediately and let the vision encoder handle it — JPEG encoding preserves the correct colors regardless of channel order, so no conversion is needed.

### Why opencv-python-headless

The standard `opencv-python` pip package includes GUI components (`cv2.imshow`, window management) that depend on display libraries. On a headless Jetson with no monitor attached, those libraries may not be present and the package may fail to install or import.

`opencv-python-headless` is the same library minus the GUI components. Since we only need frame capture and JPEG encoding — not display — headless is the correct choice and avoids the dependency issue entirely.

### Encoding Frames for the Model

The vision encoder inside Moondream2 expects image data, not raw NumPy arrays. The pipeline converts each frame to JPEG bytes, then base64-encodes those bytes into a data URI:

```python
import cv2, base64

ok, frame = cap.read()
_, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
jpeg_bytes = buf.tobytes()
img_b64 = base64.b64encode(jpeg_bytes).decode()
data_uri = f"data:image/jpeg;base64,{img_b64}"
# → passed to llm.create_chat_completion() as image_url
```

JPEG quality 90 is a good balance — it preserves enough detail for the vision encoder without excessive size. Very low quality (below ~70) can introduce JPEG artifacts that degrade model accuracy.

---

## 15. Prompt Engineering for VLMs

How you phrase a prompt has a significant effect on the quality and reliability of a VLM's output. This section captures what was learned during Phase 8 evaluation.

### Hallucination and Why It Happens

VLMs — especially smaller quantized ones — can **hallucinate**: generate confident-sounding output that does not match what is actually in the image. This is not a bug or a failure of inference; it is a fundamental property of how language models work. They predict the most statistically likely next token given their training data. When the visual signal is ambiguous (blurry text, small print, low contrast), the model fills in the gap with something plausible rather than saying it doesn't know.

**Text reading is the most hallucination-prone task** for small VLMs because:
1. Text in images is often small or partially obscured
2. The model has seen enormous amounts of text in training and will pattern-match to something familiar
3. Unlike object detection (where "wrong object" is obvious), hallucinated text can sound completely believable

In Phase 8 evaluation, Moondream2 returned `"I'm not a robot, I'm a human."` for a scene that did not contain that text — a textbook hallucination.

### The Abstention Technique

The most effective way to reduce hallucination on uncertain tasks is to **give the model an explicit exit ramp** — a clearly worded option to say it doesn't know, rather than forcing it to produce an answer.

**Without exit ramp (hallucination-prone):**
> "Is there any text visible? If so, what does it say?"

The model interprets this as: "produce text if there is any." When uncertain, it invents.

**With exit ramp:**
> "Is there any text clearly visible in this image that you can read with confidence? If you are not certain, say 'No text clearly visible.'"

The phrase "with confidence" raises the model's internal threshold for committing to an answer. The explicit fallback phrase ("No text clearly visible.") gives it a low-cost response that doesn't require fabrication.

This technique applies broadly — not just to text reading. Any task where the model might not have enough visual signal benefits from an explicit uncertainty option.

### General Prompt Guidelines for This Project

| Goal | Guidance |
|---|---|
| Concise output | Add "in one or two sentences" — models default to longer responses without a length cue |
| Reduce hallucination | Provide an explicit abstention phrase for uncertain tasks |
| Specific attributes | Ask for one thing at a time — "What color is the main object?" outperforms "Describe the object" when you only need color |
| Counting | "How many X are in this image?" works well; Moondream2 handles small counts (1–5) reliably |
| Scene context | Open-ended scene description often surfaces contextual inferences (e.g. "office being rearranged") that targeted questions miss |

### Domain-Oriented Prompting for Small VLMs

When deploying a VLM for a specific use case (e.g. transportation safety, industrial inspection), you can improve output relevance by adding a **domain anchor** to the prompt — a word or phrase that tells the model the context it's operating in.

**The key balance:** domain-specific enough to steer output, but not so narrow that the model fails when the scene doesn't perfectly match.

```
Too generic:   "Describe what you see."
Too specific:  "Describe this road vehicle and note any OSHA 1910.178 violations."
Right level:   "Describe this scene for an inspection report. Identify the environment,
                main subjects and their condition, and any safety concerns."
```

For a 1.8B quantized model, the effective prompt structure is:
1. **One domain anchor** — "inspection report", "safety assessment", "maintenance record"
2. **One structural hint** — what categories of things to observe
3. **Length cue** — "two to three sentences" or "list format"
4. **Abstention option** where hallucination risk is high

More than this (multiple paragraphs, detailed checklists, domain jargon like regulation codes) tends to be ignored or partially followed. Small models respond to simplicity and clarity, not depth of instruction.

### Prompt Labels as Stable Identifiers

In the evaluation system, each prompt has both a **label** (e.g. `"Scene Description"`) and a **prompt text** (the actual instruction). These must be treated differently:

- **Prompt text** is just a string — it can be changed freely to improve output quality
- **Prompt label** is an identifier — it becomes the frame filename (`Scene Description.jpg`), a JSON key in `results.json`, and a display heading in the ReportViewer

If you rename a label, all existing reports are broken: the `/eval/frame/` endpoint can no longer find the file (the name changed), and comparison logic between old and new runs fails (the key doesn't match). Changing a label is effectively a data migration.

**Rule:** change prompt text freely, change labels only intentionally and only when you're prepared to either migrate old data or accept that old reports will show broken frames.

### When to Annotate Frames vs. When Not To

The `annotate_frame()` function overlays the prompt and response as a caption bar on a JPEG. There are two distinct use cases and the right answer differs:

| Context | Annotate? | Reason |
|---|---|---|
| Standalone file (runs dir, snapshots) | ✅ Yes | The file may be viewed outside the app — the annotation is the only context available |
| UI-integrated report (eval frames) | ❌ No | The ReportViewer card already shows the prompt and response as text above the image; the annotation is redundant and obscures the visual content being evaluated |

When in doubt: annotate if the file will ever be viewed without the surrounding UI. Skip annotation if the UI always provides the context.

---

## 16. PyQt5 Desktop Applications on ARM/Jetson

PyQt5 is a Python binding for the Qt5 GUI framework. It was the choice for `app.py` because it's pre-installed on JetPack 7.2's system Python and works natively on the Jetson's aarch64 ARM CPU without any cross-compilation or browser runtime.

### Threading Architecture

Qt has one rule that cannot be broken: **all UI updates must happen on the main thread**. You cannot call `label.setPixmap()` or `browser.append()` from a background thread — doing so causes undefined behavior or crashes.

The correct pattern is to use `QThread` for background work and `pyqtSignal` to pass results back to the main thread:

```python
class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage)  # signal carries the result

    def run(self):  # runs on the background thread
        ok, frame = self.cap.read()
        qi = QImage(...)
        self.frame_ready.emit(qi)  # posts to main thread's event queue

# In the main thread / widget setup:
self.video_thread.frame_ready.connect(self.update_frame)  # slot runs on main thread
```

When a signal is emitted from a `QThread` and the connected slot belongs to an object in a different thread, Qt automatically uses a **queued connection** — the result is posted to the receiver's event queue and processed safely on the correct thread. You don't have to manage this manually.

This project uses four threads:
- **Main (Qt)** — UI rendering and event handling only
- **`VideoThread`** — reads camera at 30 FPS, emits `QImage` signals
- **`ModelLoader`** — loads llama.cpp model at startup, emits when done
- **`InferenceWorker`** — runs one inference call, emits `(text, tokens, elapsed)`
- **`EvalWorker`** — orchestrates a full evaluation run, emits per-prompt results

### QImage and NumPy Array Lifetime

This is the most important PyQt5 gotcha discovered in this project.

`QImage` has a constructor that takes a raw memory pointer: `QImage(data, width, height, bytes_per_line, format)`. In Python, `data` can be a `bytes` object, a `bytearray`, or a numpy memoryview. **Qt does NOT copy this data** — it stores a raw C pointer. The caller is responsible for keeping the data alive for as long as the `QImage` exists.

The dangerous pattern:
```python
qi = QImage(rgb.data.tobytes(), w, h, ch * w, QImage.Format_RGB888)
self.frame_ready.emit(qi.copy())  # CRASH at high frame rates
```

`rgb.data.tobytes()` creates a temporary `bytes` object. Python's reference counting frees it immediately after the `QImage` constructor returns, because nothing else holds a reference. `qi` now contains a dangling C pointer. On the next line, `.copy()` reads from freed heap memory — glibc's allocator detects the corruption and raises `SIGABRT`.

At 10 FPS this usually "works" because the freed memory hasn't been reused yet. At 30 FPS the higher allocation rate means the freed buffer gets recycled before `.copy()` runs, causing consistent crashes.

The correct pattern:
```python
qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
self.frame_ready.emit(qi)
```

`rgb.data` is a numpy memoryview backed by the `rgb` array, which is a named variable still in scope. The buffer is guaranteed alive until `.copy()` completes and produces a standalone `QImage` with its own memory allocation.

### ARM-Specific Crash: QGraphicsDropShadowEffect

`QGraphicsDropShadowEffect` works fine on desktop Linux/macOS, but on ARM/Qt5 it triggers a recursive repaint loop: the shadow effect invalidates its parent widget's region, which triggers another paint, which triggers another shadow recomposition, looping until the call stack overflows with `paintSiblingsRecursive → drawWidget` frames → `SIGABRT`.

This is a known Qt5 bug on embedded ARM targets. **Do not use `QGraphicsDropShadowEffect` on Jetson.** Simulate depth with gradient `QFrame` backgrounds and subtle borders instead — the dark navy aesthetic achieves the same visual layering without effects.

### Custom Tab Bars

When `QTabWidget.setTabPosition(QTabWidget.West)` is set, Qt rotates the tab text 90° to fit vertical tabs. To keep text horizontal (more readable), subclass `QTabBar` and override `paintEvent`:

```python
class LeftTabBar(QTabBar):
    _W, _H = 150, 56

    def tabSizeHint(self, index):
        return QSize(self._W, self._H)

    def minimumTabSizeHint(self, index):
        return QSize(self._W, self._H)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        opt = QStyleOptionTab()
        for i in range(self.count()):
            self.initStyleOption(opt, i)
            rect     = opt.rect
            selected = bool(opt.state & QStyle.State_Selected)
            hovered  = bool(opt.state & QStyle.State_MouseOver)
            # fill background, draw accent line, draw text...
            painter.drawText(rect, Qt.AlignCenter, self.tabText(i).strip())
        painter.end()  # always call explicitly — don't rely on GC timing on ARM
```

`initStyleOption(opt, i)` fills a `QStyleOptionTab` with the tab's current state — selected, hovered, disabled — so you can branch your drawing logic accordingly. `tabSizeHint` and `minimumTabSizeHint` must both be overridden; Qt uses the minimum hint for layout calculations and will ignore your size hint if the minimum is smaller.

### Stylesheets (QSS)

PyQt5 uses Qt Style Sheets (QSS), which look like CSS but have Qt-specific extensions.

**Gradients** use the `qlineargradient()` syntax:
```python
background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f0f23, stop:1 #1a1a35);
```
This syntax has colons and parentheses in it, which interact badly with Python string formatting (`%` substitution). Always store the gradient string in a variable and interpolate it carefully.

**Broad selectors are dangerous.** The rule `QWidget > QFrame { background: transparent; }` matches every `QFrame` that is a direct child of any `QWidget` — including Qt's own internal frame structures inside `QTabWidget`, `QSplitter`, and `QScrollArea`. Applying it at the application stylesheet level can break internal widget painting. Target your rules as specifically as possible: `QFrame#myCard { ... }` (using object names) or set stylesheets directly on the widget with `widget.setStyleSheet(...)`.

### Thread Safety: The LLM Is Not Reentrant

`llama-cpp-python`'s `Llama` object is not thread-safe. Only one thread should call `create_chat_completion()` at a time. In an app with multiple inference paths (e.g. a live Q&A thread and a background evaluation worker), both paths must be mutually excluded. The simplest correct approach is to disable all user-facing inference triggers while any inference is running — do not rely on "the user probably won't do that."

The pattern used in this project:
- `run_btn` (starts eval) is disabled while the model loads and re-enabled only on `_on_model_ready`.
- When `_run_eval()` starts, the Live tab input is also disabled to prevent the user from triggering a second inference path.
- Both are re-enabled together in `_on_eval_done()`.

### Safe App Shutdown with Background Threads

When the Qt window closes, `closeEvent` runs before the process exits. Any `QThread` that holds references to the camera, model, or UI objects must be cleaned up before those objects are released. The correct shutdown order is:

1. **Disconnect signals** from threads that can't be interrupted (e.g. `ModelLoader` which is blocked in C++ for ~10s). Disconnecting the `finished` signal prevents a late callback from accessing an already-destroyed window.
2. **Unblock waiting threads** before releasing shared resources — if `EvalWorker` is blocked on `threading.Event.wait()`, call `set_captured_frame(b"")` to unblock it, then `wait(timeout)` to let it exit cleanly.
3. **Stop the camera feed thread** with a cooperative flag (`running = False`) and `wait()`.
4. **Release the camera** only after all threads that use it have stopped.

---

## 17. FastAPI + MJPEG Streaming on Jetson

### Why FastAPI Instead of PyQt5

PyQt5 works well for a standalone desktop app on the Jetson itself. But for a robotics or inspection use case, the Jetson is often mounted on hardware with no display — you need to control it remotely. A web interface solves this: the server runs on the Jetson, and any browser on the same network becomes the UI.

FastAPI is a modern Python web framework built on top of Starlette (ASGI). It is async-first, type-annotated, and straightforward to deploy with `uvicorn`.

### MJPEG Streaming

MJPEG (Motion JPEG) is the simplest way to stream a live camera feed over HTTP. Each frame is a standalone JPEG image; the server sends them in a continuous multipart HTTP response.

```
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg

<JPEG bytes>
--frame
Content-Type: image/jpeg

<JPEG bytes>
...
```

The browser receives this as a single HTTP response that never ends. An `<img src="/stream">` tag in the browser handles it natively — no JavaScript video player, no WebSocket, no buffering logic needed. This simplicity is why MJPEG is the right choice for LAN-only streaming where bandwidth is not a concern.

In FastAPI, MJPEG is implemented with a `StreamingResponse` and a generator:

```python
def _mjpeg_generator():
    while True:
        jpeg = camera_thread.get_latest_jpeg()
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1 / 30)

@router.get("/stream")
def stream():
    return StreamingResponse(_mjpeg_generator(),
                             media_type="multipart/x-mixed-replace; boundary=frame")
```

### CameraThread Without Qt

The PyQt5 app used `VideoThread(QThread)` to read camera frames in the background. In the web architecture, there is no Qt event loop — so `QThread` is not available. The replacement is a plain `threading.Thread`:

```python
class CameraThread(threading.Thread):
    def __init__(self, cap: cv2.VideoCapture):
        super().__init__(daemon=True)
        self.cap = cap
        self.running = True
        self._jpeg: bytes | None = None
        self._lock = threading.Lock()

    def run(self):
        while self.running:
            ok, frame = self.cap.read()
            # cap.read() blocks until the camera delivers a frame.
            # Do NOT add time.sleep() here — it would halve effective FPS.
            if ok:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with self._lock:
                    self._jpeg = buf.tobytes()
```

The key insight: **`cap.read()` is a blocking call**. It waits until the camera hardware delivers the next frame — which happens at the camera's hardware FPS (30fps at 640×480 for the Logitech BRIO). Adding a `time.sleep(1/30)` after `cap.read()` does not limit to 30fps — it adds an extra 33ms on top of the natural 33ms `cap.read()` wait, effectively halving FPS to ~15fps. This was the bug that produced 10fps in early testing.

### Inference Serialization (Threading Lock)

`llama-cpp-python`'s `Llama` object is not thread-safe. In the web architecture, multiple HTTP requests can arrive concurrently (user clicks Analyze, auto-scan fires simultaneously). Both would try to call `llm.create_chat_completion()` at the same time — undefined behavior or a crash.

The solution is a `threading.Lock` with non-blocking acquire:

```python
_infer_lock = threading.Lock()

def _run_inference_locked(prompt: str) -> dict:
    if not _infer_lock.acquire(blocking=False):
        raise HTTPException(409, "Another inference is already running.")
    try:
        # ... run inference ...
    finally:
        _infer_lock.release()
```

`blocking=False` means: if the lock is already held, return `False` immediately instead of waiting. The endpoint returns HTTP 409 (Conflict) so the client can display a "busy" state rather than hanging.

### FastAPI Lifespan for Startup/Shutdown

Resources like the camera and model need to be initialized before the server starts accepting requests, and cleaned up when it shuts down. FastAPI provides a `lifespan` context manager for this:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cap = open_camera(0)
    camera = CameraThread(cap)
    camera.start()
    llm = await asyncio.to_thread(load_model)  # runs in thread, doesn't block event loop
    yield  # server runs here
    # Shutdown
    camera.stop()
    cap.release()

app = FastAPI(lifespan=lifespan)
```

`asyncio.to_thread(load_model)` is critical — `load_model()` takes ~10–33s on the Jetson (loading 877MB from disk into memory). Running it directly in an `async` function would block the entire event loop, preventing any HTTP requests from being answered during model load. Running it in a thread pool via `asyncio.to_thread` keeps the loop responsive.

### Serving the React Frontend from FastAPI

In production, the React app is built into static files (`frontend/dist/`) and served directly by the FastAPI process. This means only one port (8000) needs to be open on the Jetson — no separate web server.

```python
# Serve static assets (hashed filenames, long cache)
app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

# SPA fallback — serve index.html for all unmatched GET routes
@app.get("/{full_path:path}", include_in_schema=False)
async def spa_fallback(full_path: str):
    return FileResponse("frontend/dist/index.html")
```

**Route order matters.** The API router must be included (`app.include_router(router)`) before the catch-all `/{full_path:path}` is defined. FastAPI checks routes in registration order; if the catch-all were first, every API request would return `index.html`.

---

## 18. React Frontend Architecture for Edge AI

### Why React + Vite

React is a component-based JavaScript UI framework. Vite is its build tool — it handles TypeScript compilation, bundling, hot-reload during development, and produces an optimized `dist/` folder for production. This combination is the current standard for building browser-based UIs.

For this project, React's component model maps cleanly onto the UI structure: `CameraFeed`, `ButtonPanel`, `ResultPanel`, and `StatusBar` are all independent components that the `Dashboard` page composes.

### MJPEG in React — It's Just an `<img>` Tag

This is the most important simplification in the frontend:

```tsx
<img src="http://JETSON_IP:8000/stream" alt="Live camera feed" />
```

The browser handles the MJPEG protocol natively. No JavaScript streaming code, no WebSocket, no `canvas` element. The `src` can be a relative URL (`/stream`) when the frontend is served from the same server as the API.

### Polling for State Updates

The Jetson's model runs inference every ~10–15 seconds during auto-scan. The frontend needs to know when a new result is ready. Two options: WebSocket push (complex) or HTTP polling (simple). Polling wins here because:

- The update interval (10–15s) is slow enough that polling every 1.5–3s adds negligible overhead
- No persistent connection management needed
- Works reliably across different browsers and network conditions

```typescript
// useStatus.ts — polls GET /status every 1.5-3 seconds
useEffect(() => {
  const id = setInterval(async () => {
    const s = await api.getStatus();
    setStatus(s);
  }, intervalMs);
  return () => clearInterval(id);
}, [intervalMs]);
```

The `/status` endpoint returns `last_result` with a timestamp. The frontend tracks the last seen timestamp in a `useRef` (not state, to avoid re-render loops) and appends to the result history whenever a new timestamp appears:

```typescript
const lastSeenTimestamp = useRef<string | null>(null);

useEffect(() => {
  const lr = status?.last_result;
  if (!lr || lr.timestamp === lastSeenTimestamp.current) return;
  lastSeenTimestamp.current = lr.timestamp;
  setResults(prev => [...prev, { ...lr, source: "Auto-Scan" }]);
}, [status?.last_result?.timestamp]);
```

Manual button clicks (Analyze/Inspect) update `lastSeenTimestamp.current` to their result timestamp immediately — so the status poll effect won't re-add the same result as an "Auto-Scan" entry.

### Result History Pattern

Rather than showing only the latest inference result (which gets replaced on each new inference), the UI maintains a growing list of `ResultEntry` objects:

```typescript
interface ResultEntry {
  text: string;
  tokens: number;
  elapsed_s: number;
  timestamp: string;
  source: "Analyze" | "Inspect" | "Auto-Scan";
}
```

Each entry knows its source, so the UI can display colored badges. The list auto-scrolls to the bottom (`bottomRef.current?.scrollIntoView()`) whenever a new entry is added.

### Viewport-Contained Layout

A common pitfall in browser UIs for embedded systems: the page scrolls vertically because the camera feed pushes content below the visible area. The fix is to lock the layout to `100vh` using CSS flexbox with `min-height: 0` at every level of the flex tree:

```css
body { height: 100vh; overflow: hidden; }
.dashboard { height: 100vh; overflow: hidden; display: flex; flex-direction: column; }
.dashboard-main { flex: 1; min-height: 0; display: flex; }
.left-col  { flex: 3; min-height: 0; display: flex; flex-direction: column; }
.camera-card { flex: 1; min-height: 0; display: flex; flex-direction: column; }
.camera-img  { flex: 1; min-height: 0; object-fit: cover; }
```

`min-height: 0` is the critical rule. Without it, a flex child's minimum height defaults to the size of its content — so the camera image (640×480 native) will refuse to shrink below 480px even when the container is smaller. With `min-height: 0`, the browser is allowed to shrink it to fit.

---

## 19. Web Evaluation: Background Threads, Stats, and Security

### The Background Thread + Progress Polling Pattern

The 5-prompt evaluation suite takes ~25 seconds to complete. Running it inside a FastAPI route would block the HTTP response for 25 seconds — the browser would just wait with no feedback, and the connection might time out.

The correct pattern is to start a daemon thread and return immediately, then let the client poll for progress:

```python
# Route returns immediately
@eval_router.post("/run")
def eval_run():
    threading.Thread(target=_run_eval, daemon=True).start()
    return {"started": True}

# Background thread updates shared state
_eval_state = {"running": True, "progress": 0, "current_label": None, ...}

def _run_eval():
    for i, (label, prompt) in enumerate(PROMPTS):
        _eval_state["current_label"] = label
        _eval_state["progress"] = i
        # ... run inference ...
        _eval_state["results"].append(result)
        _eval_state["progress"] = i + 1
    _eval_state["running"] = False  # always last — signals completion

# Poll endpoint
@eval_router.get("/status")
def eval_status():
    return dict(_eval_state)  # dict() copies it — avoids mutation mid-serialization
```

The frontend uses a self-adjusting poll: fast (1.5s) while `running === true`, slow (5s) when idle. This avoids constant polling overhead without needing WebSocket push:

```typescript
async function poll() {
  const s = await api.evalStatus();
  setStatus(s);
  timer = setTimeout(poll, s.running ? 1500 : 5000);  // self-schedule
}
```

Note that `_eval_state["running"] = False` is set **after** all file writes complete. This guarantees that when the frontend detects completion and immediately fetches the report via `/eval/report/{id}`, all files (`results.json`, `report.md`, frame JPEGs) already exist.

### Using the Inference Lock in an Eval Thread

The evaluation thread uses `_infer_lock.acquire(timeout=15)` instead of `blocking=False`. This is intentional:

- `blocking=False` (used by `/analyze` and `/inspect`) returns immediately if busy — appropriate for user-facing buttons where a "busy" error is acceptable
- `timeout=15` (used by eval) waits up to 15 seconds — appropriate for a background thread that is not blocking any HTTP response; it can afford to wait out an in-flight auto-scan tick (~3–5s) rather than immediately failing

The backend also refuses to start an eval if auto-scan is active (`_state.get("autoscan")`). Auto-scan fires every 10 seconds and holds the lock for ~3–5s each time — allowing concurrent auto-scan during a 25s eval creates confusing interleaving. Refusing to start is cleaner than trying to coordinate them.

### Computing Stats and Comparison Data

Stats for a single run are computed from the `results` list once the run completes:

```python
def _compute_stats(results):
    latencies = [r["latency_s"] for r in results]
    fastest = min(results, key=lambda r: r["latency_s"])
    slowest = max(results, key=lambda r: r["latency_s"])
    return {
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
        "total_tokens":  sum(r["tokens"] for r in results),
        "fastest": {"label": fastest["label"], "latency_s": fastest["latency_s"]},
        "slowest": {"label": slowest["label"], "latency_s": slowest["latency_s"]},
    }
```

Comparison against the baseline computes per-prompt deltas. A negative `latency_delta` means the current run was faster (took less time) than the baseline:

```python
def _compute_comparison(current, previous):
    prev_map = {r["label"]: r for r in previous}
    avg_delta = round(avg_curr - avg_prev, 2)
    per_prompt = {
        r["label"]: {
            "latency_delta": round(r["latency_s"] - prev_map[r["label"]]["latency_s"], 2),
            "token_delta":   r["tokens"] - prev_map[r["label"]]["tokens"],
        }
        for r in current if r["label"] in prev_map
    }
    return {
        "avg_latency_delta": avg_delta,
        "direction": "faster" if avg_delta < 0 else "slower" if avg_delta > 0 else "same",
        "per_prompt": per_prompt,
    }
```

The comparison is only computed if the current report is **not** the baseline itself (comparing a run against itself is meaningless). The baseline is not automatically updated on each run — the user explicitly calls `POST /eval/set-baseline/{report_id}` to promote a run.

### CSS Bar Chart with Ghost Baseline

The latency chart is built entirely in CSS — no charting library needed. Each bar's width is computed as a percentage of the maximum latency in the run:

```tsx
const maxLatency = Math.max(...results.map(r => r.latency_s), 1);

// For each prompt:
<div className="latency-bar-track">
  {/* Ghost bar showing baseline */}
  {baselineLatency != null && (
    <div className="latency-bar-baseline"
         style={{ width: `${(baselineLatency / maxLatency) * 100}%` }} />
  )}
  {/* Current run bar — colored by direction */}
  <div className={`latency-bar-fill ${cmp?.latency_delta > 0 ? "bar-slower" : cmp?.latency_delta < 0 ? "bar-faster" : ""}`}
       style={{ width: `${(r.latency_s / maxLatency) * 100}%` }} />
</div>
```

`latency-bar-baseline` is rendered first (behind the current bar) with a faint color. Both divs use `position: absolute` inside the track. The current bar overlaps the ghost bar — if the current run was faster, the blue bar is shorter than the gray ghost, making the difference visible. If slower, the blue bar extends past it.

The `Math.max(..., 1)` guard prevents division by zero if all latencies are somehow 0.

### Path Security for Dynamic File Serving

When a route serves files from the filesystem using a URL parameter, **any user can craft a path**. Without validation, a URL like `/eval/frame/../../../backend/app/core/config.py` could read arbitrary files.

Two defenses are applied:

1. **Validate the `report_id` format before using it in `os.path.join`:**
   ```python
   if not re.match(r"^report_\d{8}_\d{6}$", report_id):
       raise HTTPException(400, "Invalid report ID format.")
   ```
   This regex only passes strings like `report_20260624_153045`. No dots, slashes, or wildcards can pass.

2. **Validate the filename before using it:**
   ```python
   if not re.match(r"^[\w\s]+\.jpg$", filename):
       raise HTTPException(400, "Invalid filename.")
   ```
   `\w` matches `[a-zA-Z0-9_]` and `\s` matches whitespace — the actual label names like `Scene Description.jpg` match, but `../config.py` does not (dots are not `\w` or `\s`).

### URL-Encoding Filenames with Spaces

The eval frame filenames contain spaces: `Scene Description.jpg`, `Object List.jpg`, etc. Spaces in URLs are technically invalid — browsers encode them as `%20`.

The server pre-encodes frame URLs in the JSON response using `urllib.parse.quote()`:

```python
from urllib.parse import quote
r["frame_url"] = f"/eval/frame/{report_id}/{quote(r['label'] + '.jpg')}"
# → "/eval/frame/report_20260624_153045/Scene%20Description.jpg"
```

FastAPI's path parameter handling **automatically URL-decodes** the `filename` parameter, so the route receives `"Scene Description.jpg"` (with a real space), which matches `os.path.join()` correctly.

On the frontend, when constructing a frame URL from a label name (e.g., in `EvalRunner` where `frame_url` is not yet available from the backend), the client must encode manually:

```typescript
const url = `/eval/frame/${reportId}/${encodeURIComponent(label + ".jpg")}`;
// → "/eval/frame/report_20260624_153045/Scene%20Description.jpg"
```

`encodeURIComponent()` and `urllib.parse.quote()` both produce `%20` for spaces, so they are compatible.

### The `key` Prop for Forced Component Remount

React normally reuses component instances when re-rendering the same component type at the same position in the tree. When you need to force a component to fully unmount and remount — discarding all its state and re-running its `useEffect` hooks — change the `key` prop:

```tsx
const [reportListKey, setReportListKey] = useState(0);

// After a new eval run completes:
setReportListKey(k => k + 1);  // ReportList unmounts and remounts, re-fetching the list

<ReportList key={reportListKey} ... />
```

This is simpler than threading a "refresh trigger" through props or lifting the fetch logic into the parent. Use it when: (1) the component manages its own data fetching, (2) you need to invalidate that fetch without changing the component's API, and (3) remounting is cheap (no expensive DOM setup).

---

## 20. Web Authentication: JWT, SQLite, and React Auth Patterns

### What Is JWT?

A **JSON Web Token (JWT)** is a compact, self-contained way to represent a user's identity and permissions as a signed string. Instead of storing session data on the server, the token itself carries the information.

A JWT has three parts separated by dots: `header.payload.signature`

- **Header** — algorithm info (e.g. `{"alg":"HS256","typ":"JWT"}`)
- **Payload** — the claims: who the user is, what their permissions are, when the token expires
- **Signature** — a cryptographic hash of the header and payload using a secret key; proves the token hasn't been tampered with

```python
import jwt

# Creating a token
payload = {
    "sub": "1",          # subject — the user ID (must be a string in PyJWT 2.x)
    "username": "admin",
    "is_admin": True,
    "exp": datetime.now(timezone.utc) + timedelta(hours=24),  # expiry
}
token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")

# Validating a token
payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
# Raises jwt.ExpiredSignatureError if expired, jwt.InvalidTokenError if tampered
```

**Key point:** The server never stores the token. It only needs the secret key to validate one. This means auth is stateless — any server instance with the same key can validate any token.

### The `sub` Claim Must Be a String

The JWT specification defines `sub` (subject) as a string identifier for the user. PyJWT 2.x enforces this at decode time: if `sub` is an integer (common when storing database IDs directly), `jwt.decode()` raises `InvalidTokenError: Subject must be a string` — even if the token was just created successfully.

```python
# WRONG — int from SQLite, encodes fine but fails on decode
"sub": user["id"]       # e.g. 1

# CORRECT
"sub": str(user["id"])  # e.g. "1"
```

When reading `sub` back from the decoded payload, convert to int for database queries:
```python
find_user_by_id(int(user["sub"]))
```

This was the root cause of the Phase 14 auth bug where every protected route returned 401 immediately after login.

### Bearer Token Authentication

The standard way to send a JWT with an HTTP request is in the `Authorization` header:

```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

FastAPI's `HTTPBearer` dependency extracts the token from this header automatically:

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

_security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_security)) -> dict:
    return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
```

`credentials.credentials` is the raw token string (without "Bearer "). Applying this dependency at the **router level** protects every route in the router without adding it to each route individually:

```python
router = APIRouter(dependencies=[Depends(get_current_user)])
```

**The MJPEG stream exception:** `<img src="/stream">` in HTML cannot send custom headers — the browser controls the request entirely. The `/stream` endpoint uses a `?token=` query parameter instead, validated manually inside the route handler:

```python
@stream_router.get("/stream")
def stream(token: str | None = None):
    jwt.decode(token, JWT_SECRET, algorithms=["HS256"])  # raises on invalid
    return StreamingResponse(...)
```

The frontend appends the token when constructing the stream URL:
```typescript
streamUrl: () => `/stream?token=${sessionStorage.getItem("vlmedge_token") ?? ""}`
```

### Password Hashing with bcrypt

Passwords must never be stored in plaintext. **Hashing** is a one-way transformation: the hash is stored, and at login time the submitted password is hashed again and compared — the original password is never recoverable from the stored value.

**bcrypt** is the standard algorithm for password hashing. It is intentionally slow (to resist brute-force attacks) and includes a random salt (to resist precomputed rainbow tables).

The `bcrypt` library is called directly here (helpers live in `backend/app/core/security.py`):

```python
import bcrypt

# At signup, store this hash, never the password
hashed = bcrypt.hashpw("somepassword".encode(), bcrypt.gensalt()).decode()

# At login, returns True if the password matches
bcrypt.checkpw("somepassword".encode(), hashed.encode())
```

**Why call bcrypt directly instead of through passlib?** This project originally used `passlib`'s `CryptContext` wrapper. `passlib` 1.7.4 is unmaintained and breaks against `bcrypt` 5.x: it cannot read the new version string, and its internal probe passes an over-length value that `bcrypt` 5.x rejects with a `ValueError`. That path runs on every hash and verify, so it took down signup, login, and password changes. Calling `bcrypt` directly avoids the whole failure mode. One detail: `bcrypt` only uses the first 72 bytes of a password and, as of 5.x, raises if a longer one is passed, so the helpers truncate to 72 bytes. Stored hashes are standard `$2b$` strings either way, so old and new hashes verify identically.

### SQLite for User Storage

SQLite is a file-based relational database that comes with Python's standard library (`import sqlite3`). No server process, no installation, no pip package. The entire database lives in a single `.db` file.

For a small-team web application on a local network (one Jetson, a handful of users), SQLite is entirely appropriate. The tradeoffs:

| SQLite is fine when... | Need Postgres/MySQL when... |
|---|---|
| Single server, one writer at a time | High-concurrency writes from many simultaneous clients |
| Small user base (< hundreds) | Large user base |
| No replication needed | Distributed systems, replicas |
| Simplicity > performance | Performance > simplicity |

Pattern used in this project — open, query, close (no persistent connection pool needed at this scale):

```python
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # rows behave like dicts: row["username"]
    return conn

def find_user(username_or_email: str):
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (username_or_email, username_or_email),   # parameterized — prevents SQL injection
    ).fetchone()
    conn.close()
    return dict(row) if row else None
```

Always use parameterized queries (the `?` placeholders) — never f-strings or concatenation with user input.

### React Auth Patterns

**AuthContext** is the standard way to share auth state across a React app without prop-drilling:

```typescript
// One context at the top of the tree
const AuthContext = createContext<AuthContextValue>({ user: null, ... });

export function AuthProvider({ children }) {
  const [user, setUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    // Restore session on page load. sessionStorage survives refresh but clears
    // when the browser window closes, which is the session lifetime we want.
    const token = sessionStorage.getItem("vlmedge_token");
    fetch("/auth/me", { headers: { Authorization: `Bearer ${token}` } })
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(u => setUser(u))
      .catch(() => sessionStorage.removeItem("vlmedge_token"));
  }, []);

  return <AuthContext.Provider value={{ user, login, logout }}>{children}</AuthContext.Provider>;
}

// Any component deep in the tree can access auth state
const { user, logout } = useAuth();
```

**Automatic logout on 401:** When any API call returns 401, the frontend should immediately clear the token and redirect to login. The pattern used here:

```typescript
// api.ts — dispatches a custom event on every 401
function handle401(res: Response) {
  if (res.status === 401) window.dispatchEvent(new Event("vlmedge:unauthorized"));
}

// AuthContext — always-registered listener (separate useEffect, no early return)
useEffect(() => {
  const handle = () => doLogout();
  window.addEventListener("vlmedge:unauthorized", handle);
  return () => window.removeEventListener("vlmedge:unauthorized", handle);
}, []);
```

**Critical:** The event listener `useEffect` must have no early-return condition. If the listener is only registered when a token already exists at mount time, users who arrive without a token (first visit), log in, and then encounter a 401 will have no listener active — the event fires but nothing responds. This was the second Phase 14 bug.

---

## 21. Session Lifetime, Result History, and the Library

### sessionStorage vs localStorage

Both `localStorage` and `sessionStorage` are browser key-value stores for persisting small amounts of data between JavaScript events. The critical difference is lifetime:

| | `localStorage` | `sessionStorage` |
|---|---|---|
| Browser window/tab close | **Persists** | **Cleared** |
| Page refresh | Persists | Persists |
| New tab | Shared across tabs | Separate (each tab is its own session) |
| Explicit logout | Cleared by `removeItem()` | Cleared by `removeItem()` |

For this project, `sessionStorage` is the right choice for the JWT token and result history:

- **Close browser → session ends.** The Jetson is a shared device in an inspection context. Leaving an authenticated session open in a closed browser window is a security concern. With `sessionStorage`, the token is gone when the window closes — the next user must log in.
- **Refresh → session persists.** Page refresh is a routine browser action, not a sign of leaving. Forcing re-login on every refresh makes the interface unusable. `sessionStorage` survives refreshes within the same tab.
- **Tab switch → results persist.** When the user navigates from the Live tab to the Library tab, React unmounts and remounts the Dashboard component. A plain `useState([])` would reset to empty. Storing the result array in `sessionStorage` and reading it back in the `useState` lazy initializer restores the full history on every remount.

The implementation pattern:
```typescript
// Read on mount (lazy initializer — only runs once per mount)
const [results, setResults] = useState<ResultEntry[]>(() => {
  try {
    const raw = sessionStorage.getItem("vlmedge_results");
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
});

// Write on every change
useEffect(() => {
  sessionStorage.setItem("vlmedge_results", JSON.stringify(results));
}, [results]);
```

On logout, `doLogout()` calls `sessionStorage.removeItem("vlmedge_results")` so the next user who logs in on the same tab starts with an empty history.

### Server-Supplied Source Labels

A subtle bug can arise when a status poll is used to recover inference results after a tab switch. If the frontend hardcodes the source label instead of reading it from the server, every recovered result will show the wrong badge:

```typescript
// Wrong — hardcoded, shows "Auto-Scan" even for Analyze/Inspect results
setResults(prev => [...prev, { ...lr, source: "Auto-Scan" }]);

// Correct — use the source the server recorded when the inference ran
setResults(prev => [...prev, { ...lr, source: lr.source ?? "Auto-Scan" }]);
```

The server stores `source` in `_state["last_result"]` at the time each route runs:
- `POST /analyze` → `source="Analyze"`
- `POST /inspect` → `source="Inspect"`
- `_autoscan_loop()` → `source="Auto-Scan"`

Because `source` is stored in the server's shared state dictionary and the `/status` endpoint strips only the `jpeg` field, every status poll response automatically carries the correct label — no extra endpoint or logic needed.

### Per-User Media Library: The Outputs Table Pattern

Rather than having inference routes return results and immediately forget them, every user action is logged to a SQLite `outputs` table:

```sql
CREATE TABLE outputs (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    type       TEXT NOT NULL,   -- analyze | inspect | snapshot | autoscan | record | flag
    timestamp  TEXT NOT NULL,
    file_path  TEXT,            -- absolute path on disk
    prompt     TEXT,            -- for inference types
    response   TEXT,            -- model response text
    tokens     INTEGER,
    elapsed_s  REAL,
    user_id    INTEGER
)
```

This gives every user a persistent, queryable history without any extra work at inference time — each route just calls `log_output()` after saving the file. The Library page reads this table and groups entries by date and action type.

**Per-user isolation** is enforced at the query level:
```python
# Standard user — only their own rows
cursor.execute("SELECT * FROM outputs WHERE user_id = ?", (user_id,))

# Admin — all rows with username join
cursor.execute(
    "SELECT o.*, u.username FROM outputs o JOIN users u ON o.user_id = u.id"
)
```

**Serving files with `?token=` auth:**

Library file endpoints (`/library/view/{id}`, `/library/download/{id}`) use the `?token=` query param pattern for the same reason as `/stream` and `/eval/frame/*`: `<video src>`, `<img src>`, and `<a href download>` cannot send `Authorization` headers — the browser controls those requests entirely. The token is read from the query string and validated manually inside the route handler.

The distinction between view and download is the `Content-Disposition` header:
```python
# View — browser renders inline (image shows in page, video plays in player)
headers = {"Content-Disposition": f'inline; filename="{name}"'}

# Download — browser saves to disk regardless of file type
headers = {"Content-Disposition": f'attachment; filename="{name}"'}
```

FastAPI's `FileResponse` handles HTTP Range requests automatically. Range requests are how browsers implement video seeking — they request specific byte ranges of the file rather than downloading it all at once. Without Range request support, a `<video>` element would force the user to wait for the entire file to download before they could seek or play.

### cv2.VideoWriter for Video Recording

Recording video requires a different approach than saving individual JPEG frames. `cv2.VideoWriter` writes frames to a video container file as they arrive, producing a standard file that any player can open.

```python
fourcc = cv2.VideoWriter_fourcc(*"mp4v")        # MPEG-4 Part 2 in MP4 container
writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

# In the recording loop:
frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
if writer.isOpened():
    writer.write(frame)

# On stop:
writer.release()
```

Key decisions made in this project:

- **Frame size detected dynamically.** `h, w = frame.shape[:2]` from the first captured frame, not hardcoded. This avoids mismatches if the camera switches resolution or the Logitech BRIO is swapped for a different camera.
- **`isOpened()` check before every write.** If the writer failed to open (wrong codec, invalid path, disk full), calls to `writer.write()` would silently do nothing. Checking first surfaces the failure.
- **Time-accurate frame pacing.** The recording loop tracks `last_write = time.time()` after each frame and sleeps only `interval - elapsed_since_last`. A fixed `time.sleep(1/fps)` accumulates drift — the actual FPS drifts lower than intended over long recordings. Tracking the actual write time and sleeping only the remaining budget keeps pacing accurate.
- **`mp4v` codec.** Produces MPEG-4 Part 2 video in an MP4 container. Browser codec support for this is good on Chromium (the typical browser when accessing the Jetson from a desktop). H.264 (avc1) has even wider support but requires hardware encoder access that is not reliably available on Linux without additional GStreamer pipeline setup.
- **`writer.release()` on stop.** This flushes any buffered frames and writes the MP4 file's end-of-file marker. An unreleased writer produces a file that appears to have content but may be unplayable or truncated.

---

## 22. Latency, Camera Reliability, and Responsive Layout

These are the concepts behind the Phase 16 hardening work. Written for someone meeting them for the first time.

### Why `max_tokens` is the biggest latency lever

This pipeline is **generation-bound**: almost all of the time a response takes is spent generating output tokens one at a time. On the Orin Nano the model produces roughly 22 tokens per second. That number is close to fixed, so the length of the answer, not its content, sets the response time. A 130-token answer costs about 6 seconds; an 80-token answer costs about 3.6 seconds.

`max_tokens` caps how many tokens the model may generate. If you do not set it, generation runs until the model decides to stop or hits the context limit, which is why a chatty reply can balloon the latency. Setting a cap is the cheapest possible speedup because it does not change the model, the quantization, or the hardware. It just stops generation once the answer is long enough. The tradeoff is that too low a cap truncates real answers mid-sentence, so the cap is chosen by running the eval suite and reading the outputs, not guessed. This is why the Phase 16 values start PROVISIONAL: the numbers only become trustworthy after a before-and-after measurement on the actual device.

Prefill (encoding the image and prompt before generation starts) also costs time, but it is a fixed one-time cost per call. Generation is the part that scales with answer length, so it is where the easy wins are.

### Why the model cannot serve a stale answer from its own memory

A natural worry with a chat model is that it "remembers" the previous image and repeats an old answer. For this stack that cannot happen, and it is worth understanding why.

A transformer keeps a **KV cache** (key/value cache): as it processes tokens it stores intermediate results so it does not recompute the whole history for every new token. If that cache carried over between calls, a new call could be contaminated by the previous one.

In llama-cpp-python 0.3.31, the multimodal path used by `MoondreamChatHandler` clears this cache at the start of every call. It calls `llama.reset()` and `llama._ctx.kv_cache_clear()`, then rebuilds the image embedding from the fresh JPEG bytes it was handed. There is no leftover state and no cached image between calls. So if two calls return the same text, the cause is upstream of the model: either the camera handed it the same pixels, or the sampling was near-deterministic and two similar scenes produced the same words. That distinction is exactly what the frame-hash diagnostic (below) makes visible.

### Frame staleness: the default silent failure of USB cameras

A USB camera can stop delivering frames without raising an error. It can be unplugged, brown out, or drop off the bus, and the capture call simply stops returning new images. If your code only updates its "latest frame" when a read succeeds, it will keep serving the last good frame forever. Everything downstream looks healthy: the status says the camera is ready, the video stream shows a picture, and every inference dutifully analyzes the same frozen image. The system is confidently wrong.

The defense is to treat a frame as **perishable**. Record a timestamp every time a real frame arrives, and when something asks for the latest frame, refuse to return one that is older than a threshold (here, 2 seconds). A frozen camera then reports "no frame available," which the rest of the code already knows how to handle: the inference endpoints return a clean 503 and the status shows the camera as not ready. Two more safeguards go with it: sleep briefly after a failed read so a disconnected camera does not spin a CPU core at 100 percent, and after a run of failures, release and reopen the device with a backoff so a genuinely unplugged camera does not retry forever.

A related diagnostic: hashing the exact JPEG bytes of each analyzed frame turns a vague symptom ("it sometimes repeats") into a certain diagnosis. If two inferences in a row share a hash, the camera delivered identical bytes. If the hashes differ but the text is identical, the camera is fine and the repetition is the model sampling deterministically.

### Responsive layout: one interface for phones, tablets, and desktops

A "responsive" interface adapts its layout to the screen instead of shipping a separate mobile app. The main tools are CSS **media queries**, which apply different rules at different screen widths (breakpoints), and layout that flows rather than sits at fixed pixel sizes.

Three ideas carry most of the weight:

- **Stack instead of split.** A desktop dashboard puts the camera and the results side by side in two columns. On a narrow phone there is no room for two columns, so below a breakpoint the columns become a single vertical stack that the page scrolls through. The same content, reordered top to bottom.
- **`100dvh`, not `100vh`.** The unit `100vh` means "the full height of the viewport," but on mobile Safari the address bar overlaps that height, so a `100vh` layout gets its bottom clipped. `100dvh` (dynamic viewport height) measures the actually visible area and resizes as the address bar shows and hides, so nothing is cut off.
- **Touch targets and input zoom.** Fingers are less precise than a mouse, so tap targets are enlarged (around 44 pixels) on small screens. And iOS automatically zooms the page when you focus an input whose font is smaller than 16 pixels, which is jarring, so inputs use 16 pixel text on mobile to prevent it.

The Phase 16 layer is CSS only. It adds media queries at 1024, 768, and 480 pixels and changes no components or API calls, so it cannot affect behavior, only presentation.

---

## 23. Glossary

| Term | Definition |
|------|-----------|
| **LLM** | Large Language Model — a neural network trained to understand and generate text |
| **VLM** | Vision-Language Model — an LLM extended to also process image input |
| **Quantization** | Reducing numerical precision of model weights to save memory |
| **GGUF** | File format for storing quantized LLM weights, used by llama.cpp |
| **Q4_K_M** | A mixed INT4 quantization scheme balancing size and quality |
| **Token** | The basic unit of text a language model processes (roughly a word-piece) |
| **Embedding** | A vector (list of numbers) representing the meaning of a token or image |
| **Inference** | Running a trained model to produce output (as opposed to training it) |
| **INT4 / INT8 / FP16** | Numerical formats; INT4 = 4-bit integer, FP16 = 16-bit float |
| **TOPS** | Tera Operations Per Second — a measure of AI compute throughput |
| **UMA** | Unified Memory Architecture — CPU and GPU share the same physical memory |
| **JetPack** | NVIDIA's SDK for Jetson devices, bundles CUDA, cuDNN, TensorRT, OS |
| **CUDA** | NVIDIA's parallel computing platform that enables GPU acceleration |
| **TensorRT** | NVIDIA's model optimization and inference engine for maximum GPU performance |
| **llama.cpp** | C++ inference engine for running quantized LLMs/VLMs on edge hardware |
| **llama-cpp-python** | Python bindings for llama.cpp |
| **VQA** | Visual Question Answering — asking a model questions about an image |
| **Vision encoder** | The component of a VLM that converts images to embeddings |
| **GPU offloading** | Moving model layers from CPU RAM to GPU for faster computation |
| **PyQt5** | Python bindings for the Qt5 GUI framework; used for the desktop application |
| **QThread** | Qt's thread class; background work runs in `run()`, results are sent to the main thread via signals |
| **pyqtSignal** | A Qt signal defined on a class; emitting it posts the payload to connected slots (cross-thread safe when used with QThread) |
| **QSS** | Qt Style Sheets — CSS-like syntax for styling PyQt5 widgets |
| **Queued connection** | Qt's cross-thread signal delivery mode; the slot is posted to the receiver's event queue instead of called directly |
| **FastAPI** | Modern Python async web framework; used as the backend server in Phase 11 |
| **uvicorn** | ASGI server that runs FastAPI applications; started with `python -m uvicorn server:app --host 0.0.0.0 --port 8000` |
| **MJPEG** | Motion JPEG — streaming video format where each frame is a standalone JPEG sent in a multipart HTTP response; renders natively in `<img>` tags |
| **JWT** | JSON Web Token — a signed, self-contained token encoding user identity and claims; used for stateless authentication |
| **Bearer token** | HTTP Authorization scheme: `Authorization: Bearer <token>`; the standard way to send a JWT with API requests |
| **`sub` claim** | JWT "subject" — identifies the user the token refers to; PyJWT 2.x requires this to be a string, not an integer |
| **bcrypt** | A slow, salted password hashing algorithm; intentionally expensive to prevent brute-force attacks |
| **passlib** | Formerly used to wrap bcrypt via `CryptContext`. Removed in Phase 16 because passlib 1.7.4 breaks against bcrypt 5.x; the project now calls bcrypt directly. |
| **PyJWT** | Python library for encoding and decoding JWTs |
| **SQLite** | File-based relational database included in Python's stdlib; no server process required; appropriate for small-scale local deployments |
| **AuthContext** | React context pattern for sharing login state (user, token, login/logout) across the component tree without prop-drilling |
| **localStorage** | Browser key-value storage that persists until explicitly cleared, even across browser restarts. This project uses `sessionStorage` instead (see below) so the session ends when the window closes. |
| **sessionStorage** | Browser key-value storage scoped to a tab: it survives page refresh but is cleared when the window or tab closes. Holds the JWT and Dashboard result history here, giving a session that lives as long as the browser window and forces re-login after close. |
| **HTTPBearer** | FastAPI security dependency that extracts the Bearer token from the `Authorization` header |
| **ASGI** | Asynchronous Server Gateway Interface — the async equivalent of WSGI; required by FastAPI and uvicorn |
| **React** | JavaScript UI framework for building component-based browser interfaces |
| **Vite** | Frontend build tool for React; handles TypeScript compilation, bundling, and hot-reload |
| **CameraThread** | Pure-Python `threading.Thread` subclass that continuously reads camera frames and exposes the latest JPEG via `get_latest_jpeg()` |
| **SPA fallback** | A catch-all route that serves `index.html` for any unmatched path; required for single-page React apps so that browser refreshes on sub-routes don't return 404 |
| **useRef** | React hook that holds a mutable value without triggering re-renders; used to track the last-seen inference timestamp without causing render loops |
| **useEffect** | React hook that runs side effects (data fetching, subscriptions) after render; used to detect new inference results and append them to the history |
| **`min-height: 0`** | CSS property that allows a flex child to shrink below its content's natural size; required at every level of a viewport-locked flexbox layout |
| **Eval background thread** | Pattern where a long-running task (e.g. 5-prompt evaluation) runs in a daemon thread while the HTTP response returns immediately; progress is read via a separate polling endpoint |
| **Self-adjusting poll** | A polling pattern where the next poll is scheduled with `setTimeout` inside the callback rather than `setInterval`; allows the interval to change based on current state (e.g. 1.5s when active, 5s when idle) |
| **Latency delta** | The difference in inference time between the current run and a baseline run for the same prompt; negative = current run was faster, positive = slower |
| **Baseline (eval)** | A stored reference run (`results_latest.json`) that subsequent evaluation runs are compared against; user-controlled via "Set as Baseline" |
| **Path traversal** | A security attack where a URL parameter contains `../` sequences to escape the intended directory and access arbitrary files; prevented with regex validation before `os.path.join()` |
| **`urllib.parse.quote()`** | Python function that percent-encodes a string for use in a URL; converts spaces to `%20` so filenames with spaces can be embedded in API response URLs |
| **`encodeURIComponent()`** | JavaScript equivalent of `urllib.parse.quote()`; used to construct valid URLs from strings that may contain spaces or special characters |
| **React `key` prop remount** | Setting `key={someCounter}` on a component and incrementing the counter forces React to unmount and remount the component, discarding its state and re-running all `useEffect` hooks — a clean way to force a data re-fetch |
| **Ghost bar** | In a CSS bar chart, a semi-transparent background bar showing a reference value (e.g. baseline latency) behind the current value bar, making the delta visually obvious |
| **sessionStorage** | Browser key-value storage that persists across page refreshes within the same tab but is cleared when the browser window or tab closes; used here for the JWT token and result history so that closing the browser ends the session without requiring explicit logout |
| **cv2.VideoWriter** | OpenCV class that encodes and writes camera frames to a video file in real time; requires a fourcc codec, target FPS, and frame dimensions at creation; `release()` must be called to flush buffers and finalize the container file |
| **mp4v** | A fourcc code for MPEG-4 Part 2 video; used with `cv2.VideoWriter` to produce `.mp4` files; supported by Chromium without additional codec installation |
| **HTTP Range requests** | A browser mechanism that requests specific byte ranges of a file rather than the whole thing; used by `<video>` elements to support seeking without downloading the entire file; supported automatically by FastAPI/Starlette's `FileResponse` |
| **`outputs` table** | SQLite table that logs every user action (analyze, inspect, snapshot, autoscan, record, flag) with user ID, file path, and inference metadata; powers the Library tab |
| **Library page** | A per-user media browser showing all captured outputs organized by date and action type; admins see all users' outputs; preview modal shows image + Q&A for inference types and a video player for recordings |
| **`source` field (last_result)** | A string stored in `_state["last_result"]` by `_run_inference_locked()` identifying which button triggered the inference ("Analyze", "Inspect", or "Auto-Scan"); exposed via `/status` so the frontend can show the correct badge even for results recovered after a tab switch |
| **`max_tokens`** | The cap on how many tokens the model may generate in one response. Because the pipeline is generation-bound at about 22 tokens per second, this is the main control over response latency. |
| **generation-bound** | A workload whose time is dominated by generating output token by token rather than by input processing. Shortening the output is the cheapest way to reduce latency. |
| **KV cache** | The key/value cache a transformer keeps so it does not recompute attention over the whole history for each new token. Cleared at the start of every call in the mtmd path, which is why no stale answer can carry over between inferences. |
| **`frame_hash`** | The first 12 hex characters of the md5 of an analyzed JPEG. Two consecutive inferences with the same hash prove the camera delivered identical bytes, separating a frozen camera from sampling determinism. |
| **frame staleness** | Treating a camera frame as perishable: `get_latest_jpeg(max_age_s)` returns None once the newest frame is older than the threshold, so a silently frozen USB camera becomes a clean "not ready" instead of an endlessly repeated image. |
| **`/health`** | An unauthenticated endpoint returning only booleans (`ok`, `model_ready`, `camera_ready`). Used by launch.sh and any deployment supervisor as a readiness probe without needing a token. |
| **media query** | A CSS rule that applies only at certain screen sizes (for example `@media (max-width: 768px)`), the basic tool for responsive layouts. |
| **`100dvh`** | Dynamic viewport height: the actually visible height of the screen, which resizes as the mobile browser address bar appears and hides. Used instead of `100vh` so mobile Safari does not clip the bottom of the layout. |
| **bcrypt 72-byte limit** | bcrypt only hashes the first 72 bytes of a password and, since version 5.x, raises if given more. The password helpers truncate to 72 bytes so long passwords are handled consistently and never crash. |
