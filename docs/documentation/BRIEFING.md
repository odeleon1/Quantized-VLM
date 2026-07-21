# BRIEFING.md — Project Execution Plan

**Last Updated:** Prompt 62 | June 29, 2026
**Purpose:** This is the operational plan for the project — what we're building, how the pieces fit together, what needs to be integrated, and the phased sequence to get there with minimal rework and minimal bugs.

> Relationship to other docs: CLAUDE.md tracks decisions and state. LEARNING.md teaches the underlying concepts. This document is the **execution plan** — the how and in-what-order.

---

## 1. What We're Building

A pipeline that runs a quantized Vision-Language Model (Moondream2, Q4_K_M GGUF) directly on a Jetson Orin Nano (8GB, JetPack 7.2), so the device can take a camera frame and a text prompt and generate a natural-language response — entirely on-device, no cloud dependency.

**Inputs:** Camera frame (image) + text prompt (e.g. "What do you see?")
**Outputs:** Generated text response
**Constraint driving every decision:** 8GB unified memory shared between CPU and GPU. Realistic budget for the model + inference overhead is ~5–6GB.

---

## 2. How It Works (System Flow)

```
Camera Frame
     │
     ▼
[Capture/Preprocessing]  →  resize, format conversion
     │
     ▼
[Vision Encoder]  →  image → embeddings  (inside Moondream2/llama.cpp)
     │
     ▼
[Language Model]  →  embeddings + text prompt → token generation
     │              (GPU-accelerated via CUDA offloading in llama.cpp)
     ▼
[Decode]  →  tokens → human-readable text
     │
     ▼
Output (text response)
```

**Where each piece runs:**
- Image capture: application layer (Python), reading from camera device
- Inference (vision encoder + language model): llama.cpp binary/library, with GPU layers offloaded via CUDA
- Application logic (prompt construction, output handling): Python via llama-cpp-python

---

## 3. What Needs to Be Integrated

This is the full list of components that have to come together for the system to work end-to-end:

| Component | Role | Status |
|---|---|---|
| JetPack 7.2 (CUDA, cuDNN drivers) | OS + GPU driver foundation | Already installed on device |
| llama.cpp (built from source with CUDA) | Inference engine | **Built** — b9712, CUDA 13.2, all 25 layers on GPU confirmed |
| Moondream2 GGUF model (Q4_K_M) | The model weights | **Downloaded & quantized** — 877MB Q4_K_M + 868MB mmproj F16 |
| llama-cpp-python | Python bindings to llama.cpp | **Installed** — v0.3.31, CUDA build, 22.2 t/s confirmed |
| Camera input pipeline | Supplies image frames | **Working** — Logitech BRIO on /dev/video0, OpenCV 4.13.0, 640x480 JPEG frames confirmed |
| Application/integration layer | Ties camera + prompt + inference + output together | **Built** — FastAPI backend (`backend/server.py`) + React/TypeScript frontend (`frontend/`); replaces the PyQt5 desktop app from Phase 9 |
| (Possible) Downstream integration | Whether this VLM feeds into a larger application | Open question — not yet decided |

---

## 4. Phased Plan

The phases are ordered to surface hardware/build problems early, before any application logic is written on top of an unverified foundation. Each phase has a clear exit condition — don't move to the next phase until the current one's exit condition is met.

### Phase 1 — Environment Verification ✅ COMPLETE
**Goal:** Confirm the Jetson is in a known-good state before building anything.
- Confirm JetPack 7.2 install integrity (`cat /etc/nv_tegra_release`)
- Confirm CUDA toolkit version installed and on PATH (`nvcc --version`)
- Confirm available disk space and memory headroom (`free -h`, `df -h`)
- Confirm GPU is visible and functioning (`tegrastats` or `nvidia-smi` equivalent on Jetson)

**Exit condition:** CUDA toolkit confirmed present and GPU confirmed active, before any build attempt.

**Result:** L4T R39.2.0 (JetPack 7.2), CUDA 13.2 at `/usr/local/cuda-13.2/`, CMake 3.28.3, GCC 13.3, 771GB disk free. `nvcc` was installed but not on PATH — fixed by adding `/usr/local/cuda/bin` to `~/.bashrc`.

---

### Phase 2 — llama.cpp Build with CUDA Support ✅ COMPLETE
**Goal:** Get a working llama.cpp binary with CUDA acceleration compiled specifically for JetPack 7.2 / Orin Nano's Ampere (compute capability 8.7) architecture.
- Clone llama.cpp source
- Configure CMake build with `CMAKE_CUDA_ARCHITECTURES=87` explicitly set (do not rely on auto-detection)
- Build and resolve any JetPack-7.2-specific compilation issues
- Confirm the binary runs and reports CUDA support enabled

**Exit condition:** llama.cpp binary builds successfully and reports CUDA backend available.

**Result:** Built successfully with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87`. GCC 13.3 + CUDA 13.2 was compatible (no fallback needed). Binary links `libggml-cuda.so.0`, `libcudart.so.13`, `libcublas.so.13`. Key flag: use `-DGGML_CUDA=ON`, not the deprecated `-DLLAMA_CUBLAS=ON` found in older guides.

---

### Phase 3 — Model Acquisition & Validation ✅ COMPLETE
**Goal:** Get the Moondream2 GGUF model onto the device and confirm it loads correctly.
- Download Moondream2 Q4_K_M GGUF (and the associated vision encoder file, if separate)
- Verify file integrity (checksum if available)
- Load the model in llama.cpp with a basic CLI test (text-only prompt first, then image prompt)

**Exit condition:** Model loads without errors and produces output for both a text-only and an image+text prompt via CLI.

**Result:** Official repo (`moondream/moondream2-gguf`) only publishes F16 — no pre-quantized Q4_K_M. Downloaded F16 (2.7GB) and quantized locally using `llama-quantize` → 877MB Q4_K_M. mmproj kept at F16 (868MB). Text-only test: model loaded, generated output. Image+text test: `modalities: text, vision` confirmed, image described plausibly. Known issue: missing pre-tokenizer metadata in the GGUF causes a quality degradation warning — inherent to the published file, not fixable without regenerating the model.

---

### Phase 4 — GPU Offloading Validation ✅ COMPLETE
**Goal:** Confirm inference is actually running on GPU, not silently falling back to CPU.
- Run inference with `--n-gpu-layers` set to offload all layers
- Monitor GPU utilization during inference (`tegrastats` or equivalent)
- Benchmark tokens/sec with GPU offloading vs. CPU-only as a sanity check
- Confirm memory usage stays within the realistic budget (~5–6GB)

**Exit condition:** Measurable GPU utilization during inference, and a clear tokens/sec improvement over CPU-only baseline.

**Result:** All 25 layers confirmed on CUDA0 (Orin) via `--verbose` debug output: `offloaded 25/25 layers to GPU`. Generation speed ~19–21 t/s (consistent with GPU; CPU-only would be ~1–3 t/s on this hardware). **JetPack 7.2 gotcha:** `GR3D_FREQ` in tegrastats tracks the 3D graphics rasterizer (OpenGL/Vulkan), NOT CUDA compute — it will show 0% even when CUDA inference is fully active. Use `--verbose` debug output to confirm layer offloading instead. Additionally, on Jetson's unified memory architecture, model buffer sizes report as 0.00 MiB for both CPU and CUDA — this is correct behavior (no copy needed; GPU and CPU share the same physical RAM).

---

### Phase 5 — Python Integration (llama-cpp-python) ✅ COMPLETE
**Goal:** Move from CLI testing to a Python-callable interface.
- Install llama-cpp-python compiled with CUDA support (not the default pip wheel)
- Confirm Python can load the model and replicate the CLI results
- Wrap basic inference call in a simple test script (image path + prompt → text output)

**Exit condition:** A Python script reliably reproduces the same output quality and speed as the CLI test from Phase 4.

**Result:** Created a dedicated project venv at `.venv` using `/usr/bin/python3 -m venv`. Installed llama-cpp-python 0.3.31 from source with CUDA flags: `CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87" pip install llama-cpp-python`. Compiled wheel size 265MB confirms CUDA code is included. Test script (`test_phase5.py`) confirmed both text-only and image+text inference working in Python. Internal llama profiler reported **22.20 t/s** — matching Phase 4 CLI result exactly.

**Gotchas learned:**
- Do not use wall-clock `time.time()` to measure t/s for short responses (1–14 tokens). CUDA graph warmup and first-token latency dominate those measurements and make the numbers look artificially low. The authoritative speed is in `llama_perf_context_print` from verbose output, or measure across a longer generation (50+ tokens).
- `verbose=True` floods stdout with CUDA Graph messages (`CUDA Graph id N reused`). Use `verbose=False` for the actual pipeline.
- Images are passed as base64-encoded data URIs in the OpenAI-compatible chat format: `{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}`.
- The multimodal handler is `MoondreamChatHandler` from `llama_cpp.llama_chat_format`, not a generic handler.

---

### Phase 6 — Camera Input Pipeline ✅ COMPLETE
**Goal:** Replace static test images with live camera frames.
- Identify camera hardware/interface being used
- Build a capture function that returns frames in the format the model expects
- Test capture independently of inference first (just save/display frames) before wiring into the pipeline

**Exit condition:** Camera frames are captured reliably and in the correct format, verified independently of the inference pipeline.

**Result:** Camera: Logitech BRIO USB webcam. Installed `opencv-python-headless` 4.13.0 into the project venv. Built `capture.py` module exposing `open_camera()`, `capture_frame()`, and `release_camera()`. Frames returned as JPEG bytes (90% quality). Standalone test captured 5 frames to disk — quality confirmed. Module is designed to be imported directly into the Phase 7 pipeline.

**Gotchas learned:**
- The Logitech BRIO registers as `/dev/video0` through `/dev/video3` — all four nodes point to the same physical camera. This is normal for USB cameras that expose multiple nodes for different capture formats/modes. Device 0 is the primary capture device.
- `opencv-python-headless` is the correct pip package for a headless Jetson (no display required). The full `opencv-python` package requires display libraries that may not be present.

---

### Phase 7 — End-to-End Integration ✅ COMPLETE
**Goal:** Wire camera capture → inference → output into a single running pipeline.
- Connect Phase 5 (Python inference) and Phase 6 (camera capture)
- Add basic error handling (camera failure, model failure, malformed input)
- Run a sustained test (multiple consecutive frames) to check for memory leaks or degradation over time

**Exit condition:** Pipeline runs continuously for a sustained period without crashing, memory leaking, or silent failures.

**Result:** Built `pipeline.py` — loads model once at startup, then loops: capture frame → base64 encode → run inference → print response → reset KV cache → repeat. Frames saved to `output/`. Ran two complete 10-frame sustained tests without crashes or swap usage. Memory consumption stabilized at ~1–1.3 GB above baseline (normal CUDA buffer allocation, not a leak — confirmed by second run starting with more available memory than the first run ended with). Memory warning threshold set to 1500 MB to avoid false positives from normal model overhead. Descriptions accurate across both runs.

---

### Phase 8 — Evaluation & Tuning ✅ COMPLETE
**Goal:** Assess whether output quality and performance meet the project's actual needs.
- Evaluate Moondream2 output quality against real-world test cases (scene description, VQA, incidental text reading)
- Decide if Phi-3.5-vision fallback is needed based on real results, not assumptions
- Tune quantization level, GPU layer count, and prompt structure if needed

**Exit condition:** A documented judgment call: Moondream2 is sufficient, or a justified case for switching to the fallback model.

**Result:** Built `eval.py` — a structured evaluation tool with 5 prompt types (scene description, object list, people count, subject appearance, text reading). Each run saves annotated frames and a Markdown comparison report to `output/eval reports/`. Two evaluation runs completed.

**Findings:**
- Scene description, object detection, VQA, and people counting are accurate and consistent across both runs. Object detection showed strong contextual inference (correctly identified an office being rearranged without an explicit prompt for it).
- Text reading works on large, clear text (Monster Energy can label read correctly). On ambiguous scenes the model hallucinated plausible-sounding text rather than abstaining — a known small-VLM behavior.
- Average latency: 3.3–3.4s per inference. Acceptable for non-real-time use.
- Q4_K_M quantization is sufficient — no quality degradation observed that would justify re-quantizing.

**Decision: Moondream2 is sufficient. Phi-3.5 fallback is not needed.**

**Tuning applied:** Text reading prompt updated to include an explicit abstention option to reduce hallucination risk:
> *"Is there any text clearly visible in this image that you can read with confidence? If you are not certain, say 'No text clearly visible.'"*

No other tuning was required.

---

### Phase 9 — Desktop GUI Application ✅ COMPLETE
**Goal:** Build a native desktop application that serves as the central interface for the VLM — launchable from the Jetson desktop without a terminal, with two tabs covering the two primary use modes (live inference and structured evaluation).

**Exit condition:** App launches, camera feed is live, model loads in background, a question can be submitted and answered, evaluation tab shows past reports, a new evaluation run can be started from the GUI.

**Result:** Built `app.py` — a PyQt5 desktop application with:
- **Live tab** — 30 FPS camera feed, chat-style Q&A interface, inference runs in a background QThread while the feed keeps updating
- **Evaluation tab** — list of past eval report folders (newest first), Markdown report viewer with embedded annotated frames, "Run New Evaluation" button that steps through all 5 prompt types via a capture dialog
- **Aesthetic** — deep navy background (`#09091b`), amber/blue/green/red accent palette based on SHPE election results dashboard design; left-side vertical tab bar with custom `LeftTabBar` paint override to keep text horizontal on West-position tabs
- **Status bar** — shows model loading state, GPU layer count, and ready/busy indicator

**Gotchas learned:**
- `QGraphicsDropShadowEffect` causes recursive repaint loop (`paintSiblingsRecursive` → `drawWidget`) on ARM/Qt5 and will SIGABRT. Do not use on Jetson.
- `QImage(rgb.data.tobytes(), ...)` is a use-after-free bug: `.tobytes()` creates a temporary `bytes` object; CPython frees it immediately after the constructor returns because nothing holds a reference; `QImage` keeps a raw C pointer to freed memory. At low frame rates this usually works (freed memory still intact); at 30 FPS heap pressure causes the freed buffer to be reused before `.copy()` reads it → SIGABRT from glibc's corruption detector. Fix: `QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()` — numpy `rgb.data` is a memoryview backed by the in-scope array, so the buffer is guaranteed live until `.copy()` completes.
- Custom tab bars on the West position: PyQt5 rotates tab text 90° automatically. To keep text horizontal, subclass `QTabBar`, override `paintEvent` with a `QPainter(self)` loop that uses `initStyleOption` + `drawText`, and override `tabSizeHint` / `minimumTabSizeHint` to set fixed dimensions.
- `QWidget > QFrame { background: transparent; }` in an app-level stylesheet is dangerously broad — it matches internal Qt widget structures (QTabWidget frame, QSplitter handle, etc.) and can cause paint conflicts. Use `QLabel { background: transparent; }` instead.
- `cap.set(cv2.CAP_PROP_FPS, 30)` negotiates 30 FPS with the camera at open time; without it the camera defaults to a lower rate and the thread's 33ms sleep interval gets ahead of frame availability.

---

### Phase 10 — Pipeline Optimization & Bug Fixes ✅ COMPLETE
**Goal:** Audit all four source files (`app.py`, `eval.py`, `pipeline.py`, `capture.py`) for correctness bugs, thread-safety issues, and code inconsistencies. Fix everything found before moving to the application direction decision.

**Exit condition:** All identified bugs fixed; all four files pass syntax check; findings documented.

**Bugs fixed:**

| Severity | File | Bug | Fix |
|---|---|---|---|
| Critical | `app.py` | LLM concurrency — `EvalWorker` and `InferenceWorker` could both call `llm.create_chat_completion()` simultaneously if the user switched to the Live tab during an eval run | `_run_eval()` now disables Live tab input; `_on_eval_done()` re-enables it only if model is loaded |
| High | `app.py` | Thread leak on close — `ModelLoader` (blocking in C++ for ~10s), `EvalWorker` (possibly blocked waiting for a capture dialog), not cleaned up in `closeEvent` | `closeEvent` disconnects the loader's finished signal to prevent late callbacks on a torn-down window; if `EvalWorker` is running, `set_captured_frame(b"")` unblocks it and `wait(3000)` gives it time to exit |
| Medium | `app.py` | `painter.end()` not called in `LeftTabBar.paintEvent` — relied on CPython GC to call `QPainter.__del__`, which is not guaranteed timing on ARM | Added explicit `painter.end()` at end of `paintEvent` |
| Medium | `app.py` | Double `import json` in `EvalWorker.run()` — `import json` at top of method then `import json as _json` in a nested scope | Unified to `import json as _json` at the top of `run()` |
| Medium | `pipeline.py` | `run_inference` had signature `(llm, jpeg_bytes)` with prompt hardwired as a module constant; `llm.reset()` was called in `main()` not inside the function — inconsistent with eval.py's `run_inference(llm, jpeg_bytes, prompt)` | Added `prompt=None` parameter (defaults to `PROMPT`); moved `llm.reset()` inside the function |
| Low | `capture.py` | `cap.grab()` return value ignored in flush loop — camera disconnect during flush produced "Failed to read frame" instead of a specific error | Flush loop now raises `RuntimeError` immediately if `grab()` returns False |
| Low | `eval.py` | `generate_report` divided by `len(results)` without guarding against an empty results list — would crash with a ZeroDivisionError if the baseline JSON was edited to have empty results | Early return to `generate_baseline_report` if either results list is empty |

---

### Phase 11 — Remote Web Interface (Robotics & Inspections) ✅ COMPLETE
**Goal:** Replace the PyQt5 desktop application with a FastAPI + React web interface accessible from any device on the local network. Application direction selected: Robotics & Inspections. The Jetson may be mounted on a robot or inspection unit with no display — the interface must be reachable remotely.

**Exit condition:** Web interface accessible from a remote device on the LAN, all buttons functional, camera stream live, auto-scan results appearing in the result panel.

**Result:** Full stack built and verified.

**Backend (`backend/`):**
- `server.py` — FastAPI app with lifespan context manager; opens camera and loads model at startup (`asyncio.to_thread` for non-blocking model load); serves built React frontend from `frontend/dist/` with SPA fallback
- `app/api/camera_thread.py` — Pure-Python `threading.Thread` camera loop at 30 FPS; `cap.read()` is the rate limiter — no `time.sleep()` needed or used
- `app/api/routes.py` — 9 endpoints: `GET /stream` (MJPEG), `GET /status`, `POST /analyze`, `POST /inspect`, `POST /snapshot`, `POST /record/start`, `POST /record/stop`, `POST /autoscan/start`, `POST /autoscan/stop`, `POST /flag`
- `threading.Lock` (`_infer_lock`, non-blocking acquire) prevents concurrent `llm.create_chat_completion()` calls; returns HTTP 409 if busy
- Auto-scan saves annotated frames to `output/auto_scan/scan_MMDDYY_HHMM/` per session

**Frontend (`frontend/`):**
- React + TypeScript + Vite; dark navy theme matching the PyQt5 aesthetic
- `CameraFeed` — plain `<img src="/stream">` tag; MJPEG renders natively with no JS video code
- `ButtonPanel` — Analyze (blue), Inspect (amber), Snapshot, Record (toggle), Auto-Scan (toggle), Flag (red)
- `ResultPanel` — scrollable history of all inference results with source badges (Analyze/Inspect/Auto-Scan), timestamps, token count, elapsed time; auto-scrolls to newest
- `StatusBar` — model ready state, recording/auto-scan badges, memory free
- `useStatus` hook — polls `/status` every 3s (1.5s when auto-scan active); `useEffect` detects new `last_result.timestamp` and appends to result history
- Built frontend served statically by FastAPI; accessible at `http://JETSON_IP:8000` from any LAN device

**Bugs encountered and fixed:**
| Issue | Cause | Fix |
|---|---|---|
| Camera 10fps at 1280×720 | Hardware ceiling for Logitech BRIO at that resolution | Dropped to 640×480@30fps |
| Camera thread double-throttle | `cap.read()` blocks at camera rate + extra `time.sleep(1/fps)` doubled cycle time | Removed sleep; `cap.read()` is the only rate limiter |
| Auto-scan results not appearing | Frontend `useEffect` had `if (!autoscan) return` guard on local state; state was `false` on fresh load even with backend auto-scan active | Removed guard; any new `last_result.timestamp` appends to history |
| Old frontend served after rebuild | Running server process cached old in-memory state | Restarting server + hard refresh (Cmd+Shift+R) required after each frontend rebuild |

---

### Phase 12 — Web Evaluation Interface ✅ COMPLETE
**Goal:** Expose the existing 5-prompt evaluation suite (`eval.py`) through the web UI so that model performance can be assessed and compared remotely without terminal access, and results can be reviewed alongside annotated camera frames.

**Exit condition:** Evaluation can be triggered and monitored from any browser on the LAN; per-prompt latency, token count, and comparison against a user-selected baseline are visible in the UI.

**Result:** Full evaluation workflow implemented in the web interface.

**Backend (`backend/app/api/eval_routes.py`):**
- 6 routes registered under `/eval/*` via `eval_router` (included in `server.py` before the SPA fallback)
- `POST /eval/run` — validates preconditions (model loaded, camera ready, no active auto-scan), starts `_run_eval()` in a daemon thread, returns immediately
- `GET /eval/status` — returns `_eval_state` dict: `running`, `progress`, `total`, `current_label`, `results`, `report_id`, `error`
- `GET /eval/reports` — lists all `report_YYYYMMDD_HHMMSS/` directories in `output/eval reports/`, newest first; includes `avg_latency_s`, `result_count`, `is_baseline`, `legacy` flag for old CLI-only runs
- `GET /eval/report/{report_id}` — loads `results.json`, computes `stats` (avg latency, total tokens, fastest/slowest prompt) and `comparison` (per-prompt and overall delta vs current baseline), adds `frame_url` per result
- `GET /eval/frame/{report_id}/{filename}` — serves annotated JPEG with path-traversal protection (regex validation on both params)
- `POST /eval/set-baseline/{report_id}` — copies that run's `results.json` to `BASELINE_PATH`; the user decides when to promote a run, not the system automatically

**Eval thread (`_run_eval`):**
- Iterates the 5 PROMPTS from `eval.py`; for each: grabs latest JPEG from CameraThread, acquires `_infer_lock` with 15s timeout (waits out any in-flight auto-scan tick), runs `run_inference()`, releases lock
- Saves annotated frame to `{report_dir}/{label}.jpg` **before** appending to `_eval_state["results"]` so the frame file exists by the time the frontend renders the result
- After all 5 prompts: saves `results.json`, reads baseline once (snapshot, not live), calls `generate_report()` or `generate_baseline_report()` from existing `eval.py`
- Sets `running = False` only after all file writes complete

**Frontend:**
- `App.tsx` — top nav bar ("Live" | "Evaluation" tabs) wrapping `Dashboard` and `EvalPage`; `.dashboard` height changed from `100vh` to `flex: 1` to accommodate nav bar
- `useEvalStatus` hook — self-adjusting poll: 1.5s while `running`, 5s when idle; avoids constant polling overhead when nothing is happening
- `EvalRunner` — control panel (Run button, disabled reasons, progress bar with current prompt name, completion summary: avg latency + total tokens + Set as Baseline button)
- `ReportList` — scrollable list of past reports (newest first), `key={counter}` forces remount + re-fetch after each new run
- `ReportViewer` — stats panel (summary chips: avg latency, total tokens, fastest, slowest, vs-baseline colored chip) + latency bar chart (color-coded: blue = no baseline, green = faster, red = slower; ghost bar shows baseline position) + per-prompt result cards with delta badges and annotated frames (lazy-loaded)
- Column ratio: `flex: 1` (EvalRunner) : `flex: 2` (reports) — report viewer gets majority of horizontal space

**Security:**
- `report_id` validated against `r'^report_\d{8}_\d{6}$'` on every route — no slashes, dots, or wildcards can pass
- `filename` validated against `r'^[\w\s]+\.jpg$'` — blocks `../` path traversal while allowing the space-containing label names (`Scene Description.jpg`)
- Frame URLs in API responses use `urllib.parse.quote()` server-side so `<img src>` in the browser receives a valid percent-encoded URL

---

### Phase 13 — Prompt & Button Optimization ✅ COMPLETE
**Goal:** Improve the quality and usefulness of the pre-defined inference buttons through prompt fine-tuning oriented towards the Robotics & Inspections use case.

**Result:**
- `ANALYZE_PROMPT` initially updated — framed for transportation safety inspection reports; revised again in Phase 14 to be general scene description (see Phase 14)
- `INSPECT_PROMPT` updated — structured checklist: structural damage, operational hazards, missing safety equipment, environmental conditions; asks for issue location in frame
- All 5 evaluation prompts updated — domain-oriented but kept general enough not to presuppose a transportation scene; abstention options preserved where hallucination risk exists (People Count, Text Reading)
- Eval frame images changed to raw (no text overlay) — prompt and response are already shown as captions in the ReportViewer cards
- Browser tab title changed to "VLM Edge"

---

### Phase 14 — Authentication System ✅ COMPLETE
**Goal:** Add a login system to the web interface so the application is not open to anyone on the LAN. Two role levels: standard user and admin. Admin can promote/demote other users.

**Exit condition:** Login page shown to unauthenticated visitors; all existing API routes protected behind JWT; admin panel functional; password strength feedback shown during signup.

**Result:** Full auth layer built and integrated.

**Backend:**
- `output/vlmedge.db` — SQLite database; `users` table: `id`, `username`, `email`, `password_hash`, `is_admin`, `created_at`
- `backend/app/core/database.py` — `init_db()` (creates table, seeds admin on first startup), `find_user()`, `create_user()`, `list_users()`, `set_admin()`, `update_password()`
- `backend/app/api/auth_routes.py` — `/auth/signup`, `/auth/login`, `/auth/me`, `/auth/change-password`; exports `get_current_user` and `require_admin` FastAPI dependencies
- `backend/app/api/admin_routes.py` — `/admin/users` (list), `/admin/users/{id}/promote`, `/admin/users/{id}/demote`
- `backend/app/api/routes.py` — split into `stream_router` (no auth; `/stream` validates `?token=` query param manually because `<img src>` cannot send headers) and `router` (all other routes; `dependencies=[Depends(get_current_user)]` at router level)
- `backend/app/api/eval_routes.py` — added `dependencies=[Depends(get_current_user)]` at router level
- `backend/server.py` — calls `init_db()` in lifespan startup; includes `auth_router`, `stream_router`, `router`, `eval_router`, `admin_router`

**Frontend:**
- `AuthContext.tsx` — `AuthProvider` wrapping all pages; reads `vlmedge_token` from `localStorage` on mount, validates via `GET /auth/me`; `login()` / `logout()` functions; listens for `vlmedge:unauthorized` custom event to trigger auto-logout on 401
- `LoginPage.tsx` — username or email + password; "Sign up" link
- `SignupPage.tsx` — username, email, password, confirm password; live `PasswordStrengthBar`
- `PasswordStrengthBar.tsx` — 6-criterion scoring (length≥8, length≥12, uppercase, lowercase, digit, special char); colored bar + label
- `AdminPage.tsx` — user table with Promote / Demote actions; hidden for non-admins
- `App.tsx` — wraps in `AuthProvider`; routes between login/signup/app; Admin tab only visible if `user.is_admin`; username + Log out button in nav

**Bugs found and fixed:**
| Bug | Cause | Fix |
|---|---|---|
| All protected routes returned 401 immediately after login | PyJWT 2.x requires `sub` to be a string; `user["id"]` from SQLite is an integer — token encoded fine but decoded with "Subject must be a string" | `str(user["id"])` in `_create_token()`; `int(user["sub"])` at all read-back sites |
| 401 auto-logout didn't trigger for fresh-session users | `vlmedge:unauthorized` event listener was registered inside the same `useEffect` block that returned early when no token existed at mount — users arriving without a token never got the listener | Split into two separate `useEffect` hooks: listener always registers unconditionally; session restore is a separate effect |

**`ANALYZE_PROMPT` revised** — transportation framing removed. Analyze is a general-purpose button; Inspect carries the domain-specific checklist. New prompt: *"Describe what you see in this image. Identify the environment, the main subjects or objects present, and any notable details. Be specific and factual in two to three sentences."*

---

### Phase 15 — Library, Session Management & Polish ✅ COMPLETE
**Goal:** Build a per-user media library, fix session lifetime behavior, correct result source labeling, produce real video recordings, and restrict the evaluation tab to admins.

**Exit condition:** Library tab functional with per-user isolation and admin cross-user view; browser close ends session; page refresh keeps session; result history survives tab switches with correct labels; recordings are playable MP4 files; eval tab hidden from non-admin users and eval frames display correctly.

**Result:** Six improvements shipped in one phase.

**Library tab (`backend/app/api/library_routes.py` + `frontend/src/app/pages/LibraryPage.tsx`):**
- New `outputs` SQLite table tracks every user action: `type`, `timestamp`, `file_path`, `prompt`, `response`, `tokens`, `elapsed_s`, `user_id`. All existing routes (`/analyze`, `/inspect`, `/snapshot`, `/autoscan`, `/record/stop`, `/flag`) call `log_output()` after completing.
- Three new routers: `library_router` (Bearer — user's own outputs), `library_admin_router` (Bearer + admin — all users with JOIN to get username), `library_download_router` (no dep — `?token=` query param for file serving).
- Endpoints: `GET /library/outputs`, `GET /library/admin/outputs`, `GET /library/view/{id}?token=` (inline), `GET /library/download/{id}?token=` (attachment).
- Frontend `LibraryPage`: date sidebar (newest first, auto-expands on load), action-type filter buttons per date group, media card grid (`<video preload="metadata">` for recordings, `<img>` for others), `PreviewModal` with split Q&A layout for inference types and `<video controls>` for recordings.
- Admin sidebar adds username level above dates; admin endpoint includes `username` field for grouping.
- Download button shown only for snapshot, flag, and record types.

**Video recording format:**
- Changed from discrete JPEGs to `cv2.VideoWriter` with `mp4v` codec, producing `.mp4` files.
- Frame size detected dynamically from first captured frame.
- Time-accurate pacing via `last_write` timestamp (sleep only remaining budget per frame, not a fixed sleep).
- `writer.release()` called on stop to finalize the container.

**Eval admin restriction:**
- `eval_router` upgraded from `dependencies=[Depends(get_current_user)]` to `dependencies=[Depends(require_admin)]`.
- Frontend: Evaluation nav tab wrapped in `{user.is_admin && ...}` guard (same pattern as Admin tab).

**Eval frame images:**
- Root cause: `/eval/frame/*` was inside `eval_router` (Bearer required), but `<img src>` cannot send `Authorization` headers.
- Fix: frame endpoint moved to `eval_frame_router` (no auth dep) with `token: str | None = None` query param and manual `jwt.decode()` validation.
- `api.evalFrameUrl(path)` in `api.ts` appends `?token=<sessionStorage token>` to every frame URL.

**Session lifetime:**
- JWT token moved from `localStorage` to `sessionStorage` in both `AuthContext.tsx` and `api.ts`.
- `sessionStorage` clears on browser window close → forces re-login. Survives page refresh → session continues.
- `doLogout()` explicitly clears both `vlmedge_token` and `vlmedge_results` from sessionStorage.

**Result history persistence and source labels:**
- `Dashboard.tsx` initializes `results` from `sessionStorage` via a lazy `useState` initializer. On every change, persists to `sessionStorage`. Survives all tab switches and refreshes; cleared on logout/close.
- `lastSeenTimestamp` ref seeded from stored history on mount to prevent duplicate entries.
- `_run_inference_locked()` gains a `source` parameter stored in `_state["last_result"]`. `/analyze` passes `"Analyze"`, `/inspect` passes `"Inspect"`, `_autoscan_loop` passes `"Auto-Scan"`. `/status` exposes it automatically (only strips `jpeg`).
- Status poll uses `lr.source ?? "Auto-Scan"` instead of hardcoded `"Auto-Scan"`.

**Bugs encountered and fixed:**
| Issue | Cause | Fix |
|---|---|---|
| Eval frames broken (401) | `<img src>` can't send Bearer headers; frame route was inside Bearer-protected eval_router | Moved frame endpoint to `eval_frame_router` with `?token=` validation |
| Auto-scan button resets on tab switch | `autoscan` and `recording` were local `useState(false)`, reset to false on remount | Derived from `status?.autoscan ?? false` and `status?.recording ?? false` |
| Result history lost on tab switch | `results` was local `useState([])`, reset on Dashboard unmount | Persisted to and restored from `sessionStorage` |
| Results labeled "Auto-Scan" regardless of button | Status poll hardcoded `source: "Auto-Scan"` | Server stores `source` in `last_result`; poll uses `lr.source` |
| Session persists after browser close | Token in `localStorage`, which survives window close | Moved to `sessionStorage`, which clears on close |

---

### Phase 16: Finalizing & Hardening ✅ COMPLETE
**Goal:** Make the repo run from a fresh clone, diagnose and fix the stale-response bug, tune inference latency, remove legacy code, and add a responsive phone and tablet interface. Started July 20, 2026.

**Note on gates:** all benchmarks and camera-dependent checks were run on the Jetson itself with the Logitech BRIO attached, on July 21, 2026. Numbers below are from that hardware, not from any other machine. Tasks 1, 2, 3, 4, 5, and 7 have all passed their gates. Task 6 then ran as well: flash_attn was adopted for a 14 percent end-to-end gain, while n_batch and n_threads were measured and reverted. Phase 16 is complete.

Two findings are worth reading even if the rest is skimmed. First, the Task 2 gate did its job and **failed** on the first attempt, catching answers that were being silently truncated, which forced a revision of the token caps. Second, the measured latency benefit of those caps is a **null result**: the caps are worth keeping as worst-case ceilings, but they do not speed up typical responses, and the earlier assumption that max_tokens was the single biggest latency win did not survive measurement.

**Step 0: config.py restored (DONE).** The gitignored config.py had gone missing from disk entirely, so nine backend modules failed to import and the server could not boot. Recovered the real values from the stale bytecode and rewrote config.py as a committed file that reads secrets from environment variables: JWT_SECRET comes from the env, else output/.jwt_secret, else a generated token_hex(32) written with chmod 600; ADMIN_PASSWORD is generated with token_urlsafe(12) when unset. Removed config.py from .gitignore. Verified the full import chain and that output/.jwt_secret stays out of git.

**Task 1: Camera staleness and duplicate-response diagnostics (CODE COMPLETE, gate pending).** CameraThread now timestamps every frame; get_latest_jpeg(max_age_s=2.0) returns None once the newest frame is stale, so a frozen USB camera becomes a clean 503 instead of an endlessly repeated frame. Added sleep-on-failure to stop the busy-spin, device reconnect with backoff, and a frame_age_s method surfaced in /status and the StatusBar. Each inference records an md5 frame_hash (result, last_result, and a new nullable outputs.frame_hash column via guarded migration), plus a console warning when two consecutive inferences use identical bytes. **Gate PASSED (July 21, 2026, on the Jetson with the Logitech BRIO).** With the camera unplugged mid-session: /status reported camera_ready false with frame_age_s climbing to 128.9s, /analyze returned HTTP 503 "Camera not ready" instead of analyzing the frozen frame, and /health reported camera_ready false. The server process sat at 10.5 percent CPU, where the old no-sleep retry loop would have pegged a core. Reconnect attempts backed off at 8, 10, 12, 14, 16, and 18 seconds, each logging a clear reason. On replug the camera recovered in about 3 seconds with no server restart ("Reconnect succeeded", reopened at 640x480@30fps), frame_age_s returned to 0.0, and inference resumed normally. The same-hash warning did not fire during healthy operation because consecutive analyses produced different frame hashes (11504397eac0 then 33a1f0d72b31), which is the correct result for a live camera: the diagnostic is a trap for the frozen case, not something that should trigger in normal use.

**Task 2: Inference parameters (DONE, values LOCKED after the gate forced a revision).** run_inference passes max_tokens, temperature (0.3), and repeat_penalty (1.15) explicitly instead of inheriting the handler defaults.

**Gate FAILED at the first values, then passed after raising the caps.** The provisional caps (Analyze 120, Inspect 160, eval 120) truncated real answers. /inspect stopped at exactly 160 tokens mid-item ("6. Unusual environmental conditions like extreme temperatures") and the eval Text Reading prompt stopped at exactly 120 tokens mid-phrase ("...might need additional equipment for their"). Per the rule that a truncated answer is a regression regardless of latency, the caps were raised rather than shipped:

| Constant | Provisional | Locked |
|---|---|---|
| MAX_TOKENS_ANALYZE | 120 | 160 |
| MAX_TOKENS_INSPECT | 160 | 256 |
| MAX_TOKENS_EVAL | 120 | 200 |

Re-verified after the change: /inspect returned 44 and 165 tokens with complete sentences (the 165 confirms 160 was cutting content), and all five eval prompts finished on their own.

**Before and after, full 5-prompt eval on the Jetson:**

| Prompt | Before (uncapped, t0.2, rp1.1) | After (capped, t0.3, rp1.15) |
|---|---|---|
| Scene Description | 12.02s / 84 tok | 8.15s / 75 tok |
| Object List | 5.81s / 42 tok | 5.03s / 31 tok |
| People Count | 7.07s / 62 tok | 6.10s / 47 tok |
| Subject Appearance | 7.20s / 64 tok | 6.53s / 54 tok |
| Text Reading | 6.55s / 54 tok | 7.58s / 70 tok |
| **Average** | **7.73s** | **6.68s** |
| Total tokens | 306 | 277 |

**Honest reading of that table: this is a null result on latency, not a 13 percent win.** The two runs saw different live scenes, response length varies with scene content, and the "before" run's first prompt was a 12.02s CUDA warmup outlier. Normalizing for tokens, per-token throughput is effectively identical across both (roughly 17 to 18 tokens per second on top of a fixed image prefill of about 3.5s). The model naturally stops between 30 and 90 tokens, well under any of these caps, so the caps almost never engage. Their real value is bounding worst-case latency on a rambling answer, not speeding up typical ones. Setting them too low, as the first attempt did, actively traded correctness for a speedup that does not materialise. No quality regression was observed from temperature 0.3 or repeat_penalty 1.15.

**Task 3: Fresh-clone runnability (CODE COMPLETE, gate pending).** Added an unauthenticated GET /health returning boolean readiness and pointed launch.sh at it, since the old /status poll needed a token and always waited the full 30 seconds. Documented the ADMIN and JWT_SECRET and CAMERA_INDEX environment variables in the README. Dropped the desktop-launcher path; this is a web-only interface. Also found that annotated-doc, a FastAPI 0.138 dependency, was missing from the local venv (it imported only through a leaked PYTHONPATH); a fresh uv sync installs it, so pyproject is correct. **Gate PASSED (July 21, 2026).** A throwaway clone in /tmp with the models symlinked ran start to finish with zero manual file creation. config.py was present in the clone, which is the whole point: before Step 0 a fresh clone died at import because the file was gitignored. uv sync built llama-cpp-python 0.3.31 from source with the CUDA flags honoured (nvcc confirmed in the process tree, so no silent CPU-only fallback). The server booted on port 8001, /health returned ok with model_ready and camera_ready true, the camera opened at 640x480@30fps and the model loaded in 1.8s. First boot seeded its own admin with a generated 16-character password printed to the console, and logging in with those exact credentials succeeded. The clone generated its own output/.jwt_secret at mode 600, independent of the real project's secret, and its database contained only its own admin, confirming no cross-contamination with the live deployment. No tracked files were modified during the run.

**Task 4: Legacy removal and dependency hardening (CODE COMPLETE, gate pending).** Moved the retired PyQt5 app (main.py) and the standalone pipeline.py to legacy/. Removed passlib, which is broken against bcrypt 5.x (its version probe raises on every hash and verify, so signup, login, change-password, and seeding were all failing), and call bcrypt directly through a new security.py with 72-byte-safe hashing. Existing $2b$ hashes verify unchanged. Removed the racy session_log.json write in /flag. README corrected: JetPack 7.2 is Ubuntu 24.04, storage is about 4.4GB during setup and 1.7GB after deleting the F16. **Gate PASSED (July 21, 2026).** uv sync reported the environment already in sync with no changes; passlib is absent from both the venv and uv.lock and is no longer importable, with bcrypt 5.0.0 installed. Login succeeded for the pre-existing admin account, whose hash was written by passlib before the migration, confirming that standard $2b$ hashes verify unchanged under direct bcrypt with no password reset for any existing user.

**Task 5: Concurrency and loop correctness (CODE COMPLETE, gate pending).** The MJPEG stream is now an async generator, so viewers no longer each hold a Starlette threadpool thread. Auto-scan sleeps max(0, interval minus inference time) so the tick period matches the UI. The eval suite captures the frame inside the acquired lock so the evaluated image reflects the scene at inference time. **Gate PASSED (July 21, 2026).** Two concurrent MJPEG viewers each received about 120 frame boundaries in a 4 second capture (roughly 30 fps each, 5.68MB per stream) with both served simultaneously. Auto-scan at interval 10 produced database rows 10.3, 9.2, 10.2, and 10.2 seconds apart while each inference took 6.4 to 7.2 seconds; before the fix those gaps would have been about 16 to 17 seconds. The eval suite completed with no errors.

**Task 6: Optional performance experiments (DONE, July 21, 2026). One adopted, two reverted.**

The eval suite runs against a live camera, so scene variation swamps small effects. These experiments therefore used a sharper instrument: a fixed saved frame, a fixed prompt, and `max_tokens=1` to isolate prefill, five samples per configuration, each configuration in its own process.

| Variable | Prefill median | End to end | Verdict |
|---|---|---|---|
| baseline (n_batch 512, no FA) | 3.052s | 7.34s | reference |
| **flash_attn=True** | **2.177s** | **6.28s** | **ADOPTED** |
| n_batch=1024 | 2.875s (vs 2.933 own baseline) | 7.15s | reverted |
| n_threads=6 | 3.015s | 7.46s | reverted |
| flash_attn + n_batch=1024 | 2.207s | 6.10s | n_batch adds nothing |

**flash_attn=True is the only meaningful win: prefill 28.7 percent faster, end to end 14.4 percent faster, about 1.06s off every response.** All five samples separate completely from baseline, and the runtime logs "flash attention is enabled" rather than silently falling back, confirming the sm87 kernels are actually in use. A full 5-prompt eval with it enabled averaged 6.10s with no truncation.

**n_batch=1024 was reverted.** On its own it produced a real but negligible gain (58ms, complete sample separation, but only about 2 percent of prefill and 0.8 percent end to end). Combined with flash_attn it produced nothing: 2.207s versus 2.177s with overlapping ranges. Flash attention changes the attention kernel, so the batch split stops mattering. Not worth an extra knob.

**n_threads=6 was reverted as a null result.** Prefill was within noise of baseline and end to end was marginally worse (7.46s versus 7.34s). Expected, since the GPU does the work and the CPU thread count barely participates.

**Quality check.** Because flash attention changes floating point accumulation order, output token paths shift even at the same temperature. A fixed-image comparison of three generations per configuration on the People Count prompt showed baseline answering "two people" three times out of three, and flash_attn answering "two" once and "three" twice. Both configurations hallucinated safety equipment in an office scene, which is the pre-existing small-VLM behaviour already documented in LEARNING.md. **No quality regression is concluded:** three samples cannot distinguish this from chance at temperature 0.3, and flash attention is mathematically equivalent to standard attention up to floating point associativity, so there is no mechanism by which it would systematically degrade output. Recorded here so the behaviour can be watched rather than assumed away.

**Task 7: Responsive phone and tablet interface (CODE COMPLETE, visual gate pending).** CSS-only responsive layer with a 100dvh base and breakpoints at 1024, 768, and 480 pixels. The two-column dashboard, eval, and library layouts collapse to a single scrolling column; the nav wraps with touch-sized targets; the preview modal goes full screen; inputs use 16px to stop iOS zoom on focus. No component or API changes beyond the close-button work below.

**Gate PASSED (July 21, 2026, verified on a real iPhone and iPad over the LAN).** No horizontal scroll and all controls reachable on every page. Two fixes came out of the device testing:

1. The preview modal close button collided with Safari's native video controls, which occupy both top corners on iOS and cannot be moved or restyled because the browser renders them in its own shadow DOM. Overlaying the button anywhere on the video surface is therefore unsafe. Video previews now get a dedicated header bar above the player, which cannot collide whatever Safari does inside the element. The button was also made far more visible everywhere: it was translucent white on muted grey and effectively invisible over bright frames, and is now a solid dark fill with a white border and shadow, 36px on desktop and 44px on touch.
2. On tablets the close button is hidden entirely, since from 769px up the modal is centred with a visible backdrop and tapping outside is the natural gesture. This is scoped to `(min-width: 769px) and (pointer: coarse)` so phones keep the button, where the modal is full screen and has no backdrop to tap, and desktop keeps it too. An Escape key handler was added so removing the visible control does not strand keyboard or assistive-technology users on an invisible affordance.

**Task 8: Remove Phase-N labels from source (DONE).** Stripped phase references from module docstrings, comments, and test banners so the phase history lives only in this document. Renamed test_phase5.py to test_inference.py. Legacy archives keep their headers by design.

---

## 5. Risk Areas to Watch (Bugs to Minimize Proactively)

- **Memory exhaustion** — the 8GB unified memory pool is shared with the OS and any other running processes. Monitor memory at every phase, not just at the end.
- **Silent CPU fallback** — llama.cpp can run without CUDA and just be slow, without throwing an obvious error. Phase 4 exists specifically to catch this.
- **JetPack 7.2 build incompatibilities** — most community guides target older JetPack versions. Expect to adapt steps, not copy-paste them.
- **Compute capability mismatch** — if `CMAKE_CUDA_ARCHITECTURES` is wrong or left to auto-detect, the build may succeed but silently underperform or fail at runtime.
- **Camera format mismatches** — image format/resolution mismatches between camera output and what the vision encoder expects can cause subtle quality issues rather than hard failures.

---

## 6. Working Conventions for This Plan

- Do not skip a phase's exit condition to "save time" — most bugs in projects like this come from stacking an unverified layer on top of another unverified layer
- If a phase reveals something that changes an earlier decision (e.g., GPU offloading doesn't work as expected and changes the model choice), update CLAUDE.md immediately, not at the next scheduled checkpoint
- calibrated-honesty is active for the entire duration of this project's development — flag real problems immediately, don't soften or bury them
