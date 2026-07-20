# CLAUDE.md — Project Instructions & Context

**Last Updated:** Prompt 63 | July 20, 2026
**Next Scheduled Update:** Prompt 68 (or sooner if a major topic is introduced)

---

## Project Identity

**Project Name:** Edge VLM Integration on Jetson Orin Nano
**Owner:** Orlando
**Platform:** NVIDIA Jetson Orin Nano (8GB, JetPack 7.2)
**Goal:** Deploy a quantized Vision-Language Model (VLM) for edge AI inference on a resource-constrained embedded device.

---

## Active Decisions (Locked In)

| Decision | Choice | Rationale |
|---|---|---|
| Target Model | Moondream2 (1.8B) — CONFIRMED | Edge-optimized VLM; ~1.0GB at Q4_K_M; sufficient for scene description + VQA; text reading is incidental/non-critical |
| Quantization Format | GGUF (Q4_K_M) | Best toolchain support, easiest Jetson CUDA integration |
| Runtime | llama.cpp + llama-cpp-python | Validated on Jetson hardware; simpler than TensorRT-LLM for VLMs |
| GUI Framework | React + FastAPI (web) | Replaced PyQt5 in Phase 11 — enables remote access from any device on the LAN; required for Robotics & Inspections use case where the Jetson may be mounted on hardware with no display |
| Application Direction | Robotics & Inspections | Decided in Phase 11 based on Phase 8 results — latency (~3.4s), accuracy, and VQA quality are all suitable; remote web interface is the natural fit for this use case |
| Honesty Mode | calibrated-honesty skill — PERMANENT, project-wide | Active at all times for the entire project lifecycle, not just per-conversation |

---

## Deferred / Open Questions

- Additional specialized buttons — People Count, Hazard Level, Equipment Status are candidates; deferred to Phase 16 after real-world testing informs which are actually useful
- Inference parameter tuning — implemented in Phase 16 and now passed explicitly (`max_tokens` per route, `temperature` 0.3, `repeat_penalty` 1.15), but the values are PROVISIONAL. They move to Active Decisions only after the Jetson eval-suite gate confirms no latency regression, no mid-sentence truncation, and no quality loss. See Phase 16 in BRIEFING.md.

## Resolved Questions

- **Use case** — Scene/object description, VQA, incidental text reading (large/non-critical). Confirmed Moondream2 is sufficient.
- **JetPack version** — 7.2, confirmed.
- **Fallback model needed?** — No. Phase 8 evaluation confirmed Moondream2 is sufficient for all primary use cases. Scene description, object detection, VQA, and people counting are accurate and consistent. Text reading works on large/clear text but hallucinated on ambiguous scenes — acceptable given text reading is non-critical. Phi-3.5 fallback is not warranted.
- **Quantization level** — Q4_K_M is sufficient. No quality or speed issues observed that would justify re-quantizing.
- **Model file availability** — The official `moondream/moondream2-gguf` repo on Hugging Face publishes F16 only. No pre-quantized Q4_K_M exists for download. The correct workflow: download the F16 text model (2.7GB) + mmproj F16 (868MB), then run `llama-quantize` locally to produce the Q4_K_M (877MB). The mmproj is kept at F16 — do not quantize it. Community-uploaded GGUF repos may have Q4_K_M pre-built, but mmproj availability there is inconsistent.
- **Prompt engineering** — Text reading prompt updated to include an explicit abstention option ("No text clearly visible.") to reduce hallucination risk.
- **Domain-specific prompt fine-tuning (Phase 13 + 14)** — `INSPECT_PROMPT` is scoped for transportation/safety/infrastructure inspection. `ANALYZE_PROMPT` was initially updated for transportation context in Phase 13, then revised in Phase 14 to be general scene description — no domain anchor, no safety framing — because Analyze is a general-purpose button and the domain framing was too constraining. All 5 evaluation prompts are inspection-oriented but general enough not to presuppose a transportation scene. Prompt labels left unchanged — they double as frame filenames and JSON keys, so renaming them would orphan existing report data.
- **Eval frame annotation** — Evaluation report frames are saved as clean images (no text overlay). The prompt and response are already shown as card captions in the ReportViewer. The `annotate_frame()` function is still used for the runs directory (`/analyze`, `/inspect`) where the file is a standalone artifact with no surrounding UI context.
- **GUI crashes on ARM/PyQt5** — Two distinct crashes encountered and fixed: (1) `QGraphicsDropShadowEffect` causes a recursive `paintSiblingsRecursive` → `drawWidget` loop on ARM/Qt5 and will SIGABRT every time — do not use on Jetson. (2) `QImage(rgb.data.tobytes(), ...)` passes a temporary `bytes` object whose memory CPython frees before the next line executes; `QImage` holds a raw C pointer and ends up reading freed heap, triggering glibc's corruption detector. Fix: use `rgb.data` (a numpy memoryview backed by the in-scope array) and chain `.copy()` immediately so the deep copy completes before any deallocation can occur.
- **Application direction** — Robotics & Inspections. Confirmed in Phase 11. Remote web interface is the correct architecture for a robotics/inspection unit where the Jetson may have no display attached.
- **Camera resolution and FPS** — 1280×720@10fps is the hardware ceiling for the Logitech BRIO at that resolution. Dropping to 640×480 gives 30fps. 848×480 (16:9) was attempted but the camera didn't support it at 30fps. Settled on 640×480@30fps with `object-fit: cover` in CSS to fill the 16:9 card.
- **CameraThread double-sleep bug** — `cap.read()` is already a blocking call that rate-limits to the camera's FPS. Adding `time.sleep(1/fps)` on top doubles the effective cycle time, halving FPS. Fix: remove the sleep; `cap.read()` alone is the rate limiter.
- **Session lifetime management** — JWT token moved from `localStorage` to `sessionStorage`. `sessionStorage` is cleared when the browser window/tab closes (forcing re-login) but preserved across page refreshes within the same tab (keeping the session alive). `doLogout()` explicitly clears both `vlmedge_token` and `vlmedge_results` so a subsequent login on the same tab starts clean.
- **Result history persistence and source labels** — Dashboard result history stored in `sessionStorage` (`vlmedge_results`). Initialized from sessionStorage on mount so the full history survives tab switching. The server embeds a `source` field in `last_result` ("Analyze", "Inspect", or "Auto-Scan"); the status poll uses `lr.source` instead of the previous hardcoded `"Auto-Scan"` — so results picked up after a tab switch show the correct badge.
- **Library page** — `outputs` SQLite table logs every user action (analyze, inspect, snapshot, autoscan, record, flag) with user ID, file path, and inference metadata. Library tab shows per-user media organized by date then action type. Admins see all users via username → date → type hierarchy. Preview modal: inference types (analyze/inspect/autoscan) show image left + Q&A right; snapshot/flag show full image; record shows `<video>` player. Downloads restricted to snapshot, flag, and record only. File endpoints use `?token=` query param (same pattern as `/stream` and `/eval/frame/*`) because `<a href download>`, `<video src>`, and `<img src>` cannot send `Authorization` headers.
- **Video recording format** — Recording uses `cv2.VideoWriter` with `mp4v` codec producing `.mp4` files. Frame size detected dynamically from first captured frame. Time-accurate pacing via `last_write` timestamp to avoid drift. `writer.release()` called on stop to flush and finalize the file.
- **Eval tab admin restriction** — Evaluation tab hidden from non-admin users (frontend `App.tsx` guard; backend `eval_router` uses `dependencies=[Depends(require_admin)]`).
- **Eval frame images** — Broken because `<img src>` on `/eval/frame/*` received 401 (Bearer required). Fixed by splitting into `eval_frame_router` with `?token=` query param validation — same pattern as `/stream`. Frontend constructs URL via `api.evalFrameUrl(path)` which appends the session token.
- **Auto-scan tab-switch bug** — `autoscan` and `recording` in Dashboard were local `useState`, reset to `false` on every remount. Fixed: both values are now derived from `status?.autoscan ?? false` and `status?.recording ?? false` (the server status poll), so they survive tab switches.
- **Python packaging: pip → uv** — Backend dependency management switched from `pip` + manually created `.venv` + `backend/requirements.txt` to [uv](https://docs.astral.sh/uv/), managed via a `pyproject.toml` at the repo root (`.venv` still lives at the repo root; `tool.uv.package = false` since the backend is an app, not a distributable package). `llama-cpp-python`'s CUDA build still requires `PATH`/`CMAKE_ARGS` exported before `uv sync` — same mechanism as before, just invoked through uv instead of pip. All run commands (`README.md`, `launch.sh`, `server.py` docstring) updated to `uv run ...` / `uv sync`. `backend/requirements.txt` removed.

### Phase 16 gotchas (Finalizing & Hardening)

- **Duplicate-response root cause is not the model.** Verified against the llama-cpp-python 0.3.31 source: the mtmd path in `Llava15ChatHandler.__call__` (used by `MoondreamChatHandler`) calls `llama.reset()` and `llama._ctx.kv_cache_clear()` at the start of every call and rebuilds the image embedding from the fresh bytes. There is no KV carryover and no image-embed cache between calls. The two real candidates for identical responses are a frozen camera and near-greedy sampling. The frame_hash diagnostic added in Task 1 distinguishes them: same hash means the camera delivered identical bytes (frozen), different hash with identical text means sampling determinism (addressed by Task 2's temperature bump).
- **Frozen USB camera is the default silent failure.** `CameraThread.run()` only updated `_jpeg` on a successful `cap.read()`, so if the camera stalled or dropped off the bus the last good frame was served forever: `/status` still said `camera_ready`, the stream showed the frozen image, and every inference re-analyzed identical bytes. There was also a busy-spin on read failure (no sleep) that pegged a core. Fix: timestamp every frame, reject frames older than `max_age_s` in `get_latest_jpeg`, sleep on failure, and reconnect with backoff.
- **passlib is broken against bcrypt 5.x.** passlib 1.7.4 cannot read bcrypt 5.x's version and its internal `detect_wrap_bug` probe passes an over-length value that bcrypt 5.x rejects with `ValueError`. That path runs on every hash and verify, so signup, login, change-password, and admin seeding all raised. passlib was removed entirely; `backend/app/core/security.py` calls bcrypt directly and truncates passwords to 72 bytes so bcrypt 5.x never raises. Existing `$2b$` hashes verify unchanged.
- **launch.sh always waited the full 30 seconds.** Its readiness poll hit `/status`, which requires a Bearer token since Phase 14, so `curl -sf` got 401 every iteration. Fix: an unauthenticated `GET /health` returning only boolean readiness (`ok`, `model_ready`, `camera_ready`), and launch.sh polls that.
- **config.py had gone missing from disk.** It was gitignored and the source file no longer existed, only stale bytecode in `__pycache__`, so nine backend modules failed to import and the server could not boot. The real values were recovered from the bytecode and config.py was rewritten as a committed, secrets-from-env file (see BRIEFING Phase 16 Step 0). Lesson: a gitignored file that every module imports is a single point of failure for a fresh clone.
- **annotated-doc dependency drift.** FastAPI 0.138 declares `annotated-doc` as a dependency, so a fresh `uv sync` installs it. The local `.venv` was missing it and only imported FastAPI through a leaked `PYTHONPATH` (a ROS and sibling-project setup in the shell). `pyproject.toml` is correct; the fix is to run `uv sync` on the machine so it stops depending on the leak.
- **Web-only direction confirmed.** No native desktop launcher. The `vlm-interface.desktop` file and the README desktop step were dropped; the interface is reached entirely through the browser.

---

## Document Registry

| Document | Purpose | Format |
|---|---|---|
| `CLAUDE.md` | Session instructions, active decisions, open questions | Markdown |
| `LEARNING.md` | Technical reference — concepts, tools, libraries explained | Markdown |
| `BRIEFING.md` | Execution plan — system flow, integration components, phased rollout | Markdown |

> Note: A stakeholder-facing Project Overview (.docx) was created separately for supervisors/mentors. It is not part of project documentation tracking and is not maintained in this file.

---

## Update Schedule

- **Every 5 prompts** — routine update to all three docs
- **On major topic change** — immediate update (new model, new tool, architecture decision, etc.)
- **On decision lock** — move item from "Open Questions" to "Active Decisions" table above

---

## Conventions

- **calibrated-honesty is active for the entire duration of this project's development, at all times, across all sessions and tools (Claude.ai, Claude Code, or otherwise)** — not just this conversation. Agree when correct, correct when wrong, no filler, no manufactured disagreement, no softening real problems.
- Explain the *why* behind technical decisions, not just the what
- When referencing organizations in professional writing, avoid starting sentences with "you"
- Keep LEARNING.md accessible — written for someone learning, not just as a reference dump

---

## Current Phase Status

| Phase | Status |
|---|---|
| Phase 1 — Environment Verification | ✅ Complete |
| Phase 2 — llama.cpp Build with CUDA | ✅ Complete |
| Phase 3 — Model Acquisition & Validation | ✅ Complete |
| Phase 4 — GPU Offloading Validation | ✅ Complete |
| Phase 5 — Python Integration (llama-cpp-python) | ✅ Complete |
| Phase 6 — Camera Input Pipeline | ✅ Complete |
| Phase 7 — End-to-End Integration | ✅ Complete |
| Phase 8 — Evaluation & Tuning | ✅ Complete |
| Phase 9 — Desktop GUI Application | ✅ Complete |
| Phase 10 — Pipeline Optimization & Bug Fixes | ✅ Complete |
| Phase 11 — Remote Web Interface (Robotics & Inspections) | ✅ Complete |
| Phase 12 — Web Evaluation Interface | ✅ Complete |
| Phase 13 — Prompt & Button Optimization | ✅ Complete |
| Phase 14 — Authentication System | ✅ Complete |
| Phase 15 — Library, Session Management & Polish | ✅ Complete |
| Phase 16 — Finalizing & Hardening | 🔧 In progress (code complete, hardware gates pending) |

## Session Note

Project moved to Claude Code as of June 18, 2026. Phases 1–4 completed in the first Claude Code session. Phase 9 (Desktop GUI) completed June 23, 2026 — PyQt5 app (`app.py`) built with live camera tab, evaluation tab, dark navy aesthetic, left sidebar navigation, and 30 FPS feed. Phase 10 (Optimization) completed same day — seven bugs fixed across all four source files; eval tab layout and markdown rendering also corrected.

Phase 11 (Remote Web Interface) completed June 24, 2026. PyQt5 desktop app retired as primary entry point. Replaced with FastAPI backend (`backend/server.py`) + React/TypeScript frontend (`frontend/`). Architecture: CameraThread (pure Python) feeds MJPEG stream via HTTP; all inference runs through FastAPI routes with a `threading.Lock` to prevent concurrent calls. Frontend: dark navy dashboard with live MJPEG camera feed, action buttons (Analyze, Inspect, Snapshot, Record, Auto-Scan, Flag), and a scrollable result history panel that accumulates all responses with source labels and timestamps. Auto-scan saves frames to `output/auto_scan/scan_MMDDYY_HHMM/` per session. Accessible from any device on the LAN at `http://JETSON_IP:8000`. Application direction locked: Robotics & Inspections.

Phase 12 (Web Evaluation Interface) completed June 24, 2026. Added a second page ("Evaluation" tab) to the web UI via a top navigation bar (`App.tsx`). Backend: 6 new routes under `/eval/*` — `POST /run` (starts a background eval thread), `GET /status` (progress polling), `GET /reports` (report list), `GET /report/{id}` (structured results + stats + comparison), `GET /frame/{id}/{filename}` (serves annotated JPEGs), `POST /set-baseline/{id}`. Each run executes the 5-prompt evaluation suite against the live camera using `_infer_lock` with a 15s timeout, saves annotated frames and `results.json` to `output/eval reports/report_YYYYMMDD_HHMMSS/`, and generates a comparison Markdown report vs. the current baseline. The `/report/{id}` response includes computed `stats` (avg latency, total tokens, fastest/slowest prompt) and `comparison` (per-prompt latency and token deltas vs baseline). Frontend: `EvalRunner` (control panel — Run button, progress bar, completion summary); `ReportList` (past reports with baseline badge); `ReportViewer` (summary stat chips, latency bar chart with color-coded delta bars and ghost baseline bar, per-prompt result cards with annotated frames). Column layout: 1:2 (EvalRunner : reports) so the report viewer gets more horizontal space.

Phase 13 (Prompt & Button Optimization) completed June 24, 2026. `ANALYZE_PROMPT` updated to frame output as a transportation safety inspection report covering environment, condition, personnel, and safety concerns. `INSPECT_PROMPT` updated with a structured checklist (structural damage, operational hazards, missing safety equipment, environmental conditions) and asks for issue location in the frame. All 5 evaluation prompts updated to be inspection-oriented but general enough for non-transportation scenes. Prompt labels kept unchanged to avoid orphaning existing report frame files. Eval report frames changed from annotated (Q&A overlay) to clean images — the ReportViewer cards already display the prompt and response as captions. Browser tab title changed from "frontend" to "VLM Edge".

Phase 14 (Authentication System) completed June 29, 2026. Full JWT-based auth layer added to the web interface. Backend: SQLite user database (`output/vlmedge.db`), bcrypt password hashing via passlib, PyJWT tokens with 24-hour expiry. Three new route groups: `/auth/*` (public — signup, login, me, change-password), `/admin/*` (admin-only — list users, promote, demote), all existing `/analyze`, `/inspect`, `/eval/*`, etc. routes now require Bearer token. `/stream` uses `?token=` query param since `<img src>` cannot send headers. Initial admin account seeded on first startup; credentials printed to console. Frontend: `AuthContext` with localStorage token persistence and session restore via `GET /auth/me`; `LoginPage`, `SignupPage` (with live `PasswordStrengthBar`), `AdminPage` (user table with promote/demote). Admin tab only visible to admin users. Two bugs discovered and fixed during development: (1) PyJWT 2.x requires the `sub` claim to be a string — encoding an integer user ID caused every token to fail validation silently; (2) the `vlmedge:unauthorized` event listener was only registered if a token existed at page load, meaning fresh-session users had no logout-on-401 behavior. `ANALYZE_PROMPT` also revised in this phase to remove the transportation framing — Analyze is general-purpose, Inspect is the domain-specific button. (Superseded since: `localStorage` moved to `sessionStorage` in Phase 15, and passlib was removed in favor of direct bcrypt in Phase 16.)

Phase 15 (Library, Session Management & Polish) completed June 29, 2026. Six distinct improvements shipped:

1. **Library tab** — New page visible to all authenticated users. Backend: `outputs` SQLite table logs every user action (analyze, inspect, snapshot, autoscan, record, flag) with `user_id`, `file_path`, `prompt`, `response`, `tokens`, `elapsed_s`, `timestamp`. `library_routes.py` adds four endpoints: `GET /library/outputs` (user's own), `GET /library/admin/outputs` (all users, admin only), `GET /library/view/{id}?token=` (inline file serve), `GET /library/download/{id}?token=` (attachment download). Frontend: `LibraryPage` with date sidebar navigation, action-type filter tabs, media card grid, and `PreviewModal` — inference types show image left + Q&A right; snapshot/flag show full image; record shows `<video controls>` player. Admin sidebar adds a username level above dates. Downloads restricted to snapshot, flag, record only.

2. **Video recording** — Replaced per-frame JPEG saving with `cv2.VideoWriter` using `mp4v` codec producing `.mp4` files. Frame dimensions detected dynamically from the first captured frame. Recording loop uses time-accurate pacing (`last_write` timestamp, sleep only the remaining interval) to avoid FPS drift. `writer.release()` on stop flushes and finalizes the file.

3. **Eval admin restriction** — Evaluation tab hidden in the frontend nav for non-admin users. Backend `eval_router` upgraded to `dependencies=[Depends(require_admin)]`.

4. **Eval frame images** — Fixed broken `<img src>` display in ReportViewer. Root cause: `/eval/frame/*` was inside `eval_router` which required Bearer headers — but `<img src>` cannot send them. Fix: moved the frame endpoint to a separate `eval_frame_router` with `?token=` query param validation (same pattern as `/stream`). Frontend uses `api.evalFrameUrl(path)` to append the token.

5. **Session lifetime** — JWT token moved from `localStorage` to `sessionStorage`. Closing the browser window clears the token (user must re-login). Page refresh preserves the token (session continues). Logout explicitly clears both `vlmedge_token` and `vlmedge_results` from sessionStorage.

6. **Result history persistence and correct source labels** — Dashboard result history (`results` array) stored in `sessionStorage` and restored on every mount. History survives all tab switches and page refreshes; wiped on logout or browser close. Server now stores a `source` field in `_state["last_result"]` ("Analyze", "Inspect", or "Auto-Scan") set by each calling route. Status poll uses `lr.source` instead of the previous hardcoded `"Auto-Scan"`, so results recovered after a tab switch carry the correct badge. `lastSeenTimestamp` ref seeded from stored history on mount to prevent duplicate entries.

Phase 16 (Finalizing & Hardening) started July 20, 2026. Renamed from "Future Implementations." Work done without the camera attached, so the code is complete but several gates are pending hardware. Step 0 restored the missing config.py as a committed secrets-from-env file. Task 1 added camera staleness detection and frame_hash duplicate-response diagnostics. Task 2 passes explicit inference parameters (values provisional pending the eval gate). Task 3 added the unauthenticated `/health` endpoint and fixed the launch.sh 401 wait; the interface is now web-only. Task 4 removed passlib (broken against bcrypt 5.x) in favor of direct bcrypt, moved main.py and pipeline.py to legacy/, and removed the racy session_log.json write. Task 5 fixed the sync MJPEG generator, autoscan interval math, and eval frame-capture ordering. Task 7 added a CSS-only responsive layer for phones and tablets. Task 8 stripped phase labels from source. See BRIEFING.md for per-task gate status and the full pending-gate checklist.
