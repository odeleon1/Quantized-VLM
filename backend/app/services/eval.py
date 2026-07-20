"""
Phase 8 — Evaluation script.
Runs a fixed set of prompt types on live camera frames and records
response quality and latency. Point the camera at a relevant scene
for each prompt when prompted, then press Enter to capture and infer.

Each run saves a report folder under output/eval reports/:
  report_YYYYMMDD_HHMMSS/
    report.md          — comparison vs previous run + annotated frames embedded
    scene_description.jpg
    object_list.jpg
    ...

The first run establishes the baseline (no comparison report, just the frames).
Every subsequent run generates a full comparison report.
"""

import os
import base64
import time
import json
from datetime import datetime

import cv2
import numpy as np

from app.services.capture import open_camera, capture_frame, release_camera
from app.core.config import (
    MODEL_PATH, MMPROJ_PATH, REPORTS_DIR, BASELINE_PATH, CAMERA_INDEX,
    MAX_TOKENS_ANALYZE, INFER_TEMPERATURE, INFER_REPEAT_PENALTY,
)
from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

PROMPTS = [
    (
        "Scene Description",
        "Describe this scene for an inspection report. "
        "Identify the environment, the main subjects and their visible condition, "
        "and any immediately notable features. Be specific and factual in two to three sentences.",
    ),
    (
        "Object List",
        "List the main objects, equipment, and assets visible in this image. "
        "If the condition of any item is clearly determinable, note it briefly next to each item.",
    ),
    (
        "People Count",
        "How many people are visible in this image? "
        "If any are present, note whether they appear to be wearing safety equipment "
        "such as high-visibility vests or hard hats. "
        "If no people are visible, say 'No personnel visible.'",
    ),
    (
        "Subject Appearance",
        "Describe the condition of the main subject in this image. "
        "Note any visible signs of wear, damage, corrosion, contamination, "
        "missing components, or deterioration. Be specific about what you observe.",
    ),
    (
        "Text Reading",
        "Read and report any text, labels, signs, or identification markings "
        "clearly visible in this image. "
        "If you are not certain about any text, do not guess — say 'No text clearly visible.'",
    ),
]


# ── Image annotation ──────────────────────────────────────────────────────────

def wrap_text(text, max_chars):
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + (1 if current else 0) <= max_chars:
            current += (" " if current else "") + word
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def annotate_frame(jpeg_bytes, prompt, response):
    """
    Draws a semi-transparent caption bar at the bottom of the image
    showing Q: <prompt> and A: <response>. Returns annotated JPEG bytes.
    """
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    line_h     = 22
    padding    = 10
    max_chars  = w // 10

    q_lines   = wrap_text(f"Q: {prompt}", max_chars)
    a_lines   = wrap_text(f"A: {response}", max_chars)
    all_lines = q_lines + [""] + a_lines

    bar_h = padding + len(all_lines) * line_h + padding

    overlay = img.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    for i, line in enumerate(all_lines):
        if not line:
            continue
        color = (180, 230, 255) if line.startswith("Q:") else (200, 255, 200)
        y = h - bar_h + padding + (i + 1) * line_h
        cv2.putText(img, line, (padding, y), font, font_scale, color, thickness, cv2.LINE_AA)

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes()


# ── Model ─────────────────────────────────────────────────────────────────────

def load_model():
    print("Loading model...")
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


def run_inference(
    llm,
    jpeg_bytes,
    prompt,
    max_tokens=MAX_TOKENS_ANALYZE,
    temperature=INFER_TEMPERATURE,
    repeat_penalty=INFER_REPEAT_PENALTY,
):
    """Run one image plus prompt inference.

    Sampling parameters are passed explicitly rather than inherited from the
    handler defaults. max_tokens caps generation length, which is the main
    latency control in this generation-bound pipeline.
    """
    img_b64 = base64.b64encode(jpeg_bytes).decode()
    t0 = time.time()
    response = llm.create_chat_completion(
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": prompt},
            ],
        }],
        max_tokens=max_tokens,
        temperature=temperature,
        repeat_penalty=repeat_penalty,
    )
    elapsed = time.time() - t0
    text = response["choices"][0]["message"]["content"].strip()
    tokens = response["usage"]["completion_tokens"]
    llm.reset()
    return text, tokens, elapsed


# ── Report generation ─────────────────────────────────────────────────────────

def generate_report(previous, current, report_dir):
    prev_results = previous.get("results") or []
    curr_results = current.get("results") or []
    if not prev_results or not curr_results:
        return generate_baseline_report(current, report_dir)

    prev_map = {r["label"]: r for r in prev_results}
    curr_map = {r["label"]: r for r in curr_results}

    avg_prev  = sum(r["latency_s"] for r in prev_results) / len(prev_results)
    avg_curr  = sum(r["latency_s"] for r in curr_results) / len(curr_results)
    avg_delta = avg_curr - avg_prev
    direction = "faster" if avg_delta < 0 else "slower"

    lines = []
    lines += [
        "# Evaluation Comparison Report",
        "",
        f"- **Previous run:** {previous['timestamp']}",
        f"- **Current run:**  {current['timestamp']}",
        "",
        "---",
        "",
        "## Overall Latency",
        "",
        "| | Avg latency |",
        "|---|---|",
        f"| Previous | {avg_prev:.2f}s |",
        f"| Current  | {avg_curr:.2f}s |",
        f"| Delta    | {avg_delta:+.2f}s ({abs(avg_delta):.2f}s {direction}) |",
        "",
        "---",
        "",
        "## Per-Prompt Results",
        "",
    ]

    for label, prompt in PROMPTS:
        prev = prev_map.get(label)
        curr = curr_map.get(label)
        if not prev or not curr:
            continue

        delta = curr["latency_s"] - prev["latency_s"]
        lines += [
            f"### {label}",
            "",
            f"**Prompt:** {prompt}",
            "",
            "| | Latency | Tokens | Response |",
            "|---|---|---|---|",
            f"| Previous | {prev['latency_s']:.2f}s | {prev['tokens']} | {prev['response']} |",
            f"| Current  | {curr['latency_s']:.2f}s | {curr['tokens']} | {curr['response']} |",
            f"| Delta    | {delta:+.2f}s | {curr['tokens'] - prev['tokens']:+d} | |",
            "",
            f"![{label}]({label}.jpg)",
            "",
        ]

    report_path = os.path.join(report_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


def generate_baseline_report(current, report_dir):
    """Minimal report for the first run — no comparison, just current results + frames."""
    lines = [
        "# Evaluation — Baseline Run",
        "",
        f"- **Run timestamp:** {current['timestamp']}",
        "",
        "> This is the first run. No previous results to compare against.",
        "> The next run will generate a full comparison report.",
        "",
        "---",
        "",
        "## Results",
        "",
    ]

    for r in current["results"]:
        lines += [
            f"### {r['label']}",
            "",
            f"**Prompt:** {r['prompt']}",
            "",
            f"| Latency | Tokens | Response |",
            f"|---|---|---|",
            f"| {r['latency_s']:.2f}s | {r['tokens']} | {r['response']} |",
            "",
            f"![{r['label']}]({r['label']}.jpg)",
            "",
        ]

    report_path = os.path.join(report_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    previous = None
    if os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH) as f:
            previous = json.load(f)
        print(f"Previous run found: {previous['timestamp']}")
        print("This run will generate a comparison report.\n")
    else:
        print("No previous run found — this will be the baseline.\n")

    print("=== Phase 8 — Evaluation ===\n")
    print("For each test, set up the scene, then press Enter to capture and run.\n")
    print("-" * 60)

    llm = load_model()
    cap = open_camera(CAMERA_INDEX)
    print()

    timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_dir = os.path.join(REPORTS_DIR, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(report_dir, exist_ok=True)

    results = []

    for label, prompt in PROMPTS:
        print(f"\n[{label}]")
        print(f"  Prompt : \"{prompt}\"")
        input("  Set up scene and press Enter to capture...")

        jpeg_bytes = capture_frame(cap)

        print("  Running inference...", end=" ", flush=True)
        text, tokens, elapsed = run_inference(llm, jpeg_bytes, prompt)

        print(f"{elapsed:.1f}s | {tokens} tokens")
        print(f"  Response: {text}")

        # Save annotated frame directly into the report folder
        frame_path = os.path.join(report_dir, f"{label}.jpg")
        with open(frame_path, "wb") as f:
            f.write(annotate_frame(jpeg_bytes, prompt, text))

        results.append({
            "label":     label,
            "prompt":    prompt,
            "response":  text,
            "tokens":    tokens,
            "latency_s": round(elapsed, 2),
        })

    release_camera(cap)

    current = {"timestamp": timestamp, "results": results}

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<22} {'Latency':>8}  {'Tokens':>6}  Response")
    print("-" * 60)
    for r in results:
        preview = r["response"][:45] + "…" if len(r["response"]) > 45 else r["response"]
        print(f"{r['label']:<22} {r['latency_s']:>7.1f}s  {r['tokens']:>6}  {preview}")

    # Generate report
    if previous:
        report_path = generate_report(previous, current, report_dir)
    else:
        report_path = generate_baseline_report(current, report_dir)

    # Save current run as new baseline
    with open(BASELINE_PATH, "w") as f:
        json.dump(current, f, indent=2)

    print(f"\nReport saved to {report_dir}/")
    print("\n=== Evaluation complete ===")


if __name__ == "__main__":
    main()
