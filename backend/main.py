"""
VLM Interface — desktop application for the Jetson Orin Nano VLM project.

Tab 1 (Live): live camera feed + chat — ask questions, get answers about the current frame.
Tab 2 (Evaluation): browse past eval reports and run new evaluations.

Future specialized tabs (robotics, inspection, etc.) are added to this same window.
"""

import os
import sys
import time
import threading
from datetime import datetime

import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QTextBrowser, QListWidget, QFrame,
    QSplitter, QDialog, QSizePolicy, QStatusBar,
    QStyleOptionTab, QStyle, QTabBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize, QRect
from PyQt5.QtGui import QImage, QPixmap, QPalette, QColor, QPainter

from app.services.capture import open_camera, capture_frame, release_camera
from app.services.eval import (
    load_model, run_inference, annotate_frame,
    generate_report, generate_baseline_report,
    PROMPTS, REPORTS_DIR, BASELINE_PATH,
)
from app.core.config import RUNS_DIR

CAMERA_IDX = 0
VIDEO_FPS  = 30


# ── Design tokens ─────────────────────────────────────────────────────────────

_C = {
    "bg":      "#09091b",
    "card":    "#0f0f23",
    "card2":   "#1a1a35",
    "border":  "#1e2040",
    "border2": "#2e3a5a",
    "blue":    "#2A7DE1",
    "blue_h":  "#4a97f5",
    "blue_p":  "#1a5db5",
    "blue_d":  "#0a1a32",
    "red":     "#E8453C",
    "red_h":   "#f55f57",
    "amber":   "#F7A928",
    "green":   "#1DB954",
    "text":    "#ffffff",
    "text2":   "#8899aa",
    "text3":   "#44455a",
}

# Reusable card gradient string (no %-substitution needed — no % chars in value)
_CARD_GRAD = "qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f0f23, stop:1 #1a1a35)"
_CARD_SS   = (
    "background: %s;"
    "border: 1px solid rgba(255,255,255,15);"
    "border-radius: 14px;" % _CARD_GRAD
)

# Global stylesheet — written with literal hex values to avoid %/gradient escaping issues
APP_SS = """
QWidget {
    background: #09091b;
    color: #ffffff;
    font-family: 'Ubuntu', 'DejaVu Sans', 'Segoe UI', sans-serif;
    font-size: 13px;
}
QLabel  { background: transparent; }

QTabWidget::pane { border: none; background: #09091b; }

/* ── Buttons ── */
QPushButton {
    background: #0f0f23;
    color: #8899aa;
    border: 1px solid #2e3a5a;
    border-radius: 8px;
    padding: 0 16px;
    font-size: 13px;
}
QPushButton:hover   { background: #0a1a32; border-color: #2A7DE1; color: #ffffff; }
QPushButton:pressed { background: #1a5db5; border-color: #4a97f5; color: #fff; }
QPushButton:disabled { background: #0c0c1e; color: #2e3a5a; border-color: #1e2040; }

QPushButton#primary {
    background: #2A7DE1;
    color: #fff;
    border: 1px solid #4a97f5;
    font-weight: bold;
    border-radius: 8px;
}
QPushButton#primary:hover    { background: #4a97f5; border-color: #70b7ff; }
QPushButton#primary:pressed  { background: #1a5db5; }
QPushButton#primary:disabled { background: #091428; color: #1a2e4a; border-color: #0d1e38; }

QPushButton#capture {
    background: #0a2a18;
    color: #1DB954;
    border: 1px solid #1DB954;
    font-weight: bold;
    border-radius: 8px;
}
QPushButton#capture:hover   { background: #1DB954; border-color: #3de870; color: #fff; }
QPushButton#capture:pressed { background: #148040; }

/* ── Inputs ── */
QLineEdit {
    background: #0f0f23;
    color: #ffffff;
    border: 1px solid #2e3a5a;
    border-radius: 8px;
    padding: 0 12px;
    selection-background-color: #2A7DE1;
}
QLineEdit:focus    { border-color: #2A7DE1; background: #0a1a32; }
QLineEdit:disabled { background: #0c0c1e; color: #44455a; border-color: #1e2040; }

/* ── Text browsers ── */
QTextBrowser {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f0f23, stop:1 #1a1a35);
    color: #ffffff;
    border: 1px solid rgba(255,255,255,15);
    border-radius: 14px;
    selection-background-color: #2A7DE1;
}

/* ── List widget ── */
QListWidget {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f0f23, stop:1 #1a1a35);
    color: #8899aa;
    border: 1px solid rgba(255,255,255,15);
    border-radius: 14px;
    outline: none;
}
QListWidget::item {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(255,255,255,8);
    font-size: 12px;
    font-family: 'Ubuntu Mono', 'DejaVu Sans Mono', monospace;
}
QListWidget::item:hover    { background: #0a1a32; color: #ffffff; }
QListWidget::item:selected { background: #2A7DE1; color: #fff; }

/* ── Splitter ── */
QSplitter::handle              { background: rgba(255,255,255,8); }
QSplitter::handle:horizontal   { width: 2px; }
QSplitter::handle:hover        { background: #2A7DE1; }

/* ── Scrollbars ── */
QScrollBar:vertical            { background: #09091b; width: 7px; border-radius: 3px; margin: 0; }
QScrollBar::handle:vertical    { background: #2e3a5a; border-radius: 3px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: #44455a; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal          { background: #09091b; height: 7px; border-radius: 3px; }
QScrollBar::handle:horizontal  { background: #2e3a5a; border-radius: 3px; min-width: 28px; }
QScrollBar::handle:horizontal:hover { background: #44455a; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ── Status bar ── */
QStatusBar {
    background: #0f0f23;
    border-top: 1px solid rgba(255,255,255,10);
    font-size: 12px;
    padding: 2px 0;
}

/* ── Dialog ── */
QDialog        { background: #0f0f23; }
QDialog QLabel { color: #ffffff; background: transparent; }
"""


# ── Left-side tab bar ─────────────────────────────────────────────────────────

class LeftTabBar(QTabBar):
    """Draws tabs on the left with horizontal (non-rotated) text."""
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
            rect    = opt.rect
            selected = bool(opt.state & QStyle.State_Selected)
            hovered  = bool(opt.state & QStyle.State_MouseOver)

            if selected:
                painter.fillRect(rect, QColor("#0f0f23"))
            elif hovered:
                painter.fillRect(rect, QColor("#0c0c20"))
            else:
                painter.fillRect(rect, QColor("#09091b"))

            # Right-edge blue accent for selected
            if selected:
                painter.fillRect(
                    QRect(rect.right() - 2, rect.top() + 12, 3, rect.height() - 24),
                    QColor(_C["blue"])
                )

            if selected:
                painter.setPen(QColor(_C["text"]))
            elif hovered:
                painter.setPen(QColor(_C["text2"]))
            else:
                painter.setPen(QColor(_C["text3"]))

            painter.drawText(rect, Qt.AlignCenter, self.tabText(i).strip())
        painter.end()


# ── Threads ───────────────────────────────────────────────────────────────────

class ModelLoader(QThread):
    finished = pyqtSignal(object)

    def run(self):
        self.finished.emit(load_model())


class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self, cap):
        super().__init__()
        self.cap     = cap
        self.paused  = False
        self.running = True
        self._latest_jpeg = None
        self._lock        = threading.Lock()

    def run(self):
        interval = 1.0 / VIDEO_FPS
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            with self._lock:
                self._latest_jpeg = buf.tobytes()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            # rgb.data (memoryview) stays valid while rgb is in scope;
            # .copy() makes a deep copy before rgb can be freed
            qi = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
            self.frame_ready.emit(qi)
            time.sleep(interval)

    def get_latest_jpeg(self):
        with self._lock:
            return self._latest_jpeg

    def stop(self):
        self.running = False
        self.wait()


class InferenceWorker(QThread):
    result = pyqtSignal(str, int, float)

    def __init__(self, llm, jpeg_bytes, prompt):
        super().__init__()
        self.llm        = llm
        self.jpeg_bytes = jpeg_bytes
        self.prompt     = prompt

    def run(self):
        text, tokens, elapsed = run_inference(self.llm, self.jpeg_bytes, self.prompt)
        self.result.emit(text, tokens, elapsed)


class EvalWorker(QThread):
    prompt_ready = pyqtSignal(str, str)
    prompt_done  = pyqtSignal(str, str, int, float)
    finished     = pyqtSignal()

    def __init__(self, llm, cap, video_thread):
        super().__init__()
        self.llm          = llm
        self.cap          = cap
        self.video_thread = video_thread
        self._event       = threading.Event()
        self._jpeg        = None

    def set_captured_frame(self, jpeg_bytes):
        self._jpeg = jpeg_bytes
        self._event.set()

    def run(self):
        import json as _json

        os.makedirs(REPORTS_DIR, exist_ok=True)
        previous = None
        if os.path.exists(BASELINE_PATH):
            with open(BASELINE_PATH) as f:
                previous = _json.load(f)

        timestamp  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_dir = os.path.join(
            REPORTS_DIR,
            f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        os.makedirs(report_dir, exist_ok=True)
        results = []

        for label, prompt in PROMPTS:
            self.video_thread.paused = True
            time.sleep(0.15)

            self._event.clear()
            self._jpeg = None
            self.prompt_ready.emit(label, prompt)
            self._event.wait()

            jpeg_bytes = self._jpeg
            self.video_thread.paused = False

            text, tokens, elapsed = run_inference(self.llm, jpeg_bytes, prompt)

            frame_path = os.path.join(report_dir, f"{label}.jpg")
            with open(frame_path, "wb") as f:
                f.write(annotate_frame(jpeg_bytes, prompt, text))

            results.append({
                "label": label, "prompt": prompt,
                "response": text, "tokens": tokens,
                "latency_s": round(elapsed, 2),
            })
            self.prompt_done.emit(label, text, tokens, elapsed)

        current = {"timestamp": timestamp, "results": results}
        if previous:
            generate_report(previous, current, report_dir)
        else:
            generate_baseline_report(current, report_dir)

        with open(BASELINE_PATH, "w") as f:
            _json.dump(current, f, indent=2)

        self.finished.emit()


# ── Capture dialog ────────────────────────────────────────────────────────────

class CaptureDialog(QDialog):
    def __init__(self, label, prompt, cap, parent=None):
        super().__init__(parent)
        self.cap         = cap
        self.result_jpeg = None

        self.setWindowTitle("Evaluation — Capture Frame")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(16)

        # Red→amber gradient accent bar (matches the reference component's top stripe)
        accent = QFrame()
        accent.setFixedHeight(3)
        accent.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0,"
            " stop:0 #E8453C, stop:1 #F7A928); border-radius: 2px;"
        )
        layout.addWidget(accent)

        title = QLabel(label.replace("_", " ").title())
        title.setStyleSheet(
            "color: #ffffff; font-size: 18px; font-weight: 800;"
            " letter-spacing: -0.3px; padding-top: 4px; background: transparent;"
        )
        layout.addWidget(title)

        prompt_label = QLabel(prompt)
        prompt_label.setWordWrap(True)
        prompt_label.setStyleSheet(
            "color: #8899aa; font-size: 13px; background: transparent;"
        )
        layout.addWidget(prompt_label)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setStyleSheet("background: rgba(255,255,255,15); max-height: 1px; border: none;")
        layout.addWidget(divider)

        instruction = QLabel("Point the camera at the scene, then click Capture.")
        instruction.setStyleSheet("color: #44455a; font-size: 12px; background: transparent;")
        layout.addWidget(instruction)

        layout.addSpacing(6)

        btn = QPushButton("  Capture Frame")
        btn.setObjectName("capture")
        btn.setFixedHeight(44)
        btn.clicked.connect(self._capture)
        layout.addWidget(btn)

    def _capture(self):
        self.result_jpeg = capture_frame(self.cap, flush=5)
        self.accept()


# ── Live tab ──────────────────────────────────────────────────────────────────

class LiveTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.llm               = None
        self._messages         = []   # list of (kind, text): "user" | "ai" | "thinking"
        self._current_question = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        # ── Top bar: section label + LIVE dot ──
        top_row = QHBoxLayout()
        top_row.setContentsMargins(4, 0, 6, 0)

        section = QLabel("CAMERA FEED")
        section.setStyleSheet(
            "color: #E8453C; font-size: 10px; font-weight: 800;"
            " letter-spacing: 2.5px; background: transparent;"
        )
        self._live_dot = QLabel("● LIVE")
        self._live_dot.setStyleSheet(
            "color: #E8453C; font-size: 10px; font-weight: bold; background: transparent;"
        )
        self._live_on = True
        self._live_timer = QTimer(self)
        self._live_timer.timeout.connect(self._pulse_live)
        self._live_timer.start(900)

        top_row.addWidget(section)
        top_row.addStretch()
        top_row.addWidget(self._live_dot)
        layout.addLayout(top_row)

        # ── Video card (QFrame with gradient bg + drop shadow) ──
        self._video_card = QFrame()
        self._video_card.setStyleSheet("QFrame { " + _CARD_SS + " }")
        vc_layout = QVBoxLayout(self._video_card)
        vc_layout.setContentsMargins(2, 2, 2, 2)

        self.video_label = QLabel("Starting camera…")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(280)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setStyleSheet(
            "background: transparent; color: #44455a; font-size: 12px; border-radius: 12px;"
        )
        vc_layout.addWidget(self.video_label)
        layout.addWidget(self._video_card, stretch=3)

        # ── Chat section label ──
        chat_row = QHBoxLayout()
        chat_row.setContentsMargins(4, 0, 0, 0)
        chat_lbl = QLabel("CONVERSATION")
        chat_lbl.setStyleSheet(
            "color: #2A7DE1; font-size: 10px; font-weight: 800;"
            " letter-spacing: 2.5px; background: transparent;"
        )
        chat_row.addWidget(chat_lbl)
        chat_row.addStretch()
        layout.addLayout(chat_row)

        # ── Chat history ──
        self.chat = QTextBrowser()
        self.chat.setMinimumHeight(110)
        layout.addWidget(self.chat, stretch=2)

        # ── Input row ──
        row = QHBoxLayout()
        row.setSpacing(8)
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask a question about what the camera sees…")
        self.input.setEnabled(False)
        self.input.setFixedHeight(40)
        self.input.returnPressed.connect(self.submit)
        self.send_btn = QPushButton("Ask")
        self.send_btn.setObjectName("primary")
        self.send_btn.setEnabled(False)
        self.send_btn.setFixedHeight(40)
        self.send_btn.setFixedWidth(80)
        self.send_btn.clicked.connect(self.submit)
        row.addWidget(self.input)
        row.addWidget(self.send_btn)
        layout.addLayout(row)

    def _pulse_live(self):
        self._live_on = not self._live_on
        c = "#E8453C" if self._live_on else "#2a0a08"
        self._live_dot.setStyleSheet(
            "color: %s; font-size: 10px; font-weight: bold; background: transparent;" % c
        )

    def set_model(self, llm):
        self.llm = llm
        self.input.setEnabled(True)
        self.send_btn.setEnabled(True)

    def update_frame(self, qi: QImage):
        if self.video_label.text():
            self.video_label.setText("")
        pixmap = QPixmap.fromImage(qi)
        scaled = pixmap.scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled)

    def submit(self):
        question = self.input.text().strip()
        if not question or not self.llm:
            return
        jpeg = self.window().video_thread.get_latest_jpeg()
        if jpeg is None:
            return
        self._current_question = question
        self._pending_jpeg     = jpeg
        self.input.clear()
        self._set_input_enabled(False)
        self._append("user", question)
        self._append("thinking", "")

        self._worker = InferenceWorker(self.llm, jpeg, question)
        self._worker.result.connect(self._on_result)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.start()

    def _on_result(self, text, tokens, elapsed):
        meta = (
            "<br><span style='color: #44455a; font-size: 11px;'>"
            "%.1fs &nbsp;·&nbsp; %d tok</span>" % (elapsed, tokens)
        )
        self._replace_thinking(text + meta)
        self._set_input_enabled(True)
        self.input.setFocus()
        os.makedirs(RUNS_DIR, exist_ok=True)
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RUNS_DIR, "frame_%s.jpg" % ts)
        with open(path, "wb") as f:
            f.write(annotate_frame(self._pending_jpeg, self._current_question, text))

    def _set_input_enabled(self, enabled):
        self.input.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)

    def _append(self, kind, text):
        self._messages.append((kind, text))
        self._rebuild()

    def _replace_thinking(self, text):
        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i][0] == "thinking":
                self._messages[i] = ("ai", text)
                break
        self._rebuild()

    def _rebuild(self):
        parts = ["<body style='background: #0f0f23; margin: 0; padding: 8px 4px;'>"]
        for kind, text in self._messages:
            if kind == "user":
                parts.append(
                    "<div style='text-align: right; margin: 6px 8px 6px 60px;'>"
                    "<span style='display: inline-block; background: #2A7DE1; color: #fff;"
                    " padding: 10px 16px; border-radius: 18px 18px 4px 18px;"
                    " font-size: 13px; line-height: 1.5; max-width: 90%;'>"
                    "<b>You</b><br>" + text + "</span></div>"
                )
            elif kind == "thinking":
                parts.append(
                    "<div style='text-align: left; margin: 6px 60px 6px 8px;'>"
                    "<span style='display: inline-block; background: #0a0a18; color: #44455a;"
                    " padding: 10px 16px; border-radius: 18px 18px 18px 4px;"
                    " border-left: 3px solid #2e3a5a; font-size: 13px; font-style: italic;'>"
                    "Thinking…</span></div>"
                )
            else:  # ai
                parts.append(
                    "<div style='text-align: left; margin: 6px 60px 6px 8px;'>"
                    "<span style='display: inline-block; background: #0d0d20; color: #ffffff;"
                    " padding: 10px 16px; border-radius: 18px 18px 18px 4px;"
                    " border-left: 3px solid #F7A928; font-size: 13px; line-height: 1.5; max-width: 90%;'>"
                    "<b style='color: #F7A928'>AI</b><br>" + text + "</span></div>"
                )
        parts.append("</body>")
        self.chat.setHtml("".join(parts))
        self.chat.verticalScrollBar().setValue(
            self.chat.verticalScrollBar().maximum()
        )


# ── Eval tab ──────────────────────────────────────────────────────────────────

def markdown_to_html(md: str) -> str:
    import re
    import html as h

    def _inline(text):
        """Apply bold and code-span markers to already-HTML-escaped text."""
        text = re.sub(r"\*\*(.*?)\*\*", r"<b style='color: #e0eaff'>\1</b>", text)
        text = re.sub(
            r"`(.*?)`",
            r"<code style='background: #0a0a18; color: #1DB954;"
            r" padding: 2px 6px; border-radius: 4px; font-size: 12px;'>\1</code>",
            text,
        )
        return text

    lines     = md.split("\n")
    out       = []
    in_table  = False
    table_row = 0
    in_list   = False

    for line in lines:
        is_list  = line.startswith("- ")
        is_table = line.startswith("|")

        # Close any open block when switching to a different line type
        if not is_list and in_list:
            out.append("</ul>")
            in_list = False
        if not is_table and in_table:
            out.append("</table>")
            in_table  = False
            table_row = 0

        if line.startswith("### "):
            out.append(
                "<h3 style='color: #F7A928; margin: 14px 0 5px; font-size: 14px;'>%s</h3>"
                % _inline(h.escape(line[4:]))
            )
        elif line.startswith("## "):
            out.append(
                "<h2 style='color: #ffffff; margin: 18px 0 8px; font-size: 16px;'>%s</h2>"
                % _inline(h.escape(line[3:]))
            )
        elif line.startswith("# "):
            out.append(
                "<h1 style='color: #ffffff; margin: 0 0 10px; font-size: 20px;"
                " font-weight: 800; letter-spacing: -0.3px;'>%s</h1>"
                % _inline(h.escape(line[2:]))
            )
        elif line.strip() == "---":
            out.append("<hr style='border: none; border-top: 1px solid rgba(255,255,255,12); margin: 14px 0'>")
        elif line.startswith("> "):
            out.append(
                "<blockquote style='color: #8899aa; border-left: 3px solid #2e3a5a;"
                " margin: 8px 0; padding: 8px 12px; background: #0a0a18;"
                " border-radius: 0 6px 6px 0; font-size: 12px;'>%s</blockquote>"
                % _inline(h.escape(line[2:]))
            )
        elif re.match(r"!\[.*?\]\(.*?\)", line):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", line)
            if m:
                out.append(
                    "<img src='%s'"
                    " style='max-width: 100%%; margin: 10px 0; border-radius: 8px;"
                    " border: 1px solid rgba(255,255,255,12); display: block'><br>"
                    % h.escape(m.group(2))
                )
        elif is_table:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if all(set(c) <= set("-: ") for c in cells):
                continue  # separator row — skip but don't increment table_row
            if not in_table:
                out.append(
                    "<table style='border-collapse: collapse; margin: 10px 0;"
                    " width: 100%; font-size: 12px;'>"
                )
                in_table  = True
                table_row = 0
            if table_row == 0:
                row = "".join(
                    "<th style='padding: 8px 14px; border: 1px solid rgba(255,255,255,10);"
                    " color: #e0eaff; font-weight: bold; text-align: left;'>%s</th>"
                    % _inline(h.escape(c))
                    for c in cells
                )
            else:
                row = "".join(
                    "<td style='padding: 8px 14px; border: 1px solid rgba(255,255,255,10);"
                    " color: #8899aa'>%s</td>"
                    % _inline(h.escape(c))
                    for c in cells
                )
            out.append("<tr>%s</tr>" % row)
            table_row += 1
        elif is_list:
            if not in_list:
                out.append("<ul style='margin: 6px 0 6px 16px; padding: 0;'>")
                in_list = True
            out.append(
                "<li style='color: #8899aa; margin: 4px 0; padding-left: 4px'>%s</li>"
                % _inline(h.escape(line[2:]))
            )
        elif line.strip() == "":
            out.append("<br>")
        else:
            out.append("<p style='margin: 4px 0; color: #8899aa'>%s</p>" % _inline(h.escape(line)))

    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


class EvalTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.llm          = None
        self._eval_worker = None

        # Zero-margin layout — splitter fills the entire tab area
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)

        # ── Left panel: section label + report list ──
        left = QWidget()
        left.setStyleSheet("background: #09091b;")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14, 14, 8, 14)
        ll.setSpacing(8)

        section = QLabel("REPORTS")
        section.setStyleSheet(
            "color: #F7A928; font-size: 10px; font-weight: 800;"
            " letter-spacing: 2.5px; background: transparent;"
        )
        ll.addWidget(section)

        self.report_list = QListWidget()
        self.report_list.currentTextChanged.connect(self._load_report)
        ll.addWidget(self.report_list)
        splitter.addWidget(left)

        # ── Right panel: report viewer + run button ──
        right = QWidget()
        right.setStyleSheet("background: #09091b;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 14, 14, 14)
        rl.setSpacing(10)

        self.viewer = QTextBrowser()
        self.viewer.setOpenLinks(False)
        rl.addWidget(self.viewer, stretch=1)

        self.run_btn = QPushButton("  Run New Evaluation")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.setFixedHeight(44)
        self.run_btn.clicked.connect(self._run_eval)
        rl.addWidget(self.run_btn)

        splitter.addWidget(right)
        splitter.setSizes([240, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        self.refresh_reports()

    def set_model(self, llm):
        self.llm = llm
        self.run_btn.setEnabled(True)

    def refresh_reports(self):
        self.report_list.clear()
        if not os.path.exists(REPORTS_DIR):
            return
        folders = sorted(
            [d for d in os.listdir(REPORTS_DIR) if d.startswith("report_")],
            reverse=True,
        )
        for f in folders:
            self.report_list.addItem(f)
        if self.report_list.count():
            self.report_list.setCurrentRow(0)

    def _load_report(self, folder):
        if not folder:
            return
        report_path = os.path.join(REPORTS_DIR, folder, "report.md")
        if not os.path.exists(report_path):
            return
        with open(report_path) as f:
            md = f.read()
        html = markdown_to_html(md)
        report_folder = os.path.abspath(os.path.join(REPORTS_DIR, folder))
        self.viewer.setSearchPaths([report_folder])
        self.viewer.setHtml(
            "<body style='background: #0f0f23; padding: 16px 20px;'>%s</body>" % html
        )

    def _run_eval(self):
        win = self.window()
        # Block live-tab inference so both paths can't call llm simultaneously
        win.live_tab._set_input_enabled(False)
        self.run_btn.setEnabled(False)
        self._eval_worker = EvalWorker(self.llm, win.cap, win.video_thread)
        self._eval_worker.prompt_ready.connect(self._on_prompt_ready)
        self._eval_worker.finished.connect(self._on_eval_done)
        self._eval_worker.start()

    def _on_prompt_ready(self, label, prompt):
        dialog = CaptureDialog(label, prompt, self.window().cap, self)
        if dialog.exec_() == QDialog.Accepted and dialog.result_jpeg:
            self._eval_worker.set_captured_frame(dialog.result_jpeg)
        else:
            self._eval_worker.set_captured_frame(b"")

    def _on_eval_done(self):
        self.window().video_thread.paused = False
        self.run_btn.setEnabled(True)
        live = self.window().live_tab
        live._set_input_enabled(live.llm is not None)
        self.refresh_reports()


# ── Main window ───────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLM Interface")
        self.resize(980, 740)

        self.llm = None
        self.cap = open_camera(CAMERA_IDX)

        self.tabs     = QTabWidget()
        self.tabs.setTabBar(LeftTabBar())
        self.tabs.setTabPosition(QTabWidget.West)
        self.live_tab = LiveTab()
        self.eval_tab = EvalTab()
        self.tabs.addTab(self.live_tab, "Live")
        self.tabs.addTab(self.eval_tab, "Evaluation")
        self.setCentralWidget(self.tabs)

        # ── Status bar ──
        status = QStatusBar()
        status.setSizeGripEnabled(False)
        self.setStatusBar(status)

        self._status_label = QLabel("  ⧗  Loading model…  (~10s)")
        self._status_label.setStyleSheet(
            "color: #F7A928; font-size: 12px; padding: 0 8px; font-weight: bold;"
        )
        status.addWidget(self._status_label)

        # ── Threads ──
        self.video_thread = VideoThread(self.cap)
        self.video_thread.frame_ready.connect(self.live_tab.update_frame)
        self.video_thread.start()

        self._loader = ModelLoader()
        self._loader.finished.connect(self._on_model_ready)
        self._loader.start()

    def _on_model_ready(self, llm):
        self.llm = llm
        self.live_tab.set_model(llm)
        self.eval_tab.set_model(llm)
        self._status_label.setText(
            "  ●  Moondream2 Q4_K_M  ·  25 layers GPU  ·  Ready"
        )
        self._status_label.setStyleSheet(
            "color: #1DB954; font-size: 12px; padding: 0 8px; font-weight: bold;"
        )

    def closeEvent(self, event):
        # Disconnect model loader so its finished signal can't fire into a
        # partially-destroyed window if it completes after teardown starts.
        try:
            self._loader.finished.disconnect()
        except RuntimeError:
            pass

        # If EvalWorker is blocked waiting for a capture dialog that will
        # never appear, unblock it so it exits cleanly before we release
        # the camera it holds a reference to.
        ew = self.eval_tab._eval_worker
        if ew is not None and ew.isRunning():
            ew.set_captured_frame(b"")
            ew.wait(3000)  # give it up to 3 s to finish current inference

        self.video_thread.stop()
        release_camera(self.cap)
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def _dark_palette():
    p = QPalette()
    bg   = QColor(9,  9,  27)   # #09091b
    surf = QColor(15, 15, 35)   # #0f0f23
    btn  = QColor(26, 26, 53)   # #1a1a35
    text = QColor(255, 255, 255)
    hi   = QColor(42, 125, 225) # #2A7DE1
    p.setColor(QPalette.Window,          surf)
    p.setColor(QPalette.WindowText,      text)
    p.setColor(QPalette.Base,            bg)
    p.setColor(QPalette.AlternateBase,   btn)
    p.setColor(QPalette.Text,            text)
    p.setColor(QPalette.Button,          btn)
    p.setColor(QPalette.ButtonText,      text)
    p.setColor(QPalette.Highlight,       hi)
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.ToolTipBase,     surf)
    p.setColor(QPalette.ToolTipText,     text)
    return p


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(_dark_palette())
    app.setStyleSheet(APP_SS)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
