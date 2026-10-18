from __future__ import annotations

import html
import os
import socket
import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# IMPORTANT FIX FOR pythonw.exe
# ----------------------------------------------------------------------
# When a Windows GUI program is started with pythonw.exe, sys.stdout and
# sys.stderr can be None. Uvicorn's logging formatter calls .isatty() on
# these streams, which causes:
#
#   AttributeError: 'NoneType' object has no attribute 'isatty'
#   ValueError: Unable to configure formatter 'default'
#
# Give GUI-mode Python real file-like streams before Gradio/Uvicorn starts.
# We write them to a log file so startup/inference errors remain inspectable.
# ----------------------------------------------------------------------
LOG_PATH = ROOT / "tg_unet_desktop.log"

_log_stream = open(
    LOG_PATH,
    "a",
    encoding="utf-8",
    buffering=1,
)

if sys.stdout is None:
    sys.stdout = _log_stream

if sys.stderr is None:
    sys.stderr = _log_stream

if sys.stdin is None:
    try:
        sys.stdin = open(os.devnull, "r", encoding="utf-8")
    except Exception:
        pass

print()
print("=" * 88)
print(time.strftime("%Y-%m-%d %H:%M:%S"))
print("Starting TG-UNet embedded Gradio desktop shell")
print(f"Log file: {LOG_PATH}")
print("=" * 88)

# Keep the exact same offline model behavior as the current project.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QObject, QThread, QUrl, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QStatusBar,
)
from PySide6.QtWebEngineWidgets import QWebEngineView

_GRADIO_DEMO = None

LOADING_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Loading</title>
<style>
  html,body{
    height:100%;margin:0;
    background:#0f172a;color:#e5e7eb;
    font-family:"Segoe UI","Microsoft YaHei UI",sans-serif;
  }
  .wrap{
    min-height:100%;display:flex;align-items:center;justify-content:center;
    padding:40px;box-sizing:border-box;
  }
  .card{
    width:min(760px,92vw);
    border:1px solid #263244;border-radius:18px;
    background:#111827;padding:34px 38px;box-sizing:border-box;
  }
  h1{font-size:28px;margin:0 0 10px;color:#f8fafc}
  p{margin:7px 0;color:#94a3b8;font-size:15px;line-height:1.6}
  .row{display:flex;align-items:center;gap:15px;margin-top:24px}
  .spinner{
    width:28px;height:28px;border-radius:50%;
    border:3px solid #334155;border-top-color:#3b82f6;
    animation:spin 1s linear infinite;
  }
  .status{color:#dbeafe;font-weight:600}
  .note{margin-top:18px;font-size:13px;color:#64748b}
  @keyframes spin{to{transform:rotate(360deg)}}
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>Text-Guided Photovoltaic Defect Segmentation System</h1>
    <p>General Model + Presence Gate v2 + Expert Routing</p>
    <div class="row">
      <div class="spinner"></div>
      <div>
        <div class="status">Loading models and starting the local interface…</div>
        <p>The application will enter the detection interface automatically.</p>
      </div>
    </div>
    <div class="note">No external browser will be opened.</div>
  </div>
</div>
</body>
</html>
"""


def _error_html(message: str, details: str) -> str:
    return f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
html,body{{margin:0;background:#0f172a;color:#e5e7eb;
font-family:"Segoe UI","Microsoft YaHei UI",sans-serif}}
.wrap{{padding:35px;max-width:1000px;margin:auto}}
.card{{background:#111827;border:1px solid #7f1d1d;border-radius:14px;padding:26px}}
h2{{color:#fecaca;margin-top:0}}
pre{{white-space:pre-wrap;word-break:break-word;background:#0b1220;
border:1px solid #334155;border-radius:8px;padding:16px;color:#cbd5e1}}
</style>
</head>
<body>
<div class="wrap"><div class="card">
<h2>Application startup failed</h2>
<p>{html.escape(message)}</p>
<pre>{html.escape(details)}</pre>
<p>Log file: {html.escape(str(LOG_PATH))}</p>
</div></div>
</body>
</html>
"""


def _choose_port() -> int:
    for port in range(7860, 7881):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.08)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_until_ready(url: str, timeout: float = 180.0) -> None:
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        try:
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "TG-UNet-Desktop"},
            )
            with urllib.request.urlopen(request, timeout=1.0) as response:
                if 200 <= int(response.status) < 500:
                    return
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)

    raise TimeoutError(
        f"Local Gradio server did not become ready in {timeout:.0f}s. "
        f"Last error: {last_error}"
    )


class BackendWorker(QObject):
    ready = Signal(str)
    failed = Signal(str, str)

    @Slot()
    def run(self):
        global _GRADIO_DEMO

        try:
            # Importing app.py keeps the user's existing Gradio UI unchanged.
            import app as gradio_app

            _GRADIO_DEMO = gradio_app.demo
            port = _choose_port()
            url = f"http://127.0.0.1:{port}"

            print(f"Launching embedded Gradio backend: {url}")

            _GRADIO_DEMO.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False,
                inbrowser=False,
                show_error=True,
                prevent_thread_lock=True,
            )

            _wait_until_ready(url)
            print("Embedded Gradio backend ready.")
            self.ready.emit(url)

        except Exception as exc:
            details = traceback.format_exc()
            print(details, file=sys.stderr)
            self.failed.emit(
                f"{type(exc).__name__}: {exc}",
                details,
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            "Text-Guided Photovoltaic Defect Segmentation System"
        )
        self.resize(1560, 940)
        self.setMinimumSize(1100, 720)

        self.web = QWebEngineView(self)
        self.web.setHtml(LOADING_HTML, QUrl("about:blank"))
        self.setCentralWidget(self.web)

        status = QStatusBar()
        self.setStatusBar(status)
        self.statusBar().showMessage(
            "Loading General model, Presence Gate and experts..."
        )

        self.web.page().profile().downloadRequested.connect(
            self._handle_download
        )

        self._thread = QThread(self)
        self._worker = BackendWorker()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.ready.connect(self._on_ready)
        self._worker.failed.connect(self._on_failed)

        self._worker.ready.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    @Slot(str)
    def _on_ready(self, url: str):
        self.statusBar().showMessage(
            "Ready | Embedded local interface"
        )
        self.web.setUrl(QUrl(url))

    @Slot(str, str)
    def _on_failed(self, message: str, details: str):
        self.statusBar().showMessage("Startup failed")
        self.web.setHtml(
            _error_html(message, details),
            QUrl("about:blank"),
        )

    @Slot(object)
    def _handle_download(self, item):
        suggested = item.downloadFileName() or "result.zip"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Result",
            str(Path.home() / "Downloads" / suggested),
            "All files (*.*)",
        )

        if not path:
            item.cancel()
            return

        target = Path(path)
        item.setDownloadDirectory(str(target.parent))
        item.setDownloadFileName(target.name)
        item.accept()

    def closeEvent(self, event):
        global _GRADIO_DEMO

        try:
            if _GRADIO_DEMO is not None:
                _GRADIO_DEMO.close()
        except Exception:
            pass

        try:
            _log_stream.flush()
        except Exception:
            pass

        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("TG-UNet DefectSeg")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
