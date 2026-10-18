"""Small offline localhost UI for validating either inference architecture."""

import base64
import io
import json
import os
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

import cv2
import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from inference_engine import DefectInferenceEngine


PAGE = Path(__file__).with_name("local_ui.html")
ENGINE = None
LOCK = Lock()
MAX_UPLOAD = 15 * 1024 * 1024


def png_data(image):
    if image is None:
        return None
    image = np.asarray(image)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, content = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode result image")
    return "data:image/png;base64," + base64.b64encode(content).decode("ascii")


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/":
            self.reply(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/status":
            self.reply(200, json.dumps({
                "mode": "shared MoE experiment" if ENGINE.use_shared_moe else "Final V3 baseline",
                "device": str(ENGINE.device),
            }).encode(), "application/json; charset=utf-8")
        else:
            self.reply(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        if self.path != "/api/detect":
            return self.reply(404, b"Not found", "text/plain; charset=utf-8")
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= MAX_UPLOAD:
            return self.reply(413, b"Upload must be under 15 MB", "text/plain; charset=utf-8")
        try:
            data = self.rfile.read(length)
            mime = self.headers.get("Content-Type", "")
            if "multipart/form-data" not in mime:
                raise ValueError("Expected image upload")
            message = BytesParser(policy=default).parsebytes(
                f"Content-Type: {mime}\r\nMIME-Version: 1.0\r\n\r\n".encode() + data
            )
            fields = {part.get_param("name", header="content-disposition"): part
                      for part in message.iter_parts()}
            image_bytes = fields["image"].get_payload(decode=True)
            expression = fields["text"].get_content().strip()
            if not expression:
                raise ValueError("Please enter an English reference description")
            if len(image_bytes) > MAX_UPLOAD:
                raise ValueError("Image is too large")
            bgr = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Unable to decode image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            with LOCK:
                result = ENGINE.infer(rgb, expression)
            info = result["info"]
            response = {
                "mode": "shared MoE experiment" if ENGINE.use_shared_moe else "Final V3 baseline",
                "overlay": png_data(result["overlay"]),
                "mask": png_data(result["binary_mask"]),
                "status": result["status"],
                "target_probability": info.get("Presence Probability (Target)"),
                "expert": info.get("Selected Expert"),
                "time": info.get("Inference Time (s)"),
                "ground_truth": info.get("GT Match Status"),
            }
            self.reply(200, json.dumps(response).encode(), "application/json; charset=utf-8")
        except (ValueError, KeyError, TypeError) as exc:
            self.reply(400, json.dumps({"error": str(exc)}).encode(), "application/json; charset=utf-8")
        except Exception as exc:
            self.reply(500, json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(), "application/json; charset=utf-8")


if __name__ == "__main__":
    ENGINE = DefectInferenceEngine()
    port = int(os.environ.get("TG_LOCAL_PORT", "8790"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Local UI: http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()
