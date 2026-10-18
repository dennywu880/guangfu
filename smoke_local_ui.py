"""Verify localhost pages and single-image inference for both model modes."""

import json
from pathlib import Path
from urllib.request import Request, urlopen

from dataset import RefELDataset


ROOT = Path(__file__).resolve().parent
DATASET = RefELDataset(ROOT, split="val", image_size=384, random_text=False)
INDEX = next(i for i in range(100) if int(DATASET[i]["mask"].sum()) > 0)
REF = DATASET.refs[INDEX]
IMAGE = DATASET.image_dir / DATASET.images_by_id[REF["image_id"]]["file_name"]
TEXT = DATASET.get_text(REF)[0]


def test(port):
    base = f"http://127.0.0.1:{port}"
    with urlopen(base + "/") as response:
        html = response.read().decode("utf-8")
        assert "光伏 EL 缺陷检测" in html
    with urlopen(base + "/api/status") as response:
        status = json.load(response)
    boundary = b"local-ui-smoke-test"
    body = b"".join([
        b"--" + boundary + b"\r\nContent-Disposition: form-data; name=\"text\"\r\n\r\n" + TEXT.encode() + b"\r\n",
        b"--" + boundary + b"\r\nContent-Disposition: form-data; name=\"image\"; filename=\"sample.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n",
        IMAGE.read_bytes(), b"\r\n--" + boundary + b"--\r\n",
    ])
    request = Request(base + "/api/detect", data=body, headers={
        "Content-Type": "multipart/form-data; boundary=" + boundary.decode(),
    })
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    assert result["mode"] == status["mode"]
    assert result["overlay"].startswith("data:image/png;base64,")
    assert result["mask"].startswith("data:image/png;base64,")
    assert result["expert"] in {"tiny", "ultra_tiny", "larger"}, result["status"]
    print(f"{base}: {result['mode']} {result['expert']} "
          f"{result['time']:.2f}s, image responses OK", flush=True)


if __name__ == "__main__":
    test(8790)
    test(8791)
