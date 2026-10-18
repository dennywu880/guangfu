from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path


ENV_PYTHONW = Path(r"D:\Anaconda\envs\defect_ris\pythonw.exe")


def message_box(title: str, text: str, icon: int = 0x10) -> None:
    ctypes.windll.user32.MessageBoxW(
        None,
        text,
        title,
        icon,
    )


def main() -> int:
    # build_embedded_exe.bat copies the final EXE to the project root.
    if getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parent

    script = root / "gradio_desktop_window.py"

    if not ENV_PYTHONW.exists():
        message_box(
            "TG-UNet",
            "Python environment not found:\n"
            f"{ENV_PYTHONW}\n\n"
            "Expected conda environment: defect_ris",
        )
        return 1

    if not script.exists():
        message_box(
            "TG-UNet",
            "Desktop window script not found:\n"
            f"{script}\n\n"
            "Keep TG_UNet_DefectSeg.exe and "
            "gradio_desktop_window.py in the same project folder.",
        )
        return 2

    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"

    try:
        creationflags = getattr(
            subprocess,
            "CREATE_NO_WINDOW",
            0,
        )

        subprocess.Popen(
            [
                str(ENV_PYTHONW),
                str(script),
            ],
            cwd=str(root),
            env=env,
            creationflags=creationflags,
        )
        return 0

    except Exception as exc:
        message_box(
            "TG-UNet",
            "Could not start the desktop interface:\n"
            f"{type(exc).__name__}: {exc}",
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
