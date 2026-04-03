from __future__ import annotations

import subprocess
import sys
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE_VENV = APP_ROOT / ".venv" / "Scripts" / "python.exe"
ENVIRONMENT_CHECK = "import cv2, numpy, torch, fastapi, uvicorn; from PIL import Image"


def supports_live_gui(python_executable: Path) -> bool:
    if not python_executable.exists():
        return False
    result = subprocess.run(
        [str(python_executable), "-c", ENVIRONMENT_CHECK],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def resolve_python_executable() -> Path:
    for candidate in (WORKSPACE_VENV, Path(sys.executable)):
        if supports_live_gui(candidate):
            return candidate
    raise FileNotFoundError(
        "No compatible Python environment was found for the RT-DETR web app. "
        "The launcher checked the workspace .venv and the current interpreter."
    )


def main() -> None:
    python_executable = resolve_python_executable()
    command = [
        str(python_executable),
        str(APP_ROOT / "launch_rtdetr_app.py"),
    ]
    subprocess.run(command, cwd=str(APP_ROOT), check=False)


if __name__ == "__main__":
    main()