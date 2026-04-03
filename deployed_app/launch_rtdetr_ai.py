from __future__ import annotations

import subprocess
import sys
from shutil import which
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENT_CHECK = (
    "import cv2, jinja2, multipart, numpy, torch, ultralytics, uvicorn; "
    "from PIL import Image; "
    "from fastapi import FastAPI"
)


def python_candidates() -> list[Path]:
    candidates = [
        APP_ROOT / ".venv" / "Scripts" / "python.exe",
        APP_ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]

    for command_name in ("python3", "python"):
        resolved = which(command_name)
        if resolved:
            candidates.append(Path(resolved))

    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved_candidate = candidate.resolve() if candidate.exists() else candidate
        if resolved_candidate in seen:
            continue
        seen.add(resolved_candidate)
        unique_candidates.append(candidate)
    return unique_candidates


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
    for candidate in python_candidates():
        if supports_live_gui(candidate):
            return candidate
    raise FileNotFoundError(
        "No compatible Python environment was found for the RT-DETR web app. "
        "The launcher checked the workspace .venv, the current interpreter, and system Python commands."
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