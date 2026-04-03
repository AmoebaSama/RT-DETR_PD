from __future__ import annotations

import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn


BIND_HOST = os.environ.get("RTDETR_HOST", "127.0.0.1")
PORT = int(os.environ.get("RTDETR_PORT", "8000"))
HEALTHCHECK_URL = f"http://127.0.0.1:{PORT}"


def _runtime_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


os.environ.setdefault("RTDETR_ROOT", str(_runtime_root()))

from rtdetr.web.app import app


def _display_urls() -> list[str]:
    urls = [f"http://127.0.0.1:{PORT}"]
    if BIND_HOST not in {"127.0.0.1", "localhost"}:
        try:
            lan_ip = socket.gethostbyname(socket.gethostname())
        except OSError:
            lan_ip = None
        if lan_ip and not lan_ip.startswith("127."):
            urls.append(f"http://{lan_ip}:{PORT}")
    return urls


def _maybe_open_browser(url: str) -> None:
    should_open = os.environ.get("RTDETR_OPEN_BROWSER", "1").lower() not in {"0", "false", "no"}
    if not should_open:
        return
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.5)
        return probe.connect_ex((host, port)) == 0


def _healthcheck(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
            return response.status == 200
    except Exception:
        return False


def _wait_for_server(url: str, timeout_seconds: float = 45.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _healthcheck(url):
            return True
        time.sleep(0.5)
    return False


def main() -> int:
    if _is_port_open("127.0.0.1", PORT) and _healthcheck(HEALTHCHECK_URL):
        _maybe_open_browser(HEALTHCHECK_URL)
        return 0

    config = uvicorn.Config(app=app, host=BIND_HOST, port=PORT, log_level="warning")
    server = uvicorn.Server(config)
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    if not _wait_for_server(HEALTHCHECK_URL):
        print("RT-DETR GUI failed to start.")
        server.should_exit = True
        server_thread.join(timeout=5)
        return 1

    urls = _display_urls()
    print("RT-DETR GUI is running at:")
    for url in urls:
        print(url)
    print("Close this window to stop the local server.")
    _maybe_open_browser(urls[0])

    try:
        while server_thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.should_exit = True
        server_thread.join(timeout=5)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())