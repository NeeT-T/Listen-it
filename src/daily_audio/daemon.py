"""Persistent model server that keeps Whisper loaded between CLI calls."""

import json
import os
import socket
import subprocess
import sys
import time

_SOCKET_DIR = "/tmp"
_STARTUP_TIMEOUT = 30  # seconds to wait for daemon to be ready


def socket_path(model: str) -> str:
    return f"{_SOCKET_DIR}/daily-audio-{model}.sock"


def pid_path(model: str) -> str:
    return f"{_SOCKET_DIR}/daily-audio-{model}.pid"


# ── Server ────────────────────────────────────────────────────────────────────

def _handle(conn: socket.socket, model_name: str) -> None:
    from daily_audio.transcriber import transcribe_file

    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            return
        buf += chunk

    req = json.loads(buf.split(b"\n", 1)[0].decode())

    try:
        for ts, text in transcribe_file(
            req["audio_path"],
            model_name=model_name,
            language=req.get("language"),
            quality=req.get("quality", "fast"),
        ):
            conn.sendall((json.dumps({"ts": ts, "text": text}) + "\n").encode())
    except Exception as exc:
        conn.sendall((json.dumps({"error": str(exc)}) + "\n").encode())

    conn.sendall((json.dumps({"done": True}) + "\n").encode())


def run_server(model_name: str) -> None:
    from daily_audio.transcriber import get_model

    sock = socket_path(model_name)
    pid = pid_path(model_name)

    if os.path.exists(sock):
        os.unlink(sock)

    # Write PID so callers can check if we are alive
    with open(pid, "w") as f:
        f.write(str(os.getpid()))

    get_model(model_name)  # load once, keep in memory

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
        srv.bind(sock)
        srv.listen(8)
        try:
            while True:
                conn, _ = srv.accept()
                with conn:
                    _handle(conn, model_name)
        finally:
            for path in (sock, pid):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass


# ── Client ────────────────────────────────────────────────────────────────────

def _is_running(model: str) -> bool:
    sock = socket_path(model)
    if not os.path.exists(sock):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect(sock)
        return True
    except OSError:
        return False


def _start_daemon(model: str) -> None:
    subprocess.Popen(
        [sys.argv[0], "daemon", "--model", model],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + _STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if _is_running(model):
            return
        time.sleep(0.3)
    raise TimeoutError(f"Daemon para modelo '{model}' não iniciou em {_STARTUP_TIMEOUT}s")


def ensure_running(model: str) -> None:
    if not _is_running(model):
        _start_daemon(model)


def transcribe_via_daemon(
    audio_path: str,
    model_name: str = "base",
    language: str | None = None,
    quality: str = "fast",
):
    ensure_running(model_name)

    req = json.dumps({"audio_path": audio_path, "language": language, "quality": quality}) + "\n"

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.connect(socket_path(model_name))
        s.sendall(req.encode())

        buf = b""
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                data = json.loads(line.decode())
                if data.get("done"):
                    return
                if "error" in data:
                    raise RuntimeError(data["error"])
                yield data["ts"], data["text"]


def stop_daemon(model: str) -> bool:
    """Send SIGTERM to the daemon. Returns True if it was running."""
    p = pid_path(model)
    if not os.path.exists(p):
        return False
    try:
        with open(p) as f:
            pid = int(f.read().strip())
        os.kill(pid, 15)  # SIGTERM
        return True
    except (ValueError, ProcessLookupError, FileNotFoundError):
        return False
