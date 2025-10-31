#!/usr/bin/env python3
"""
Start the GitHub Pages-style test runner locally on the first available port
and print the URL to open.

Usage:
  python3 server.py

The server will serve the contents of the `web_tester/` directory and will continue
running until interrupted (Ctrl+C).
"""

import contextlib
import os
import socket
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import subprocess
import signal


def find_free_port(start: int = 8000, end: int = 8999) -> int:
    for port in range(start, end + 1):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{end}")

def list_pids_on_port(port: int) -> list[int]:
    try:
        out = subprocess.check_output(["lsof", "-ti", f"tcp:{port}"], stderr=subprocess.DEVNULL)
        lines = out.decode().strip().splitlines()
        return [int(pid) for pid in lines if pid.strip()]
    except subprocess.CalledProcessError:
        return []
    except FileNotFoundError:
        # lsof not available
        return []

def first_free_port_after(start: int, end: int) -> int:
    for port in range(start, end + 1):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}-{end}")

def kill_pids(pids: list[int]) -> None:
    # Try SIGTERM, then SIGKILL
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    # Give a brief grace period
    try:
        import time
        time.sleep(0.2)
    except Exception:
        pass
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            pass


def main() -> int:
    # Script is in web_tester/ directory, serve from current directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # We're already in web_tester/, so serve from current directory
    os.chdir(script_dir)

    try:
        print("Checking port 8000...")
        try:
            sys.stdout.flush()
        except Exception:
            pass
        pids_8000 = list_pids_on_port(8000)
        if pids_8000:
            print(f"Port 8000 is in use by PID(s): {' '.join(str(pid) for pid in pids_8000)}")
            try:
                sys.stdout.flush()
            except Exception:
                pass
            ans = input("Kill these and use port 8000? [y/N]: ").strip().lower()
            if ans == 'y':
                kill_pids(pids_8000)
                # Wait briefly and verify port is free; retry a few times
                import time
                for _ in range(10):
                    time.sleep(0.1)
                    if not list_pids_on_port(8000):
                        break
                if list_pids_on_port(8000):
                    print("Port 8000 still in use after kill attempts; falling back to a free port >8000")
                    port = first_free_port_after(8001, 8999)
                else:
                    port = 8000
            else:
                # Fall back to first free port after 8000
                port = first_free_port_after(8001, 8999)
        else:
            port = 8000
    except KeyboardInterrupt:
        print("\nAborted by user before server start.")
        return 130

    class DocsHandler(SimpleHTTPRequestHandler):
        # Serve files from the current working directory (web_tester/)
        def log_message(self, format: str, *args) -> None:
            # Keep standard logging behavior
            super().log_message(format, *args)

    try:
        server = ThreadingHTTPServer(("", port), DocsHandler)
    except OSError as e:
        # If binding fails (e.g., race), fall back to first free port >8000
        print(f"Failed to bind to :{port} ({e}). Trying next free port...")
        port = first_free_port_after(8001, 8999)
        server = ThreadingHTTPServer(("", port), DocsHandler)

    url = f"http://localhost:{port}"
    print(f"Serving test runner from web_tester/ at: {url}")
    print(f"Open {url} in your browser")
    print("Press Ctrl+C to stop.")
    try:
        sys.stdout.flush()
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


