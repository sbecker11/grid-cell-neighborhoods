#!/usr/bin/env python3
"""
Start the GitHub Pages-style test runner locally on the first available port
and print the URL to open.

Usage:
  python3 start_test_runner.py

The server will serve the contents of the `docs/` directory and will continue
running until interrupted (Ctrl+C).
"""

import contextlib
import os
import socket
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler


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


def main() -> int:
    repo_root = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(repo_root, "docs")
    if not os.path.isdir(docs_dir):
        print("docs/ directory not found next to this script.", file=sys.stderr)
        return 1

    # Change working directory to docs/ so relative paths in index.html work
    os.chdir(docs_dir)

    port = find_free_port(8000, 8999)

    class DocsHandler(SimpleHTTPRequestHandler):
        # Serve files from the current working directory (docs/)
        def log_message(self, format: str, *args) -> None:
            # Keep standard logging behavior
            super().log_message(format, *args)

    server = ThreadingHTTPServer(("", port), DocsHandler)

    url = f"http://localhost:{port}"
    print(f"Serving test runner from docs/ at: {url}")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


