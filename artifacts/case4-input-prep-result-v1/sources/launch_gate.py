"""Wait for an identity-bound launch token before execing the Case4 worker.

The input worker cannot import patient arrays before the supervisor has captured
the child's kernel PID/start identity and recorded a durable launch receipt.
EOF, an invalid token, or an invalid path exits without exec.
"""
import os
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 3:
        raise SystemExit(2)
    descriptor = int(sys.argv[1])
    worker_path = Path(sys.argv[2])
    if descriptor < 3 or worker_path.is_symlink():
        raise SystemExit(2)
    worker = worker_path.resolve(strict=True)
    if not worker.is_file():
        raise SystemExit(2)
    try:
        token = os.read(descriptor, 1)
    finally:
        os.close(descriptor)
    if token != b"G":
        raise SystemExit(3)
    os.execv(sys.executable, [sys.executable, "-I", str(worker)])


if __name__ == "__main__":
    main()
