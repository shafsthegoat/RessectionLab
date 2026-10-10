#!/usr/bin/env python3
"""Project ReMIND TRAIN headers or convert an explicit public-support crop."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("headers", "crop", "crop-mr"))
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--case-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headers", type=Path)
    parser.add_argument("--headers-sha256")
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args()
    if args.phase != "headers" and (args.headers is None or args.headers_sha256 is None):
        parser.error("crop requires --headers and --headers-sha256")
    from resectionlab.remind_planning_qc import run_headers, run_crop
    return run_headers(args) if args.phase == "headers" else run_crop(args)


if __name__ == "__main__": raise SystemExit(main())
