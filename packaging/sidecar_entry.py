"""Frozen Electron engine: JSONL only; no desktop framework imports."""
from multiprocessing import freeze_support
from resectionlab.desktop_bridge import main

if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
