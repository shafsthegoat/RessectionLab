"""Frozen Electron engine: JSONL only; no desktop framework imports."""
from multiprocessing import freeze_support
import sys

if __name__ == "__main__":
    freeze_support()
    if sys.argv[1:] == ["--native-probe"]:
        from resectionlab.native_runtime_probe import main
    else:
        from resectionlab.desktop_bridge import main
    raise SystemExit(main())
