"""Explicit existing source and separately installed, root-pinned runtime paths."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
OLD_RUNNER=ROOT/'build/train045-focused-batched-v1/runner'
OLD_MODEL=ROOT/'build/train045-focused-batched-v1/model'
DEPS=ROOT/'build/local-goal-ppo-deps-v1/site'
for p in reversed((HERE,OLD_RUNNER,OLD_MODEL,ROOT/'src',DEPS)):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
