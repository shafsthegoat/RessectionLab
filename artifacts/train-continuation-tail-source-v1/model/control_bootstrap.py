"""Isolated generated-only test entry, existing dependencies and no bytecode."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
if not sys.flags.isolated or not sys.dont_write_bytecode:
    raise RuntimeError('Require isolated -I -B bootstrap')
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
import pytest
raise SystemExit(pytest.main(['-q', '-p', 'no:cacheprovider',
    str(ROOT/'tests/test_spatial_continuation_tail.py'),
    str(ROOT/'tests/test_spatial_continuation_planner.py'),
    '--basetemp='+str(HERE/'controls-01-tmp')]))
