"""Load the exact staged admission overlay before tiny runner glue controls."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parents[1];ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import resectionlab
resectionlab.__path__.insert(0,str(HERE/'stage/resectionlab'))
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',str(HERE/'tests/test_runner.py')]))
