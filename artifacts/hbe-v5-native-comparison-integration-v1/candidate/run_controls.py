"""Load only staged comparator candidates; run tiny generated controls."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import scripts
scripts.__path__ = [str(HERE/'stage/scripts'), *scripts.__path__]
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/test_mechanics_hbe_v5_comparison.py'),
    str(HERE/'test_generated.py'),'--basetemp='+str(HERE/os.environ.get('HBE_GENERATED_BASETEMP','generated-controls-current')),*sys.argv[1:]]))
