import pathlib,sys
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider','--basetemp='+str(HERE/'pytest-temp'),str(HERE.parent/'runner/test_runner.py')]))
