from pathlib import Path
import sys,importlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'src'))
import resectionlab
resectionlab.__path__.insert(0,str(HERE/'stage/resectionlab'))
for name in ('native_resection','native_proposals','public_target_context','native_spatial_task','evaluation','native_spatial_evaluation'):
    module=importlib.import_module('resectionlab.'+name)
    assert Path(module.__file__).parent==HERE/'stage/resectionlab'
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/test_native_resection.py'),str(ROOT/'tests/test_native_spatial_evaluation.py')]))
