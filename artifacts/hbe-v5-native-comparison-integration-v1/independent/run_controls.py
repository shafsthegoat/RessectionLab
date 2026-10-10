import os,sys
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;AUTHOR=ROOT/'build/hbe-v5-native-comparison-preparation-v1'
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
import scripts
scripts.__path__=[str(AUTHOR/'stage/scripts'),*scripts.__path__]
# An audit hook also covers os.open, beyond the author Path.open test guard.
def guard(event,args):
 if event in ('subprocess.Popen','os.system','os.posix_spawn'):raise AssertionError('No process launch in generated review')
 if event=='open' and isinstance(args[0],(str,bytes)):
  path=Path(os.fsdecode(args[0])).resolve()
  if any(path.is_relative_to(ROOT/p) for p in ('data/mechanics','outputs/mechanics')):raise AssertionError('No actual-native/measured payload read in generated review')
sys.addaudithook(guard)
import pytest
raise SystemExit(pytest.main(['-q','-p','no:cacheprovider',str(ROOT/'tests/test_mechanics_hbe_v5_comparison.py'),str(AUTHOR/'test_generated.py'),str(HERE/'extra_generated.py'),'--basetemp='+str(HERE/'generated-tests')]))
