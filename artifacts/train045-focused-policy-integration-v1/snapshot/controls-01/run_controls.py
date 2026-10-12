"""Root-owned generated integration suite using the verified bounded supervisor."""
from pathlib import Path
import hashlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
helper = ROOT / 'build/continuation-tail-v1/model/run_controls.py'
raw = helper.read_bytes()
if hashlib.sha256(raw).hexdigest() != '1d2a1b072d5bb67a2ce20568570333a3cd0ca461b788b8e361baba95359bb394':
    raise ValueError('Reviewed control supervisor changed')
source = raw.decode().replace('8*1024**2', '1024**2').replace('8MiB output cap', '1MiB output cap')
namespace = {'__file__': str(helper), '__name__': 'focused_control_supervision'}
exec(compile(source, str(helper), 'exec'), namespace)
namespace['HERE'] = HERE
namespace['FILES'] = tuple(sorted(str(p.relative_to(ROOT))
    for folder in ('model', 'runner', 'independent', 'controls-01')
    for p in (HERE.parent / folder).glob('*.py')))
raise SystemExit(namespace['main']())
