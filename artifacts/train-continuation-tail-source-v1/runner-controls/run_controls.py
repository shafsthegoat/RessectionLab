"""Reuse exact previously successful generated-suite process supervision."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
helper=HERE.parent/'model/run_controls.py'
raw=helper.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='1d2a1b072d5bb67a2ce20568570333a3cd0ca461b788b8e361baba95359bb394'
ns={'__file__':str(helper),'__name__':'runner_control_supervision'}
exec(compile(raw,str(helper),'exec'),ns)
ns['HERE']=HERE
ns['FILES']=tuple(str(p.relative_to(ROOT)) for p in sorted((HERE.parent/'runner').glob('*.py')))+('src/resectionlab/spatial_continuation_tail.py','src/resectionlab/spatial_continuation_planner.py')
raise SystemExit(ns['main']())
