"""Reuse previously verified supervision, with a 1MiB control-output cap."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
helper=ROOT/'build/continuation-tail-v1/model/run_controls.py'
raw=helper.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='1d2a1b072d5bb67a2ce20568570333a3cd0ca461b788b8e361baba95359bb394'
source=raw.decode().replace('8*1024**2','1024**2').replace('8MiB output cap','1MiB output cap')
ns={'__file__':str(helper),'__name__':'endgame_control_supervision'}
exec(compile(source,str(helper),'exec'),ns)
ns['HERE']=HERE
ns['FILES']=tuple(str(p.relative_to(ROOT)) for p in sorted((HERE.parent/'replay-recovery').glob('*.py')))+('src/resectionlab/spatial_continuation_tail.py','src/resectionlab/spatial_continuation_planner.py')
raise SystemExit(ns['main']())
