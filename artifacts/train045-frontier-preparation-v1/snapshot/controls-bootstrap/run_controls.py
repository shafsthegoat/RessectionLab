"""Reuse verified single-child60s/1GiB controls with1MiB output allowance."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
helper=ROOT/'build/continuation-tail-v1/model/run_controls.py'
raw=helper.read_bytes()
assert hashlib.sha256(raw).hexdigest()=='1d2a1b072d5bb67a2ce20568570333a3cd0ca461b788b8e361baba95359bb394'
source=raw.decode().replace('8*1024**2','1024**2').replace('8MiB output cap','1MiB output cap')
ns={'__file__':str(helper),'__name__':'frontier_control_supervision'}
exec(compile(source,str(helper),'exec'),ns)
ns['HERE']=HERE
ns['FILES']=tuple(str((HERE.parent/n).relative_to(ROOT)) for n in ('worker.py','frontier_protocol.py','frontier_selector.py','common.py','inputs.py','bootstrap_smoke.py'))
raise SystemExit(ns['main']())
