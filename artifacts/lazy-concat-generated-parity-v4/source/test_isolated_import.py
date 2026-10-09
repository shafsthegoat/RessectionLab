"""Exact `python -I` worker import and direct phase-RSS check, no model main."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock


here = Path(__file__).resolve().parent
if not sys.flags.isolated or str(here) in sys.path:
    raise SystemExit("test must start with Python -I and no script-directory path")
spec = importlib.util.spec_from_file_location("isolated_pair_worker", here / "pair_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
sampler_module = sys.modules[worker.FastDarwinSampler.__module__]
if Path(sampler_module.__file__).resolve() != here / "darwin_fast_sampler.py":
    raise SystemExit("early sampler import did not resolve to pinned local source")
output = io.StringIO()
with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("child spawned")):
    with redirect_stdout(output):
        worker.phase("isolated_generated_import_only")
record = json.loads(output.getvalue().strip())
if (record.get("phase") != "isolated_generated_import_only" or
        record.get("phase_rss_source") != "direct_libproc_self_pid_no_subprocess" or
        record.get("self_rss_kib", 0) <= 0):
    raise SystemExit("direct phase-RSS contract failed")
print(json.dumps({"status": "isolated_import_and_no_ps_phase_pass",
                  "sampler_path": str(Path(sampler_module.__file__).resolve()),
                  "self_rss_kib": record["self_rss_kib"]}, sort_keys=True))
