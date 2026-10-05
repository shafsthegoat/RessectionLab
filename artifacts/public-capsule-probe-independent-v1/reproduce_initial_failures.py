"""Run five synthetic regressions against the preserved initial runner only.

No patient bundle is opened and no subprocess is launched by these fixtures.
Expected initial outcome: five failures showing missing guards/diagnostics.
"""
from pathlib import Path
import importlib.util
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
SOURCE = ROOT / "artifacts/public-capsule-probe-review-v1/pre-review-script.py"
spec = importlib.util.spec_from_file_location("initial_public_cache_probe_review", SOURCE)
initial = importlib.util.module_from_spec(spec)
spec.loader.exec_module(initial)


class PreservedRunner:
    def pytest_collection_modifyitems(self, items):
        for item in items:
            item.module.probe = initial


if __name__ == "__main__":
    name = "tests/test_public_capsule_probe_review.py"
    raise SystemExit(pytest.main([
        name + "::test_common_uncached_setup_preview_count_is_bound",
        name + "::test_trace_hook_restoration_failure_keeps_incomplete_diagnostic",
        name + "::test_launcher_never_publishes_failed_or_timed_out_payload[result_failed]",
        name + "::test_launcher_never_publishes_failed_or_timed_out_payload[worker_runtime_hash]",
        name + "::test_launcher_never_publishes_failed_or_timed_out_payload[malformed_result]",
        "-q", "--tb=short",
    ], plugins=[PreservedRunner()]))
