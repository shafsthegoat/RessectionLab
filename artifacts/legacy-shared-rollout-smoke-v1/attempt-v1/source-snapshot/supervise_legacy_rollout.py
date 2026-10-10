"""One bounded generated-only legacy RL256/search rollout; root schedules execution.

This reuses the repository's existing sampled-RSS/wall subprocess supervisor.
It reserves a fresh attempt directory before spawning and never retries. The
child also reserves its JSON result before checkpoint loading or search.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FROZEN_SUPERVISOR = ROOT / "build/matched-estimate-methods-v1/candidate/frozen_preflight_supervisor.py"

VERSION = "legacy-RL256-shared-transition-supervision-v1"
MAX_WALL_SECONDS = 90
MAX_RSS_BYTES = 1024 * 1024 * 1024
SOURCE_SHA256 = {
    "build/integration-first-policy-v1/legacy_checkpoint_rollout.py": "0156ada24f0b1443f30db1a3a00415e11dbda674eb9274bf9acb718e503aa9a3",
    "build/matched-estimate-methods-v1/candidate/frozen_preflight_supervisor.py": "8d3db9c04e4b2e040f6b6f5bc153618f277233fb93c12c2cabdf418493e58048",
    "src/resectionlab/core.py": "094bb902552117ee0be1ee4ed79ae0b09a7005e48d24ca4a9a4674ff54af077b",
    "src/resectionlab/native_resection.py": "0cbbe2904f6b87cfb97d555121cc70690989d70a971361ec417735a7061e3a3c",
    "src/resectionlab/native_spatial_task.py": "503acba2bfe14dc6f744bd2099e45b018cfa2011de4e35b30e0e8c70abdb6255",
    "src/resectionlab/spatial_policy.py": "5801f1b59b5eed90edeb33e5182d57547de35dd0fd326c76c4e427f89c05b6a4",
    "src/resectionlab/spatial_observations.py": "48a2a9d1c7faee791f7f327ac928e133f2d0f80f60a045a868802bb75785d711",
    "src/resectionlab/shared_episode.py": "81d6b1faac3433f1a2eb09ff4ef70e773b69922c056495caf1a879fbbc381a4d",
    "src/resectionlab/sequential_spatial_observation.py": "c8791eccbc45c96449ba7dd9578881a486654d2cd65e092157a79cde3b8673a3",
    "src/resectionlab/observed_search.py": "cdbacc079fa3e4f8d49a356e76dfdc77e4484ef4bb1a3383970a0e937c72febd",
    "src/resectionlab/evaluation.py": "624db1b0f520c45d91b7da2fd3312f75acab3f2573a31621fa33e38343dc6010",
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sources() -> dict[str, str]:
    if not SOURCE_SHA256:
        raise RuntimeError("Source bindings have not been frozen")
    for relative, expected in SOURCE_SHA256.items():
        if sha(ROOT / relative) != expected:
            raise ValueError(f"Bound generated integration source changed: {relative}")
    return dict(SOURCE_SHA256)


def bound_supervisor():
    # Load the previously reviewed fixed lifecycle controller only after its
    # exact source bytes pass check_sources(); no patient entry point is used.
    spec = importlib.util.spec_from_file_location("bound_preflight_supervisor", FROZEN_SUPERVISOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Bound subprocess supervisor is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.supervise_worker


def run_attempt(output: Path) -> dict:
    # Existing attempts fail before source/checkpoint validation or worker spawn.
    output.mkdir(parents=True, exist_ok=False)
    started = {"status": "started", "version": VERSION, "automatic_retry": False,
        "max_wall_seconds": MAX_WALL_SECONDS, "max_sampled_worker_rss_bytes": MAX_RSS_BYTES,
        "scope": "generated_legacy_aspirate_only"}
    (output / "attempt.json").write_text(json.dumps(started, sort_keys=True, indent=2) + "\n")
    try:
        source_hashes = check_sources()
        started["source_sha256"] = source_hashes
        (output / "attempt.json").write_text(json.dumps(started, sort_keys=True, indent=2) + "\n")
        # The imported supervisor starts a fresh child session, samples RSS,
        # applies fixed wall/RSS stops, and retains worker/supervisor logs.
        os.environ["PYTHONPATH"] = str(ROOT / "src")
        result = bound_supervisor()([sys.executable, "-B", str(HERE / "legacy_checkpoint_rollout.py"),
            "--execute", "--output", str(output / "comparison.json")], output,
            {"max_wall_seconds": MAX_WALL_SECONDS, "max_rss_bytes": MAX_RSS_BYTES},
            hashlib.sha256(json.dumps(started, sort_keys=True).encode()).hexdigest())
        if check_sources() != source_hashes:
            raise RuntimeError("Bound sources changed during the supervised attempt")
        if result["status"] != "complete":
            return result
        comparison_path = output / "comparison.json"
        if not comparison_path.is_file() or comparison_path.stat().st_size > 2 * 1024 * 1024:
            raise RuntimeError("Completed child has no bounded comparison JSON")
        comparison = json.loads(comparison_path.read_text())
        if (comparison.get("status") != "complete" or comparison.get("clinical_validation") is not None
                or comparison.get("new_training_updates") != 0
                or not all(comparison.get(key) is True for key in
                    ("same_source", "same_environment", "same_initial_observation"))):
            raise RuntimeError("Completed child output is not a generated-only comparison")
        (output / "completion.json").write_text(json.dumps({"status": "complete", "version": VERSION,
            "comparison_sha256": sha(comparison_path), "supervision": result},
            sort_keys=True, indent=2, allow_nan=False) + "\n")
        return result
    except BaseException as error:
        (output / "preflight-or-completion-failure.json").write_text(json.dumps({"status": "failed",
            "error_type": type(error).__name__, "error": str(error), "automatic_retry": False},
            sort_keys=True, indent=2) + "\n")
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_attempt(args.output_dir)
    print(json.dumps({"status": result["status"], "output_dir": str(args.output_dir)}, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
