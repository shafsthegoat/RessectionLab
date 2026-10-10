"""One generated RL256 transfer in an owned, bounded child; no retries."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading


def _activate_stage():
    here = Path(__file__).resolve()
    root = next((parent for parent in here.parents if (parent / "pyproject.toml").is_file()), None)
    if root is None:
        raise RuntimeError("A local source checkout is required for this development selector")
    sys.path.insert(0, str(root / "src"))
    import resectionlab
    resectionlab.__path__.insert(0, str(here.parent))
    return root


ROOT = _activate_stage()


def _require_parent_lease():
    """Stop the child if the owning app process disappears unexpectedly."""
    encoded = os.environ.get("RESECTIONLAB_PARENT_LEASE_FD")
    if encoded is None or not encoded.isdecimal() or int(encoded) < 3:
        raise RuntimeError("Owned transfer child requires a parent lease")
    fd = int(encoded)
    os.fstat(fd)
    def watch():
        try:
            # The parent never writes; EOF means it exited or released the job.
            while os.read(fd, 1):
                pass
        finally:
            os._exit(143)
    threading.Thread(target=watch, name="transfer-parent-lease", daemon=True).start()


def _write_reserved(handle, value):
    handle.seek(0)
    handle.truncate()
    json.dump(value, handle, sort_keys=True, separators=(",", ":"), allow_nan=False)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())


def execute(output: Path):
    # Fail before any checkpoint read if an attempt already exists.
    with output.open("x") as handle:
        _write_reserved(handle, {"status": "started", "scope": "generated_transfer_only",
            "automaticRetry": False})
        stage = "fixed_loader"
        try:
            import torch
            torch.set_num_threads(1)
            from resectionlab.fixed_rl256_loader import load_fixed_rl256_policy
            from resectionlab.development_episode import make_development_task
            from resectionlab.legacy_aspiration_projection import (
                AspirationOnlyNativeTask, BoundAspirationTransferPolicy,
                execute_matched_aspiration_transfer_pair)
            from resectionlab.legacy_transfer_episode import (
                CHECKPOINT_SHA, execute_transfer_episode_from_pair,
                transfer_authorship, validate_transfer_planning)
            from resectionlab.shared_vascular_evaluation import _preflight

            policy, identity, original_receipt = load_fixed_rl256_policy()
            if identity.checkpoint_sha256 != CHECKPOINT_SHA:
                raise RuntimeError("Fixed loader returned another policy identity")
            stage = "nominal_actor_and_search"
            target = AspirationOnlyNativeTask(make_development_task())
            bound = BoundAspirationTransferPolicy(policy, identity, target)
            pair = execute_matched_aspiration_transfer_pair(target, bound,
                max_calls=24, beam_width=4, seconds=10.)
            stage = "sealed_native_episode_and_geometry"
            display, episode = execute_transfer_episode_from_pair(pair)
            if not validate_transfer_planning(make_development_task(), episode):
                raise RuntimeError("Generated transfer trace did not replay")
            replay_task, _, _, _ = _preflight(episode, None)  # No private reference load.
            if replay_task.case.source_hash != episode["sourceHash"]:
                raise RuntimeError("Independent generated source replay differs")
            result = {"status": "complete", "scope": "generated_transfer_not_training_domain",
                "checkpointSha256": CHECKPOINT_SHA, "originalLoaderReceipt": original_receipt,
                "pair": pair, "caseHash": display.semantic_hash, "episode": episode,
                "episodeAuthorship": transfer_authorship(episode, live_backend_run=True),
                "newOptimizerUpdates": 0, "clinicalValidation": False}
            _write_reserved(handle, result)
            return result
        except BaseException as error:
            _write_reserved(handle, {"status": "failed", "stage": stage,
                "errorType": type(error).__name__, "error": str(error),
                "automaticRetry": False, "scope": "generated_transfer_only"})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _require_parent_lease()
    execute(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
