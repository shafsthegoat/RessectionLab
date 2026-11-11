"""Small data-free public-call receipt for source, installed and frozen runtimes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
import sysconfig
import tempfile


def checkpoint_probe() -> dict:
    """Verify stock-copy qualification and the actual rollback copy dispatch."""
    from . import native_backend, surgical_engine
    genuine = surgical_engine._engine_copy_genuine
    unchanged = surgical_engine._engine_copy_hooks_unchanged()
    assert genuine and unchanged, "Stock deepcopy hooks did not qualify in this runtime"
    components = [{"cells": [2, 1]}]
    state = {"flapComponents": components, "componentAlias": components,
             "removed": [1], "debris": {"2": .5}, "nested": {"position": [1., -0., 3.]}}
    before = native_backend.backend_identity()
    copied = surgical_engine._microstep_checkpoint(state)
    after = native_backend.backend_identity()
    calls = after["counters"]["engineCopyCalls"] - before["counters"]["engineCopyCalls"]
    fallbacks = after["counters"]["engineCopyFallbackCalls"] - before["counters"]["engineCopyFallbackCalls"]
    assert copied == state and copied is not state
    assert copied["flapComponents"] is copied["componentAlias"] is components
    assert copied["removed"] is not state["removed"] and copied["debris"] is not state["debris"]
    assert copied["nested"]["position"] is not state["nested"]["position"]
    assert (calls >= 1 and fallbacks == 0) if after["selected"] == "cpp" else calls == 0
    return {"passed": True, "stockCopyHooksQualified": genuine,
            "stockCopyHooksUnchanged": unchanged, "componentExemptionPreserved": True,
            "selected": after["selected"], "nativeCalls": calls, "nativeFallbackCalls": fallbacks}


def gc_proof_probe() -> dict:
    """Qualify cancellation-style owner cleanup in the actual loaded runtime."""
    import gc
    import numpy as np
    from .generated_cpp_fixture import joint_source
    from .surgical_scene import SurgicalScene
    from . import surgical_proof_cache as proofs
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        garbage = [joint_source()]
        garbage.append(garbage)
        del garbage
        scene = SurgicalScene(np.zeros((4, 4, 4), np.uint16), np.eye(4), [], {}, {})
        epoch = proofs.implementation_epoch()
        assert epoch is not None
        with proofs.proof_scope(scene, {}):
            assert gc.collect() > 0
            assert proofs.implementation_epoch() == epoch
        assert proofs.implementation_epoch() == epoch
        return {"passed": True, "genuineOwnerGcEpochStable": True}
    finally:
        if enabled:
            gc.enable()


def replay_probe() -> dict:
    """Exercise real bridge admission and observed frames before a long UI run."""
    from .desktop_bridge import BridgeSession, _Request
    from .surgical_frame_replay import SurgicalFrameCache
    from .surgical_replay_cache import SurgicalReplayCache
    from .surgical_verified_cache import VerifiedSurgicalCache
    with tempfile.TemporaryDirectory(prefix="resectionlab-replay-probe-") as directory:
        root = Path(directory)
        session = BridgeSession(root / "transfers", run_dir=root / "runs")
        # Source/installed probes also keep all disposable cache data in this
        # temporary directory. Frozen probes use the production session routing.
        if not getattr(sys, "frozen", False):
            session.surgical_replay_cache = SurgicalReplayCache(
                verified_cache=VerifiedSurgicalCache(directory=root / "verified"))
            session.surgical_frame_cache = SurgicalFrameCache(directory=root / "observed")
        try:
            def call(operation, **args):
                return session.execute(operation, args, _Request("runtime-replay"), lambda *_: None)
            case = call("createSurgicalFixture")
            key = case["caseHash"]
            current = call("surgicalRehearsal", caseHash=key, command="create")
            accepted = call("surgicalRehearsal", caseHash=key, command="step",
                expectedStateId=current["view"]["stateId"],
                action={"slot": "left", "operation": "inspect", "duration": .1})
            assert accepted["outcome"]["status"] == "accepted"
            initial = call("surgicalRehearsal", caseHash=key, command="seekFrame", position=0, frameIndex=0,
                           expectedStateId=accepted["view"]["stateId"])
            final = call("surgicalRehearsal", caseHash=key, command="seekFrame", position=0,
                         frameIndex=initial["playback"]["frameCount"] - 1,
                         expectedStateId=accepted["view"]["stateId"])
            assert initial["playback"]["readOnly"] and not initial["changed"] and not final["changed"]
            assert final["view"]["stateId"] == accepted["view"]["stateId"]
            unchanged = call("surgicalRehearsal", caseHash=key, command="inspect")
            assert unchanged["view"]["stateId"] == accepted["view"]["stateId"]
            return {"passed": True, "acceptedActions": 1,
                    "frameCount": initial["playback"]["frameCount"],
                    "initialStateId": initial["view"]["stateId"],
                    "endpointStateId": final["view"]["stateId"],
                    "acceptedStateUnchanged": True}
        finally:
            session.close()


def probe() -> dict:
    import numpy as np
    from . import geometry, native_backend, surgical_motion
    from .surgical_idle_native import strict_live_identity as idle_identity
    from .surgical_prefix_json_native import strict_live_identity as prefix_identity

    before = native_backend.backend_identity()
    scene = geometry.GeometryScene(np.zeros((9, 8, 7), dtype=bool), np.eye(4))
    cells = geometry.capsule_voxel_indices(scene, [1., 2., 1.], [6., 5., 5.], .75)
    assert cells.dtype == np.int64 and cells.ndim == 2 and cells.shape[1] == 3 and len(cells) > 0
    assert not cells.flags.writeable
    checkpoint = checkpoint_probe()
    motion = {"sourceQualified": surgical_motion._rotation_captures_genuine,
              "helpersUnchanged": surgical_motion._rotation_helpers_unchanged()}
    assert all(motion.values()), "Stock SciPy angular helpers did not qualify in this runtime"
    replay = replay_probe()
    proof_lifecycle = gc_proof_probe()
    return {"schema": "resectionlab-native-runtime-probe-v1", "passed": True,
            "runtime": {"python": sys.version, "executable": sys.executable,
                        "architecture": platform.machine(), "abi": sysconfig.get_config_var("SOABI"),
                        "frozen": bool(getattr(sys, "frozen", False)),
                        "packagePath": str(Path(geometry.__file__).resolve().parent),
                        "sysPath": list(sys.path)},
            "before": before, "backend": native_backend.backend_identity(),
            "publicCall": {"name": "geometry.capsule_voxel_indices", "cells": len(cells),
                           "orderedCellsSha256": hashlib.sha256(cells.tobytes()).hexdigest()},
            "replay": replay,
            "checkpoint": checkpoint,
            "motionCache": motion,
            "proofLifecycle": proof_lifecycle,
            "olderOptionalHelpers": {"idle": idle_identity(), "prefixJson": prefix_identity()}}


def main() -> int:
    try:
        result = probe()
    except Exception as error:
        print(json.dumps({"schema": "resectionlab-native-runtime-probe-v1", "passed": False,
                          "errorType": type(error).__name__, "error": str(error)}))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
