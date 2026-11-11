#!/usr/bin/env python3
"""Verify one real runtime, including absent-extension and integrity failures.

Install wheels separately into an environment outside the checkout, then pass
that interpreter and site-packages directory. Frozen engines use --executable.
All fault injection changes temporary copies; qualified inputs stay untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def run_probe(command: list[str], mode: str, directory: Path) -> dict:
    environment = dict(os.environ, RESSECTIONLAB_NUMERICAL_BACKEND=mode, PYTHONNOUSERSITE="1")
    for name in ("PYTHONPATH", "PYTHONHOME", "RESSECTIONLAB_IDLE_THERMAL_BUILD",
                 "RESSECTIONLAB_PREFIX_JSON_BUILD"):
        environment.pop(name, None)
    result = subprocess.run(command, cwd=directory, env=environment, text=True,
                            capture_output=True, timeout=90)
    lines = result.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError(f"Probe produced no JSON ({result.returncode}): {result.stderr[-2000:]}")
    record = json.loads(lines[-1])
    record["exitCode"] = result.returncode
    return record


def python_command(python: Path, package_root: Path) -> list[str]:
    code = ("import sys;sys.path.insert(0," + repr(str(package_root)) + ");"
            "from resectionlab.native_runtime_probe import main;raise SystemExit(main())")
    return [str(python), "-I", "-c", code]


def successful(record: dict, native: bool, package_root: Path) -> None:
    assert record["passed"] and record["exitCode"] == 0, record
    assert record["replay"]["passed"] and record["replay"]["acceptedActions"] == 1
    assert record["replay"]["acceptedStateUnchanged"] and record["replay"]["frameCount"] >= 2
    assert record["proofLifecycle"]["passed"] and record["proofLifecycle"]["genuineOwnerGcEpochStable"]
    checkpoint = record["checkpoint"]
    assert checkpoint["passed"] and checkpoint["stockCopyHooksQualified"] and checkpoint["stockCopyHooksUnchanged"]
    assert checkpoint["componentExemptionPreserved"] and checkpoint["nativeFallbackCalls"] == 0
    assert checkpoint["selected"] == ("cpp" if native else "reference")
    assert checkpoint["nativeCalls"] >= 1 if native else checkpoint["nativeCalls"] == 0
    assert record["motionCache"]["sourceQualified"] and record["motionCache"]["helpersUnchanged"]
    trajectory = record["trajectory"]
    assert trajectory["passed"] and trajectory["physicalSubsteps"] == 33
    assert trajectory["batchSizes"] == ([32, 1] if native else [])
    assert trajectory["scalarCalls"] == (0 if native else 33)
    segment_batch = record["segmentBatch"]
    assert segment_batch["passed"] and segment_batch["conflict"] is False
    assert segment_batch["nativeCalls"] == (1 if native else 0)
    assert segment_batch["submittedPairs"] == segment_batch["examinedPairs"] == (4 if native else 0)
    assert Path(record["runtime"]["packagePath"]).is_relative_to(package_root), record["runtime"]
    backend = record["backend"]
    assert backend["selected"] == ("cpp" if native else "reference"), backend
    for helper in record["olderOptionalHelpers"].values():
        assert (json.loads(helper) if isinstance(helper, str) else helper)["mode"] == "python", helper
    if native:
        assert backend["binarySha256"] and backend["buildIdentity"], backend
        assert backend["counters"]["physicalSubstepCalls"] > 0, "Physical executor was not native"
        assert backend["counters"]["trajectoryCalls"] > 0, "Trajectory executor was not native"
        assert Path(backend["binaryPath"]).is_relative_to(package_root), backend
        assert backend["counters"]["bindingCalls"] > record["before"]["counters"]["bindingCalls"], "Public call did not enter C++"
    else:
        assert backend["counters"]["bindingCalls"] == record["before"]["counters"]["bindingCalls"] == 0


def verify(args) -> dict:
    with tempfile.TemporaryDirectory(prefix="ressectionlab-runtime-") as temporary:
        work = Path(temporary).resolve()
        if args.runtime == "frozen":
            assert args.executable and not args.package_root
            executable = args.executable.resolve(strict=True)
            package_root = executable.parent / "_internal"
            command = [str(executable), "--native-probe"]
            copied = work / "engine"
            shutil.copytree(executable.parent, copied, symlinks=True)
            isolated_package = copied / "_internal/resectionlab"
            isolated_root = copied / "_internal"
            isolated_command = [str(copied / executable.name), "--native-probe"]
        else:
            assert args.package_root and not args.executable
            package_root = args.package_root.resolve(strict=True)
            if args.runtime == "wheel":
                checkout = Path(__file__).resolve().parents[1]
                assert not package_root.is_relative_to(checkout), "Install the wheel outside the checkout"
            command = python_command(args.python.absolute(), package_root)
            isolated_root = work / "package"
            isolated_package = isolated_root / "resectionlab"
            shutil.copytree(package_root / "resectionlab", isolated_package,
                            ignore=shutil.ignore_patterns("__pycache__"))
            isolated_command = python_command(args.python.absolute(), isolated_root)

        cpp = run_probe(command, "cpp", work)
        successful(cpp, True, package_root)
        reference = run_probe(command, "reference", work)
        successful(reference, False, package_root)
        assert reference["publicCall"] == cpp["publicCall"], "Reference/C++ ordered cells differ"
        assert reference["replay"] == cpp["replay"], "Reference/C++ accepted replay differs"
        for key in ("physicalSubsteps", "elapsedSeconds", "quaternionsSha256"):
            assert reference["trajectory"][key] == cpp["trajectory"][key], "Reference/C++ trajectory differs"

        manifest_path = isolated_package / "_geometry_cpp.build.json"
        original = manifest_path.read_bytes()
        manifest = json.loads(original)
        manifest["binarySha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        integrity = run_probe(isolated_command, "auto", work)
        assert not integrity["passed"] and integrity["exitCode"] != 0, "Integrity failure silently fell back"
        assert integrity["errorType"] == "NativeIntegrityError", integrity
        assert "identity" in integrity["error"].lower() or "integrity" in integrity["error"].lower(), integrity
        manifest_path.write_bytes(original)

        binary = isolated_package / manifest["binaryName"]
        binary.unlink()
        unavailable = run_probe(isolated_command, "auto", work)
        successful(unavailable, False, isolated_root)
        assert unavailable["backend"]["fallbackReason"] == "extension-unavailable"
        assert unavailable["publicCall"] == reference["publicCall"]
        assert unavailable["replay"] == reference["replay"]
        explicit = run_probe(isolated_command, "cpp", work)
        assert not explicit["passed"] and explicit["exitCode"] != 0, "Explicit C++ accepted a missing extension"
        assert explicit["errorType"] == "NativeUnavailableError", explicit
        return {"schema": "resectionlab-cpp-runtime-verification-v1", "runtime": args.runtime,
                "passed": True, "cpp": cpp, "reference": reference,
                "unavailableAuto": unavailable, "unavailableCpp": explicit, "integrityAuto": integrity}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("source", "wheel", "frozen"), required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve previous evidence; choose a new output file")
    try:
        result = verify(args)
    except Exception as error:
        result = {"schema": "resectionlab-cpp-runtime-verification-v1", "runtime": args.runtime,
                  "passed": False, "errorType": type(error).__name__, "error": str(error)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"runtime": args.runtime, "passed": result["passed"], "output": str(args.output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
