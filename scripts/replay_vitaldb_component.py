#!/usr/bin/env python3
"""Local-only VitalDB numeric QC and deterministic observed replay CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from resectionlab.vitaldb_observed import ObservedReplay, VitalDBError, retrospective_qc, check_original_request


def _output_paths(input_path: Path, paths: list[Path | None]) -> None:
    """Outputs must be new files in explicit artifact/output areas, never inputs."""
    allowed = [ROOT / 'artifacts/vitaldb-recorded-component-v1', ROOT / 'outputs',
               ROOT / 'build/vitaldb-replay']
    resolved = []
    for path in (p for p in paths if p is not None):
        target = path.resolve()
        if target == input_path.resolve() or (path.exists() and input_path.exists() and path.samefile(input_path)):
            raise VitalDBError('Output aliases the authenticated input')
        if target in resolved:
            raise VitalDBError('Output and snapshot paths alias each other')
        if path.exists() or path.is_symlink():
            raise VitalDBError('Output already exists; overwrite refused')
        if not any(target.is_relative_to(directory.resolve()) for directory in allowed):
            raise VitalDBError('Output must be in artifacts/vitaldb-recorded-component-v1, outputs, or build/vitaldb-replay')
        if not path.parent.is_dir():
            raise VitalDBError('Output parent directory must already exist')
        resolved.append(target)


def _write_new(path: Path, text: str) -> None:
    # Exclusive creation rejects a file/symlink installed after initial validation.
    with path.open('x') as handle:
        handle.write(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("qc", "replay", "original-check"))
    parser.add_argument("--input", type=Path, default=ROOT / "data/vitaldb-1.0.0/mirror-01/0003.vital")
    parser.add_argument("--clock", type=float, help="Required for replay: unshifted native source seconds, not elapsed time")
    parser.add_argument("--reopen", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    # Failure reporting must never write to an unvalidated destination either.
    try:
        _output_paths(args.input, [args.output, args.snapshot])
    except (ValueError, OSError) as exc:
        print(json.dumps({'status': 'refused', 'error': str(exc), 'no_output_written': True}))
        return 2
    try:
        if args.mode in ("qc", "original-check"):
            if args.reopen or args.snapshot or args.clock is not None:
                parser.error("Clock and snapshot options only apply to replay")
            result = (retrospective_qc(args.input) if args.mode == "qc" else check_original_request(args.input))
        else:
            if args.clock is None:
                parser.error("Replay requires --clock in unshifted native source seconds")
            if args.reopen and args.reopen.stat().st_size > 16384:
                raise VitalDBError("Snapshot size bound exceeded")
            replay = (ObservedReplay.reopen(args.input, json.loads(args.reopen.read_text()))
                      if args.reopen else ObservedReplay(args.input))
            result = replay.advance(args.clock)
            if args.snapshot:
                _write_new(args.snapshot, json.dumps(replay.snapshot(), indent=2, allow_nan=False) + "\n")
        text = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output:
            _write_new(args.output, text)
        else:
            print(text, end="")
        return 2 if result.get("status") == "refused" else 0
    except (ValueError, OSError) as exc:
        result = {"status": "refused", "error": str(exc), "input": str(args.input),
                  "partial_import_admitted": False}
        text = json.dumps(result, indent=2) + "\n"
        if args.output:
            try:
                _write_new(args.output, text)
            except OSError:
                print(text, end="")
        else:
            print(text, end="")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
