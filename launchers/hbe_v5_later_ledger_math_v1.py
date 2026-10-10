"""Pure arithmetic for later HBE rows; caller must authenticate source evidence.

This is not an admission gate. In particular, a caller-supplied charge is never
proof that a policy sidecar or native receipt exists. The execution core must
first run the frozen predecessor validator and independently verify every
source, release, receipt and sidecar behind the charge.
"""
from __future__ import annotations

import math
import re

HEX64 = re.compile(r"[0-9a-f]{64}\Z")

LEDGER_KEYS = frozenset({
    "sha256", "native_seconds", "readout_seconds", "prep_seconds",
    "output_bytes", "native_calls", "combined_wall_seconds",
    "combined_output_bytes", "supplement_readout_seconds",
    "supplement_prep_seconds", "supplement_output_bytes",
    "supplement_replay_calls",
})
CHARGE_KEYS = frozenset({
    "ordinal", "native_receipt_sha256", "sidecar_sha256",
    "native_prep_seconds", "launcher_inclusive_prep_seconds",
    "reserved_sidecar_output_bytes",
})


def _finite(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative resource value required")
    return float(value)


def charge_old_baseline(old: dict, expected_old: dict, charges: list[dict],
                        index: int) -> dict:
    """Add authenticated sidecar deltas once to the exact frozen old baseline.

    `expected_old` must be derived from the independently verified old-chain
    receipt ancestry; it cannot be an asserted property of a generated charge.
    Rows 10 and 11 have, respectively, two and three extension predecessors.
    """
    if type(index) is not int or index not in (10, 11):
        raise ValueError("Only declared later-row indices are supported")
    if (not isinstance(old, dict) or not isinstance(expected_old, dict)
            or set(old) != LEDGER_KEYS or old != expected_old):
        raise ValueError("Uncharged frozen old-chain baseline required")
    hashes = old["sha256"]
    if (not isinstance(hashes, list) or len(hashes) != index
            or not all(isinstance(item, str) and HEX64.fullmatch(item)
                       for item in hashes)
            or type(old["native_calls"]) is not int
            or old["native_calls"] != index):
        raise ValueError("Frozen predecessor count or hashes differ")
    if (type(old["output_bytes"]) is not int
            or type(old["combined_output_bytes"]) is not int
            or type(old["supplement_output_bytes"]) is not int):
        raise ValueError("Frozen predecessor output units differ")
    if not isinstance(charges, list) or len(charges) != index - 8:
        raise ValueError("Every declared extension predecessor is required")
    for key in LEDGER_KEYS - {"sha256", "native_calls", "supplement_replay_calls"}:
        _finite(old[key])
    if (type(old["supplement_replay_calls"]) is not int
            or old["supplement_replay_calls"] < 0
            or not math.isclose(old["combined_wall_seconds"],
                old["native_seconds"] + old["readout_seconds"] +
                old["prep_seconds"] + old["supplement_readout_seconds"] +
                old["supplement_prep_seconds"], rel_tol=0, abs_tol=1e-9)
            or old["combined_output_bytes"] !=
                old["output_bytes"] + old["supplement_output_bytes"]):
        raise ValueError("Frozen old-chain arithmetic differs")
    prep_delta = 0.0
    output_delta = 0
    for ordinal, charge in enumerate(charges, start=8):
        if (not isinstance(charge, dict) or set(charge) != CHARGE_KEYS
                or type(charge["ordinal"]) is not int
                or charge["ordinal"] != ordinal
                or charge["native_receipt_sha256"] != hashes[ordinal]
                or not isinstance(charge["sidecar_sha256"], str)
                or not HEX64.fullmatch(charge["sidecar_sha256"])
                or type(charge["reserved_sidecar_output_bytes"]) is not int
                or charge["reserved_sidecar_output_bytes"] != 1024**2):
            raise ValueError("Extension charge identity, order or reservation differs")
        native_prep = _finite(charge["native_prep_seconds"])
        launcher_prep = _finite(charge["launcher_inclusive_prep_seconds"])
        if launcher_prep < native_prep or launcher_prep >= 150:
            raise ValueError("Extension preparation delta or frozen cap differs")
        prep_delta += launcher_prep - native_prep
        output_delta += charge["reserved_sidecar_output_bytes"]
    result = dict(old)
    result["sha256"] = list(hashes)
    result["prep_seconds"] += prep_delta
    result["output_bytes"] += output_delta
    result["combined_wall_seconds"] += prep_delta
    result["combined_output_bytes"] += output_delta
    return {"old_native_receipt_ledger": dict(old),
            "authenticated_charge_order": [item["ordinal"] for item in charges],
            "incremental_prep_seconds": prep_delta,
            "incremental_output_bytes": output_delta,
            "charged_cumulative_ledger": result}
