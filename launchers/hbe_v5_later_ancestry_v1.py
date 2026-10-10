"""Small saved-metadata admission for ordinal-9 policy ancestry.

Candidate tracked location: launchers/hbe_v5_later_ancestry_v1.py. The caller
must first run the frozen full predecessor chain and exact SHA-bound reads of
the native receipt and one-file sidecar. This module performs no file I/O and
does not treat a sidecar assertion as an observed physical measurement.
"""
from __future__ import annotations

import math

ROW9_NATIVE_SHA = "d2476c6355a4c23421c829eff4b9a5db1899b4013b73bacf7e2701eda0def144"
ROW9_SIDECAR_SHA = "656dbcdc6edb69a516c68f0e5ff81ab22c2afe4732c7ad74f89efd595435f8eb"
ROW9_INNER_SHA = "f0a3ff4c9fc4a19eada8448d0837274adff0efaa33b60b2bf903bf78463745b6"
ROW9_OUTER_SHA = "47d9d12a810fc03c43bf4715e2e6e04fe3c0c2e7b935a2cd464616a56a55fe7b"
ROW9_COMMIT = "e804aa88446d609b7e135973a1159f463b767cf2"
ROW9_STATUS = "passed_numerical_software_only_with_v2_cumulative_ledger_v1"
ROW9_RUN = "tension:N16:S60:reference"
ONE_MIB = 1024**2


def _finite(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Finite nonnegative saved resource required")
    return value


def _same_ledger(actual: dict, expected: dict) -> bool:
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    for key, value in expected.items():
        other = actual[key]
        if type(value) in (int, float) and type(other) in (int, float):
            if not math.isclose(value, other, rel_tol=0, abs_tol=1e-9):
                return False
        elif other != value:
            return False
    return True


def row9_charge(*, old_after9: dict, row8_exact: dict, sidecar: dict,
                sidecar_sha256: str, native: dict, native_sha256: str,
                remaining, row9_policy: dict) -> dict:
    """Return two arithmetic claims only after exact saved-event checks.

    The caller must bind `sidecar_sha256` and `native_sha256` to stat-checked
    saved bytes, and must obtain `old_after9` from frozen validation at index
    10 or the validated prior-ledger fields in row 10's native receipt. It is
    always the complete old ledger through ordinal 9.
    """
    if (sidecar_sha256 != ROW9_SIDECAR_SHA or native_sha256 != ROW9_NATIVE_SHA
            or not isinstance(sidecar, dict) or not isinstance(native, dict)
            or sidecar.get("schema") != "hbe-v5-ordinal9-continuation-sidecar-v1"
            or sidecar.get("status") != ROW9_STATUS
            or sidecar.get("source_commit") != ROW9_COMMIT
            or sidecar.get("envelope_sha256") != ROW9_OUTER_SHA
            or sidecar.get("inner_release_sha256") != ROW9_INNER_SHA
            or sidecar.get("policy") != row9_policy
            or sidecar.get("native_receipt_sha256") != ROW9_NATIVE_SHA
            or sidecar.get("native_receipt_status") !=
               "passed_numerical_software_only"
            or sidecar.get("native_calls_authorized") != 1
            or sidecar.get("native_calls_attempted") != 1
            or sidecar.get("hbe_readout_calls_attempted") != 1
            or native.get("run_id") != ROW9_RUN
            or native.get("ordinal") != 9
            or native.get("source_commit") != ROW9_COMMIT
            or native.get("release_sha256") != ROW9_INNER_SHA
            or native.get("status") != "passed_numerical_software_only"
            or native.get("no_retry") is not True
            or native.get("native_calls_attempted") != 1
            or native.get("readout_calls_attempted") != 1):
        raise ValueError("Exact independently reviewed ordinal-9 event required")
    source = sidecar.get("extension_sources_before")
    if (not isinstance(source, dict)
            or source != sidecar.get("extension_sources_after")
            or source.get("source_commit") != ROW9_COMMIT
            or source.get("source_hashes") !=
               sidecar.get("extension_sources_terminal", {}).get("source_hashes")):
        raise ValueError("Ordinal-9 source closure changed during execution")
    for stage in ("native", "readout"):
        cleanup = sidecar.get("owned_stage_cleanup", {}).get(stage, {})
        if (cleanup.get("contained") is not True
                or cleanup.get("direct_child_reaped") is not True
                or cleanup.get("fallback_used") is not False
                or cleanup.get("errors") != []
                or cleanup.get("remaining_members") != []):
            raise ValueError("Ordinal-9 owned child cleanup differs")
    for key, count, size in (("preflight_hint_audit", 16, 2_849_433_519),
                             ("final_hint_audit", 32, 5_698_867_038)):
        hint = sidecar.get(key, {})
        if (hint.get("eligible_files") != count
                or hint.get("hinted_file_opens") != count
                or hint.get("eligible_bytes") != size):
            raise ValueError("Ordinal-9 complete hint coverage differs")
    for key, floor in (("host_before_validation", 54),
                       ("host_before_native_reservation", 45),
                       ("host_immediately_before_native_supervision", 45)):
        host = sidecar.get(key, {})
        if (host.get("kernel_pressure_mask") != 1
                or type(host.get("available_percent")) is not int
                or host["available_percent"] < floor):
            raise ValueError("Ordinal-9 sampled host gate differs")
    charge = sidecar.get("extension_resource_charge", {})
    prep = _finite(charge.get("launcher_inclusive_prep_seconds_with_finalization_reserve"))
    native_prep = _finite(native.get("prep_elapsed_seconds"))
    native_seconds = _finite(native.get("native_stage", {}).get("elapsed_seconds"))
    readout_seconds = _finite(native.get("readout_stage", {}).get("elapsed_seconds"))
    closed_native = charge.get("native_closed_output_bytes_at_check")
    if (prep < native_prep or prep >= remaining.PREP_WALL
            or type(closed_native) is not int or closed_native < 0
            or charge.get("per_row_prep_wall_seconds") != remaining.PREP_WALL
            or charge.get("native_stage_seconds") != native_seconds
            or charge.get("readout_stage_seconds") != readout_seconds
            or charge.get("sidecar_reserved_output_bytes") != ONE_MIB
            or charge.get("charged_current_output_bytes") !=
               closed_native + ONE_MIB):
        raise ValueError("Ordinal-9 wrapper resource charge differs")
    prior = sidecar.get("old_validation_identity", {}).get("previous")
    row8 = row8_exact.get("surcharge", {})
    if (not isinstance(prior, dict) or not isinstance(row8, dict)
            or sidecar.get("v2_ordinal8_exact_evidence") != row8_exact
            or sidecar.get("old_validation_identity", {}).get("index") != 9
            or sidecar.get("old_validation_identity", {}).get("run_id") != ROW9_RUN
            or native.get("prior_receipt_sha256") != prior.get("sha256")
            or len(prior.get("sha256", [])) != 9
            or prior["sha256"][-1] != row8.get("source_ordinal8_native_receipt_sha256")):
        raise ValueError("Ordinal-8 surcharge or old predecessor identity differs")
    expected_charged_prior = dict(prior)
    expected_charged_prior["v2_surcharge_applied"] = row8
    for field, value in (("prep_seconds", row8.get("prep_wall_seconds")),
                         ("combined_wall_seconds", row8.get("prep_wall_seconds")),
                         ("output_bytes", row8.get("output_bytes")),
                         ("combined_output_bytes", row8.get("output_bytes"))):
        expected_charged_prior[field] += _finite(value)
    if not _same_ledger(sidecar.get("charged_previous"), expected_charged_prior):
        raise ValueError("Ordinal-8 predecessor charge missing or duplicated")
    aggregate = remaining._check_aggregate(
        expected_charged_prior, native_seconds, readout_seconds, prep,
        closed_native + ONE_MIB, 1)
    if charge.get("aggregate_with_extension") != aggregate:
        raise ValueError("Ordinal-9 frozen aggregate charge differs")
    expected_after = dict(prior)
    expected_after["sha256"] = list(prior["sha256"]) + [ROW9_NATIVE_SHA]
    for field, increment in (("native_seconds", native_seconds),
                             ("readout_seconds", readout_seconds),
                             ("prep_seconds", native_prep),
                             ("output_bytes", closed_native),
                             ("combined_wall_seconds", native_seconds +
                              readout_seconds + native_prep),
                             ("combined_output_bytes", closed_native),
                             ("native_calls", 1)):
        expected_after[field] += increment
    if not _same_ledger(old_after9, expected_after):
        raise ValueError("Frozen old ledger does not include exact ordinal-9 result")
    supplemental = sidecar.get("supplemental_ledger_after_ordinal9", {})
    if (supplemental.get("schema") != "hbe-v5-ordinal9-continuation-ledger-v1"
            or supplemental.get("through_ordinal") != 9
            or supplemental.get("ordinal9_native_receipt_sha256") != ROW9_NATIVE_SHA
            or supplemental.get("ordinal9_inclusive_prep_seconds") != prep
            or supplemental.get("ordinal9_reserved_sidecar_output_bytes") != ONE_MIB
            or supplemental.get("ordinal8_surcharge") != row8):
        raise ValueError("Ordinal-9 supplemental ledger identity differs")
    after_charge = dict(expected_charged_prior)
    after_charge["sha256"] = list(prior["sha256"]) + [ROW9_NATIVE_SHA]
    for field, increment in (("native_seconds", native_seconds),
                             ("readout_seconds", readout_seconds),
                             ("prep_seconds", prep),
                             ("output_bytes", closed_native + ONE_MIB),
                             ("combined_wall_seconds", native_seconds +
                              readout_seconds + prep),
                             ("combined_output_bytes", closed_native + ONE_MIB),
                             ("native_calls", 1)):
        after_charge[field] += increment
    if not _same_ledger(supplemental.get("charged_cumulative_ledger"), after_charge):
        raise ValueError("Ordinal-9 supplemental ledger arithmetic differs")
    return {"ordinal": 9, "native_receipt_sha256": ROW9_NATIVE_SHA,
            "sidecar_sha256": ROW9_SIDECAR_SHA,
            "native_prep_seconds": native_prep,
            "launcher_inclusive_prep_seconds": prep,
            "reserved_sidecar_output_bytes": ONE_MIB}
