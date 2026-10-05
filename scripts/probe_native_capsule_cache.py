#!/usr/bin/env python3
"""Bounded synthetic reference/cache/reversal probe; no patient or training input."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import fields, is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from resectionlab import geometry, native_proposals, native_resection as native
from resectionlab import evaluation, experimental_capsule_cache as cache_module
from resectionlab.evaluation import independent_check_native_history
from resectionlab.experimental_capsule_cache import ExactCapsuleCoverCache, inject_native_capsule_cache


def encoded(value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        if is_dataclass(item):
            return {field.name: getattr(item, field.name) for field in fields(item)}
        raise TypeError(type(item).__name__)
    return json.dumps(value, default=convert, sort_keys=True, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Retain the previous probe; choose a new output path")
    if not 2 <= args.repetitions <= 20:
        raise ValueError("Use two to twenty synthetic repetitions")
    paths = [Path(module.__file__) for module in (geometry, native_proposals, native, evaluation, cache_module)] + [Path(__file__)]
    identities = {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}
    tissue = np.zeros((9, 9, 10), bool)
    tissue[1:8, 1:8, 2:9] = True
    labels = np.zeros(tissue.shape, np.int16)
    labels[3:6, 3:6, 4:8] = 1
    config = native.NativeResectionConfig(tissue, labels, np.eye(4),
        geometry.AccessWindow([4, 4, 1.5], [0, 0, 1], 6), native.NATIVE_GENERIC_TOOLS,
        "synthetic-capsule-cache-probe", "explicit 9x9x10 synthetic source tissue")
    prepared = perf_counter()
    cache = ExactCapsuleCoverCache(config)
    preparation_seconds = perf_counter() - prepared
    reference, phases, histories = None, [], {}

    def peak_rss():
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024))

    def signature(engine, result):
        return (encoded(result), engine.state_hash, encoded(engine.history),
                *(getattr(engine, field).tobytes() for field in
                  ("remaining_mask", "removed_mask", "contact_mask", "connected_free_mask")))

    for mode in ("reference", "cached", "reference_reversal"):
        engine = native.NativeResectionEngine(config)
        with inject_native_capsule_cache(cache) if mode == "cached" else nullcontext():
            for repetition in range(args.repetitions):
                before = cache.stats()
                started = perf_counter()
                engine.reset()
                result = engine.execute_stroke("native-wide-aspiration", [4, 4, 7])
                elapsed = perf_counter() - started
                assert result.feasible
                actual = signature(engine, result)
                if reference is None:
                    reference = actual
                assert actual == reference
                after = cache.stats()
                phases.append({"mode": mode, "repetition": repetition, "seconds": elapsed,
                    "cache_hits_delta": after["hits"] - before["hits"],
                    "cache_misses_delta": after["misses"] - before["misses"],
                    "cache_query_seconds_delta": after["query_seconds"] - before["query_seconds"],
                    "cache_compute_seconds_delta": after["compute_seconds"] - before["compute_seconds"],
                    "process_cumulative_peak_rss_bytes": peak_rss()})
        histories[mode] = engine.history
        assert native.capsule_voxel_indices is geometry.capsule_voxel_indices

    case = SimpleNamespace(mri=np.zeros(tissue.shape), affine=config.affine,
                           frame="RAS+", semantic_hash=config.source_hash)
    audits = {}
    for mode in ("reference", "cached"):
        started = perf_counter()
        audit = independent_check_native_history(case, config.tools, histories[mode],
            tissue_mask=config.tissue_mask, access=config.access, hard_exclusion=config.hard_exclusion)
        assert audit.feasible
        audits[mode] = {"seconds": perf_counter() - started, "receipt": audit.to_dict()}
    assert audits["reference"]["receipt"] == audits["cached"]["receipt"]

    hard = np.zeros_like(tissue)
    hard[4, 4, 4] = True
    blocked_config = replace(config, hard_exclusion=hard)
    blocked_cache = ExactCapsuleCoverCache(blocked_config)
    blocked_reference = native.NativeResectionEngine(blocked_config).preview_stroke("native-wide-aspiration", [4, 4, 7])
    with inject_native_capsule_cache(blocked_cache):
        blocked_engine = native.NativeResectionEngine(blocked_config)
        blocked_actual = blocked_engine.preview_stroke("native-wide-aspiration", [4, 4, 7])
    assert encoded(blocked_actual) == encoded(blocked_reference)
    assert not blocked_actual.feasible and blocked_actual.reason.startswith("HARD_GEOMETRY")
    assert not blocked_engine.removed_mask.any() and blocked_cache.stats()["calls"] == 0
    assert native.capsule_voxel_indices is geometry.capsule_voxel_indices
    assert identities == {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}
    report = {"source_files_sha256": identities, "config_hash": config.fingerprint,
        "phases": phases, "cache_preparation_seconds": preparation_seconds,
        "cache": cache.stats(), "histories": histories, "independent_audits": audits,
        "negative_hard_barrier": {"preview": json.loads(encoded(blocked_actual)),
                                  "cache": blocked_cache.stats(), "removed_cells": 0},
        "exact_history_masks_ancestry_and_reversal": True, "query_symbol_restored": True,
        "timing_scope": "Synthetic local repetitions with no quiet-window guarantee; not a patient or whole-episode speedup claim.",
        "rss_scope": "Cumulative process peak RSS, not isolated cache allocation cost.",
        "patient_cases_used": 0, "gradient_updates": 0, "final_or_stress_worlds_used": False,
        "clinical_deficit_probability": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"phases": phases, "cache": cache.stats(),
                      "independent_audits_passed": 2, "negative_hard_barrier_preserved": True}))


if __name__ == "__main__":
    main()
