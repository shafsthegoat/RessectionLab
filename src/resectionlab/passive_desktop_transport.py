"""Live-only transport for the fixed generated passive v1 episode.

The saved-workspace replay authority is a separate consumer. This module does
not admit patient data, learned selectors, or a v2 registration decision.
"""
from __future__ import annotations

from dataclasses import asdict
import json

import numpy as np

from .core import CaseData, SourceRef, array_digest, semantic_digest
from .persistent_passive import (PersistentPassiveTask, VERSION,
                                 make_passive_development_task, run_passive_development_episode)
from .structural_evidence import structural_frame_hash


def passive_display_case(task: PersistentPassiveTask) -> CaseData:
    if type(task) is not PersistentPassiveTask or getattr(task, "_decision_protocol", None) is not None:
        raise ValueError("Only the fixed generated v1 task can supply the passive display")
    source = task.case
    return CaseData(VERSION, source.structural_intensity,
        {"generated_nominal_target": source.nominal_target > 0}, source.affine_ras_mm,
        (SourceRef(VERSION, "generated://" + VERSION,
                   native_frame="RAS+", provenance="simulated"),), frame="RAS+",
        brain_mask=source.observed_support,
        metadata={"is_synthetic": True, "evidence_kind": "generated_software_fixture",
                  "patient_admission": False, "source_task_hash": source.source_hash,
                  "scope": "declared pre-existing free tunnel and passive tool geometry only"})


def passive_source_binding(task: PersistentPassiveTask, case: CaseData) -> dict:
    if type(case) is not CaseData or type(task) is not PersistentPassiveTask:
        raise TypeError("Exact generated native task and display CaseData are required")
    if (getattr(task, "_decision_protocol", None) is not None or
            array_digest(case.mri) != array_digest(task.case.structural_intensity) or
            array_digest(case.brain_mask) != array_digest(task.case.observed_support) or
            array_digest(case.compartments["generated_nominal_target"]) !=
            array_digest(task.case.nominal_target > 0) or
            array_digest(case.affine) != array_digest(task.case.affine_ras_mm)):
        raise ValueError("Display differs from the fixed generated native source")
    body = {"caseHash": case.semantic_hash,
            "sourceHash": task.case.source_hash,
            "decisionModelHash": task.decision_model_hash,
            "structuralIntensityHash": array_digest(case.mri),
            "supportHash": array_digest(case.brain_mask),
            "nominalTargetMaskHash": array_digest(case.compartments["generated_nominal_target"]),
            "affineHash": array_digest(case.affine),
            "sourceFrameHash": structural_frame_hash(case),
            "tools": [asdict(tool) for tool in task.case.tools]}
    return {**body, "bindingHash": semantic_digest(body)}


def execute_persistent_passive_episode(*, paired=True, inspection_status="unresolved",
                                       cancelled=None) -> dict:
    if paired is not True or inspection_status not in ("verified", "unresolved"):
        raise ValueError("Only the fixed paired generated v1 episode is admitted")
    task = make_passive_development_task(inspection_status=inspection_status,
                                          paired=True, cancelled=cancelled)
    episode = run_passive_development_episode(inspection_status=inspection_status,
                                               paired=True, cancelled=cancelled)
    case = passive_display_case(task)
    binding = passive_source_binding(task, case)
    if (episode["sourceHash"] != binding["sourceHash"] or
            episode["decisionModelHash"] != binding["decisionModelHash"]):
        raise RuntimeError("Executed passive episode differs from its display source")
    canonical = json.dumps({key: value for key, value in episode.items() if key != "episodeId"},
                           sort_keys=True, separators=(",", ":"), allow_nan=False)
    if episode["episodeId"] != semantic_digest({key: value for key, value in episode.items()
                                                  if key != "episodeId"}):
        raise RuntimeError("Passive episode identity changed before publication")
    return {"case": case, "episode": episode,
            "episodeCanonicalJson": canonical,
            "sourceBinding": binding}
