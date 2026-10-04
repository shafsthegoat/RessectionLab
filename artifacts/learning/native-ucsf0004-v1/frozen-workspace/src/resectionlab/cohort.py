"""Conservative patient grouping and preoperative input audit.

Collection release totals are descriptive metadata, never evaluated sample
counts. This utility checks a small explicit record registry and its local
source evidence without downloading anything or inventing missing identities.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


OUTER_SPLITS = {"development", "final_evaluation", "excluded", "unassigned"}
USES = {"global_development", "population_pretraining", "case_optimization",
        "checkpoint_selection", "final_scoring", "viewer_qc"}
IDENTITY_STATES = {"verified_primary", "mirror_attributed", "unknown"}


def _timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Timestamps must include a timezone")
    return result


def input_exclusion_reason(item: Mapping[str, Any], planning_as_of: str | None) -> str | None:
    """Return why an input cannot enter the primary preoperative optimizer.

    A source declaration that an image was acquired preoperatively can support
    an imaging-only replay when calendar dates are deidentified. It cannot
    establish when a molecular or clinical result became available.
    """
    kind = item.get("kind")
    if kind not in {"imaging", "source_annotation", "molecular", "clinical", "outcome"}:
        return "unsupported_input_kind"
    if not item.get("source_evidence_ids"):
        return "input_provenance_unknown"
    if item.get("operative_phase") == "postoperative" or kind == "outcome":
        return "postoperative_information_excluded"
    if item.get("evidence_type", "observed") not in {"observed", "estimated"}:
        return "nonobserved_context_excluded_from_primary"
    available_at = item.get("available_at")
    if available_at is not None:
        if planning_as_of is None:
            return "planning_cutoff_unknown"
        try:
            available = _timestamp(available_at)
            cutoff = _timestamp(planning_as_of)
            measured = item.get("measurement_at")
            if measured is not None and _timestamp(measured) > available:
                return "availability_precedes_measurement"
        except (TypeError, ValueError):
            return "invalid_or_naive_timestamp"
        if available > cutoff:
            return "available_after_planning_cutoff"
        return None
    if (kind in {"imaging", "source_annotation"}
            and item.get("operative_phase") == "preoperative"
            and item.get("availability_basis") == "source_declares_preoperative_images"):
        if kind == "source_annotation" and item.get("track") != "annotation_assisted":
            return "source_annotation_requires_annotation_assisted_track"
        return None
    return "availability_time_unknown"


def check_evidence(spec: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Verify source bytes and a specific identity assertion, without fetching."""
    root = project_root.resolve()
    raw_path = spec.get("path", "")
    if not raw_path or Path(raw_path).is_absolute() or ".." in Path(raw_path).parts:
        return {"verified": False, "reason": "unsafe_evidence_path"}
    path = (root / raw_path).resolve()
    if not path.is_relative_to(root):
        return {"verified": False, "reason": "evidence_path_escapes_project"}
    if not path.is_file():
        return {"verified": False, "reason": "source_evidence_missing"}
    try:
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != spec.get("sha256"):
            return {"verified": False, "reason": "source_evidence_hash_mismatch"}
        kind = spec.get("kind")
        if kind == "delimited_row":
            rows = list(csv.DictReader(payload.decode("utf-8-sig").splitlines(), delimiter=spec.get("delimiter", ",")))
            match = spec.get("match", {})
            if not match:
                return {"verified": False, "reason": "identity_match_unspecified"}
            found = [row for row in rows if all(row.get(k) == v for k, v in match.items())]
            if len(found) != 1:
                return {"verified": False, "reason": "identity_row_missing_or_ambiguous"}
            if any(found[0].get(k) != v for k, v in spec.get("expect", {}).items()):
                return {"verified": False, "reason": "identity_assertion_mismatch"}
        elif kind == "contains_text":
            texts = spec.get("texts", [])
            if not texts or not all(text in payload.decode("utf-8") for text in texts):
                return {"verified": False, "reason": "source_assertion_missing"}
        elif kind == "json_fields":
            fields = spec.get("expect", {})
            data = json.loads(payload)
            if not fields or any(data.get(k) != v for k, v in fields.items()):
                return {"verified": False, "reason": "source_assertion_mismatch"}
        elif kind != "sha256":
            return {"verified": False, "reason": "unsupported_evidence_check"}
    except (OSError, UnicodeError, ValueError, TypeError):
        return {"verified": False, "reason": "unreadable_evidence"}
    return {"verified": True, "reason": None}


class _Groups:
    def __init__(self):
        self.parents: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parents.setdefault(item, item)
        if self.parents[item] != item:
            self.parents[item] = self.find(self.parents[item])
        return self.parents[item]

    def join(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        self.parents[max(a, b)] = min(a, b)


def audit_registry(registry: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    """Check known overlap conservatively; unknown identity cannot establish independence.

    ``case_optimization`` on an outer-final patient is allowed: it is part of
    the frozen per-case inference procedure. Population pretraining or global
    development on that patient is not. Within-patient world separation belongs
    to the independent experiment protocol, not a different patient identifier.
    """
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported cohort registry schema")
    evidence_specs = registry.get("evidence", [])
    evidence = {item["id"]: check_evidence(item, project_root) for item in evidence_specs}
    evidence_by_id = {item["id"]: item for item in evidence_specs}
    if len(evidence) != len(evidence_specs):
        raise ValueError("Duplicate evidence IDs")
    records = registry.get("records", [])
    by_id = {item["record_id"]: item for item in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate record IDs")
    aliases = _Groups()
    issues: list[dict[str, Any]] = []

    def issue(code: str, record_ids: list[str], detail: str, severity: str = "error") -> None:
        issues.append({"code": code, "record_ids": sorted(record_ids), "detail": detail, "severity": severity})

    def supported(refs: list[str]) -> bool:
        return bool(refs) and all(evidence.get(ref, {}).get("verified") for ref in refs)

    for link in registry.get("identity_links", []):
        names = link.get("identities", [])
        if len(names) < 2 or any(not isinstance(name, str) or ":" not in name for name in names):
            raise ValueError("Identity links require at least two namespaced identifiers")
        # Even unconfirmed reported overlap is grouped conservatively.
        for name in names[1:]:
            aliases.join(names[0], name)
        if not supported(link.get("evidence_ids", [])):
            issue("UNVERIFIED_IDENTITY_LINK", [], f"Cannot independently verify link {link['id']}", "limitation")

    groups = _Groups()
    known: dict[str, bool] = {}
    identity_to_record: dict[str, str] = {}
    for record_id, record in by_id.items():
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Record IDs must be nonempty strings")
        groups.find(record_id)
        if record.get("outer_split") not in OUTER_SPLITS:
            raise ValueError(f"Unsupported outer split on {record_id}")
        if not set(record.get("uses", [])).issubset(USES):
            raise ValueError(f"Unsupported use on {record_id}")
        if record.get("identity_status") not in IDENTITY_STATES:
            raise ValueError(f"Unsupported identity status on {record_id}")
        identity = record.get("identity")
        if identity is not None and (not isinstance(identity, str) or ":" not in identity):
            raise ValueError("Patient identifiers must include a source namespace")
        refs = record.get("identity_evidence_ids", [])
        identity_is_supported = any(
            identity in evidence_by_id.get(ref, {}).get("supports_identities", []) for ref in refs
        )
        known[record_id] = bool(identity and record["identity_status"] == "verified_primary"
                                and identity_is_supported and supported(refs))
        if identity:
            canonical = aliases.find(identity)
            if canonical in identity_to_record:
                groups.join(record_id, identity_to_record[canonical])
            identity_to_record[canonical] = record_id
        if not known[record_id]:
            severity = "error" if record["outer_split"] == "final_evaluation" else "limitation"
            issue("PATIENT_IDENTITY_NOT_VERIFIED", [record_id],
                  "Source identity or source bytes are missing, mirror-attributed, or unverified; exclude independence claims.", severity)

    for record_id, record in by_id.items():
        parent_id = record.get("derivative_of")
        if parent_id is None:
            continue
        if parent_id not in by_id:
            issue("UNRESOLVED_DERIVATIVE_PARENT", [record_id], f"Missing parent {parent_id}")
            known[record_id] = False
            continue
        parent = by_id[parent_id]
        if (known[record_id] and known[parent_id]
                and aliases.find(record["identity"]) != aliases.find(parent["identity"])):
            issue("CONTRADICTORY_DERIVATIVE_IDENTITY", [record_id, parent_id], "A derivative names a different verified patient from its parent.")
        groups.join(record_id, parent_id)
        seen = {record_id}
        ancestor = parent_id
        while ancestor in by_id:
            if ancestor in seen:
                issue("CYCLIC_DERIVATIVE_LINEAGE", sorted(seen), "Derivative parentage contains a cycle.")
                break
            seen.add(ancestor)
            ancestor = by_id[ancestor].get("derivative_of")

    grouped: dict[str, list[str]] = {}
    for record_id in by_id:
        grouped.setdefault(groups.find(record_id), []).append(record_id)
    summaries = []
    for record_ids in grouped.values():
        splits = {by_id[r]["outer_split"] for r in record_ids}
        uses = {use for r in record_ids for use in by_id[r].get("uses", [])}
        if {"development", "final_evaluation"}.issubset(splits):
            issue("CROSS_SPLIT_PATIENT_LEAKAGE", record_ids, "One patient, visit family, or derivative group appears in development and final evaluation.")
        if "final_evaluation" in splits and "population_pretraining" in uses:
            issue("FINAL_PATIENT_USED_FOR_POPULATION_PRETRAINING", record_ids, "Population pretraining cannot include an outer-final patient's visits or derivatives.")
        if "final_evaluation" in splits and "global_development" in uses:
            issue("FINAL_PATIENT_USED_FOR_GLOBAL_DEVELOPMENT", record_ids, "Global method development cannot use an outer-final patient.")
        summaries.append({"record_ids": sorted(record_ids), "outer_splits": sorted(splits),
                          "primary_identity_verified": all(known[r] for r in record_ids)})

    inputs = []
    for record_id, record in by_id.items():
        for item in record.get("inputs", []):
            reason = input_exclusion_reason(item, record.get("planning_as_of"))
            if not supported(item.get("source_evidence_ids", [])):
                reason = "input_source_evidence_unverified"
            inputs.append({"record_id": record_id, "input_id": item["input_id"],
                           "permitted_preoperative_input": reason is None, "exclusion_reason": reason})
            if item.get("used_by_primary_optimizer", False) and reason is not None:
                issue("UNAVAILABLE_INPUT_USED_BY_OPTIMIZER", [record_id], f"{item['input_id']}: {reason}")

    errors = [item for item in issues if item["severity"] == "error"]
    final_records = [r for r in records if r["outer_split"] == "final_evaluation"]
    return {
        "registry_valid": not errors, "record_count": len(records),
        "registered_patient_groups": len(summaries),
        "groups_with_verified_primary_identity": sum(g["primary_identity_verified"] for g in summaries),
        "final_patient_groups": sum("final_evaluation" in g["outer_splits"] for g in summaries),
        "independence_claim_status": ("no_final_patients_registered" if not final_records else
                                      "blocked_by_registry_errors" if errors else "known_identity_overlap_checks_passed"),
        "groups": sorted(summaries, key=lambda g: g["record_ids"]), "evidence": evidence,
        "inputs": inputs, "issues": issues,
        "limits": ["Release counts are not evaluated sample counts", "No undeclared or undiscoverable cross-dataset identity overlap is ruled out",
                   "Source-byte equivalence and identity are separate from anatomical eligibility", "Within-patient world isolation requires separate protocol checks"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("manifests/cohort_registry.json"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        report = audit_registry(json.loads(args.registry.read_text()), args.project_root)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2))
    return 0 if report["registry_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
