"""Independent constructed-table controls; no patient data or fitting."""
import csv
from dataclasses import FrozenInstanceError
import hashlib
import io

import pytest

from resectionlab import rhuh_outcomes as rhuh


def row(patient_id="RHUH-0001", **overrides):
    result = {header: "unused analytic token" for header in rhuh.RHUH_HEADERS}
    result.update({
        rhuh.PATIENT_ID_FIELD: patient_id,
        rhuh.DEFICIT_FIELD: "No",
        rhuh.PREOPERATIVE_KPS_FIELD: "80",
        rhuh.POSTOPERATIVE_KPS_FIELD: "70",
        rhuh.PREOPERATIVE_CE_VOLUME_FIELD: "12.5",
    })
    result.update(overrides)
    return result


def encode(rows, headers=None):
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    headers = rhuh.RHUH_HEADERS if headers is None else headers
    writer.writerow(headers)
    writer.writerows([record[header] for header in headers] for record in rows)
    return out.getvalue().encode("utf-8")


def parse(rows):
    raw = encode(rows)
    return rhuh.parse_rhuh_csv(raw, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_all_disallowed_fields_and_container_identity_are_outside_projection():
    initial = row()
    modified = row()
    allowed = {rhuh.PATIENT_ID_FIELD, rhuh.PREOPERATIVE_KPS_FIELD,
               rhuh.PREOPERATIVE_CE_VOLUME_FIELD}
    for field in rhuh.RHUH_HEADERS:
        if field not in allowed:
            modified[field] = 'changed, quoted "analytic" token\nsecond line'
    modified[rhuh.DEFICIT_FIELD] = "Major Persistent"
    modified[rhuh.POSTOPERATIVE_KPS_FIELD] = "30"
    before, after = parse([initial]), parse([modified])
    left, right = before.records[0], after.records[0]
    assert before.source.sha256 != after.source.sha256
    assert left.any_recorded_postoperative_deficit is False
    assert right.any_recorded_postoperative_deficit is True
    assert left.postoperative_kps != right.postoperative_kps
    assert dict(left.raw_fields) == initial
    assert dict(right.raw_fields) == modified
    assert left.preoperative_projection().to_dict() == right.preoperative_projection().to_dict()
    assert left.preoperative_projection().semantic_hash == right.preoperative_projection().semantic_hash


@pytest.mark.parametrize("field,value", [
    ("Preoperative KPS", "90"),
    ("Preoperative  contrast enhancing tumor volume (cm3)", "13.5"),
])
def test_each_permitted_value_changes_feature_identity(field, value):
    before = parse([row()]).records[0].preoperative_projection()
    after = parse([row(**{field: value})]).records[0].preoperative_projection()
    assert before.semantic_hash != after.semantic_hash
    assert before.to_dict() != after.to_dict()


def test_exact_raw_tokens_and_unknown_target_are_retained_separately():
    first = row(**{rhuh.DEFICIT_FIELD: "NA", rhuh.POSTOPERATIVE_KPS_FIELD: ""})
    second = row("RHUH-0002", **{rhuh.DEFICIT_FIELD: "No"})
    data = parse([first, second])
    missing, observed = data.records
    assert missing.deficit_category is None
    assert missing.any_recorded_postoperative_deficit is None
    assert missing.postoperative_kps is None
    assert missing.raw_fields[rhuh.DEFICIT_FIELD] == "NA"
    assert missing.raw_fields[rhuh.POSTOPERATIVE_KPS_FIELD] == ""
    assert observed.any_recorded_postoperative_deficit is False
    with pytest.raises(TypeError):
        missing.raw_fields[rhuh.DEFICIT_FIELD] = "No"
    with pytest.raises((FrozenInstanceError, AttributeError)):
        missing.patient_id = "RHUH-0002"
    copy = missing.preoperative_projection().to_dict()
    copy["preoperative_kps"] = -999
    assert missing.preoperative_projection().preoperative_kps == 80


def test_duplicate_identity_rejected_even_when_targets_disagree():
    with pytest.raises(ValueError):
        parse([row(), row(**{rhuh.DEFICIT_FIELD: "Transient"})])


@pytest.mark.parametrize("field,value", [
    ("Patient ID", "RHUH-0041"),
    ("Patient ID", "RHUH-0001 "),
    ("Postoperative Neurological Deficit", "No neurological deficit"),
    ("Preoperative KPS", "80.5"),
    ("Preoperative  contrast enhancing tumor volume (cm3)", "inf"),
    ("Preoperative  contrast enhancing tumor volume (cm3)", "-0.1"),
])
def test_malformed_identity_category_and_numeric_values_fail(field, value):
    with pytest.raises(ValueError):
        parse([row(**{field: value})])


def test_unpinned_loader_does_not_accept_a_plausible_analytic_table(tmp_path):
    raw = encode([row()])
    path = tmp_path / "analytic.csv"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        rhuh.load_rhuh_csv(path)
    parsed = rhuh.load_rhuh_csv(path, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert len(parsed.records) == 1
    changed = raw.replace(b"12.5", b"12.6")
    path.write_bytes(changed)
    with pytest.raises(ValueError):
        rhuh.load_rhuh_csv(path, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_duplicate_header_cannot_shadow_a_permitted_field():
    headers = list(rhuh.RHUH_HEADERS)
    unrelated = next(i for i, key in enumerate(headers) if key not in {
        rhuh.PATIENT_ID_FIELD, rhuh.PREOPERATIVE_KPS_FIELD,
        rhuh.POSTOPERATIVE_KPS_FIELD, rhuh.DEFICIT_FIELD,
        rhuh.PREOPERATIVE_CE_VOLUME_FIELD})
    headers[unrelated] = rhuh.PREOPERATIVE_KPS_FIELD
    raw = encode([row()], headers)
    with pytest.raises(ValueError):
        rhuh.parse_rhuh_csv(raw, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_imaging_interval_never_dates_an_outcome_or_authorizes_a_plan():
    record = parse([row(**{"Days from earliest imaging to surgery ": "1"})]).records[0]
    projection = record.preoperative_projection()
    for timing in (record.preoperative_timing, record.postoperative_timing, projection.timing):
        assert timing.assessment_time is None
        assert timing.available_at is None
        assert timing.missing_reason == "time_unreported"
    assert record.unresolved_interpretation["baseline_relative_new_or_worsened"] == "not_collected"
    assert projection.primary_planner_authorized is False
    assert projection.to_dict()["primary_planner_authorized"] is False
    assert projection.to_dict()["scope"] == "preoperative_observational_research"
