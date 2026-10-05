"""Constructed CSV controls only; no patient inference, fitting or protected data."""
import csv
from dataclasses import FrozenInstanceError, replace
import hashlib
import io
import json

import pytest

from resectionlab import rhuh_outcomes as rh


def row(patient_id="RHUH-0001", **updates):
    values = dict.fromkeys(rh.RHUH_HEADERS, "uninterpreted source token")
    values.update({rh.PATIENT_ID_FIELD: patient_id, rh.PREOPERATIVE_KPS_FIELD: "80",
        rh.POSTOPERATIVE_KPS_FIELD: "70", rh.PREOPERATIVE_CE_VOLUME_FIELD: "12.50",
        rh.DEFICIT_FIELD: "Minor Persistent"})
    values.update(updates)
    return values


def payload(rows=None, headers=rh.RHUH_HEADERS):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(headers)
    for values in [row()] if rows is None else rows:
        writer.writerow([values[h] for h in headers])
    return stream.getvalue().encode("utf-8")


def parse(data):
    return rh.parse_rhuh_csv(data, expected_sha256=hashlib.sha256(data).hexdigest())


def test_exact_raw_snapshot_and_source_binding_survive_caller_mutation():
    raw = row(**{"Age": "  55  ", "Sex": 'source, "quote"'})
    data = payload([raw])
    dataset = parse(data)
    item = dataset.records[0]
    assert dict(item.raw_fields) == raw
    assert tuple(item.raw_fields) == rh.RHUH_HEADERS
    assert dataset.source.sha256 == hashlib.sha256(data).hexdigest()
    assert item.source is dataset.source
    direct = rh.RHUHOutcomeRecord(item.patient_id, raw, item.source)
    raw[rh.DEFICIT_FIELD] = "Major Persistent"
    assert direct.deficit_category == "Minor Persistent"
    with pytest.raises(TypeError):
        item.raw_fields["Age"] = "99"
    with pytest.raises(FrozenInstanceError):
        item.patient_id = "RHUH-0002"
    with pytest.raises(TypeError):
        item.unresolved_interpretation["affected_neurological_domain"] = "motor"


@pytest.mark.parametrize("category,expected", [("No", False), ("Transient", True),
    ("Minor Persistent", True), ("Major Persistent", True)])
def test_nominal_category_and_binary_recorded_status(category, expected):
    item = parse(payload([row(**{rh.DEFICIT_FIELD: category})])).records[0]
    assert item.deficit_category.value == category
    assert item.any_recorded_postoperative_deficit is expected
    assert item.preoperative_kps == 80 and item.postoperative_kps == 70
    assert item.preoperative_ce_volume_cm3 == 12.5


@pytest.mark.parametrize("token", ["", " NA ", "n/a", "NaN", "null", "none", "missing"])
def test_missing_is_never_no_or_zero_and_source_token_is_retained(token):
    fields = [rh.DEFICIT_FIELD, rh.PREOPERATIVE_KPS_FIELD,
        rh.POSTOPERATIVE_KPS_FIELD, rh.PREOPERATIVE_CE_VOLUME_FIELD]
    item = parse(payload([row(**dict.fromkeys(fields, token))])).records[0]
    assert all(item.raw_fields[f] == token for f in fields)
    assert item.deficit_category is None
    assert item.any_recorded_postoperative_deficit is None
    assert item.preoperative_kps is None and item.postoperative_kps is None
    assert item.preoperative_ce_volume_cm3 is None
    projection = item.preoperative_projection()
    assert projection.preoperative_kps is None and projection.preoperative_ce_volume_cm3 is None
    no = parse(payload([row(**{rh.DEFICIT_FIELD: "No"})])).records[0]
    assert no.any_recorded_postoperative_deficit is False


def test_timing_and_interpretation_do_not_invent_a_cutoff_or_domain():
    item = parse(payload()).records[0]
    for timing, phase in [(item.preoperative_timing, "preoperative"),
                          (item.postoperative_timing, "postoperative")]:
        assert timing.phase == phase
        assert timing.assessment_time is None and timing.available_at is None
        assert timing.missing_reason == "time_unreported"
    assert item.unresolved_interpretation["recovery_trajectory"] == "not_collected"
    projection = item.preoperative_projection()
    assert not projection.primary_planner_authorized
    assert projection.to_dict()["scope"] == "preoperative_observational_research"
    assert projection.timing.available_at is None
    with pytest.raises(ValueError, match="init=False"):
        replace(projection, primary_planner_authorized=True)


def test_projection_has_only_two_values_and_excluded_edits_never_change_hash():
    original_rows = [row(), row("RHUH-0002", **{rh.DEFICIT_FIELD: "No",
        rh.POSTOPERATIVE_KPS_FIELD: "90", "Extent of resection [EOR]  %": "100"})]
    baseline = parse(payload(original_rows))
    edited = []
    permitted = {rh.PATIENT_ID_FIELD, rh.PREOPERATIVE_KPS_FIELD, rh.PREOPERATIVE_CE_VOLUME_FIELD}
    for i, original in enumerate(original_rows):
        # Permute every excluded field across patients, then reorder rows.
        other = original_rows[1 - i]
        edited.append({key: value if key in permitted else other[key] for key, value in original.items()})
    altered = parse(payload(edited[::-1]))
    assert baseline.source.sha256 != altered.source.sha256
    before = {r.patient_id: r.preoperative_projection() for r in baseline.records}
    after = {r.patient_id: r.preoperative_projection() for r in altered.records}
    assert before == after
    assert {k: p.semantic_hash for k, p in before.items()} == {k: p.semantic_hash for k, p in after.items()}
    manifest = before["RHUH-0001"].to_dict()
    assert set(manifest) == {"schema", "collection_id", "source_version", "patient_id",
        "preoperative_kps", "preoperative_ce_volume_cm3", "units", "phase", "assessment_time",
        "available_at", "scope", "primary_planner_authorized"}
    assert baseline.source.sha256 not in json.dumps(manifest)
    assert "postoperative" not in json.dumps(manifest)
    equivalent = parse(payload([row(**{rh.PREOPERATIVE_KPS_FIELD: "+80.0",
        rh.PREOPERATIVE_CE_VOLUME_FIELD: "1.250e1"})])).records[0].preoperative_projection()
    assert equivalent.semantic_hash == before["RHUH-0001"].semantic_hash


@pytest.mark.parametrize("updates", [{rh.PREOPERATIVE_KPS_FIELD: "70"},
    {rh.PREOPERATIVE_CE_VOLUME_FIELD: "12.51"}, {rh.PATIENT_ID_FIELD: "RHUH-0002"},
    {rh.PREOPERATIVE_KPS_FIELD: ""}, {rh.PREOPERATIVE_CE_VOLUME_FIELD: ""}])
def test_permitted_values_and_patient_identity_affect_projection_hash(updates):
    original = parse(payload()).records[0].preoperative_projection()
    changed = parse(payload([row(**updates)])).records[0].preoperative_projection()
    assert original.semantic_hash != changed.semantic_hash


@pytest.mark.parametrize("field,value", [(rh.DEFICIT_FIELD, "No "), (rh.DEFICIT_FIELD, "Mild"),
    (rh.DEFICIT_FIELD, "0"), (rh.PREOPERATIVE_KPS_FIELD, "79.5"),
    (rh.POSTOPERATIVE_KPS_FIELD, "101"), (rh.PREOPERATIVE_KPS_FIELD, "-1"),
    (rh.PREOPERATIVE_KPS_FIELD, "Infinity"), (rh.PREOPERATIVE_CE_VOLUME_FIELD, "-0.1"),
    (rh.PREOPERATIVE_CE_VOLUME_FIELD, "inf"), (rh.PREOPERATIVE_CE_VOLUME_FIELD, "1e999"),
    (rh.PREOPERATIVE_CE_VOLUME_FIELD, "1e-999"), (rh.PREOPERATIVE_CE_VOLUME_FIELD, "1_000")])
def test_invalid_category_or_number_is_rejected_at_ingestion(field, value):
    with pytest.raises(ValueError):
        parse(payload([row(**{field: value})]))


@pytest.mark.parametrize("patient_id", ["RHUH-0000", "RHUH-0041", "RHUH-1", "BTC-0001", " RHUH-0001"])
def test_collection_ids_are_strict(patient_id):
    with pytest.raises(ValueError, match="Patient ID"):
        parse(payload([row(patient_id)]))


def test_duplicate_ids_wrong_columns_and_empty_table_fail_closed():
    for data in [payload([row(), row()]), payload([],),
        payload(headers=rh.RHUH_HEADERS[:-1]), payload(headers=rh.RHUH_HEADERS + ("Age",)),
        payload(headers=tuple(reversed(rh.RHUH_HEADERS))),
        payload().replace(b"Days from earliest imaging to surgery ", b"Days from earliest imaging to surgery")]:
        with pytest.raises(ValueError):
            parse(data)
    with pytest.raises(ValueError, match="row width"):
        parse(payload() + b"\r\n")


def test_hash_check_precedes_decoding_and_file_read_is_bounded(tmp_path):
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rh.parse_rhuh_csv(b"\xff malformed", expected_sha256="0" * 64)
    with pytest.raises(ValueError, match="Malformed"):
        parse(b"\xff")
    for data in [payload().replace(b"80", b"8\0", 1), b"x" * (rh.MAX_CSV_BYTES + 1),
                 payload().split(b"\r\n")[0] + b'\r\n"unterminated']:
        with pytest.raises(ValueError):
            parse(data)
    path = tmp_path / "table.csv"
    data = payload()
    path.write_bytes(data)
    assert rh.load_rhuh_csv(path, expected_sha256=hashlib.sha256(data).hexdigest()) == parse(data)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        rh.load_rhuh_csv(path)  # Analytic fixture is not the pinned public release.
    path.write_bytes(b"x" * (rh.MAX_CSV_BYTES + 17))
    with pytest.raises(ValueError, match="size limit"):
        rh.load_rhuh_csv(path)


def test_direct_construction_cannot_bypass_scalar_and_source_rules():
    source = rh.RHUHSource("1" * 64)
    for raw in [row(**{rh.DEFICIT_FIELD: "invented"}), row(**{"Age": ["mutable"]})]:
        with pytest.raises(ValueError):
            rh.RHUHOutcomeRecord("RHUH-0001", raw, source)
    with pytest.raises(ValueError, match="raw source"):
        rh.RHUHOutcomeRecord("RHUH-0002", row(), source)
    for score, volume in [(True, 1.0), (80, True), (80, float("nan")), (80, -1.0)]:
        with pytest.raises(ValueError):
            rh.RHUHPreoperativeProjection("RHUH-0001", score, volume)
    with pytest.raises(ValueError, match="release"):
        rh.RHUHSource("1" * 64, source_version="outcome-derived-version")
    item = rh.RHUHOutcomeRecord("RHUH-0001", row(), source)
    records = [item]
    dataset = rh.RHUHOutcomes(records, source)
    records.clear()
    assert dataset.records == (item,)
    with pytest.raises(ValueError, match="same source"):
        rh.RHUHOutcomes((item,), rh.RHUHSource("2" * 64))
