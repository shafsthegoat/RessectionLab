"""Evaluator-owned RHUH release reader; no planner, role assignment or model inputs beyond two fields.

The mixed pre/post file digest authenticates ingestion only. The separate permitted
projection deliberately excludes that digest so outcome edits cannot change its identity.
Source-labelled phases support a declared observational research task, not a planning
cutoff: individual assessment/availability timestamps are unreported.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
import hashlib
import io
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping


RHUH_COLLECTION_ID = "RHUH-GBM"
RHUH_SOURCE_VERSION = "TCIA-v1-2023-06-09"
RHUH_SOURCE_URI = "https://www.cancerimagingarchive.net/wp-content/uploads/clinical_data_TCIA_RHUH-GBM.csv"
RHUH_CSV_SHA256 = "32d638906d34aaf8f66f5ec41c53c044216aed73bac22c776fb399bf2f741728"
MAX_CSV_BYTES = 65_536
MAX_FIELD_CHARS = 4096
MISSING_TOKENS = frozenset({"", "na", "n/a", "nan", "null", "none", "missing"})
PATIENT_ID_FIELD = "Patient ID"
DEFICIT_FIELD = "Postoperative Neurological Deficit"
PREOPERATIVE_KPS_FIELD = "Preoperative KPS"
POSTOPERATIVE_KPS_FIELD = "Postoperative KPS"
PREOPERATIVE_CE_VOLUME_FIELD = "Preoperative  contrast enhancing tumor volume (cm3)"
RHUH_HEADERS = (
    PATIENT_ID_FIELD,
    "Days from earliest imaging to surgery ",
    "Age",
    "Sex",
    PREOPERATIVE_KPS_FIELD,
    "Previous treatment = (no, surgery, surgery + QT/RT)",
    "Histopathological subtype",
    "WHO grade",
    "IDH status = (mutant [mut], wild type [wt], NOS)",
    "Operative adjuncts = (5' aminolevulinic acid [5ALA], Sodium Fluorescein - Yellow 560 [FL], Neuronavigation [NAV],  Intraoperative monitoring [IONM], Direct electrical stimulation [DES], intraoperative ultrasound [ioUS])",
    PREOPERATIVE_CE_VOLUME_FIELD,
    "Postoperative contrast enhancing residual tumor (cm3)",
    "Preoperative T2/FLAIR abnormality  (cm3) ",
    "Postoperative T2/FLAIR abmnormality (cm3)",
    "Extent of resection [EOR]  %",
    "EOR = (Gross total resection [GTR : 100%], Near total resection [NTR : > 95%], Subtotal resection [STR : 91 - 94%], Partial resection [PR : < 90 %]",
    "Adjuvant therapy (Radiotherapy [RT], Temozolomide [TMZ])",
    "Radiotherapy treatment details (technique/dose/number of fractions)",
    DEFICIT_FIELD,
    POSTOPERATIVE_KPS_FIELD,
    "Progression free survival [PFS] (days)",
    "Overall survival [OS] (days)",
    "Right Censored",
)


class DeficitCategory(StrEnum):
    """Nominal source labels: no severity order or equal-distance harm mapping."""

    NO = "No"
    TRANSIENT = "Transient"
    MINOR_PERSISTENT = "Minor Persistent"
    MAJOR_PERSISTENT = "Major Persistent"


def _patient_id(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"RHUH-00(?:0[1-9]|[1-3][0-9]|40)", value) is None:
        raise ValueError("Patient ID must be a released collection ID RHUH-0001 through RHUH-0040")


def _digest(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Expected a lowercase hexadecimal SHA-256 digest")


def _release(version: str, uri: str) -> None:
    if type(version) is not str or version != RHUH_SOURCE_VERSION or type(uri) is not str or uri != RHUH_SOURCE_URI:
        raise ValueError("Only the audited RHUH release identity and source URI are supported")


def _missing(token: str) -> bool:
    return token.strip().casefold() in MISSING_TOKENS


def _number(token: str, name: str) -> Decimal | None:
    if _missing(token):
        return None
    if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", token.strip()) is None:
        raise ValueError(f"{name} must be a finite decimal or explicit missing token")
    try:
        value = Decimal(token.strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid numeric token for {name}") from exc
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return value


def _kps(token: str, name: str) -> int | None:
    value = _number(token, name)
    if value is None:
        return None
    # Broad score bounds, never the narrower observed 50–90 source range.
    if not 0 <= value <= 100 or value != value.to_integral_value():
        raise ValueError(f"{name} must be an integer score from 0 to 100")
    return int(value)


def _volume(token: str) -> float | None:
    value = _number(token, PREOPERATIVE_CE_VOLUME_FIELD)
    if value is None:
        return None
    result = float(value)
    if value < 0 or not math.isfinite(result) or (value != 0 and result == 0):
        raise ValueError("Preoperative CE volume must be finite, nonnegative and representable in cm3")
    return 0.0 if result == 0 else result


@dataclass(frozen=True, slots=True)
class RHUHSource:
    sha256: str
    source_version: str = RHUH_SOURCE_VERSION
    source_uri: str = RHUH_SOURCE_URI
    collection_id: str = field(default=RHUH_COLLECTION_ID, init=False)

    def __post_init__(self) -> None:
        _digest(self.sha256)
        _release(self.source_version, self.source_uri)


@dataclass(frozen=True, slots=True)
class RHUHAssessmentTiming:
    phase: str
    assessment_time: None = field(default=None, init=False)
    available_at: None = field(default=None, init=False)
    missing_reason: str = field(default="time_unreported", init=False)

    def __post_init__(self) -> None:
        if type(self.phase) is not str or self.phase not in {"preoperative", "postoperative"}:
            raise ValueError("RHUH timing requires an explicit source-labelled phase")


@dataclass(frozen=True, slots=True)
class RHUHPreoperativeProjection:
    """Only source-labelled preoperative research values; no mixed-container lineage."""

    patient_id: str
    preoperative_kps: int | None
    preoperative_ce_volume_cm3: float | None
    source_version: str = RHUH_SOURCE_VERSION
    collection_id: str = field(default=RHUH_COLLECTION_ID, init=False)
    primary_planner_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _patient_id(self.patient_id)
        _release(self.source_version, RHUH_SOURCE_URI)
        score = self.preoperative_kps
        if score is not None and (type(score) is not int or not 0 <= score <= 100):
            raise ValueError("Preoperative KPS must be an integer score from 0 to 100 or None")
        volume = self.preoperative_ce_volume_cm3
        if volume is not None:
            if type(volume) not in {int, float} or not math.isfinite(volume) or volume < 0:
                raise ValueError("Preoperative CE volume must be finite nonnegative cm3 or None")
            object.__setattr__(self, "preoperative_ce_volume_cm3", 0.0 if volume == 0 else float(volume))

    @property
    def timing(self) -> RHUHAssessmentTiming:
        return RHUHAssessmentTiming("preoperative")

    def to_dict(self) -> dict:
        return {
            "schema": "rhuh-preoperative-projection-v1",
            "collection_id": self.collection_id,
            "source_version": self.source_version,
            "patient_id": self.patient_id,
            "preoperative_kps": self.preoperative_kps,
            "preoperative_ce_volume_cm3": self.preoperative_ce_volume_cm3,
            "units": {"preoperative_kps": "KPS_score_points", "preoperative_ce_volume_cm3": "cm3"},
            "phase": self.timing.phase,
            "assessment_time": None,
            "available_at": None,
            "scope": "preoperative_observational_research",
            "primary_planner_authorized": False,
        }

    @property
    def semantic_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class RHUHOutcomeRecord:
    patient_id: str
    raw_fields: Mapping[str, str]
    source: RHUHSource

    def __post_init__(self) -> None:
        _patient_id(self.patient_id)
        if type(self.source) is not RHUHSource or not isinstance(self.raw_fields, Mapping):
            raise ValueError("Record requires typed RHUH provenance and raw field mapping")
        raw = dict(self.raw_fields)
        if set(raw) != set(RHUH_HEADERS) or any(type(k) is not str or type(v) is not str or len(v) > MAX_FIELD_CHARS or "\0" in v for k, v in raw.items()):
            raise ValueError("Record must preserve all exact RHUH headers and bounded string tokens")
        if raw[PATIENT_ID_FIELD] != self.patient_id:
            raise ValueError("Patient identity does not match the raw source record")
        object.__setattr__(self, "raw_fields", MappingProxyType({key: raw[key] for key in RHUH_HEADERS}))
        # Validate now, including direct construction, not only when a property is read.
        self.deficit_category
        self.preoperative_kps
        self.postoperative_kps
        self.preoperative_ce_volume_cm3

    @property
    def deficit_category(self) -> DeficitCategory | None:
        token = self.raw_fields[DEFICIT_FIELD]
        return None if _missing(token) else DeficitCategory(token)

    @property
    def preoperative_kps(self) -> int | None:
        return _kps(self.raw_fields[PREOPERATIVE_KPS_FIELD], PREOPERATIVE_KPS_FIELD)

    @property
    def postoperative_kps(self) -> int | None:
        return _kps(self.raw_fields[POSTOPERATIVE_KPS_FIELD], POSTOPERATIVE_KPS_FIELD)

    @property
    def preoperative_ce_volume_cm3(self) -> float | None:
        return _volume(self.raw_fields[PREOPERATIVE_CE_VOLUME_FIELD])

    @property
    def any_recorded_postoperative_deficit(self) -> bool | None:
        category = self.deficit_category
        return None if category is None else category is not DeficitCategory.NO

    @property
    def preoperative_timing(self) -> RHUHAssessmentTiming:
        return RHUHAssessmentTiming("preoperative")

    @property
    def postoperative_timing(self) -> RHUHAssessmentTiming:
        return RHUHAssessmentTiming("postoperative")

    @property
    def unresolved_interpretation(self) -> Mapping[str, str]:
        return MappingProxyType({
            "affected_neurological_domain": "not_collected",
            "baseline_relative_new_or_worsened": "not_collected",
            "severity_rubric": "definition_ambiguous",
            "persistence_duration": "time_unreported",
            "recovery_trajectory": "not_collected",
        })

    def preoperative_projection(self) -> RHUHPreoperativeProjection:
        return RHUHPreoperativeProjection(self.patient_id, self.preoperative_kps,
            self.preoperative_ce_volume_cm3, self.source.source_version)


@dataclass(frozen=True, slots=True)
class RHUHOutcomes:
    records: tuple[RHUHOutcomeRecord, ...]
    source: RHUHSource

    def __post_init__(self) -> None:
        if type(self.source) is not RHUHSource:
            raise ValueError("Dataset requires typed RHUH source provenance")
        records = tuple(self.records)
        if not 1 <= len(records) <= 40 or any(type(r) is not RHUHOutcomeRecord or r.source != self.source for r in records):
            raise ValueError("Dataset requires 1–40 records bound to the same source")
        if len({r.patient_id for r in records}) != len(records):
            raise ValueError("Duplicate RHUH patient ID")
        object.__setattr__(self, "records", records)


def parse_rhuh_csv(data: bytes, *, expected_sha256: str,
                   source_version: str = RHUH_SOURCE_VERSION,
                   source_uri: str = RHUH_SOURCE_URI) -> RHUHOutcomes:
    """Authenticate exact bytes before parsing; explicit digests permit analytic fixtures.

    This parser does not grant outcome access or assign patient roles. The caller must
    enforce its declared study access before supplying evaluator-owned source bytes.
    """
    source = RHUHSource(expected_sha256, source_version, source_uri)
    if type(data) is not bytes or not 0 < len(data) <= MAX_CSV_BYTES:
        raise ValueError("RHUH CSV must be nonempty bytes within the size limit")
    if hashlib.sha256(data).hexdigest() != source.sha256:
        raise ValueError("RHUH source SHA-256 mismatch; no records parsed")
    try:
        decoded = data.decode("utf-8")
        if "\0" in decoded:
            raise ValueError("NUL is not valid RHUH CSV text")
        reader = csv.reader(io.StringIO(decoded, newline=""), strict=True)
        if tuple(next(reader)) != RHUH_HEADERS:
            raise ValueError("RHUH CSV header mismatch (including spelling, order and whitespace)")
        records = []
        for row in reader:
            if len(row) != len(RHUH_HEADERS) or len(records) >= 40:
                raise ValueError("RHUH CSV row width or record count exceeds the release schema")
            raw = dict(zip(RHUH_HEADERS, row, strict=True))
            records.append(RHUHOutcomeRecord(raw[PATIENT_ID_FIELD], raw, source))
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise ValueError("Malformed RHUH CSV") from exc
    return RHUHOutcomes(tuple(records), source)


def load_rhuh_csv(path: str | Path, *, expected_sha256: str = RHUH_CSV_SHA256) -> RHUHOutcomes:
    """Bounded read of the pinned public release; never changes or downloads source."""
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_CSV_BYTES + 1)
    return parse_rhuh_csv(data, expected_sha256=expected_sha256)
