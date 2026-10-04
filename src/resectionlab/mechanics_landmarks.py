"""Separate observed and sealed landmarks for a displacement-only experiment.

This module does no file/image IO and grants no data-access authorization. Its
caller must first follow the committed acquisition/access declaration. MINC tag
sources are parsed without converting destination tokens. Only the six selected
observations enter ForwardLandmarks; validation is opened through a distinct
function bound to an already persisted model/prediction freeze.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import re

import numpy as np

from .core import array_digest, immutable_array, semantic_digest

DISPLACEMENT_ROLE = "before_us_to_during_us"
REGISTRATION_ROLE = "mri_to_before_us"
PARTITION_RULE = "source_fps_six_rank3_v1"
MAX_TAG_BYTES = 1024 * 1024
MAX_POINTS = 4096
_NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
_INTEGER = re.compile(r"[+-]?\d+\Z")
_TOKEN = re.compile(r'"[^"\r\n]*"|[^\s";]+|;')


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r"(?:sha256:)?[0-9a-f]{64}", value) is None:
        raise ValueError("Expected an exact lowercase SHA256 identity")
    return "sha256:" + value.removeprefix("sha256:")


def _text(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("Expected bounded nonempty identity text")
    return value


def _points(value, count=None):
    raw = np.asarray(value)
    if (raw.dtype.kind not in "iuf" or raw.ndim != 2 or raw.shape[1:] != (3,)
            or not 1 <= len(raw) <= MAX_POINTS or (count is not None and len(raw) != count)
            or not np.isfinite(raw).all() or np.any(np.abs(raw) > 1e6)):
        raise ValueError("Expected finite real physical points within the declared parser bounds")
    return immutable_array(raw, np.float64)


def _frame(value):
    raw = np.asarray(value)
    if (raw.dtype.kind not in "iuf" or raw.shape != (4, 4) or not np.isfinite(raw).all()
            or not np.array_equal(raw[3], [0, 0, 0, 1])
            or not np.allclose(raw[:3, :3].T @ raw[:3, :3], np.eye(3), rtol=0, atol=1e-12)
            or np.any(np.abs(raw[:3, 3]) > 1e6)):
        raise ValueError("Frame QC must declare a rigid world-mm to common-RAS-mm transform")
    return immutable_array(raw, np.float64)


def _transform(points, affine):
    return _points(np.asarray(points) @ affine[:3, :3].T + affine[:3, 3])


def _ids(value, count):
    result = tuple(value)
    if (len(result) != count or len(set(result)) != count
            or any(type(i) is not int or not 1 <= i <= MAX_POINTS for i in result)):
        raise ValueError("Landmark row IDs must be unique one-based integers")
    return result


@dataclass(frozen=True)
class LandmarkFrameQC:
    """Explicit caller-supplied image/frame audit; never inferred from six columns."""
    source_image_sha256: str
    destination_image_sha256: str
    source_world_to_ras_mm: np.ndarray
    destination_world_to_ras_mm: np.ndarray
    report_sha256: str
    convention: str
    _seal: str = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("source_image_sha256", "destination_image_sha256", "report_sha256"):
            object.__setattr__(self, name, _hash(getattr(self, name)))
        for name in ("source_world_to_ras_mm", "destination_world_to_ras_mm"):
            object.__setattr__(self, name, _frame(getattr(self, name)))
        _text(self.convention)
        object.__setattr__(self, "_seal", self._digest())

    def _digest(self):
        return semantic_digest({"source": self.source_image_sha256, "destination": self.destination_image_sha256,
            "source_transform": array_digest(self.source_world_to_ras_mm),
            "destination_transform": array_digest(self.destination_world_to_ras_mm),
            "report": self.report_sha256, "convention": self.convention})

    def assert_intact(self):
        if self._digest() != self._seal:
            raise ValueError("Frame QC was changed")


@dataclass(frozen=True)
class LandmarkPairBinding:
    patient_group: str
    role: str
    tag_sha256: str
    source_image_sha256: str
    destination_image_sha256: str
    frame_qc: LandmarkFrameQC | None = None
    _seal: str = field(init=False, repr=False)

    def __post_init__(self):
        _text(self.patient_group)
        if self.role not in {DISPLACEMENT_ROLE, REGISTRATION_ROLE}:
            raise ValueError("Unknown landmark pair role")
        for name in ("tag_sha256", "source_image_sha256", "destination_image_sha256"):
            object.__setattr__(self, name, _hash(getattr(self, name)))
        self._check_frame()
        object.__setattr__(self, "_seal", self.audit_hash)

    def _check_frame(self):
        if self.frame_qc is not None:
            if not isinstance(self.frame_qc, LandmarkFrameQC):
                raise ValueError("Frame QC must be a typed declaration")
            self.frame_qc.assert_intact()
            if (self.source_image_sha256 != self.frame_qc.source_image_sha256
                    or self.destination_image_sha256 != self.frame_qc.destination_image_sha256):
                raise ValueError("Frame QC is bound to different images")

    @property
    def audit_hash(self):
        return semantic_digest({"patient_group": self.patient_group, "role": self.role,
            "tag_sha256": self.tag_sha256, "source_image_sha256": self.source_image_sha256,
            "destination_image_sha256": self.destination_image_sha256,
            "frame_qc": None if self.frame_qc is None else self.frame_qc._digest()})

    def assert_intact(self):
        self._check_frame()
        if self.audit_hash != self._seal:
            raise ValueError("Landmark source binding was changed")


def _records(payload, binding):
    """Tokenize bounded documented writer syntax, without parsing target numbers.

    Each record is one line: six coordinate tokens, optional three auxiliary
    tokens (weight/integer IDs), optional quoted label. Unknown variants fail.
    Labels/auxiliaries never affect row IDs, selection, weighting or model input.
    """
    if not isinstance(binding, LandmarkPairBinding):
        raise TypeError("A typed pair binding is required")
    binding.assert_intact()
    if type(payload) is not bytes or not 1 <= len(payload) <= MAX_TAG_BYTES:
        raise ValueError("Tag payload exceeds bounds or is not immutable bytes")
    if _hash(hashlib.sha256(payload).hexdigest()) != binding.tag_sha256:
        raise ValueError("Tag bytes differ from the declared source")
    try:
        text = payload.decode("ascii")
    except UnicodeError as error:
        raise ValueError("Only bounded ASCII MINC tag syntax is supported") from error
    lines = []
    for line in text.splitlines():
        quoted, clean = False, []
        for character in line:
            if character == '"':
                quoted = not quoted
            if character == "%" and not quoted:
                break
            clean.append(character)
        if quoted:
            raise ValueError("Unterminated tag label")
        line = "".join(clean).strip()
        if line:
            lines.append(line)
    if not lines or lines[0] != "MNI Tag Point File":
        raise ValueError("Missing MINC tag header")
    match = re.match(r"\AVolumes\s*=\s*2\s*;\s*Points\s*=", "\n".join(lines[1:]))
    if match is None:
        raise ValueError("Exactly two tag volumes and a Points section are required")
    body = "\n".join(lines[1:])[match.end():]
    records, terminated = [], False
    for line in body.splitlines():
        if not line.strip():
            continue
        if terminated:
            raise ValueError("Unexpected content after tag terminator")
        matches = list(_TOKEN.finditer(line))
        position = 0
        for token in matches:
            if line[position:token.start()].strip():
                raise ValueError("Unsupported tag syntax")
            position = token.end()
        if line[position:].strip():
            raise ValueError("Unsupported tag syntax")
        tokens = [item.group() for item in matches]
        if ";" in tokens:
            if tokens[-1] != ";" or tokens.count(";") != 1:
                raise ValueError("Malformed tag terminator")
            terminated = True
            tokens.pop()
        if not tokens:
            continue
        if len(tokens) < 6 or any(token.startswith('"') for token in tokens[:6]):
            raise ValueError("Each paired landmark needs six coordinate tokens")
        suffix = tokens[6:]
        if suffix and suffix[-1].startswith('"'):
            suffix = suffix[:-1]
        if suffix and (len(suffix) != 3 or _NUMBER.fullmatch(suffix[0]) is None
                       or any(_INTEGER.fullmatch(item) is None for item in suffix[1:])):
            raise ValueError("Unsupported tag auxiliary fields")
        records.append(tuple(tokens[:6]))
        if len(records) > MAX_POINTS:
            raise ValueError("Too many landmark records")
    if not terminated or not records:
        raise ValueError("Missing tag records or terminator")
    return records


def _coordinate_tokens(tokens):
    if any(_NUMBER.fullmatch(token) is None for token in tokens):
        raise ValueError("Accessed landmark coordinate is not a finite decimal number")
    return _points([[float(token) for token in tokens]])[0]


@dataclass(frozen=True)
class LandmarkSources:
    binding: LandmarkPairBinding
    source_world_mm: np.ndarray
    frame: str = field(default="world_mm_unverified", init=False)
    _seal: str = field(init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.binding, LandmarkPairBinding):
            raise TypeError("A typed pair binding is required")
        object.__setattr__(self, "source_world_mm", _points(self.source_world_mm))
        object.__setattr__(self, "_seal", self.source_hash)

    @property
    def row_ids(self):
        return tuple(range(1, len(self.source_world_mm) + 1))

    @property
    def source_hash(self):
        # Full paired-file and destination-image hashes remain audit data, never
        # influence source selection/model identity through withheld values.
        return semantic_digest({"patient": self.binding.patient_group, "role": self.binding.role,
            "source_image": self.binding.source_image_sha256,
            "source_world_mm": array_digest(self.source_world_mm), "row_ids": self.row_ids})

    def assert_intact(self):
        self.binding.assert_intact()
        if self.source_hash != self._seal or self.frame != "world_mm_unverified":
            raise ValueError("Landmark source snapshot changed")


def parse_tag_sources(payload: bytes, binding: LandmarkPairBinding) -> LandmarkSources:
    rows = _records(payload, binding)
    return LandmarkSources(binding, np.array([_coordinate_tokens(row[:3]) for row in rows]))


@dataclass(frozen=True)
class LandmarkPartition:
    source_hash: str
    boundary_ids: tuple[int, ...]
    validation_ids: tuple[int, ...]
    rank_ratio: float
    rule: str = PARTITION_RULE

    def __post_init__(self):
        object.__setattr__(self, "source_hash", _hash(self.source_hash))
        object.__setattr__(self, "boundary_ids", _ids(self.boundary_ids, 6))
        object.__setattr__(self, "validation_ids", _ids(self.validation_ids, len(self.validation_ids)))
        combined = self.boundary_ids + self.validation_ids
        if (len(self.validation_ids) < 6 or len(set(combined)) != len(combined)
                or sorted(combined) != list(range(1, len(combined) + 1))
                or self.rule != PARTITION_RULE or not np.isfinite(self.rank_ratio)
                or not 1e-6 < self.rank_ratio <= 1):
            raise ValueError("Invalid source-only six-observation partition")

    @property
    def partition_hash(self):
        return semantic_digest({"source_hash": self.source_hash, "boundary_ids": self.boundary_ids,
            "validation_ids": self.validation_ids, "rank_ratio": self.rank_ratio, "rule": self.rule})


def partition_displacement_sources(sources: LandmarkSources) -> LandmarkPartition:
    if not isinstance(sources, LandmarkSources) or sources.binding.role != DISPLACEMENT_ROLE:
        raise ValueError("Displacement partition requires before-US/during-US sources")
    sources.assert_intact()
    points = sources.source_world_mm
    if len(points) < 12 or len(np.unique(points, axis=0)) != len(points):
        raise ValueError("Require at least 12 unique source landmarks; duplicates cannot cross roles")
    chosen = [0]
    distance = np.sum((points - points[0]) ** 2, axis=1)
    for _ in range(5):
        distance[chosen] = -1
        # np.argmax returns the first exact tie: original one-based row order.
        selected = int(np.argmax(distance))
        chosen.append(selected)
        distance = np.minimum(distance, np.sum((points - points[selected]) ** 2, axis=1))
    centered = points[chosen] - points[chosen].mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.
    if ratio <= 1e-6:
        raise ValueError("Selected source observations fail frozen rank3 volumetric eligibility")
    return LandmarkPartition(sources.source_hash, tuple(i + 1 for i in chosen),
        tuple(i + 1 for i in range(len(points)) if i not in chosen), ratio)


@dataclass(frozen=True)
class ForwardLandmarks:
    """Allowlisted observed input only: no raw paired file, V or image payload."""
    patient_group: str
    source_image_sha256: str
    partition_hash: str
    boundary_ids: tuple[int, ...]
    source_ras_mm: np.ndarray
    observed_ras_mm: np.ndarray
    source_world_to_ras_mm: np.ndarray
    destination_world_to_ras_mm: np.ndarray
    _seal: str = field(init=False, repr=False)

    def __post_init__(self):
        _text(self.patient_group)
        for name in ("source_image_sha256", "partition_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name)))
        object.__setattr__(self, "boundary_ids", _ids(self.boundary_ids, 6))
        for name in ("source_ras_mm", "observed_ras_mm"):
            object.__setattr__(self, name, _points(getattr(self, name), 6))
        for name in ("source_world_to_ras_mm", "destination_world_to_ras_mm"):
            object.__setattr__(self, name, _frame(getattr(self, name)))
        object.__setattr__(self, "_seal", self.forward_hash)

    def _manifest(self):
        return {"schema_version": 1, "role": "observed_internal_displacements_not_tool_boundaries",
            "patient_group": self.patient_group, "source_image_sha256": self.source_image_sha256,
            "partition_hash": self.partition_hash, "frame": "RAS+", "units": "mm",
            "boundary_ids": self.boundary_ids, "source_ras_mm": self.source_ras_mm.tolist(),
            "observed_ras_mm": self.observed_ras_mm.tolist(),
            "source_world_to_ras_mm": self.source_world_to_ras_mm.tolist(),
            "destination_world_to_ras_mm": self.destination_world_to_ras_mm.tolist()}

    @property
    def forward_hash(self):
        return semantic_digest(self._manifest())

    def to_manifest(self):
        self.assert_intact()
        return self._manifest()

    @property
    def displacement_ras_mm(self):
        self.assert_intact()
        return immutable_array(self.observed_ras_mm - self.source_ras_mm)

    def assert_intact(self):
        if self.forward_hash != self._seal:
            raise ValueError("Forward landmark input changed")


def _checked_partition(payload, binding, partition):
    sources = parse_tag_sources(payload, binding)
    expected = partition_displacement_sources(sources)
    if not isinstance(partition, LandmarkPartition) or partition != expected:
        raise ValueError("Partition is stale, altered or belongs to another source")
    if binding.frame_qc is None:
        raise ValueError("WORLD_FRAME_UNVERIFIED: explicit image/frame QC is required")
    return sources, _records(payload, binding), binding.frame_qc


def build_forward_landmarks(payload, binding, partition) -> ForwardLandmarks:
    sources, rows, qc = _checked_partition(payload, binding, partition)
    selected = np.array(partition.boundary_ids) - 1
    destination = np.array([_coordinate_tokens(rows[i][3:6]) for i in selected])
    return ForwardLandmarks(binding.patient_group, binding.source_image_sha256,
        partition.partition_hash, partition.boundary_ids,
        _transform(sources.source_world_mm[selected], qc.source_world_to_ras_mm),
        _transform(destination, qc.destination_world_to_ras_mm),
        qc.source_world_to_ras_mm, qc.destination_world_to_ras_mm)


@dataclass(frozen=True)
class LandmarkModelFreeze:
    forward_hash: str
    partition_hash: str
    model_sha256: str
    prediction_sha256: str
    source_audit_hash: str
    _seal: str = field(init=False, repr=False)

    def __post_init__(self):
        for name in ("forward_hash", "partition_hash", "model_sha256", "prediction_sha256", "source_audit_hash"):
            object.__setattr__(self, name, _hash(getattr(self, name)))
        object.__setattr__(self, "_seal", self.freeze_hash)

    @property
    def freeze_hash(self):
        return semantic_digest({name: getattr(self, name) for name in
            ("forward_hash", "partition_hash", "model_sha256", "prediction_sha256", "source_audit_hash")})

    def assert_intact(self):
        if self.freeze_hash != self._seal:
            raise ValueError("Model/prediction freeze was changed")


def freeze_landmark_model(forward, *, source_binding, model_sha256, prediction_sha256):
    """Bind outputs and sealed-source audit; persist before evaluator access.

    The audit binding is kept outside ForwardLandmarks: its full paired-file
    hash must not become a solver seed, model input or forward-model identity.
    This function cannot prove a caller persisted the record or that an external
    solver respected the allowlist; those remain execution-runner obligations.
    """
    if type(forward) is not ForwardLandmarks:
        raise TypeError("Only allowed forward observations can bind a model freeze")
    forward.assert_intact()
    if not isinstance(source_binding, LandmarkPairBinding):
        raise TypeError("A separate sealed-source audit binding is required")
    source_binding.assert_intact()
    qc = source_binding.frame_qc
    if (source_binding.role != DISPLACEMENT_ROLE or qc is None
            or source_binding.patient_group != forward.patient_group
            or source_binding.source_image_sha256 != forward.source_image_sha256
            or not np.array_equal(qc.source_world_to_ras_mm, forward.source_world_to_ras_mm)
            or not np.array_equal(qc.destination_world_to_ras_mm, forward.destination_world_to_ras_mm)):
        raise ValueError("Sealed-source audit does not match allowed forward input")
    return LandmarkModelFreeze(forward.forward_hash, forward.partition_hash,
        model_sha256, prediction_sha256, source_binding.audit_hash)


@dataclass(frozen=True)
class ValidationLandmarks:
    """Evaluator-only physical coordinates, separate from forward input type."""
    row_ids: tuple[int, ...]
    source_ras_mm: np.ndarray
    observed_ras_mm: np.ndarray
    freeze_hash: str
    tag_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "source_ras_mm", _points(self.source_ras_mm))
        object.__setattr__(self, "observed_ras_mm", _points(self.observed_ras_mm, len(self.source_ras_mm)))
        object.__setattr__(self, "row_ids", _ids(self.row_ids, len(self.source_ras_mm)))
        for name in ("freeze_hash", "tag_sha256"):
            object.__setattr__(self, name, _hash(getattr(self, name)))


def reveal_validation_landmarks(payload, binding, partition, freeze) -> ValidationLandmarks:
    if type(freeze) is not LandmarkModelFreeze:
        raise ValueError("A bound model/prediction freeze is required before validation access")
    freeze.assert_intact()
    if not isinstance(binding, LandmarkPairBinding):
        raise TypeError("A typed pair binding is required")
    binding.assert_intact()
    if freeze.source_audit_hash != binding.audit_hash:
        raise ValueError("Sealed source differs from the model/prediction freeze")
    forward = build_forward_landmarks(payload, binding, partition)
    if freeze.forward_hash != forward.forward_hash or freeze.partition_hash != partition.partition_hash:
        raise ValueError("Model freeze does not match this partition and observed inputs")
    sources, rows, qc = _checked_partition(payload, binding, partition)
    selected = np.array(partition.validation_ids) - 1
    destination = np.array([_coordinate_tokens(rows[i][3:6]) for i in selected])
    return ValidationLandmarks(partition.validation_ids,
        _transform(sources.source_world_mm[selected], qc.source_world_to_ras_mm),
        _transform(destination, qc.destination_world_to_ras_mm), freeze.freeze_hash, binding.tag_sha256)


@dataclass(frozen=True)
class RegistrationLandmarks:
    """Baseline MRI/before-US alignment data; cannot be a displacement forward DTO."""
    source_ras_mm: np.ndarray
    observed_ras_mm: np.ndarray
    tag_sha256: str

    def __post_init__(self):
        object.__setattr__(self, "source_ras_mm", _points(self.source_ras_mm))
        object.__setattr__(self, "observed_ras_mm", _points(self.observed_ras_mm, len(self.source_ras_mm)))
        object.__setattr__(self, "tag_sha256", _hash(self.tag_sha256))


def read_registration_landmarks(payload, binding) -> RegistrationLandmarks:
    if not isinstance(binding, LandmarkPairBinding) or binding.role != REGISTRATION_ROLE:
        raise ValueError("Baseline registration requires its separate MRI/before-US pair role")
    sources = parse_tag_sources(payload, binding)
    if binding.frame_qc is None:
        raise ValueError("WORLD_FRAME_UNVERIFIED: explicit image/frame QC is required")
    rows = _records(payload, binding)
    destination = np.array([_coordinate_tokens(row[3:6]) for row in rows])
    return RegistrationLandmarks(_transform(sources.source_world_mm, binding.frame_qc.source_world_to_ras_mm),
        _transform(destination, binding.frame_qc.destination_world_to_ras_mm), binding.tag_sha256)
