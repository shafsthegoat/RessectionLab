"""Frozen within-patient simulation assumptions and honest uncertainty fields.

The rigid ensembles here are declared registration sensitivity scenarios. They
are not diffusion reconstructions, tissue mechanics, or clinical probabilities.
Sample one latent world per episode; expose only ``actor_observation`` to an actor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation


def _canonical(value: Any) -> str:
    def encode(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, Mapping):
            return dict(obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        raise TypeError(f"Unsupported fingerprint input: {type(obj).__name__}")
    return json.dumps(value, default=encode, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


def _immutable_array(value: Any, dtype: Any = float) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _binary_mask(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "biuf" or not np.isfinite(array).all() or not np.isin(array, [0, 1]).all():
        raise ValueError(f"{name} must contain finite binary values")
    return array.astype(bool, copy=True)


@dataclass(frozen=True)
class FrozenDecisionModel:
    """Canonical snapshots cannot change through a caller-owned mutable mapping."""

    case_hash: str
    configuration_json: str

    @classmethod
    def create(cls, *, case_hash: str, geometry: Any, objectives: Any, tools: Any,
               world_generator: Any, action_primitives: Any,
               planning_as_of: str | None = None) -> "FrozenDecisionModel":
        if not isinstance(case_hash, str) or not case_hash:
            raise ValueError("A case content hash is required")
        return cls(case_hash, _canonical({
            "case_hash": case_hash, "geometry": geometry, "objectives": objectives,
            "tools": tools, "world_generator": world_generator,
            "action_primitives": action_primitives, "planning_as_of": planning_as_of,
        }))

    def __post_init__(self) -> None:
        if not isinstance(self.case_hash, str) or not self.case_hash or not isinstance(self.configuration_json, str):
            raise ValueError("Frozen decision model requires string case identity and canonical configuration")
        value = json.loads(self.configuration_json)
        if not isinstance(value, dict) or value.get("case_hash") != self.case_hash:
            raise ValueError("Decision-model case does not match configuration")
        object.__setattr__(self, "configuration_json", _canonical(value))

    @property
    def fingerprint(self) -> str:
        return content_hash(json.loads(self.configuration_json))

    def assert_matches(self, current: "FrozenDecisionModel") -> None:
        if self.fingerprint != current.fingerprint:
            raise ValueError("DECISION_MODEL_CHANGED: start a new versioned experiment")

    def to_dict(self) -> dict[str, Any]:
        return {"fingerprint": self.fingerprint, **json.loads(self.configuration_json)}


class WorldRole(str, Enum):
    OPTIMIZATION = "optimization"
    SELECTION = "selection"
    FINAL_EVALUATION = "final_evaluation"
    STRESS = "stress"


@dataclass(frozen=True)
class WorldGeneratorConfig:
    family: str = "rigid_gaussian"
    translation_scale_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_scale_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_center_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    version: str = "rigid-registration-v1"
    parameter_basis: str = "declared_uncalibrated_sensitivity_scenario"

    def __post_init__(self) -> None:
        if self.family not in {"rigid_gaussian", "rigid_uniform", "rigid_directional_stress"}:
            raise ValueError("Unsupported coherent perturbation family")
        for name in ("translation_scale_mm", "rotation_scale_deg", "rotation_center_mm"):
            values = tuple(float(v) for v in getattr(self, name))
            if len(values) != 3 or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain three finite physical values")
            if name != "rotation_center_mm" and min(values) < 0:
                raise ValueError("Perturbation scales must be nonnegative")
            object.__setattr__(self, name, values)
        if not isinstance(self.version, str) or not self.version or not isinstance(self.parameter_basis, str) or not self.parameter_basis:
            raise ValueError("World version and parameter basis are required")

    @property
    def deterministic(self) -> bool:
        return not any(self.translation_scale_mm + self.rotation_scale_deg)

    @property
    def fingerprint(self) -> str:
        return content_hash(asdict(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorldPartitionManifest:
    role: WorldRole
    case_hash: str
    generator: WorldGeneratorConfig
    seeds: tuple[int, ...]
    planning_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", WorldRole(self.role))
        seeds = tuple(self.seeds)
        if not seeds or any(not isinstance(s, (int, np.integer)) or isinstance(s, bool) or s < 0 for s in seeds):
            raise ValueError("World seeds must be nonempty nonnegative integers")
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate world seed within partition")
        if not isinstance(self.case_hash, str) or not self.case_hash:
            raise ValueError("World partition must identify a case")
        if not isinstance(self.generator, WorldGeneratorConfig):
            raise ValueError("World partition needs a typed immutable generator")
        if self.planning_hash is not None and (not isinstance(self.planning_hash, str) or not self.planning_hash):
            raise ValueError("Planning hash must be a nonempty string")
        object.__setattr__(self, "seeds", tuple(int(s) for s in seeds))

    @property
    def partition_hash(self) -> str:
        return content_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role.value, "case_hash": self.case_hash, "planning_hash": self.planning_hash,
                "generator": self.generator.to_dict(), "seeds": list(self.seeds),
                "interpretation": "deterministic_optimization" if self.generator.deterministic
                else "model_conditioned_registration_sensitivity"}


@dataclass(frozen=True)
class WorldPartitions:
    optimization: WorldPartitionManifest
    selection: WorldPartitionManifest
    final_evaluation: WorldPartitionManifest
    stress: WorldPartitionManifest

    def __post_init__(self) -> None:
        manifests = (self.optimization, self.selection, self.final_evaluation, self.stress)
        expected = tuple(WorldRole)
        if tuple(m.role for m in manifests) != expected:
            raise ValueError("Partitions have incorrect roles")
        if len({m.case_hash for m in manifests}) != 1:
            raise ValueError("Partitions must belong to one case")
        all_seeds = [seed for manifest in manifests for seed in manifest.seeds]
        if len(set(all_seeds)) != len(all_seeds):
            raise ValueError("World seeds overlap across partitions")
        if len({m.generator.fingerprint for m in manifests[:3]}) != 1:
            raise ValueError("Optimization, selection and evaluation must share frozen assumptions")
        if self.stress.generator.family == self.optimization.generator.family:
            raise ValueError("Stress tests must use a withheld perturbation family")

    def to_dict(self) -> dict[str, Any]:
        return {m.role.value: m.to_dict() for m in
                (self.optimization, self.selection, self.final_evaluation, self.stress)}


def generate_partitions(case_hash: str, config: WorldGeneratorConfig, master_seed: int, *,
                        optimization: int = 16, selection: int = 8,
                        final_evaluation: int = 16, stress: int = 8,
                        stress_config: WorldGeneratorConfig | None = None,
                        planning_hash: str | None = None) -> WorldPartitions:
    if not isinstance(master_seed, int) or isinstance(master_seed, bool) or master_seed < 0:
        raise ValueError("Master seed must be a nonnegative integer")
    if stress_config is None:
        stress_config = WorldGeneratorConfig(
            family="rigid_uniform" if config.family != "rigid_uniform" else "rigid_gaussian",
            translation_scale_mm=config.translation_scale_mm,
            rotation_scale_deg=config.rotation_scale_deg,
            rotation_center_mm=config.rotation_center_mm)
    manifests = []
    for role, count in zip(WorldRole, (optimization, selection, final_evaluation, stress)):
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError("Every partition requires a positive integer world count")
        seeds = tuple(int.from_bytes(hashlib.sha256(
            f"{planning_hash or case_hash}:{master_seed}:{role.value}:{i}".encode()).digest()[:8], "big")
            for i in range(count))
        manifests.append(WorldPartitionManifest(role, case_hash,
                         stress_config if role is WorldRole.STRESS else config, seeds, planning_hash))
    return WorldPartitions(*manifests)


def validate_training_partitions(optimization: WorldPartitionManifest,
                                 selection: WorldPartitionManifest,
                                 decision_model: FrozenDecisionModel | None = None) -> None:
    if optimization.role is not WorldRole.OPTIMIZATION or selection.role is not WorldRole.SELECTION:
        raise ValueError("Training accepts optimization and checkpoint-selection worlds only")
    if optimization.case_hash != selection.case_hash:
        raise ValueError("Training partitions belong to different cases")
    if set(optimization.seeds) & set(selection.seeds):
        raise ValueError("Optimization and selection worlds overlap")
    if optimization.generator.fingerprint != selection.generator.fingerprint:
        raise ValueError("Training assumptions differ between partitions")
    if decision_model is not None:
        if optimization.case_hash != decision_model.case_hash:
            raise ValueError("Training partition and frozen case do not match")
        configured = json.loads(decision_model.configuration_json)["world_generator"]
        if configured not in (optimization.generator.to_dict(), optimization.generator.fingerprint):
            # Canonical JSON converts immutable tuples to lists.
            if _canonical(configured) != _canonical(optimization.generator.to_dict()):
                raise ValueError("World generator differs from frozen decision model")


@dataclass(frozen=True)
class LatentWorld:
    """Simulator-only episode state. Do not pass this object to a policy actor."""

    world_id: str
    model_version: str
    _matrix_bytes: bytes
    deterministic: bool

    @property
    def anatomy_transform_mm(self) -> np.ndarray:
        return np.frombuffer(self._matrix_bytes, dtype=np.float64).reshape(4, 4)

    def transform_points(self, points_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_mm, dtype=float)
        if points.shape[-1:] != (3,) or not np.isfinite(points).all():
            raise ValueError("Structure points must have finite physical xyz coordinates")
        matrix = self.anatomy_transform_mm
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    def actor_observation(self) -> dict[str, Any]:
        # World ID, seed and sampled transform are deliberately absent. The
        # actor receives source estimates from the environment, never latent truth.
        return {"world_model_version": self.model_version,
                "uncertainty_mode": "deterministic" if self.deterministic else "registration_scenario"}


@dataclass(frozen=True)
class WorldGenerator:
    config: WorldGeneratorConfig

    def sample(self, manifest: WorldPartitionManifest, index: int) -> LatentWorld:
        if manifest.generator.fingerprint != self.config.fingerprint:
            raise ValueError("Manifest generator does not match frozen generator")
        if not isinstance(index, int) or index < 0 or index >= len(manifest.seeds):
            raise IndexError("World index outside partition")
        return self.sample_seed(manifest.seeds[index], world_id=content_hash({
            "partition": manifest.partition_hash, "index": index}))

    def sample_seed(self, seed: int, *, world_id: str | None = None) -> LatentWorld:
        """Simulator convenience; caller must enforce its permitted seed manifest."""
        if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool) or seed < 0:
            raise ValueError("World seed must be a nonnegative integer")
        rng = np.random.default_rng(seed)
        if self.config.family == "rigid_gaussian":
            draw = rng.normal(size=6)
        elif self.config.family == "rigid_uniform":
            draw = rng.uniform(-1.0, 1.0, size=6)
        else:
            # Correlated directional stress, deliberately different from iid
            # Gaussian registration parameters. Still one rigid anatomy.
            draw = np.full(6, rng.choice([-1.0, 1.0])) * rng.uniform(0.75, 1.0)
        translation = draw[:3] * self.config.translation_scale_mm
        rotvec = np.deg2rad(draw[3:] * self.config.rotation_scale_deg)
        rotation = Rotation.from_rotvec(rotvec).as_matrix()
        center = np.asarray(self.config.rotation_center_mm)
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = center + translation - rotation @ center
        return LatentWorld(world_id or content_hash({"generator": self.config.fingerprint, "seed": int(seed)}),
                           self.config.version, matrix.astype(np.float64).tobytes(),
                           self.config.deterministic)


@dataclass(frozen=True)
class HaloField:
    distance_mm: np.ndarray
    values: np.ndarray
    known_coverage: np.ndarray
    source: str
    frame: str
    scale_mm: float
    qc_state: str
    method: str = "H0 physical voxel-center distance Gaussian proximity surrogate"

    def __post_init__(self) -> None:
        for name in ("distance_mm", "values"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))
        object.__setattr__(self, "known_coverage", _immutable_array(self.known_coverage, bool))


def physical_proximity_halo(mask: np.ndarray, affine: np.ndarray, scale_mm: float, *,
                            source: str, coverage_mask: np.ndarray | None = None,
                            frame: str = "RAS+", qc_state: str = "unreviewed") -> HaloField:
    """H0 distance to estimated structure voxel centers, with millimeter sampling.

    Orthogonal oblique grids are supported. Shear must be resampled by an explicit
    imaging operation first; treating sheared axes as Euclidean spacings is wrong.
    Empty reconstructions remain unknown even if a supplied coverage mask is full.
    """
    structure = _binary_mask(mask, "Halo structure")
    matrix = np.asarray(affine, dtype=float)
    if structure.ndim != 3 or 0 in structure.shape:
        raise ValueError("Halo structure must be a nonempty 3-D grid")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Invalid voxel-to-world affine")
    basis = matrix[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if np.any(spacing <= 0) or not np.allclose(basis.T @ basis, np.diag(spacing**2), atol=1e-7):
        raise ValueError("Physical distance transform requires orthogonal voxel axes; shear unsupported")
    if not np.isfinite(scale_mm) or scale_mm <= 0 or not isinstance(source, str) or not source:
        raise ValueError("A positive physical halo scale and source are required")
    if not isinstance(frame, str) or not frame or not isinstance(qc_state, str) or not qc_state:
        raise ValueError("Halo coordinate frame and QC state are required")
    coverage = np.ones(structure.shape, bool) if coverage_mask is None else _binary_mask(coverage_mask, "Halo coverage")
    if coverage.shape != structure.shape:
        raise ValueError("Coverage and structure shapes differ")
    if not structure.any() or qc_state == "failed":
        coverage[:] = False
    distances = distance_transform_edt(~structure, sampling=spacing) if structure.any() else np.full(structure.shape, np.nan)
    distances[~coverage] = np.nan
    values = np.exp(-distances**2 / (2 * scale_mm**2))
    return HaloField(distances, values, coverage, source, frame, float(scale_mm), qc_state)


@dataclass(frozen=True)
class EnsembleSupport:
    support: np.ndarray
    membership_count: np.ndarray
    assessed_count: np.ndarray
    total_worlds: int
    source: str
    meaning: str = "H1 empirical anatomical support in the supplied reconstruction ensemble; not clinical risk"

    def __post_init__(self) -> None:
        for name, dtype in (("support", float), ("membership_count", np.int64), ("assessed_count", np.int64)):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), dtype))


def anatomical_ensemble_support(masks: Sequence[np.ndarray | None], *, source: str,
                                coverage_masks: Sequence[np.ndarray | None] | None = None,
                                shape: tuple[int, int, int] | None = None) -> EnsembleSupport:
    """Summarize actual supplied realizations; never fabricate missing masks.

    Partial or missing coverage remains NaN. Counts are retained so a viewer can
    explain why support is unknown rather than silently shrinking the denominator.
    """
    if not masks or not isinstance(source, str) or not source:
        raise ValueError("At least one realization and its source are required")
    known = [_binary_mask(mask, "Ensemble structure") for mask in masks if mask is not None]
    if shape is None:
        if not known:
            raise ValueError("Specify a grid shape when all reconstruction worlds are missing")
        shape = known[0].shape
    if len(shape) != 3 or any(n <= 0 for n in shape):
        raise ValueError("A positive 3-D grid shape is required")
    if coverage_masks is not None and len(coverage_masks) != len(masks):
        raise ValueError("Coverage must match every realization")
    numerator = np.zeros(shape, np.int64)
    assessed = np.zeros(shape, np.int64)
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        mask = _binary_mask(mask, "Ensemble structure")
        if mask.shape != shape:
            raise ValueError("Ensemble grids differ")
        coverage = np.ones(shape, bool) if coverage_masks is None else coverage_masks[i]
        if coverage is None:
            continue
        coverage = _binary_mask(coverage, "Ensemble coverage")
        if coverage.shape != shape:
            raise ValueError("Ensemble coverage grid differs")
        assessed += coverage
        numerator += mask & coverage
    support = np.full(shape, np.nan)
    complete = assessed == len(masks)
    support[complete] = numerator[complete] / len(masks)
    return EnsembleSupport(support, numerator, assessed, len(masks), source)
