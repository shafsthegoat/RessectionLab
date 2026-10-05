"""Saved-JSON and mocked-boundary PAT05 review; no arrays, patient or model I/O."""
import copy
from dataclasses import make_dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import pat05_access_preparation as p
from resectionlab.core import freeze_json, thaw_json

PROFILE_RAW = (ROOT / p.PROFILE_PATH).read_bytes()
LEARNING_RAW = (ROOT / p.LEARNING_PATH).read_bytes()
RECEIPT_RAW = (ROOT / p.RECEIPT_PATH).read_bytes()
PROFILE = json.loads(PROFILE_RAW)
LEARNING = json.loads(LEARNING_RAW)
EXECUTED = json.loads(RECEIPT_RAW)
GRID = EXECUTED["initial_task_metrics"]["native_grid_reconciliation"]
MetadataReward = make_dataclass("MetadataReward", [(key, float) for key in LEARNING["objective"]])


def records():
    return p.PAT05HistoricalRecords(PROFILE_RAW, LEARNING_RAW, RECEIPT_RAW)


def canonical_fixture():
    old = PROFILE["access_derivation"]
    result = {key: copy.deepcopy(old[key]) for key in ("annotation_centroid_voxels", "representative_voxel",
        "selected_boundary_voxel", "depth_to_annotation_representative_mm", "cortical_access_permitted")}
    result.update(rule="annotation-centroid-nearest-cell-six-axis-shortest-exit-radius6-v1",
                  selection_uses_reward_or_native_preview=False)
    # Test-only current-rule metadata coordinates; these are not historical observations.
    boundaries = [[19.5, 138., 151.], [139.5, 138., 151.], [46., 46.5, 151.],
                  [46., 180.5, 151.], [46., 138., 105.5], [46., 138., 201.5]]
    result["six_axis_exit_distances_mm"] = [{**copy.deepcopy(row), "boundary_voxel": boundary}
        for row, boundary in zip(old["six_axis_exit_distances_mm"], boundaries, strict=True)]
    return result


def changed(value):
    if isinstance(value, bool): return not value
    if isinstance(value, float): return math.nextafter(value, math.inf)
    if isinstance(value, int): return value + 1
    if isinstance(value, str): return value + "-changed"
    if isinstance(value, list):
        result = copy.deepcopy(value); result[0] = changed(result[0]); return result
    raise AssertionError("Unhandled metadata type")


class MetadataVector:
    def __init__(self, values): self.values = copy.deepcopy(values)
    def tolist(self): return copy.deepcopy(self.values)


class MetadataAccess:
    def __init__(self, *, center_mm, normal_inward, radius_mm, window_id):
        self.center_mm, self.normal_inward = MetadataVector(center_mm), MetadataVector(normal_inward)
        self.radius_mm, self.window_id = radius_mm, window_id


def fake_boundary(monkeypatch, *, grid=None):
    state = {"derivations": 0, "frozen_checks": 0, "binary_checks": 0, "casts": 0}
    cast_sentinel, support_sentinel = object(), object()
    affine_metadata = copy.deepcopy(GRID["original_affine_ras_mm"])
    class Nominal:
        def astype(self, dtype):
            assert dtype is bool
            state["casts"] += 1
            return cast_sentinel
    nominal = Nominal()
    class Source:
        source_hash = EXECUTED["initial_task_metrics"]["source_hash"]
        nominal_target = nominal
        observed_support = support_sentinel
        affine_ras_mm = affine_metadata
        access = MetadataAccess(**LEARNING["member"]["access"])
        _grid_record = freeze_json(copy.deepcopy(GRID if grid is None else grid))
        @property
        def reference_target(self):
            pytest.fail("Evaluator-only field was read")
    class Task:
        case = Source()
        decision_model_hash = EXECUTED["decision_model_hash"]
        _state_seal = "unchanged-mocked-initial-state"
        _steps = 0
        terminated = False
        _planning = False
        _engine = SimpleNamespace(revision=0, history=())
        reward_spec = MetadataReward(**LEARNING["objective"])
        max_steps = LEARNING["settings"]["max_steps"]
        def _assert_frozen(self): state["frozen_checks"] += 1
    def isin(value, allowed):
        assert value is nominal and allowed == (0., 1.)
        state["binary_checks"] += 1
        return SimpleNamespace(all=lambda: True)
    def derive(target, support, affine, subject):
        assert target is cast_sentinel and support is support_sentinel
        assert affine is affine_metadata and subject == "sub-PAT05"
        state["derivations"] += 1
        return copy.deepcopy(PROFILE["access"]), canonical_fixture()
    monkeypatch.setattr(p, "NativeSpatialTask", Task)
    monkeypatch.setattr(p, "AccessWindow", MetadataAccess)
    monkeypatch.setattr(p, "np", SimpleNamespace(isin=isin))
    monkeypatch.setattr(p, "derive_access", derive)
    # Existing complete-grid metadata functions stay real and use the saved 20 fields.
    for name in ("_decode_case", "_construct_task", "load_prepared_training_case"):
        monkeypatch.setattr(p.coverage.prep, name, lambda *a, **k: pytest.fail("Patient/preparation boundary reached"))
    return Task(), state


def test_success_authenticates_history_and_labels_only_new_current_fields(monkeypatch):
    task, calls = fake_boundary(monkeypatch)
    old = copy.deepcopy(PROFILE)
    result = p.canonical_pat05_access_metadata(task, records())
    output = thaw_json(result)
    assert calls == {"derivations": 1, "frozen_checks": 2, "binary_checks": 1, "casts": 1}
    assert output["access_derivation"] == canonical_fixture()
    assert len(output["complete_native_grid_binding"]["complete_grid"]) == 20
    assert output["complete_native_grid_binding"]["complete_grid"] == GRID
    assert output["newly_rederived_fields"] == ["six_axis_exit_distances_mm[*].boundary_voxel"]
    assert output["current_shared_rule_metadata"] == ["rule", "selection_uses_reward_or_native_preview"]
    assert "boundary_voxel" not in " ".join(output["historically_verified_fields"]).replace("selected_boundary_voxel", "")
    assert output["historical_provenance"] == {key: old["access_derivation"][key]
        for key in ("procedure", "track", "source_grid_only_no_native_engine_execution")}
    assert PROFILE == old and all("boundary_voxel" not in row for row in PROFILE["access_derivation"]["six_axis_exit_distances_mm"])
    assert not output["historical_records_rewritten"] and not output["new_access_selected"]
    assert not output["clinical_use_permitted"] and not output["complete_stroke_certified"]
    with pytest.raises(TypeError): result["access_derivation"]["rule"] = "changed"


@pytest.mark.parametrize("name", ["profile", "learning", "receipt"])
def test_semantically_identical_but_resealed_raw_metadata_refused_before_task(name):
    original = records()
    bad = replace(original, **{name: getattr(original, name) + b"\n"})
    with pytest.raises(ValueError, match="bytes changed"):
        p.canonical_pat05_access_metadata(object(), bad)


def test_loader_opens_only_three_bounded_saved_json_records(monkeypatch):
    actual_open, opened = Path.open, []
    permitted = {ROOT / key for key in p.RECORD_SHA256}
    class Stream:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def read(self, count):
            assert count == p.MAX_RECORD_BYTES + 1
            return self.stream.read(count)
    def limited(path, *args, **kwargs):
        assert path in permitted, "Only saved metadata can be opened"
        opened.append(path)
        return Stream(actual_open(path, *args, **kwargs))
    monkeypatch.setattr(Path, "open", limited)
    assert p.load_pat05_historical_records().decoded() == records().decoded()
    assert len(opened) == 3 and set(opened) == permitted


def test_decoded_records_are_fresh_copies_of_exact_original_bytes():
    bound = records()
    decoded = bound.decoded()
    decoded[0]["access_derivation"]["representative_voxel"][0] += 1
    assert bound.decoded() == (PROFILE, LEARNING, EXECUTED)
    for name, raw in zip((p.PROFILE_PATH, p.LEARNING_PATH, p.RECEIPT_PATH),
                         (bound.profile, bound.learning, bound.receipt), strict=True):
        assert hashlib.sha256(raw).hexdigest() == p.RECORD_SHA256[name]


@pytest.mark.parametrize("field", ["path", "sha256", "json_pointer"])
def test_complete_grid_receipt_identity_cannot_be_substituted(monkeypatch, field):
    task, calls = fake_boundary(monkeypatch)
    binding = p.coverage.pat05_complete_grid_binding(copy.deepcopy(LEARNING))
    binding["receipt"][field] += "changed"
    monkeypatch.setattr(p.coverage, "pat05_complete_grid_binding", lambda original: binding)
    with pytest.raises(ValueError, match="pinned historical contract"):
        p.canonical_pat05_access_metadata(task, records())
    assert calls["derivations"] == 0


@pytest.mark.parametrize("condition", ["steps", "terminated", "planning", "revision", "history"])
def test_noninitial_task_refuses_metadata_before_rederivation(monkeypatch, condition):
    task, calls = fake_boundary(monkeypatch)
    if condition == "steps": task._steps = 1
    elif condition == "terminated": task.terminated = True
    elif condition == "planning": task._planning = True
    elif condition == "revision": task._engine.revision = 1
    else: task._engine.history = ("untrusted old operation",)
    with pytest.raises(ValueError, match="unexecuted"):
        p.canonical_pat05_access_metadata(task, records())
    assert calls["derivations"] == 0


@pytest.mark.parametrize("field", sorted(GRID))
def test_all_twenty_actual_grid_fields_are_exact_before_rederivation(monkeypatch, field):
    grid = copy.deepcopy(GRID); grid[field] = changed(grid[field])
    task, calls = fake_boundary(monkeypatch, grid=grid)
    with pytest.raises(p.coverage.PAT05TaskBindingMismatch) as error:
        p.canonical_pat05_access_metadata(task, records())
    assert calls["derivations"] == 0
    assert error.value.details["mismatched_fields"] == ["native_grid." + field]
    assert error.value.details["actual"]["native_grid"] == grid


@pytest.mark.parametrize("kind", ["missing", "extra", "bool_to_number"])
def test_incomplete_or_reinterpreted_grid_is_not_downgraded_to_old_eight(monkeypatch, kind):
    grid = copy.deepcopy(GRID)
    if kind == "missing": del grid["source_image_hash"]
    elif kind == "extra": grid["unrecorded"] = "new"
    else: grid["resampled"] = 0
    task, calls = fake_boundary(monkeypatch, grid=grid)
    with pytest.raises(p.coverage.PAT05TaskBindingMismatch):
        p.canonical_pat05_access_metadata(task, records())
    assert calls["derivations"] == 0


@pytest.mark.parametrize("field", ["annotation_centroid_voxels", "representative_voxel", "selected_boundary_voxel",
    "depth_to_annotation_representative_mm", "cortical_access_permitted"])
def test_each_recorded_canonical_scalar_or_vector_mismatch_refuses_metadata(monkeypatch, field):
    task, calls = fake_boundary(monkeypatch)
    good = p.derive_access
    def wrong(*args):
        geometry, canonical = good(*args); canonical[field] = changed(canonical[field]); return geometry, canonical
    monkeypatch.setattr(p, "derive_access", wrong)
    with pytest.raises(ValueError, match="derivation differs"):
        p.canonical_pat05_access_metadata(task, records())


@pytest.mark.parametrize("index", range(6))
@pytest.mark.parametrize("field", ["axis", "outward_sign", "distance_mm"])
def test_all_six_ordered_exit_triples_must_match_exactly(monkeypatch, index, field):
    task, _ = fake_boundary(monkeypatch)
    good = p.derive_access
    def wrong(*args):
        geometry, canonical = good(*args)
        row = canonical["six_axis_exit_distances_mm"][index]; row[field] = changed(row[field])
        return geometry, canonical
    monkeypatch.setattr(p, "derive_access", wrong)
    with pytest.raises(ValueError, match="six ordered"):
        p.canonical_pat05_access_metadata(task, records())


@pytest.mark.parametrize("field", sorted(PROFILE["access"]))
def test_original_selected_access_remains_exact(monkeypatch, field):
    task, _ = fake_boundary(monkeypatch)
    good = p.derive_access
    def wrong(*args):
        geometry, canonical = good(*args); geometry[field] = changed(geometry[field]); return geometry, canonical
    monkeypatch.setattr(p, "derive_access", wrong)
    with pytest.raises(ValueError, match="selected access"):
        p.canonical_pat05_access_metadata(task, records())


@pytest.mark.parametrize("field", ["source_hash", "decision_model_hash"])
def test_executed_source_and_model_hashes_gate_rederivation(monkeypatch, field):
    task, calls = fake_boundary(monkeypatch)
    if field == "source_hash": task.case.source_hash += "changed"
    else: task.decision_model_hash += "changed"
    with pytest.raises(ValueError, match="executed learning source/model"):
        p.canonical_pat05_access_metadata(task, records())
    assert calls["derivations"] == 0


@pytest.mark.parametrize("kind", ["objective", "horizon"])
def test_learning_objective_and_horizon_remain_fixed(monkeypatch, kind):
    task, calls = fake_boundary(monkeypatch)
    if kind == "objective": task.reward_spec.normal_per_mm3 += .001
    else: task.max_steps = 4
    with pytest.raises(p.coverage.PAT05TaskBindingMismatch) as error:
        p.canonical_pat05_access_metadata(task, records())
    assert calls["derivations"] == 0
    assert error.value.details["mismatched_fields"] == ["objective" if kind == "objective" else "max_steps"]


def test_post_derivation_grid_change_fails_second_exact_comparison(monkeypatch):
    task, calls = fake_boundary(monkeypatch)
    good = p.derive_access
    def mutate(*args):
        result = good(*args)
        grid = copy.deepcopy(GRID); grid["support_hash"] += "changed"
        task.case._grid_record = freeze_json(grid)
        return result
    monkeypatch.setattr(p, "derive_access", mutate)
    with pytest.raises(p.coverage.PAT05TaskBindingMismatch):
        p.canonical_pat05_access_metadata(task, records())
    assert calls["derivations"] == 1


def test_post_derivation_state_change_refuses_metadata(monkeypatch):
    task, calls = fake_boundary(monkeypatch)
    good = p.derive_access
    def mutate(*args):
        result = good(*args); task._state_seal += "changed"; return result
    monkeypatch.setattr(p, "derive_access", mutate)
    with pytest.raises(RuntimeError, match="changed while preparing"):
        p.canonical_pat05_access_metadata(task, records())
    assert calls["frozen_checks"] == 2
