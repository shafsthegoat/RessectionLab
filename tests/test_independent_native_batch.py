"""Opt-in batch parity on complete small numerical histories, never patients."""

import ast
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab import evaluation
from resectionlab import independent_geometry_batch as batch
from resectionlab.geometry import AccessWindow
from resectionlab.native_resection import NATIVE_GENERIC_TOOLS, NativeResectionConfig, NativeResectionEngine
from test_worlds_evaluation import native_history_fixture


def test_production_kernel_preserves_reviewed_prototype_executable_math():
    root = Path(__file__).resolve().parents[1]
    def code(path):
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not (
            isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
            or isinstance(node, ast.Assign) and any(isinstance(name, ast.Name) and name.id == "VERSION" for name in node.targets))]
        return ast.dump(tree, include_attributes=False)
    assert code(root / "scripts/independent_geometry_batch_prototype.py") == code(root / "src/resectionlab/independent_geometry_batch.py")


def _audit(case, tool, access, tissue, history, **kwargs):
    return evaluation.independent_check_native_history(case, (tool,), history,
        tissue_mask=tissue, access=access, **kwargs)


@pytest.mark.parametrize("failure", [None, "missing_contact", "premature_removal", "enlarged_tip", "macro_accounting", "macro_tip", "hard_exclusion"])
def test_full_certificates_match_for_accepted_and_rejected_histories(failure):
    case, tool, access, tissue, history = native_history_fixture()
    kwargs = {}
    if failure == "missing_contact":
        history[0]["microsteps"][0]["contact_indices_native"] = []
        reason = "unrecorded_partial_active_tissue_contact"
    elif failure == "premature_removal":
        history[0]["microsteps"][0]["removed_indices_native"] = [[3, 3, 3]]
        reason = "native_removed_cell_not_fully_contained"
    elif failure == "enlarged_tip":
        history[0]["microsteps"][0]["active_radius_mm"] = 3.0
        reason = "native_active_envelope_differs_from_frozen_tool"
    elif failure == "macro_accounting":
        history[0]["removed_indices_native"] = []
        reason = "native_macro_removal_accounting_mismatch"
    elif failure == "macro_tip":
        history[0]["tip_mm"] = [3, 3, 4]
        reason = "native_macro_tip_mismatch"
    elif failure == "hard_exclusion":
        hard = np.zeros_like(tissue); hard[3, 3, 2] = True
        kwargs["hard_exclusion"] = hard
        reason = "native_full_tool_hard_constraint_failure"
    else:
        reason = None
    before = deepcopy(history)
    reference = _audit(case, tool, access, tissue, history, **kwargs)
    assert reference.failures == (() if reason is None else (reason,))
    for size in (1, 7, 256):
        actual = _audit(case, tool, access, tissue, history, distance_backend="batch", distance_batch_size=size, **kwargs)
        assert actual.to_dict() == reference.to_dict()
    assert history == before


def test_batch_cannot_borrow_tissue_removed_at_microstep_endpoint():
    case, tool, _, tissue, history = native_history_fixture()
    tool = replace(tool, tip_radius_mm=.9, shaft_radius_mm=.1, tip_length_mm=.1)
    access = AccessWindow((3, 3, 2.49), (0, 0, 1), 2)
    initial, cutting = history[0]["microsteps"]
    initial.update(tip_start_mm=(3, 3, 2.49), tip_end_mm=(3, 3, 2.49),
        active_stroke_start_mm=(3, 3, 2.39), active_stroke_end_mm=(3, 3, 2.49), active_radius_mm=.9)
    cutting.update(tip_start_mm=(3, 3, 2.49), tip_end_mm=(3, 3, 2.99),
        active_stroke_start_mm=(3, 3, 2.39), active_stroke_end_mm=(3, 3, 2.99), active_radius_mm=.9)
    scalar = _audit(case, tool, access, tissue, history)
    assert scalar.failures == ("native_shaft_collides_with_remaining_tissue",)
    assert _audit(case, tool, access, tissue, history, distance_backend="batch", distance_batch_size=1).to_dict() == scalar.to_dict()


def _two_cut_fixture(frame):
    tissue = np.zeros((9, 9, 10), dtype=bool)
    tissue[1:8, 1:8, 3:9] = True
    matrix = np.eye(4)
    if frame != "identity":
        matrix[:3, :3] = Rotation.from_euler("xyz", [13, -17, 23], degrees=True).as_matrix() @ np.diag([1.3, 1.1, .7])
        matrix[:3, 3] = [31, -12, 48]
    if frame == "mirrored_lps":
        matrix[:3, 0] *= -1
    point = lambda value: matrix[:3, :3] @ value + matrix[:3, 3]
    axis = matrix[:3, 2] / np.linalg.norm(matrix[:3, 2])
    access = AccessWindow(point([4, 4, 2.5]), axis, 4)
    tool = NATIVE_GENERIC_TOOLS[0]
    config = NativeResectionConfig(tissue, tissue.astype(np.int16), matrix, access,
        (tool,), f"two-cut-unit-{frame}", "Explicit numerical unit box, no patient data")
    engine = NativeResectionEngine(config)
    for x in (3, 5):
        result = engine.execute_stroke(tool.tool_id, point([x, 4, 5]), entry_mm=point([x, 4, 2.5]))
        assert result.feasible, result.reason
    case = SimpleNamespace(mri=tissue.astype(float), affine=matrix, semantic_hash=config.source_hash, frame="RAS+")
    if frame == "mirrored_lps":
        case.affine = np.diag([-1., -1., 1., 1.]) @ matrix
        case.frame = "LPS+"
    return case, tool, access, tissue, deepcopy(engine.history)


@pytest.mark.parametrize("frame", ["identity", "anisotropic_oblique", "mirrored_lps"])
def test_complete_two_cut_contact_shaft_and_occupancy_prefixes_match(frame, monkeypatch):
    case, tool, access, tissue, history = _two_cut_fixture(frame)
    contacts, shafts, prefixes = [], [], []
    original_contacts = evaluation._native_active_contacts
    original_collision = evaluation._cell_collision
    original_extend = evaluation._extend_independent_free_space

    def traced_contacts(remaining, *args, **kwargs):
        result = original_contacts(remaining, *args, **kwargs)
        contacts.append((remaining.tobytes(), tuple(sorted(result))))
        return result

    def traced_collision(scene, a, b, radius, **kwargs):
        result = original_collision(scene, a, b, radius, **kwargs)
        if np.any(scene.forbidden_mask):
            shafts.append((scene.forbidden_mask.tobytes(), np.asarray(a).tobytes(), np.asarray(b).tobytes(), radius, result))
        return result

    def traced_extend(remaining, free, keys, **kwargs):
        original_extend(remaining, free, keys, **kwargs)
        prefixes.append((remaining.tobytes(), free.tobytes(), tuple(sorted(keys))))

    monkeypatch.setattr(evaluation, "_native_active_contacts", traced_contacts)
    monkeypatch.setattr(evaluation, "_cell_collision", traced_collision)
    monkeypatch.setattr(evaluation, "_extend_independent_free_space", traced_extend)
    scalar = _audit(case, tool, access, tissue, history)
    assert scalar.feasible and scalar.action_count == 2
    expected = deepcopy((contacts, shafts, prefixes))
    assert len(contacts) == sum(len(record["microsteps"]) for record in history)
    assert len(contacts) == len(shafts) == len(prefixes)
    contacts.clear(); shafts.clear(); prefixes.clear()
    actual = _audit(case, tool, access, tissue, history, distance_backend="batch", distance_batch_size=7)
    assert actual.to_dict() == scalar.to_dict()
    assert (contacts, shafts, prefixes) == expected
    assert tissue.sum() > 0  # borrowed source remains occupied, never mutated


def test_default_and_general_geometry_never_enter_batch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Default scalar checks must not enter the opt-in backend")
    monkeypatch.setattr(evaluation, "_batch_cell_contacts", forbidden)
    case, tool, access, tissue, history = native_history_fixture()
    assert _audit(case, tool, access, tissue, history).feasible
    scene = SimpleNamespace(forbidden_mask=tissue, affine=np.eye(4))
    assert evaluation._cell_collision(scene, np.array([3., 3, 1]), np.array([3., 3, 4]), .2) == (3., 3., 3.)


@pytest.mark.parametrize("backend,size", [("other", 256), (None, 256), ([], 256), ("batch", 0), ("batch", 4097), ("scalar", True)])
def test_invalid_opt_in_configuration_refused(backend, size):
    case, tool, access, tissue, history = native_history_fixture()
    with pytest.raises(ValueError):
        _audit(case, tool, access, tissue, history, distance_backend=backend, distance_batch_size=size)


def test_box_arrays_stay_bounded_and_current_mask_controls_each_call(monkeypatch):
    original = batch.segment_box_contact_indices
    sizes = []
    def recorded(start, end, lower, upper, *args, **kwargs):
        sizes.append(len(lower))
        return original(start, end, lower, upper, *args, **kwargs)
    monkeypatch.setattr(batch, "segment_box_contact_indices", recorded)
    remaining = np.ones((5, 5, 5), bool)
    arguments = (np.eye(3), np.ones(3), np.zeros(3), np.array([2., 2, 0]), np.array([2., 2, 4]), 1.25)
    kwargs = dict(distance_backend="batch", distance_batch_size=7, cancelled=None)
    first = evaluation._native_active_contacts(remaining, *arguments, **kwargs)
    remaining[2, 2, 2] = False
    second = evaluation._native_active_contacts(remaining, *arguments, **kwargs)
    assert first - second == {(2, 2, 2)}
    assert max(sizes) <= 7


def test_cancellation_inside_batch_returns_incomplete_without_committing(monkeypatch):
    case, tool, access, tissue, history = native_history_fixture()
    original = batch._distances_chunk
    entered = []
    def recorded(*args):
        entered.append(True)
        return original(*args)
    monkeypatch.setattr(batch, "_distances_chunk", recorded)
    checked = _audit(case, tool, access, tissue, history, distance_backend="batch",
                     distance_batch_size=1, cancelled=lambda: bool(entered))
    assert entered
    assert checked.failures == ("independent_validation_cancelled",)
    assert not checked.complete_tool_checked and not checked.frontier_checked
    assert checked.contained_source_tissue_volume_mm3 == 0
    assert tissue[3, 3, 3]
