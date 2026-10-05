"""Cheap metamorphic checks of the experiment-only proposal rule."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[2] / "scripts" / "probe_native_frontier.py"
SPEC = importlib.util.spec_from_file_location("native_frontier_probe", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(affine=None):
    affine = np.eye(4) if affine is None else affine
    tissue = np.ones((17, 17, 20), bool)
    labels = np.zeros(tissue.shape, np.int16)
    labels[3:14, 3:14, 5:19] = 1
    normal = affine[:3, 2] / np.linalg.norm(affine[:3, 2])
    access = SimpleNamespace(center_mm=affine[:3, :3] @ (8., 8., -.5) + affine[:3, 3],
                             normal_inward=normal, radius_mm=6.)
    tools = (SimpleNamespace(tool_id="fine", tip_radius_mm=1.25, shaft_radius_mm=.45),
             SimpleNamespace(tool_id="wide", tip_radius_mm=2.25, shaft_radius_mm=1.1))
    config = SimpleNamespace(affine=affine, access=access, tools=tools,
                              target_labels=labels, tissue_mask=tissue)
    return SimpleNamespace(config=config, remaining_mask=tissue.copy())


def test_pure_deterministic_rule_preserves_source_and_generates_target_endpoints():
    engine = fixture()
    labels = engine.config.target_labels.copy()
    remaining = engine.remaining_mask.copy()
    one = MODULE.axis_rays(engine)
    assert one == MODULE.axis_rays(engine)
    assert len(one[0]) == 26
    for row in one[0]:
        assert row["entry_mm"][2] == -.5
        for tip in row["endpoints_mm"]:
            assert labels[tuple(np.rint(tip).astype(int))] > 0
    np.testing.assert_array_equal(engine.config.target_labels, labels)
    np.testing.assert_array_equal(engine.remaining_mask, remaining)


def test_visible_remaining_cavity_changes_proposals_without_changing_labels():
    engine = fixture()
    before = MODULE.axis_rays(engine)[0]
    engine.remaining_mask[8, 8, 10:] = False
    after = MODULE.axis_rays(engine)[0]
    assert before[0]["endpoints_mm"][0] == [8., 8., 18.]
    assert after[0]["endpoints_mm"][0] == [8., 8., 9.]
    assert engine.config.target_labels[8, 8, 18] == 1


def test_patient_rigid_frame_change_transforms_every_ray_without_resampling():
    angle = .4
    rotation = np.array(((np.cos(angle), -np.sin(angle), 0.),
                         (np.sin(angle), np.cos(angle), 0.), (0., 0., 1.)))
    affine = np.eye(4)
    affine[:3, :3] = rotation
    affine[:3, 3] = (12., -3., 6.)
    before, _ = MODULE.axis_rays(fixture())
    after, _ = MODULE.axis_rays(fixture(affine))
    assert len(before) == len(after)
    for one, two in zip(before, after, strict=True):
        assert one["ray_id"] == two["ray_id"]
        np.testing.assert_allclose(rotation @ one["entry_mm"] + affine[:3, 3], two["entry_mm"])
        np.testing.assert_allclose(np.asarray(one["endpoints_mm"]) @ rotation.T + affine[:3, 3], two["endpoints_mm"])


def test_narrow_access_filters_entire_radius_without_changing_dimensions():
    engine = fixture()
    engine.config.access.radius_mm = 2.
    rows, rejected = MODULE.axis_rays(engine)
    assert [row["tool_id"] for row in rows] == ["fine"]
    assert all(row["reason"] == "FULL_TOOL_APERTURE_PREFILTER" for row in rejected)
    assert engine.config.tools[1].tip_radius_mm == 2.25


@pytest.mark.parametrize("kind", ["tilted", "fractional", "sheared"])
def test_unsupported_access_abstains_instead_of_changing_selected_ray(kind):
    engine = fixture()
    if kind == "tilted":
        engine.config.access.normal_inward = np.array([1., 0., 1.]) / np.sqrt(2)
    elif kind == "fractional":
        engine.config.access.center_mm[0] += .25
    else:
        engine.config.affine[0, 1] = .2
    with pytest.raises(ValueError, match="EXPERIMENT_UNSUPPORTED"):
        MODULE.axis_rays(engine)
