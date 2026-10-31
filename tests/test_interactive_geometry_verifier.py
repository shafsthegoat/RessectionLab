"""Adversarial controls for the independent exported-program geometry audit."""
import importlib.util
from pathlib import Path

import numpy as np

from resectionlab.geometry import GeometryScene, ToolGeometry, ToolPose


SPEC = importlib.util.spec_from_file_location('interactive_geometry_verifier',
    Path(__file__).parents[1] / 'scripts/verify_interactive_geometry.py')
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


def route(end=(20., 20., 20.)):
    pose = lambda p: {'position': list(p), 'quaternion': [0., 0., 0., 1.]}
    return {'actionProgram': [{'operation': 'place', 'pose': pose((20., 20., 10.))},
                              {'operation': 'move', 'pose': pose(end)}],
            'withdrawalProgram': [{'operation': 'deactivate'},
                                  {'operation': 'withdraw', 'pose': pose((20., 20., 10.))}]}


def certify(mask, program=None, other_tools=()):
    scene = GeometryScene(mask, np.eye(4))
    shaft = ToolGeometry('audit-shaft', .5, .5, 10., tip_length_mm=2.)
    full = ToolGeometry('audit-cusa', 1.5, .5, 10., tip_length_mm=1e-10)
    return VERIFIER.certify_route(program or route(), shaft=shaft, full=full,
        shaft_scene=scene, working_scene=scene, other_tools=other_tools)


def test_independent_checker_clear_route_and_exact_reverse():
    assert all(result.feasible for result in certify(np.zeros((40, 40, 40), bool)))


def test_independent_checker_refuses_obstacle_intersecting_shaft_only():
    mask = np.zeros((40, 40, 40), bool)
    mask[20, 20, 5] = True
    insertion, withdrawal, whole = certify(mask)
    assert not insertion.feasible and not withdrawal.feasible and not whole.feasible
    assert any('shaft' in failure.detail for failure in insertion.failures)


def test_independent_checker_refuses_working_sphere_when_shaft_clears():
    mask = np.zeros((40, 40, 40), bool)
    mask[22, 20, 20] = True
    insertion, withdrawal, whole = certify(mask)
    assert insertion.feasible and withdrawal.feasible and not whole.feasible
    assert any('tip' in failure.detail for failure in whole.failures)


def test_independent_checker_refuses_static_secondary_tool_in_free_space():
    secondary = (ToolGeometry('secondary', .5, .5, 10., tip_length_mm=2.),
                 ToolPose((20., 20., 15.), (0., 0., 1.)))
    insertion, withdrawal, whole = certify(np.zeros((40, 40, 40), bool), other_tools=(secondary,))
    assert insertion.feasible and withdrawal.feasible and not whole.feasible
    assert any(failure.reason == 'TOOL_COLLISION' for failure in whole.failures)


def test_independent_checker_refuses_shortcut_withdrawal():
    import pytest
    program = route()
    program['withdrawalProgram'][-1]['pose']['position'][0] += 1
    with pytest.raises(ValueError, match='original outside pose'):
        certify(np.zeros((40, 40, 40), bool), program)


def test_independent_checker_does_not_certify_tissue_effect_program():
    import pytest
    program = route()
    program['actionProgram'][-1]['operation'] = 'apply'
    with pytest.raises(ValueError, match='tissue-effect program'):
        certify(np.zeros((40, 40, 40), bool), program)
