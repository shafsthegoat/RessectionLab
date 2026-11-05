"""Independent-checker falsification controls on generated whole-cell geometry."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from resectionlab.surgical_scene import SurgicalScene

spec = importlib.util.spec_from_file_location('two_tool_geometry_auditor', Path(__file__).resolve().parents[1]/'scripts/verify_two_tool_geometry.py')
auditor = importlib.util.module_from_spec(spec); spec.loader.exec_module(auditor)


def slot(tool, xyz):
    return {'instrument': tool, 'position': xyz, 'quaternion': [0., 0., 0., 1.],
            'parameters': {'radiusMm': 1.5, 'depthMm': 2.}, 'active': False, 'inserted': True, 'grip': []}


def scene(labels, affine=None):
    return SurgicalScene(labels, np.eye(4) if affine is None else affine,
        [{'id': 1, 'name': 'generated non-target', 'material': 'soft_tissue'},
         {'id': 2, 'name': 'generated target', 'material': 'tumor', 'target': True}],
        {'kind': 'generated audit falsification'}, {})


def frame(before, after, *, op='apply', active=True):
    return {'hand': 'left', 'operation': op, 'active': active, 'before': before,
            'after': after, 'removedBefore': [], 'dt': .1}


def test_controlled_target_working_contact_is_distinct_from_shaft_and_non_target():
    before = {'left': slot('cusa', [20., 20., 17.]), 'right': slot('frazier_suction', [-20., -20., -20.])}
    after = copy.deepcopy(before); after['left']['position'] = [20., 20., 18.5]
    labels = np.zeros((40, 40, 40), np.uint16); labels[20, 20, 20] = 2
    assert auditor.check_geometry([frame(before, after)], scene(labels), set())['passed']
    labels[20, 20, 20] = 1
    assert not auditor.check_geometry([frame(before, after)], scene(labels), set())['passed']
    labels[20, 20, 20] = 2; labels[20, 20, 10] = 2
    result = auditor.check_geometry([frame(before, after)], scene(labels), set())
    assert not result['passed'] and any('shaft' in row['check'] for row in result['failures'])


def test_retreat_proof_rejects_forward_motion_and_other_hand_conflict():
    start = {'left': slot('cusa', [20., 20., 17.]), 'right': slot('frazier_suction', [-20., -20., -20.])}
    contact = copy.deepcopy(start); contact['left']['position'] = [20., 20., 18.5]
    forward = copy.deepcopy(contact); forward['left']['position'] = [20., 20., 19.]
    labels = np.zeros((40, 40, 40), np.uint16); labels[20, 20, 20] = 2
    result = auditor.check_geometry([frame(start, contact), frame(contact, forward, op='withdraw', active=False)], scene(labels), set())
    assert not result['passed'] and any(row['check'] == 'nonaxial-or-forward-contact-exit' for row in result['failures'])
    other = copy.deepcopy(contact); other['right']['position'] = contact['left']['position'][:]
    assert not auditor.check_geometry([frame(other, other)], scene(labels), set())['passed']


def test_submillimeter_crop_keeps_full_radius_obstacles():
    labels = np.zeros((40, 40, 40), np.uint16); labels[26, 20, 20] = 1
    affine = np.diag([.25, .25, .25, 1.])
    tools = {'left': slot('cusa', [5., 5., 5.]), 'right': slot('frazier_suction', [-20., -20., -20.])}
    result = auditor.check_geometry([frame(tools, tools)], scene(labels, affine), set())
    assert not result['passed'] and result['cropHi'][0] >= 26
