"""Refuse filesystem escape without touching acquired source data."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('vascular_probe', Path(__file__).parents[1]/'scripts/data_integrated_vascular_probe.py')
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


@pytest.mark.parametrize('name', ['/tmp/outside', '../outside', 'nested/name', 'a..b', '', 'has space'])
def test_output_name_cannot_escape_or_reinterpret_scope(tmp_path, name):
    with pytest.raises(ValueError, match='plain immutable token'):
        probe.output_path(tmp_path, name)


def test_checked_inputs_reject_relative_and_symlink_escape(tmp_path):
    with pytest.raises(ValueError): probe.checked_input(tmp_path, '../outside')
    (tmp_path/'escape').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(ValueError): probe.checked_input(tmp_path, 'escape/outside')
    assert probe.output_path(tmp_path, '006-SSS-v1') == tmp_path/'integrated-v1/probes/006-SSS-v1'


def test_setup_distinguishes_display_geometry_from_physical_motion():
    import copy
    before = {'slots': {'right': {'position': [1., 2., 3.], 'quaternion': [0., 0., 0., 1.],
        'instrument': 'cusa', 'active': False, 'inserted': False, 'grip': [], 'parameters': {}}},
        'removed': [], 'deformation': {}, 'metrics': {'elapsedSeconds': 0., 'travelMm': 0.}}
    after = copy.deepcopy(before); after['slots']['right']['contactGeometry'] = [{'part': 'shaft'}]
    assert all(probe.setup_invariants(before, after).values())
    after['slots']['right']['position'][0] += .01
    assert not probe.setup_invariants(before, after)['poses_unchanged']
