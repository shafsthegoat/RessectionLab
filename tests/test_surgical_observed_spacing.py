"""Nominal proposals use the observation's physical sampling grid only."""
import numpy as np
import pytest
from resectionlab.surgical_observed_policy import candidates


def observation(spacing=None):
    local = np.zeros((2, 8, 9, 9, 9))
    local[0, 1, 5, 4, 4] = 1.
    obs = {'local': local, 'tool_parameters': np.full((2, 12), .01),
           'slots': np.zeros((2, 12))}
    obs['slots'][:, 6] = 1.
    if spacing is not None: obs['local_spacing_mm'] = np.asarray(spacing)
    return obs


def test_declared_spacing_changes_nominal_sample_geometry_without_changing_action_step():
    legacy, actions = candidates(observation())
    one, one_actions = candidates(observation([1.]))
    two, two_actions = candidates(observation([2.]))
    np.testing.assert_array_equal(legacy, one)
    assert actions == one_actions == two_actions
    assert one[1, 1] != two[1, 1]
    # The nominal +2 mm action is the same physical request on either grid.
    assert actions[1]['pose']['position'] == [2., 0., 0.]


@pytest.mark.parametrize('spacing', [[0.], [-1.], [np.nan], [1., 2.], []])
def test_invalid_declared_spacing_is_not_silently_replaced(spacing):
    with pytest.raises(ValueError, match='local_spacing_mm'):
        candidates(observation(spacing))
