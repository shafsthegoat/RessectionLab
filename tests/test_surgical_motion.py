import numpy as np
import pytest
from scipy.spatial.transform import Rotation, Slerp

from resectionlab.geometry import GeometryScene, capsule_voxel_indices, point_segment_distances
from resectionlab.surgical_motion import axial_dissector_sweep, envelopes, swept_envelopes
from resectionlab.surgical_tool_profiles import PROFILE_INDEX
from resectionlab.surgical_tools import TOOLS


DISSECTOR = TOOLS['dissector']
PARAMETERS = {'radiusMm': .16, 'depthMm': .10}


@pytest.mark.parametrize('direction', [-1., 1.])
def test_axial_capsule_cover_contains_interpolated_centerlines_and_radii(direction):
    rng = np.random.default_rng(2011)
    for _ in range(12):
        start_rotation = Rotation.random(random_state=rng)
        end_rotation = Rotation.from_rotvec(rng.normal(size=3) * 1e-8) * start_rotation
        interpolation = Slerp([0., 1.], Rotation.from_quat([start_rotation.as_quat(), end_rotation.as_quat()]))
        origin = rng.uniform(-4095., 4095., 3)
        delta = start_rotation.apply([2e-7, -2e-7, direction * .49])
        start = {'position': origin, 'quaternion': interpolation(0.).as_quat()}
        end = {'position': origin + delta, 'quaternion': interpolation(1.).as_quat()}
        covers = axial_dissector_sweep(DISSECTOR, PARAMETERS, start, end)
        assert covers[1][2] < PARAMETERS['radiusMm'] + 1e-6
        for t in np.linspace(0., 1., 33):
            actual = envelopes(DISSECTOR, PARAMETERS, origin + t * delta, interpolation(t).as_quat())
            for (a, b, radius, part), (lo, hi, bound, role) in zip(actual, covers):
                assert part == role
                residual = point_segment_distances(np.linspace(a, b, 17), lo, hi)
                assert radius + residual.max() <= bound


def test_axial_sweep_keeps_a_voxel_missed_by_both_endpoint_capsules():
    cell = (2, 2, 4)
    mask = np.zeros((5, 5, 12), bool)
    mask[cell] = True
    scene = GeometryScene(mask, np.diag([.05, .05, .05, 1.]))
    start = {'position': [.1, .1, 0.], 'quaternion': [0., 0., 0., 1.]}
    end = {**start, 'position': [.1, .1, .49]}

    def touches(capsule):
        return any(tuple(index) == cell for index in capsule_voxel_indices(scene, *capsule[:3]))

    assert not any(touches(envelopes(DISSECTOR, PARAMETERS, **pose)[1]) for pose in (start, end))
    assert touches(axial_dissector_sweep(DISSECTOR, PARAMETERS, start, end)[1])


@pytest.mark.parametrize(('tool_name', 'overrides', 'position', 'angle'), [
    ('dissector', {}, [0., 0., .1], 2e-5),
    ('dissector', {}, [2e-5, 0., .1], 0.),
    ('dissector', {}, [0., 0., .51], 0.),
    ('dissector', {'contactGeometryRevision': 2}, [0., 0., .1], 0.),
    ('dissector', {'operativeToolProfileV1': PROFILE_INDEX['dissector'],
                   'operativeToolAssemblyRevision': 2}, [0., 0., .1], 0.),
    ('cusa', {}, [0., 0., .1], 0.),
])
def test_axial_sweep_uses_exact_legacy_cover_outside_its_profile(tool_name, overrides, position, angle):
    parameters = {**PARAMETERS, **overrides}
    start = {'position': [0., 0., 0.], 'quaternion': [0., 0., 0., 1.]}
    end = {'position': position, 'quaternion': Rotation.from_rotvec([angle, 0., 0.]).as_quat()}
    result = axial_dissector_sweep(TOOLS[tool_name], parameters, start, end)
    expected = swept_envelopes(TOOLS[tool_name], parameters, start, end)
    assert len(result) == len(expected)
    for (a, b, radius, part), (old_a, old_b, old_radius, old_part) in zip(result, expected):
        np.testing.assert_array_equal(a, old_a)
        np.testing.assert_array_equal(b, old_b)
        assert (radius, part) == (old_radius, old_part)
