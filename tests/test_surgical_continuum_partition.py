"""Source-corner identity, induced connectivity and explicit partition refusal."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_continuum import voxel_mesh
from resectionlab.surgical_continuum_partition import source_partition
from resectionlab.surgical_scene import SurgicalScene


def geometry(cells, blocked=()):
    scene = SurgicalScene(np.ones((3, 3, 1), np.uint16), np.eye(4),
        [{'id': 1, 'name': 'Generated soft block', 'material': 'soft_tissue'}],
        {'kind': 'generated partition control'}, {})
    mesh = voxel_mesh(scene, cells, blocked)
    return dict(cells=np.asarray(mesh.cells), shape=np.asarray(scene.shape), nodes=mesh.nodes,
                elementNodes=mesh.element_nodes, affine=scene.affine,
                freeDofs=np.ones(mesh.nodes.size, dtype=bool))


def test_induced_core_joins_sectors_through_exterior_without_losing_source_corners():
    # Diagonal core cells do not share a face. The full block connects their
    # common geometric edge through exterior source cells.
    core, reference = geometry([0, 4]), geometry(range(9))
    summary, arrays = source_partition(core, reference)
    assert summary['coreNodeMerges'] and summary['inducedCoreNodes'] < summary['originalCoreNodes']
    assert set(arrays['coreCells']).isdisjoint(arrays['exteriorCells'])
    assert set(arrays['coreCells']) | set(arrays['exteriorCells']) == set(range(9))
    rows = arrays['coreElementRowsInReference']
    mapped = arrays['originalCoreNodeToReference'][core['elementNodes']]
    np.testing.assert_array_equal(mapped, reference['elementNodes'][rows])
    induced = arrays['coreNodesInReference'][arrays['inducedCoreElementNodes']]
    np.testing.assert_array_equal(induced, mapped)
    assert summary['mechanicalResponseComputed'] is False
    assert summary['nonzeroHistoryTransferSupported'] is False


def test_source_corner_mapping_refuses_a_split_even_with_identical_coordinates():
    core = geometry([0, 1])
    reference = geometry(range(9), [(0, 1), (1, 4), (3, 4)])
    with pytest.raises(ValueError, match='OWNER_SPLIT'):
        source_partition(core, reference)


@pytest.mark.parametrize('change,error', [
    (lambda d: d['cells'].__setitem__(1, d['cells'][0]), 'SOURCE_CELLS'),
    (lambda d: d['nodes'].__setitem__((0, 0), float('nan')), 'NODE_OWNERSHIP'),
    (lambda d: d['freeDofs'].__setitem__(slice(None), False), None),
    (lambda d: d['affine'].__setitem__((0, 3), 1.), 'DIFFERENT_SOURCE_GRID'),
])
def test_malformed_geometry_or_changed_supports_are_explicit(change, error):
    core, reference = geometry([0, 1]), geometry(range(9))
    reference = copy.deepcopy(reference); change(reference)
    if error:
        with pytest.raises(ValueError, match=error): source_partition(core, reference)
    else:
        summary, _ = source_partition(core, reference)
        assert summary['interfaceFreeDofs'] == summary['exteriorExclusiveFreeDofs'] == 0


def test_geometry_partition_owns_export_arrays_and_enforces_batch_bound():
    core, reference = geometry([0, 1]), geometry(range(9))
    _, arrays = source_partition(core, reference)
    arrays['coreCells'][0] = 100
    assert core['cells'][0] == 0
    with pytest.raises(ValueError, match='BATCH_COLUMNS'):
        source_partition(core, reference, batch_columns=65)
