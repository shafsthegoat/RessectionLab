"""Offline source-corner partition preparation; no solve or history transfer."""
from __future__ import annotations

import numpy as np

GEOMETRY_KEYS = ('cells', 'shape', 'nodes', 'elementNodes', 'affine', 'freeDofs')


def _validate(data):
    cells, shape, nodes, elements, affine, free = (np.asarray(data[k]) for k in GEOMETRY_KEYS)
    if (cells.ndim != 1 or cells.dtype.kind not in 'iu' or not len(cells)
            or len(cells) > 16000 or len(np.unique(cells)) != len(cells)):
        raise ValueError('PARTITION_SOURCE_CELLS')
    if (shape.shape != (3,) or shape.dtype.kind not in 'iu' or np.any(shape <= 0)
            or np.any(cells < 0) or np.any(cells >= np.prod(shape))):
        raise ValueError('PARTITION_SOURCE_SHAPE')
    if (nodes.ndim != 2 or nodes.shape[1] != 3 or not len(nodes) or nodes.size > 60000
            or not np.isfinite(nodes).all() or elements.shape != (len(cells), 8)
            or elements.dtype.kind not in 'iu' or np.any(elements < 0)
            or np.any(elements >= len(nodes)) or len(np.unique(elements)) != len(nodes)
            or any(len(set(row)) != 8 for row in elements.tolist())):
        raise ValueError('PARTITION_NODE_OWNERSHIP')
    if free.dtype.kind != 'b' or free.shape != (nodes.size,):
        raise ValueError('PARTITION_SUPPORT_MASK')
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError('PARTITION_AFFINE')
    return cells, shape, nodes, elements, affine, free


def source_partition(core, reference, *, batch_columns=64):
    """Use source-cell/corner ownership, including exterior-induced node joins.

    The induced core takes its connectivity and supports from the complete
    reference. Original core aliases are exported explicitly; no nonzero field
    is merged or projected. This geometry witness grants no mechanics gate.
    """
    if type(batch_columns) is not int or not 1 <= batch_columns <= 64:
        raise ValueError('PARTITION_BATCH_COLUMNS')
    cc, cs, cn, ce, ca, cf = _validate(core)
    rc, rs, rn, re, ra, rf = _validate(reference)
    if not np.array_equal(cs, rs) or not np.array_equal(ca, ra):
        raise ValueError('PARTITION_DIFFERENT_SOURCE_GRID')
    lookup = {int(cell): row for row, cell in enumerate(rc)}
    if any(int(cell) not in lookup for cell in cc):
        raise ValueError('PARTITION_CORE_NOT_SUBSET')
    rows = np.asarray([lookup[int(cell)] for cell in cc], dtype=np.int64)
    if not np.array_equal(cn[ce], rn[re[rows]]):
        raise ValueError('PARTITION_SOURCE_CORNER_COORDINATES')
    mapping = np.full(len(cn), -1, dtype=np.int64)
    for original, target in zip(ce.flat, re[rows].flat):
        if mapping[original] not in (-1, target):
            raise ValueError('PARTITION_CORE_OWNER_SPLIT')
        mapping[original] = target
    exterior_mask = np.ones(len(rc), dtype=bool)
    exterior_mask[rows] = False
    if not exterior_mask.any():
        raise ValueError('PARTITION_EXTERIOR_REQUIRED')
    core_nodes = np.unique(re[rows])
    exterior_nodes = np.unique(re[exterior_mask])
    interface = np.intersect1d(core_nodes, exterior_nodes)
    exclusive = np.setdiff1d(exterior_nodes, core_nodes)
    if not len(interface):
        raise ValueError('PARTITION_INTERFACE_REQUIRED')
    owners = {}
    for original, target in enumerate(mapping):
        owners.setdefault(int(target), []).append(original)
    merges = [{'referenceNode': target, 'originalCoreNodes': originals,
               'originallyConstrained': bool(not cf.reshape((-1, 3))[originals].any())}
              for target, originals in owners.items() if len(originals) > 1]
    interface_free = int(rf.reshape((-1, 3))[interface].sum())
    exterior_free = int(rf.reshape((-1, 3))[exclusive].sum())
    columns = min(batch_columns, interface_free)
    summary = {
        'coreElements': len(cc), 'exteriorElements': int(exterior_mask.sum()),
        'referenceElements': len(rc), 'originalCoreNodes': len(cn),
        'inducedCoreNodes': len(core_nodes), 'interfaceNodes': len(interface),
        'interfaceFreeDofs': interface_free, 'exteriorExclusiveFreeDofs': exterior_free,
        'coreNodeMerges': merges, 'denseInterfaceFloat64Bytes': 8 * interface_free**2,
        'unbatchedExteriorBasisFloat64Bytes': 8 * exterior_free * interface_free,
        'batchRhsAndResponseFloat64Bytes': 16 * exterior_free * columns,
        'batches': (interface_free + batch_columns - 1) // batch_columns,
        'storageEstimateExcludes': 'Sparse matrices, factor fill, assembly, copies and solver workspaces',
        'mechanicalResponseComputed': False, 'nonzeroHistoryTransferSupported': False,
        'physicalValidationPassed': False, 'physiologicalRL': 'hold',
    }
    arrays = {
        'coreCells': cc.copy(), 'exteriorCells': rc[exterior_mask].copy(),
        'coreElementRowsInReference': rows, 'originalCoreNodeToReference': mapping,
        'coreNodesInReference': core_nodes, 'interfaceNodesInReference': interface,
        'exteriorExclusiveNodesInReference': exclusive,
        'inducedCoreElementNodes': np.searchsorted(core_nodes, re[rows]),
    }
    return summary, arrays
