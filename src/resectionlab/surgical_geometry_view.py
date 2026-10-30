"""Read-only conditional occupancy from declared source fields, never an engine.

This mirrors the existing center-raster ownership rules for comparison only.
It performs no action, no mechanics admission and no physical-state mutation.
"""
from types import SimpleNamespace
import numpy as np

from .surgical_occupancy import MaterialOccupancy
from .surgical_physics_v2 import raster


def raster_view(scene,removed,cells,corner_displacement_mm):
    scene.require_cache_source();ids=np.asarray(cells);u=np.asarray(corner_displacement_mm,float)
    if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or len(ids)>16000 or len(np.unique(ids))!=len(ids)
            or np.any(ids<0) or np.any(ids>=scene.labels.size) or u.shape!=(len(ids),8,3) or not np.isfinite(u).all()):
        raise ValueError('GEOMETRY_VIEW_SOURCE_FIELD')
    gone=np.asarray(removed)
    if gone.size==0:gone=np.empty(0,dtype=np.int64)
    if gone.ndim!=1 or gone.dtype.kind not in 'iu' or np.any(gone<0) or np.any(gone>=scene.labels.size):
        raise ValueError('GEOMETRY_VIEW_REMOVED_IDS')
    removed=set(map(int,gone))
    if removed.intersection(map(int,ids)):raise ValueError('GEOMETRY_VIEW_REMOVED_SOURCE_FIELD')
    displacement={str(int(cell)):value.tolist() for cell,value in zip(ids,u.mean(axis=1))}
    pairs=raster(SimpleNamespace(scene=scene,state={'removed':sorted(removed),'deformation':displacement}))
    cleared=removed|{a for a,b in pairs};memberships={};base=scene.labels.ravel();by_label={r['id']:r for r in scene.regions}
    for original,target in pairs:
        if not base[original]:raise ValueError('GEOMETRY_VIEW_UNOWNED_SOURCE')
        if base[target] and target not in cleared:
            if by_label[int(base[target])]['material'] not in ('soft_tissue','tumor'):raise ValueError('DISPLACEMENT_COLLISION')
            memberships.setdefault(target,[]).append(target)
        memberships.setdefault(target,[]).append(original)
    moved={};precedence={r['id']:rank for rank,r in enumerate(scene.regions)}
    for target,originals in memberships.items():
        memberships[target]=sorted(set(originals));moved[target]=max(originals,key=lambda c:(precedence[int(base[c])],c))+1
    return MaterialOccupancy(scene.labels,cleared,moved,memberships),pairs
