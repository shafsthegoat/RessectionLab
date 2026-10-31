"""Conservative algebraic partitions for a declared CUSA tip segment.

This changes no physical domain, support, material, field or contact rule.
Geometric coverage never certifies factor reuse: exact source-graph/exterior
and both coupling checks remain necessary after every actual removal.
"""
from itertools import product

import numpy as np

from .surgical_continuum_partition import _validate


def corridor_partition(geometry,contacts,start,requested,radius_mm,max_center_displacement_mm):
    """Select whole source-element DOFs for a bounded absolute-pose corridor.

    Every possible resolved tip must lie on [start,requested]. A candidate
    displaced source center is at most U from rest. Nearest-voxel rounding can
    move its raster center by one cell half-diagonal, and the legacy capsule
    supercover adds another. Thus rest-center distance <= r+U+2H+1e-9 is
    necessary for candidate ownership. Only geometry.cells are selected;
    outside-domain source owners/removals still require refusal. This is a
    numerical selection guarantee for an orthogonal grid, not a contact witness
    or a public-scheduler trajectory guarantee. Query fields must obey U first.
    """
    cells,shape,nodes,en,affine,free=_validate(geometry)
    a,b=(np.asarray(v,float) for v in (start,requested))
    if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (a,b)):
        raise ValueError('CORRIDOR_POSE')
    for value in (radius_mm,max_center_displacement_mm):
        if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value<0:
            raise ValueError('CORRIDOR_DISTANCE_BOUND')
    A=affine[:3,:3]
    if (not np.array_equal(affine[3],[0.,0.,0.,1.]) or abs(np.linalg.det(A))<1e-9
            or not np.allclose(A.T@A,np.diag(np.diag(A.T@A)),rtol=0,atol=1e-12)):
        raise ValueError('CORRIDOR_ORTHOGONAL_SOURCE_GRID')
    corners=np.array(list(product((-.5,.5),repeat=3)))
    centers=np.array(np.unravel_index(cells,tuple(shape))).T@A.T+affine[:3,3]
    expected=centers[:,None,:]+corners@A.T
    if np.max(np.abs(nodes[en]-expected),initial=0.)>1e-10:
        raise ValueError('CORRIDOR_SOURCE_CORNER_OWNERSHIP')
    half=float(np.linalg.norm(corners@A.T,axis=1).max())
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        try:
            bound=float(np.nextafter(float(radius_mm)+float(max_center_displacement_mm)+2*half+1e-9,np.inf))
            direction=b-a;length=float(direction@direction)
            t=np.clip((centers-a)@direction/length,0.,1.) if length else np.zeros(len(cells))
            distance=np.linalg.norm(centers-a-t[:,None]*direction,axis=1)
        except FloatingPointError as error:raise ValueError('CORRIDOR_DISTANCE_OVERFLOW') from error
    if not np.isfinite(bound) or not np.isfinite(distance).all():raise ValueError('CORRIDOR_DISTANCE_OVERFLOW')
    rows=np.flatnonzero(distance<=bound)
    lookup={int(cell):i for i,cell in enumerate(cells)}
    if any(c['sourceCell'] not in lookup for c in contacts):raise ValueError('CORRIDOR_CONTACT_OUTSIDE_DOMAIN')
    contact_rows=np.asarray([lookup[c['sourceCell']] for c in contacts],int)
    def full_free(indices):
        full=np.unique((3*en[indices,:,None]+np.arange(3)).ravel())
        return full[free[full]]
    selected=full_free(np.unique(np.r_[rows,contact_rows]));original_free=np.flatnonzero(free)
    core=np.searchsorted(original_free,selected).astype(np.int64)
    if not 0<len(core)<len(original_free) or len(core)>6000:raise ValueError('CORRIDOR_CORE_LIMIT')
    summary={'sourceCells':len(cells),'physicalTotalDofs':nodes.size,'physicalFreeDofs':len(original_free),
        'corridorSourceCells':len(rows),'coreFreeDofs':len(core),'exteriorFreeDofs':len(original_free)-len(core),
        'radiusMm':float(radius_mm),'maxCenterDisplacementMm':float(max_center_displacement_mm),
        'sourceHalfDiagonalMm':half,'restCenterBoundMm':bound,'startMm':a.tolist(),'requestedMm':b.tolist(),
        'factorRetentionCertified':False,'physicalValidationPassed':False,'largerDomainPublicActionsAdmitted':False,
        'physiologicalRL':'hold','largeScaleRL':'hold'}
    arrays={'coreFreeDofs':core,'corridorSourceIds':cells[rows].astype(np.int64,copy=True),
        'corridorElementRows':rows,'springFullFreeDofs':full_free(contact_rows)}
    for v in arrays.values():v.flags.writeable=False
    return summary,arrays
