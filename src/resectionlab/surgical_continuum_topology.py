"""Offline source-graph rebuilds with inherited supports and a stable partition.

No live domain/cap change or episode field is owned here. Every rebuilt node
inherits an original source-corner lineage; new contacts never supply support.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from .surgical_continuum import voxel_mesh
from .surgical_continuum_events import _geometry


class _DeclaredOwners:
    """Mesher membership view, not a generated patient image or material map."""
    def __init__(self, shape, cells):
        self.size = int(np.prod(shape)); self.cells = set(map(int, cells)); self.flat = self

    def __getitem__(self, cell):
        return int(int(cell) in self.cells)


class SourceDomain:
    """Rebuild one frozen domain under removals and additional unhealed cuts.

    The ordered original core free-DOF declaration classifies every descendant.
    A node that loses its last core *element* remains in I if other owners keep
    it alive. This policy only fixes algebraic partition lineage; it does not
    certify any exterior matrix/factor reuse.
    """
    def __init__(self, geometry, core_free_dofs, *, cut_faces=()):
        g = _geometry(geometry)
        self.geometry = {key: np.asarray(geometry[key]).copy() for key in
            ('cells','shape','nodes','elementNodes','affine','freeDofs','muPa','nu')}
        self.cells = g[0].copy(); self.shape = g[1].copy()
        self.lookup = {int(cell): row for row,cell in enumerate(self.cells)}
        self.original_free = np.flatnonzero(g[5]); core = np.asarray(core_free_dofs)
        if (core.ndim != 1 or core.dtype.kind not in 'iu' or not 0 < len(core) < len(self.original_free)
                or len(core) > 6000 or len(np.unique(core)) != len(core)
                or np.any(core < 0) or np.any(core >= len(self.original_free))):
            raise ValueError('TOPOLOGY_CORE_FREE_PARTITION')
        self.original_core = core.astype(np.int64, copy=True)
        self.is_original_core = np.zeros(len(self.original_free),bool); self.is_original_core[core] = True
        self.core_rank = np.full(len(self.original_free),len(self.original_free),dtype=np.int64)
        self.core_rank[core] = np.arange(len(core))
        self.baseline_cuts = self._cuts(cut_faces)
        baseline = self._build(set(), self.baseline_cuts)
        parents = baseline['originalNodeForNew']
        # Reconstruct the supplied graph from its declared cuts before a later
        # deletion could expose an omitted old cut. Coordinates alone cannot
        # establish source-sector connectivity.
        if len(parents) != len(g[2]) or len(np.unique(parents)) != len(g[2]):
            raise ValueError('TOPOLOGY_BASELINE_GRAPH')

    def _cuts(self, cuts):
        array = np.asarray(cuts)
        if array.size == 0: return set()
        if array.ndim != 2 or array.shape[1] != 2 or array.dtype.kind not in 'iu':
            raise ValueError('TOPOLOGY_CUT_FACE')
        result = set()
        for a,b in array:
            a,b = int(a),int(b)
            if a not in self.lookup or b not in self.lookup:
                raise ValueError('TOPOLOGY_CUT_OUTSIDE_DECLARED_DOMAIN')
            ca,cb = np.asarray(np.unravel_index(a,tuple(self.shape))),np.asarray(np.unravel_index(b,tuple(self.shape)))
            if np.abs(ca-cb).sum() != 1: raise ValueError('TOPOLOGY_CUT_FACE')
            result.add(tuple(sorted((a,b))))
        return result

    def _build(self, removed, cuts):
        original = self.geometry; retained = sorted(set(self.lookup)-removed)
        if not retained: raise ValueError('TOPOLOGY_EMPTY_DOMAIN')
        scene = SimpleNamespace(shape=tuple(self.shape), affine=original['affine'],
            labels=_DeclaredOwners(self.shape,self.cells), regions=[{'id':1,'material':'soft_tissue'}])
        # Explicit offline export bound; the live 24k-DOF cap is unchanged.
        mesh = voxel_mesh(scene,retained,cuts,max_dofs=60000)
        rows = np.asarray([self.lookup[cell] for cell in mesh.cells]); en = mesh.element_nodes
        ancestors = original['elementNodes'][rows].ravel(); target = en.ravel()
        parents = np.full(len(mesh.nodes),len(original['nodes']),dtype=np.int64)
        np.minimum.at(parents,target,ancestors)
        if not np.array_equal(ancestors,parents[target]):
            raise ValueError('TOPOLOGY_SOURCE_JOIN_NOT_ADMITTED')
        if np.max(np.abs(mesh.nodes-original['nodes'][parents]),initial=0.) > 1e-10:
            raise ValueError('TOPOLOGY_SOURCE_CORNER_COORDINATES')
        full_lineage = (3*parents[:,None]+np.arange(3)).ravel()
        free = original['freeDofs'][full_lineage]
        original_free = np.searchsorted(self.original_free,full_lineage[free])
        if not np.array_equal(self.original_free[original_free],full_lineage[free]):
            raise ValueError('TOPOLOGY_FREE_LINEAGE')
        core = np.flatnonzero(self.is_original_core[original_free])
        core = core[np.argsort(self.core_rank[original_free[core]],kind='stable')]
        if not 0 < len(core) < len(original_free) or len(core) > 6000:
            raise ValueError('TOPOLOGY_DERIVED_PARTITION')
        data = dict(cells=np.asarray(mesh.cells),shape=original['shape'].copy(),
            nodes=original['nodes'][parents].copy(),elementNodes=en.copy(),affine=original['affine'].copy(),
            freeDofs=free,muPa=original['muPa'][rows].copy(),nu=original['nu'][rows].copy())
        return {'geometry':data,'coreFreeDofs':core,'originalNodeForNew':parents,
            'originalFreeDofForNew':original_free,'removedSourceIds':sorted(removed),
            'cutFaces':[list(face) for face in sorted(cuts)]}

    def after(self, removed_source_ids=(), *, additional_cut_faces=()):
        owners = np.asarray(removed_source_ids)
        if owners.size == 0: owners = np.empty(0,dtype=np.int64)
        if owners.ndim != 1 or owners.dtype.kind not in 'iu':
            raise ValueError('TOPOLOGY_REMOVED_SOURCE_OWNER')
        removed = set(map(int,owners))
        if not removed <= set(self.lookup): raise ValueError('TOPOLOGY_REMOVAL_OUTSIDE_DECLARED_DOMAIN')
        return self._build(removed,self.baseline_cuts | self._cuts(additional_cut_faces))
