"""Source-preserving material adapters for the instrument workbench.

Simulation cells are whole affine cells. Material ownership is distinct from
annotation membership. Names never determine a tool's mechanical behavior.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import numpy as np
from .core import array_digest, immutable_array, semantic_digest, thaw_json

MATERIALS = ('unknown', 'bone', 'soft_tissue', 'tumor', 'membrane', 'vessel', 'fluid', 'functional_region')
KINDS = {'skull': 'bone', 'brain': 'soft_tissue', 'tissue': 'soft_tissue', 'tumor': 'tumor',
         'dura': 'membrane', 'membrane': 'membrane', 'vessels': 'vessel', 'vessel': 'vessel',
         'artery': 'vessel', 'vein': 'vessel', 'sinus': 'vessel', 'bone': 'bone', 'soft_tissue': 'soft_tissue'}


@dataclass
class SurgicalScene:
    labels: np.ndarray
    affine: np.ndarray
    regions: list[dict]
    source: dict
    properties: dict

    def __post_init__(self):
        labels = np.asarray(self.labels)
        a = np.asarray(self.affine, float)
        if labels.ndim != 3 or labels.dtype.kind not in 'iu' or labels.min() < 0:
            raise ValueError('A nonnegative 3D categorical volume is required')
        if a.shape != (4, 4) or not np.isfinite(a).all() or abs(np.linalg.det(a[:3, :3])) < 1e-9 or not np.allclose(a[3], [0, 0, 0, 1]):
            raise ValueError('A nonsingular affine in RAS millimeters is required')
        if labels.size > 128_000_000:
            raise ValueError('Simulation grid exceeds 128 million cells; choose a smaller declared resolution')
        self.regions = copy.deepcopy(self.regions)
        ids = [r['id'] for r in self.regions]
        if len(set(ids)) != len(ids) or any(type(i) is not int or not 1 <= i < 65536 for i in ids):
            raise ValueError('Region IDs must be distinct positive uint16 values')
        if any(r['material'] not in MATERIALS for r in self.regions) or not set(np.unique(labels)) <= {0, *ids}:
            raise ValueError('Every occupied label requires a declared material, including unknown')
        self.labels = immutable_array(labels, np.uint16)
        self.affine = immutable_array(a, np.float64)
        self.inverse = immutable_array(np.linalg.inv(a))
        self.spacing = immutable_array(np.linalg.norm(a[:3, :3], axis=0))
        self.voxel_mm3 = abs(float(np.linalg.det(a[:3, :3])))
        self.source = copy.deepcopy(self.source)
        self.properties = copy.deepcopy(self.properties)
        self._validate_properties()
        self.identity = semantic_digest({'labels': array_digest(self.labels), 'affine': a.tolist(),
                                         'regions': self.regions, 'source': self.source, 'properties': self.properties})
        # Prototyping code may still edit metadata explicitly. Disposable
        # performance caches must refuse such an unversioned source edit.
        self._cache_arrays = tuple((v, v.shape, v.dtype.str, v.strides)
                                   for v in (self.labels, self.affine, self.inverse, self.spacing))
        self._cache_metadata = semantic_digest([self.regions, self.source, self.properties])
        self._cache_identity = self.identity
        self._cache_voxel_mm3 = self.voxel_mm3

    def require_cache_source(self):
        arrays = (self.labels, self.affine, self.inverse, self.spacing)
        if (self.identity != self._cache_identity or self.voxel_mm3 != self._cache_voxel_mm3 or any(
                current is not original or current.shape != shape or current.dtype.str != dtype or current.strides != strides
                for current, (original, shape, dtype, strides) in zip(arrays, self._cache_arrays))
                or semantic_digest([self.regions, self.source, self.properties]) != self._cache_metadata):
            raise ValueError('Source scene changed; construct a newly versioned scene before cached replay')

    def _validate_properties(self):
        """Reject invalid explicit physics before any transition can run."""
        from .surgical_physics import validate_tissue_physics
        validate_tissue_physics(self)
        from .surgical_physics_v2 import validate
        validate(self)
        from .surgical_cusa_v3 import validate as validate_cusa
        validate_cusa(self)
        # Reject a continuum declaration even when its required V2 parent is
        # missing; it must never silently fall through to historical mechanics.
        from .surgical_continuum import validate as validate_continuum
        validate_continuum(self)
        from .surgical_vascular_graph import validate as validate_vascular_graph
        validate_vascular_graph(self)
        from .surgical_bipolar_thermal import validate as validate_bipolar_thermal
        validate_bipolar_thermal(self)
        def cell(value):
            if type(value) is not int or not 0 <= value < self.labels.size:
                raise ValueError('Property cell must index this source grid')
        for field in ('attachments', 'strands', 'bleedingSources', 'functionalEvidence'):
            records = self.properties.get(field, [])
            if not isinstance(records, list): raise ValueError(field + ' must be a list')
            ids = set()
            for row in records:
                if not isinstance(row, dict) or not isinstance(row.get('id'), str) or row['id'] in ids:
                    raise ValueError(field + ' requires distinct string IDs')
                ids.add(row['id'])
                if 'cell' in row: cell(row['cell'])
                for value in row.get('cells', []): cell(value)
                if field in ('attachments', 'strands') and not row.get('cells'):
                    raise ValueError(field + ' requires declared source cells')
                if field in ('bleedingSources', 'functionalEvidence') and 'cell' not in row:
                    raise ValueError(field + ' requires a declared source cell')
                for name in ('rateMm3PerSecond', 'coagulationExposure', 'reservoirMm3', 'exposure'):
                    if name in row and (type(row[name]) not in (int, float) or not np.isfinite(row[name]) or row[name] < 0):
                        raise ValueError(name + ' must be finite and nonnegative')
                if field == 'functionalEvidence' and row.get('probe') not in ('cortical', 'subcortical'):
                    raise ValueError('Functional evidence requires declared cortical/subcortical coverage')
        for value in self.properties.get('boneAnchors', []): cell(value)

    @property
    def shape(self): return self.labels.shape

    def points(self, ids):
        indices = np.column_stack(np.unravel_index(np.asarray(ids, dtype=np.int64), self.shape))
        return indices @ self.affine[:3, :3].T + self.affine[:3, 3]

    def descriptor(self):
        from .surgical_environment import descriptor
        return {'sceneId': self.identity, 'shape': list(self.shape), 'affine': self.affine.tolist(),
                'environment': descriptor(self),
                'regions': copy.deepcopy(self.regions), 'source': copy.deepcopy(self.source),
                'resolutionMm': self.spacing.tolist(), 'frame': 'RAS+', 'units': 'mm',
                'approximations': ['whole-cell removal', ('continuous coupled displacement with shared grid raster' if self.properties.get('tissuePhysicsV2') else 'reversible lattice patch displacement'),
                                   'geometric contact; no calibrated force or clinical risk']}


def from_case(case, mapping=None):
    """Adapt declared source labels; last region wins only in the explicit map."""
    if case.frame not in ('RAS+', 'LPS+'):
        raise ValueError('Declare a RAS+ or LPS+ source frame')
    layers = thaw_json(case.metadata).get('anatomy_layers', {})
    masks = dict(case.compartments)
    if case.brain_mask is not None and 'brain' not in masks:
        masks = {'brain': case.brain_mask, **masks}
    # Bundle JSON canonicalization sorts dictionary keys. Explicit ownership
    # order must survive saving; it cannot depend on insertion/alphabetic order.
    adapter = thaw_json(case.metadata).get('surgical_adapter') or {}
    precedence = adapter.get('ownershipPrecedence')
    if precedence is not None:
        if len(precedence) != len(masks) or set(precedence) != set(masks):
            raise ValueError('Saved ownership precedence must name every annotation exactly once')
        masks = {name: masks[name] for name in precedence}
    if not masks:
        raise ValueError('No segmentation supplied; import a labelmap or assign regions first')
    regions = mapping or [{'id': i + 1, 'name': name, 'material': KINDS.get(layers.get(name, {}).get('kind'), 'unknown'),
                           'target': layers.get(name, {}).get('kind') == 'tumor',
                           'color': layers.get(name, {}).get('color', '#92a7b8')}
                          for i, name in enumerate(masks)]
    if len(regions) != len(masks) or {r['name'] for r in regions} != set(masks):
        raise ValueError('Mapping must name every source region exactly once')
    labels = np.zeros(case.mri.shape, np.uint16)
    for r in regions:
        labels[masks[r['name']]] = r['id']
    affine = case.affine.copy()
    if case.frame == 'LPS+': affine = np.diag([-1., -1., 1., 1.]) @ affine
    return SurgicalScene(labels, affine, regions, {'caseHash': case.semantic_hash, 'caseId': case.case_id,
        'provenance': 'source annotations and declared cached estimates',
        'adapter': thaw_json(case.metadata).get('surgical_adapter'),
        'annotationHashes': {name: array_digest(mask) for name, mask in masks.items()},
        'overlapRule': 'last mapped region owns the material; original memberships retained',
        **({'workingGridRefinement': thaw_json(case.metadata['working_grid_refinement'])}
           if 'working_grid_refinement' in case.metadata else {})}, thaw_json(case.metadata).get('surgical_properties', {}))


def engineering_scene():
    labels = np.zeros((64, 56, 48), np.uint16)
    labels[6:25, 8:27, 20:24] = 1
    labels[32:55, 8:29, 20:34] = 2
    labels[38:49, 14:25, 24:31] = 3
    labels[6:25, 35:49, 22:23] = 4
    labels[33:52, 38:40, 22:24] = 5
    regions = [{'id': i, 'name': name, 'material': material, 'target': i == 3, 'color': color}
               for i, name, material, color in [(1, 'Tile 17', 'bone', '#e5d6b8'),
               (2, 'Block Q', 'soft_tissue', '#c799b0'), (3, 'Insert 9', 'tumor', '#db6b95'),
               (4, 'Sheet A', 'membrane', '#82c8d8'), (5, 'Strand B', 'vessel', '#e7645c')]]
    from .surgical_effects import fixture_properties
    return SurgicalScene(labels, np.eye(4), regions, {'caseId': 'instrument-engineering-fixture',
        'provenance': 'explicit generated engineering geometry; not patient anatomy'},
        fixture_properties(labels))
