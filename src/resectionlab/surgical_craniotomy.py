"""Source-bound cranial access declarations and read-only bone topology checks.

Reviewed declarations are required inputs, not clinical or independent-review
certificates. This module performs no cut, grasp, displacement or removal, and
does not qualify a patient task or the physics/RL paths. Scene validation runs
before scene.identity exists, so bindings never depend on it or its cache.
"""
from __future__ import annotations

from dataclasses import dataclass
import builtins as _validation_builtins
import itertools
import math
import re

import numpy as np
from scipy.ndimage import label

from .core import array_digest, semantic_digest
from . import native_backend

PROPERTY = 'cranialAccessV1'
SCHEMA = 'resectionlab.cranial-access.v1'
BASE_CASE_HASH = 'cranialAccessBaseCaseHash'
_PROOFS = ('window', 'support', 'attachments', 'exit')
_FIELDS = {'schema', 'sourceBinding', 'targetBinding', 'attachmentsBinding',
           'windowCells', 'flapCells', 'kerfCells', 'retainedSkullCells',
           'boneAnchorCells', 'entry', 'exit', 'evidence', 'contentHash'}


def _invalid(reason):
    raise ValueError('CRANIOTOMY_CONTRACT_INVALID: ' + reason)


def _blocked(reason):
    raise ValueError('CRANIOTOMY_' + reason)


@dataclass(frozen=True)
class EvidenceProof:
    status: str
    provenance: str
    evidence_sha256: str | None


@dataclass(frozen=True)
class EntryGeometry:
    center_ras_mm: tuple[float, float, float]
    normal_ras_unit: tuple[float, float, float]
    u_ras_unit: tuple[float, float, float]
    half_extents_mm: tuple[float, float]
    maximum_inward_mm: float
    maximum_angle_degrees: float


@dataclass(frozen=True)
class ExitGeometry:
    axis: int
    side: int
    plane_voxel_boundary: float
    minimum_voxel_boundary: tuple[float, float, float]
    maximum_voxel_boundary: tuple[float, float, float]


@dataclass(frozen=True)
class CranialAccessContract:
    source_binding: str
    target_binding: str
    attachments_binding: str
    window_cells: tuple[int, ...]
    flap_cells: tuple[int, ...]
    kerf_cells: tuple[int, ...]
    retained_skull_cells: tuple[int, ...]
    bone_anchor_cells: tuple[int, ...]
    entry: EntryGeometry
    exit: ExitGeometry
    evidence_scope: str
    proofs: tuple[EvidenceProof, ...]  # window, support, attachments, exit
    content_hash: str

    def as_dict(self):
        """A detached declaration, also used to revalidate caller-made objects."""
        return {'schema': SCHEMA, 'sourceBinding': self.source_binding,
            'targetBinding': self.target_binding, 'attachmentsBinding': self.attachments_binding,
            'windowCells': list(self.window_cells), 'flapCells': list(self.flap_cells),
            'kerfCells': list(self.kerf_cells), 'retainedSkullCells': list(self.retained_skull_cells),
            'boneAnchorCells': list(self.bone_anchor_cells),
            'entry': {'centerRasMm': list(self.entry.center_ras_mm),
                'normalRasUnit': list(self.entry.normal_ras_unit), 'uRasUnit': list(self.entry.u_ras_unit),
                'halfExtentsMm': list(self.entry.half_extents_mm),
                'maximumInwardMm': self.entry.maximum_inward_mm,
                'maximumAngleDegrees': self.entry.maximum_angle_degrees},
            'exit': {'axis': self.exit.axis, 'side': self.exit.side,
                'planeVoxelBoundary': self.exit.plane_voxel_boundary,
                'minimumVoxelBoundary': list(self.exit.minimum_voxel_boundary),
                'maximumVoxelBoundary': list(self.exit.maximum_voxel_boundary)},
            'evidence': {'scope': self.evidence_scope, **{name: {
                'status': proof.status, 'provenance': proof.provenance,
                'evidenceSha256': proof.evidence_sha256} for name, proof in zip(_PROOFS, self.proofs)}},
            'contentHash': self.content_hash}


@dataclass(frozen=True)
class BoneComponent:
    cells: tuple[int, ...]
    content_hash: str
    volume_mm3: float
    contract_hash: str


def binding(scene):
    """Stable before/after adapter original-target binding; excludes only self.

    Target annotation hashes bind original overlays when available. Runtime
    eligibility separately verifies the actual immutable overlay memberships.
    Exit bounds are the entire handling workspace, intersected with any other
    native task domain by the execution guard. Entry normal points INWARD.
    """
    targets = [r for r in scene.regions if r.get('target')]
    ids = sorted(r['id'] for r in targets)
    source = dict(scene.source)
    if BASE_CASE_HASH in source:
        # from_case computes this from the full CaseData manifest with ONLY
        # this property removed. The induced caseHash includes this property;
        # substituting the computed base avoids its self-referential digest.
        base = source.pop(BASE_CASE_HASH)
        if (type(source.get('caseHash')) is not str or type(base) is not str or
                not re.fullmatch('sha256:[a-f0-9]{64}', base)):
            _invalid('computed base case binding is invalid')
        source['caseHash'] = base
    return {
        'sourceBinding': semantic_digest({'labels': array_digest(scene.labels),
            'affine': scene.affine.tolist(), 'regions': scene.regions,
            'source': source, 'properties': {k: v for k, v in scene.properties.items()
                                                  if k != PROPERTY}}),
        'targetBinding': semantic_digest({'ownedTarget': array_digest(np.isin(scene.labels, ids)),
            'targetRegions': targets, 'originalAnnotationHashes': {
                r['name']: scene.source.get('annotationHashes', {}).get(r['name']) for r in targets}}),
        'attachmentsBinding': semantic_digest({'present': 'attachments' in scene.properties,
                                               'records': scene.properties.get('attachments')})}


def contract_hash(contract):
    """Detached raw JSON declarations receive an ordinary semantic seal."""
    return semantic_digest({k: v for k, v in contract.items() if k != 'contentHash'})


def _cells(value, size, name, *, empty=False):
    if (type(value) not in (list, tuple) or (not value and not empty) or
            any(type(c) is not int or not 0 <= c < size for c in value) or
            list(value) != sorted(set(value))):
        _invalid(name + ' requires sorted distinct source integer cells')
    return tuple(value)


_CELL_FUNCTION, _CELL_CODE = _cells, _cells.__code__
_CELL_INVALID, _CELL_INVALID_CODE = _invalid, _invalid.__code__
_CELL_NUMPY = tuple((name, function, implementation, getattr(implementation,'__code__',None),
        getattr(implementation,'__defaults__',None),
        None if getattr(implementation,'__kwdefaults__',None) is None else
            tuple(implementation.__kwdefaults__.items()))
    for name in ('isin', 'flatnonzero') for function in (getattr(np, name),)
    for implementation in (getattr(function,'_implementation',None),))
_CELL_NAMESPACE = globals()
_CELL_BINDINGS = tuple((name, getattr(_validation_builtins, name)) for name in
    ('type', 'list', 'tuple', 'any', 'int', 'len', 'sorted', 'set', 'map', 'ValueError'))
_CELL_FALLBACK = object()
_CELL_ARRAY_TYPE = np.ndarray


def _native_cells_supported():
    return (_cells is _CELL_FUNCTION and _cells.__code__ is _CELL_CODE and
        _invalid is _CELL_INVALID and _invalid.__code__ is _CELL_INVALID_CODE and
        all(code is not None and getattr(np,name) is function and function._implementation is implementation and
            implementation.__code__ is code and implementation.__defaults__ is defaults and
            (implementation.__kwdefaults__ is None if keywords is None else
             type(implementation.__kwdefaults__) is dict and
             len(implementation.__kwdefaults__)==len(keywords) and
             all(type(key) is str for key in implementation.__kwdefaults__) and
             all(implementation.__kwdefaults__.get(key,_CELL_FALLBACK) is value for key,value in keywords))
            for name,function,implementation,code,defaults,keywords in _CELL_NUMPY))


def _vector(value, count, name):
    if (type(value) not in (list, tuple) or len(value) != count or
            any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        _invalid(name + ' requires finite explicit coordinates')
    return tuple(value)  # Preserve JSON numeric representation for exact seals.


def _number(value, name, low, high=math.inf, *, strict_low=False):
    if (type(value) not in (int, float) or not math.isfinite(value) or
            value > high or (value <= low if strict_low else value < low)):
        _invalid(name + ' is outside its declared bounds')
    return value


def _entry(raw):
    if type(raw) is not dict or set(raw) != {'centerRasMm', 'normalRasUnit', 'uRasUnit',
            'halfExtentsMm', 'maximumInwardMm', 'maximumAngleDegrees'}:
        _invalid('exact entry geometry required')
    center = _vector(raw['centerRasMm'], 3, 'entry center')
    normal = _vector(raw['normalRasUnit'], 3, 'entry inward normal')
    u = _vector(raw['uRasUnit'], 3, 'entry tangent')
    if (abs(np.linalg.norm(normal)-1.) > 1e-6 or abs(np.linalg.norm(u)-1.) > 1e-6 or
            abs(np.dot(normal, u)) > 1e-6):
        _invalid('entry axes must be orthonormal')
    half = _vector(raw['halfExtentsMm'], 2, 'entry half extents')
    if min(half) <= 0:
        _invalid('positive entry half extents required')
    return EntryGeometry(center, normal, u, half,
        _number(raw['maximumInwardMm'], 'entry inward depth', 0., strict_low=True),
        _number(raw['maximumAngleDegrees'], 'entry maximum angle', 0., 80.))


def _exit(raw, shape):
    if type(raw) is not dict or set(raw) != {'axis', 'side', 'planeVoxelBoundary',
            'minimumVoxelBoundary', 'maximumVoxelBoundary'}:
        _invalid('exact exit/workspace geometry required')
    if type(raw['axis']) is not int or raw['axis'] not in (0, 1, 2) or type(raw['side']) is not int or raw['side'] not in (-1, 1):
        _invalid('exit axis/side must be explicit integers')
    low = _vector(raw['minimumVoxelBoundary'], 3, 'workspace minimum')
    high = _vector(raw['maximumVoxelBoundary'], 3, 'workspace maximum')
    if np.any(np.asarray(low) < -.5) or np.any(np.asarray(high) > np.asarray(shape)-.5) or np.any(np.asarray(low) >= high):
        _invalid('workspace must lie inside full native voxel-face extents')
    plane = _number(raw['planeVoxelBoundary'], 'exit plane', low[raw['axis']], high[raw['axis']])
    return ExitGeometry(raw['axis'], raw['side'], plane, low, high)


def _proofs(raw):
    if type(raw) is not dict or set(raw) != {'scope', *_PROOFS} or raw['scope'] not in ('idealized-engineering', 'patient-source-review'):
        _invalid('explicit evidence scope and four coverage proofs required')
    proofs = []
    for name in _PROOFS:
        proof = raw[name]
        if (type(proof) is not dict or set(proof) != {'status', 'provenance', 'evidenceSha256'} or
                proof['status'] not in ('unknown', 'unreviewed', 'reviewed') or
                type(proof['provenance']) is not str or not proof['provenance'].strip()):
            _invalid('invalid ' + name + ' evidence declaration')
        digest = proof['evidenceSha256']
        if (digest is not None and (type(digest) is not str or not re.fullmatch('[a-f0-9]{64}', digest))) or proof['status'] == 'reviewed' and digest is None:
            _invalid('reviewed ' + name + ' evidence requires a SHA256')
        proofs.append(EvidenceProof(proof['status'], proof['provenance'], digest))
    return raw['scope'], tuple(proofs)


def _window_geometry(scene, cells, entry, workspace):
    normal, u = np.asarray(entry.normal_ras_unit), np.asarray(entry.u_ras_unit)
    v = np.cross(normal, u)
    corners = np.asarray(list(itertools.product((-.5, .5), repeat=3)))
    for start in range(0, len(cells), 4096):
        indices = np.column_stack(np.unravel_index(cells[start:start+4096], scene.shape))
        ijk = indices[:, None, :] + corners
        if (np.any(ijk < np.asarray(workspace.minimum_voxel_boundary)-1e-9) or
                np.any(ijk > np.asarray(workspace.maximum_voxel_boundary)+1e-9)):
            _invalid('window cells extend outside handling workspace')
        local = ijk @ scene.affine[:3, :3].T + scene.affine[:3, 3] - entry.center_ras_mm
        if (np.any(np.abs(local @ u) > entry.half_extents_mm[0]+1e-9) or
                np.any(np.abs(local @ v) > entry.half_extents_mm[1]+1e-9) or
                np.any(local @ normal < -1e-9) or
                np.any(local @ normal > entry.maximum_inward_mm+1e-9)):
            _invalid('window full affine cells exceed the entry rectangle/depth')


def validate_contract(scene, contract=None):
    """Return immutable declarations; unknown proofs remain inspectable, blocked.

    windowCells are the full-thickness allowed bone-work column and contain
    F union K. F/K are disjoint; R is exactly all other categorical source bone.
    """
    raw = scene.properties.get(PROPERTY) if contract is None else contract
    if raw is None:
        if PROPERTY in scene.properties:
            _invalid('present declaration cannot be null')
        return None
    if isinstance(raw, CranialAccessContract):
        try:
            raw = raw.as_dict()
        except (AttributeError, TypeError):
            _invalid('malformed typed contract')
    if type(raw) is not dict or set(raw) != _FIELDS or raw.get('schema') != SCHEMA:
        _invalid('exact versioned declaration required')
    if raw['contentHash'] != contract_hash(raw) or any(raw[k] != v for k, v in binding(scene).items()):
        _invalid('content or source/target/attachment binding differs')
    parsed = _CELL_FALLBACK
    if native_backend._module is not None and _native_cells_supported():
        from .surgical_scene import SurgicalScene
        parsed = native_backend.cranial_cells_partition(raw,scene,(SurgicalScene,_CELL_ARRAY_TYPE),
            _CELL_NAMESPACE,_validation_builtins.__dict__,_CELL_BINDINGS,_CELL_FALLBACK)
    if parsed is _CELL_FALLBACK:
        parsed = [_cells(raw[name], scene.labels.size, name, empty=name == 'boneAnchorCells')
                  for name in ('windowCells', 'flapCells', 'kerfCells', 'retainedSkullCells', 'boneAnchorCells')]
        window, flap, kerf, retained, anchors = map(set, parsed)
        bone = set(map(int, np.flatnonzero(np.isin(scene.labels, [r['id'] for r in scene.regions if r['material'] == 'bone']))))
        if flap & kerf or retained & (flap | kerf) or flap | kerf | retained != bone:
            _invalid('F/K/R must partition all categorical source bone')
        if not (flap | kerf) <= window <= bone or not anchors <= retained:
            _invalid('window must contain F union K; anchors must belong to retained skull')
        if not set(scene.properties.get('boneAnchors', ())) <= anchors:
            _invalid('declared source bone anchors cannot be omitted or reassigned')
    entry, exit_geometry = _entry(raw['entry']), _exit(raw['exit'], scene.shape)
    scope, proofs = _proofs(raw['evidence'])
    _window_geometry(scene, parsed[0], entry, exit_geometry)
    return CranialAccessContract(raw['sourceBinding'], raw['targetBinding'], raw['attachmentsBinding'],
        *parsed, entry, exit_geometry, scope, proofs, raw['contentHash'])


def require_evidence(contract):
    """Require declared reviewed coverage; never grants patient qualification."""
    if not isinstance(contract, CranialAccessContract):
        _blocked('CONTRACT_REQUIRED')
    if (len(contract.proofs) != len(_PROOFS) or
            any(not isinstance(proof, EvidenceProof) for proof in contract.proofs)):
        _blocked('COVERAGE_PROOFS_REQUIRED')
    _proofs(contract.as_dict()['evidence'])
    for name, proof in zip(_PROOFS, contract.proofs):
        if proof.status != 'reviewed':
            _blocked(name.upper() + '_EVIDENCE_REQUIRED')
    return contract


def _original_targets(scene):
    ids = set(map(int, np.flatnonzero(np.isin(scene.labels, [r['id'] for r in scene.regions if r.get('target')]))))
    memberships = getattr(scene, '_original_target_memberships', None)
    if memberships is None:
        if any(r['name'] in scene.source.get('annotationHashes', {})
               for r in scene.regions if r.get('target')):
            _blocked('ORIGINAL_TARGET_EVIDENCE_REQUIRED')
        if not ids:
            _blocked('NONEMPTY_TARGET_EVIDENCE_REQUIRED')
        return ids
    proof = getattr(scene, '_original_target_proof', None)
    if proof is None or proof[0] is not memberships or proof[1] != scene.identity:
        _blocked('ORIGINAL_TARGET_BINDING_INVALID')
    names = {r['name'] for r in scene.regions if r.get('target')}
    if set(memberships) != names:
        _blocked('ORIGINAL_TARGET_BINDING_INVALID')
    for name, values in memberships.items():
        if not isinstance(values, np.ndarray) or values.flags.writeable or values.ndim != 1:
            _blocked('ORIGINAL_TARGET_BINDING_INVALID')
        cells = _cells(values.tolist(), scene.labels.size, 'original target')
        mask = np.zeros(scene.shape, bool); mask.ravel()[list(cells)] = True
        if array_digest(mask) != scene.source.get('annotationHashes', {}).get(name):
            _blocked('ORIGINAL_TARGET_BINDING_INVALID')
        ids.update(cells)
    if not ids:
        _blocked('NONEMPTY_TARGET_EVIDENCE_REQUIRED')
    return ids


def validate_bone_scope(engine, contract=None):
    """Before effects, forbid both kerf and flap damage to target/protection.

    Original memberships may be attached after scene initialization. Unknown
    original target evidence blocks here rather than defaulting to ownership.
    This does not assert reviewed coverage or authorize an execution.
    """
    from .surgical_engine import SurgicalEngine
    from .surgical_transition_meter import MeteredEngine
    if not isinstance(engine, (SurgicalEngine, MeteredEngine)):
        _blocked('NATIVE_ENGINE_REQUIRED')
    contract = validate_contract(engine.scene, contract)
    if contract is None:
        _blocked('CONTRACT_REQUIRED')
    scene = engine.scene
    intended = set(contract.flap_cells) | set(contract.kerf_cells)
    if intended & _original_targets(scene):
        _blocked('TARGET_OVERLAP')
    protected = set(getattr(engine, '_protected_ids', ()))
    for row in scene.source.get('protectedAnatomyV1', []):
        protected.update(row['cells'])
    protected.update(map(int, np.flatnonzero(np.isin(scene.labels,
        [r['id'] for r in scene.regions if r.get('protected') is True or r.get('critical') is True]))))
    if intended & protected:
        _blocked('PROTECTED_STRUCTURE')
    return contract


def eligible_bone_component(engine, contract=None, seed_cell=None, *, allow_detached=None):
    """Resolve exact source IDs only; no cut/grasp/removal or clinical admission.

    Conservative26-connectivity blocks residual face/edge/corner bone bridges.
    Retained skull components must each have an intact declared anchor.
    """
    contract = require_evidence(validate_bone_scope(engine, contract))
    scene, state = engine.scene, engine.state
    flap, kerf, retained = map(set, (contract.flap_cells, contract.kerf_cells, contract.retained_skull_cells))
    if seed_cell is None:
        seed_cell = contract.flap_cells[0]
    if type(seed_cell) is not int or seed_cell not in flap:
        _blocked('FLAP_SEED_INVALID')
    removed = set(_cells(state['removed'], scene.labels.size, 'removed state', empty=True))
    if not kerf <= removed:
        _blocked('KERF_INCOMPLETE')
    if removed & (flap | retained):
        _blocked('INTENDED_BONE_MEMBERSHIP_CHANGED')
    detached = state.get('detachedFragments', [])
    if allow_detached is not None:
        if (type(allow_detached) is not dict or sum(h is allow_detached for h in detached) != 1 or
                allow_detached.get('kind') != 'bone-flap-v1' or
                allow_detached.get('contractHash') != contract.content_hash or
                allow_detached.get('componentHash') != semantic_digest(list(contract.flap_cells)) or
                tuple(allow_detached.get('cells', ())) != contract.flap_cells):
            _blocked('DETACHED_BINDING_INVALID')
        offsets = [allow_detached.get(name) for name in ('offset', 'baseOffset')]
        if any(type(offset) not in (list, tuple) or len(offset) != 3 or
               any(type(n) is not int for n in offset) for offset in offsets):
            _blocked('DETACHED_TRANSFORM_INVALID')
        destination = np.column_stack(np.unravel_index(contract.flap_cells, scene.shape))+offsets[0]
        if (np.any(destination-.5 < np.asarray(contract.exit.minimum_voxel_boundary)-1e-9) or
                np.any(destination+.5 > np.asarray(contract.exit.maximum_voxel_boundary)+1e-9)):
            _blocked('DETACHED_TRANSFORM_INVALID')
    held = set(c for h in [*state['holds'].values(), *detached]
               if 'target' not in h and h is not allow_detached for c in h['cells'])
    if held & flap:
        _blocked('MATERIAL_ALREADY_HELD')
    if held & retained:
        _blocked('RETAINED_SKULL_SUPPORT_CHANGED')
    if not contract.bone_anchor_cells:
        _blocked('SUPPORT_ANCHORS_REQUIRED')
    if 'attachments' not in scene.properties:
        _blocked('ATTACHMENT_COVERAGE_REQUIRED')
    for attachment in scene.properties['attachments']:
        if flap.intersection(attachment['cells']):
            if attachment.get('kind') not in ('periosteal', 'soft'):
                _blocked('ATTACHMENT_MODEL_UNSUPPORTED')
            if attachment['id'] not in state['separatedAttachments']:
                _blocked('SEPARATE_ATTACHMENTS_BEFORE_GRASP')
    bone = np.isin(scene.labels, [r['id'] for r in scene.regions if r['material'] == 'bone'])
    bone.ravel()[list(removed)] = False
    components, _ = label(bone, structure=np.ones((3, 3, 3), bool))
    identifier = int(components.ravel()[seed_cell])
    cells = tuple(map(int, np.flatnonzero(components == identifier)))
    if set(cells) != flap:
        _blocked('BONE_BRIDGE_OR_FLAP_COMPONENT_MISMATCH')
    indices = np.column_stack(np.unravel_index(cells, scene.shape))
    local = np.zeros(tuple(np.ptp(indices, axis=0)+1), bool)
    local[tuple((indices-indices.min(0)).T)] = True
    if label(local)[1] != 1:
        _blocked('FLAP_NOT_FACE_CONNECTED')
    anchored = set(map(int, components.ravel()[list(contract.bone_anchor_cells)]))
    retained_components = set(map(int, components.ravel()[list(retained)]))
    if 0 in anchored or not retained_components <= anchored:
        _blocked('RETAINED_SKULL_SUPPORT_INCOMPLETE')
    geometry = contract.exit; axis = geometry.axis
    occupied = np.nonzero(scene.labels)
    bound = (max(occupied[axis])+.5 if geometry.side > 0 else min(occupied[axis])-.5)
    if geometry.side*(geometry.plane_voxel_boundary-bound) < -1e-9:
        _blocked('EXIT_NOT_SOURCE_EXTERIOR')
    width = float(np.ptp(indices[:, axis])+1)
    room = (geometry.maximum_voxel_boundary[axis]-geometry.plane_voxel_boundary if geometry.side > 0
            else geometry.plane_voxel_boundary-geometry.minimum_voxel_boundary[axis])
    if room <= width+1e-9:
        _blocked('EXIT_WORKSPACE_INSUFFICIENT')
    return BoneComponent(cells, semantic_digest({'contract': contract.content_hash, 'cells': list(cells)}),
                         len(cells)*scene.voxel_mm3, contract.content_hash)
