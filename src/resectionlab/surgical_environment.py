"""Shared, source-preserving contract for new desktop/RL research environments.

Display presets never add anatomy. Historical scenes keep their original physics;
only an explicit new-environment operation selects the standard V2 profile.
"""
import copy
import numpy as np
from .core import semantic_digest
from .surgical_scene import SurgicalScene
from .surgical_physics_v2 import default_profile

SCHEMA = 'resectionlab.surgical-environment.v1'
ROLES = {'brain': 'soft_tissue', 'tumor': 'tumor', 'bone': 'bone', 'vessels': 'vessel'}
COLORS = {'soft_tissue': '#b9c5c9', 'tumor': '#9863db', 'bone': '#e5d6b8',
          'vessel': '#e54491', 'membrane': '#82c8d8', 'unknown': '#92a7b8'}


def load_environment(path):
    """Read the same initial environment selected in a saved desktop workspace.

    Reset starts with intact anatomy, not the saved action prefix. Legacy files
    retain their source-defined physics and identity for existing protection data.
    """
    from .imaging import load_case, read_case_artifacts
    from .surgical_scene import from_case
    case = load_case(path)
    record = read_case_artifacts(path).get('surgical_toolbox')
    if not record or record.get('environmentSchema') != SCHEMA:
        return from_case(case)
    scene = standardize(from_case(case, record['mapping']), physics_profile=record['physicsProfile'])
    snapshots = record['branches'].values()
    if not record['branches'] or any(snapshot.get('sceneId') != scene.identity for snapshot in snapshots):
        raise ValueError('Saved environment does not match the source, mapping and physics profile')
    return scene


def default_target_names(case):
    """Share the existing source task selector with bounded rehearsal.

    Explicit per-layer task declarations take precedence. The two historical
    source adapter keys retain their established primary-target meaning.
    """
    layers = case.metadata.get('anatomy_layers', {})
    declared = {name: row['rehearsal_target'] for name, row in layers.items() if 'rehearsal_target' in row}
    if declared:
        return [name for name, selected in declared.items() if selected is True and name in case.compartments]
    for name in ('monai_tumor_core', 'enhancing_target'):
        if name in case.compartments:
            return [name]
    return [name for name in case.compartments if layers.get(name, {}).get('kind') == 'tumor']


def standardize(scene, target_names=None, physics_profile=None):
    """Create a distinct, explicitly versioned scene; never modify the source."""
    properties = copy.deepcopy(scene.properties)
    properties.pop('tissuePhysics', None)
    if physics_profile is not None:
        properties['tissuePhysicsV2'] = copy.deepcopy(physics_profile)
    else:
        properties.setdefault('tissuePhysicsV2', default_profile())
    source = copy.deepcopy(scene.source)
    source['environmentSchema'] = SCHEMA
    regions = copy.deepcopy(scene.regions)
    for region in regions:
        region['color'] = COLORS.get(region['material'], region.get('color', '#92a7b8'))
        if target_names is not None:
            region['target'] = region['name'] in target_names
    return SurgicalScene(scene.labels, scene.affine, regions, source, properties)


def descriptor(scene):
    materials = {row['material'] for row in scene.regions if np.any(scene.labels == row['id'])}
    profile = scene.properties.get('tissuePhysicsV2') or scene.properties.get('tissuePhysics')
    target_cells = sum(int(np.count_nonzero(scene.labels == row['id'])) for row in scene.regions if row.get('target', False))
    return {'schema': SCHEMA, 'frame': 'RAS+', 'units': 'mm',
            'grid': {'shape': list(scene.shape), 'affine': scene.affine.tolist(),
                     'spacingMm': scene.spacing.tolist(), 'voxelVolumeMm3': scene.voxel_mm3},
            'physicsSchema': profile.get('schema') if profile else None,
            'physicsHash': semantic_digest(profile),
            'standardized': scene.source.get('environmentSchema') == SCHEMA,
            'anatomy': {role: ('supplied' if material in materials else 'unknown') for role, material in ROLES.items()},
            'target': {'regionIds': [r['id'] for r in scene.regions if r.get('target', False)],
                       'initialVolumeMm3': target_cells * scene.voxel_mm3},
            'occupancyRule': 'later mapped regions own overlaps; source memberships retained',
            'coverageMeaning': 'Supplied masks do not establish completeness. Unlabeled and outside-image anatomy remain unknown.',
            'functionalAnatomy': 'unknown unless explicit source evidence is supplied',
            'clinicalValidation': False}


WORKING_GRID_SCHEMA = 'resectionlab.surgical-working-grid-refinement.v1'
MAX_WORKING_GRID_CELLS = 128_000_000


def refine_working_grid(case, factor=2):
    """Return a factor-two computational grid over exactly the same source cells.

    Each immutable parent cell becomes eight constant-valued child cells. MRI
    replication is derived interpolation, not a new acquired observation. This
    creates a new case/scene identity and never migrates a native surgical state.
    Grid-bound evidence and nonvoxel/cell-indexed surgical properties require a
    separate explicit adapter and are refused here. Physical tool and task
    parameters are not changed by this operation.
    """
    import math
    from .core import CaseData, array_digest, thaw_json

    if not isinstance(case, CaseData):
        raise TypeError('Working-grid refinement requires an immutable CaseData')
    if type(factor) is not int or factor != 2:
        raise ValueError('Working-grid refinement supports factor 2 only')
    child_shape = tuple(int(size)*factor for size in case.mri.shape)
    if math.prod(child_shape) > MAX_WORKING_GRID_CELLS:
        raise ValueError('Refined working grid exceeds 128 million cells')
    if (case.structural_evidence or case.critical_evidence or case.prior_proposals or
            case.functional_evidence is not None):
        raise ValueError('Grid-bound evidence requires an explicit refinement adapter')
    metadata = thaw_json(case.metadata)
    if 'working_grid_refinement' in metadata:
        raise ValueError('This case already has a refined computational working grid')
    embedded = ('surgical_toolbox', 'surgical_prior_scenes', 'surgical_geometry',
                'surgical_state', 'surgical_episode', 'surgical_actions')
    if any(metadata.get(key) for key in embedded):
        raise ValueError('Embedded surgical data cannot migrate to a refined grid')
    properties = metadata.get('surgical_properties', {})
    if not isinstance(properties, dict) or set(properties)-{'tissuePhysics', 'tissuePhysicsV2'}:
        raise ValueError('Nonvoxel surgical properties require an explicit refinement adapter')

    parent_affine = np.asarray(case.affine)
    child_affine = parent_affine.copy()
    child_affine[:3, :3] = parent_affine[:3, :3]/factor
    child_affine[:3, 3] += parent_affine[:3, :3]@np.full(3, -.25)
    child_to_parent = np.eye(4)
    child_to_parent[:3, :3] /= factor
    child_to_parent[:3, 3] = -.25
    metadata['working_grid_refinement'] = {
        'schema': WORKING_GRID_SCHEMA, 'factor': factor, 'factorPerAxis': factor,
        'parentAffine': case.affine.tolist(), 'parentShape': list(case.mri.shape),
        'parentFrame': case.frame, 'childFrame': case.frame,
        'method': 'nearest piecewise-constant parent-cell subdivision',
        'sourceImageIsInterpolated': True, 'noNewAnatomy': True,
        'purpose': 'Computational-grid convergence study; no new acquired resolution',
        'parentCaseId': case.case_id, 'parentCaseHash': case.semantic_hash,
        'parentPlanningHash': case.planning_hash,
        'parentGrid': {'shape': list(case.mri.shape), 'affine': case.affine.tolist(), 'frame': case.frame,
            'spacingMm': list(case.spacing_mm), 'voxelVolumeMm3': case.voxel_volume_mm3},
        'childGrid': {'shape': list(child_shape), 'affine': child_affine.tolist(), 'frame': case.frame,
            'spacingMm': (np.asarray(case.spacing_mm)/factor).tolist(),
            'voxelVolumeMm3': case.voxel_volume_mm3/factor**3},
        'mapping': {'childCenterToParentVoxelAffine': child_to_parent.tolist(),
            'parentCellForChildIndex': 'floor(child_ijk / 2)',
            'childIndicesForParentCell': '2 * parent_ijk + offset, each offset in {0,1}',
            'constantParentCellReplication': True, 'samePhysicalCellUnion': True},
        'mriSampling': 'Derived nearest piecewise-constant replication of the parent working image; not a new observation',
        'anatomySampling': 'Exact parent voxel-cell subdivision; active/source compartments and brain mask replicated without new labels',
        'sourceAdapterScope': 'Unchanged surgical_adapter sampling/grid fields describe the parent source derivative; childGrid above describes this computational grid',
        'nativeStateMigrated': False, 'clinicalValidation': False,
        'clinicalInterpretation': 'unavailable; discretization comparison only',
    }

    replicated_masks = {}

    def subdivide(array):
        key = array_digest(array) if array.dtype.kind == 'b' else None
        if key is not None and key in replicated_masks:
            return replicated_masks[key]
        result = np.asarray(array)
        for axis in range(3):
            result = np.repeat(result, factor, axis=axis)
        if key is not None:
            replicated_masks[key] = result
        return result

    return CaseData(case_id=case.case_id+'-working-grid-2x',
        mri=subdivide(case.mri), compartments={name: subdivide(mask)
            for name, mask in case.compartments.items()},
        source_compartments={name: subdivide(mask)
            for name, mask in case.source_compartments.items()},
        brain_mask=None if case.brain_mask is None else subdivide(case.brain_mask),
        affine=child_affine, source_refs=case.source_refs, context=case.context,
        revision=case.revision+1, unknowns=case.unknowns, frame=case.frame,
        metadata=metadata)
