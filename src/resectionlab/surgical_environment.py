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
