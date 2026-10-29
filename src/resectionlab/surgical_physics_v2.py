"""Explicit versioned research mechanics; never silently upgrades a saved scene."""
import copy
from types import SimpleNamespace
import numpy as np
from . import surgical_deformation_v2 as brain
from . import surgical_drilling_v2 as bone
from . import surgical_vessels_v2 as vessel

SCHEMA = 'resectionlab.tissue-physics.v2'

def default_profile():
    return {'schema': SCHEMA, 'provenance': {'kind': 'assumed-research', 'description':
        'Uncalibrated engineering parameters supplied explicitly for simulator experiments; not patient measurements.'},
        'brain': brain.profile(), 'bone': copy.deepcopy(bone.DEFAULT_PROFILE),
        'vessel': dict(model='compliant-wounds-v2', pressurePa=1000., reservoirMm3=100.,
            refillConductanceMm3PerPaS=.001, leakConductanceMmPerPaS=.0001,
            mechanicalDoseNs=2., thermalDoseCs=200., coagulationDose=2., clotRatePerSecond=.02)}

def enabled(engine): return 'tissuePhysicsV2' in engine.scene.properties

def validate(scene):
    p = scene.properties.get('tissuePhysicsV2')
    if p is None: return
    if scene.properties.get('tissuePhysics') is not None: raise ValueError('AMBIGUOUS_PHYSICS_VERSION')
    if not isinstance(p, dict) or set(p) != {'schema', 'provenance', 'brain', 'bone', 'vessel'} or p['schema'] != SCHEMA:
        raise ValueError('COMPLETE_EXPLICIT_V2_PHYSICS_REQUIRED')
    provenance = p['provenance']
    if not isinstance(provenance, dict) or provenance.get('kind') != 'assumed-research' or not isinstance(provenance.get('description'), str) or not provenance['description'].strip():
        raise ValueError('EXPLICIT_ASSUMED_RESEARCH_PROVENANCE_REQUIRED')
    brain._settings(SimpleNamespace(scene=scene)); bone.validate(p['bone']); vessel.validate(p['vessel'])
    from .surgical_continuum import validate as validate_continuum
    validate_continuum(scene)
    from .surgical_vascular_graph import validate as validate_vascular_graph
    validate_vascular_graph(scene)

def initialize(engine):
    if enabled(engine):
        validate(engine.scene)
        brain.initialize(engine); bone.initialize(engine); vessel.initialize(engine)
        from .surgical_bipolar_thermal import initialize as initialize_bipolar_thermal
        initialize_bipolar_thermal(engine)

def raster(engine):
    removed = set(engine.state['removed']); pairs = []
    for key, displacement in sorted(engine.state.get('deformation', {}).items(), key=lambda item:int(item[0])):
        cell = int(key)
        if cell in removed: continue
        ijk = np.rint(np.asarray(np.unravel_index(cell, engine.scene.shape)) + engine.scene.inverse[:3,:3] @ displacement).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= engine.scene.shape): raise ValueError('DISPLACEMENT_OUTSIDE_GRID')
        pairs.append([cell, int(np.ravel_multi_index(ijk, engine.scene.shape))])
    return pairs

def descriptor(engine):
    from .surgical_continuum import descriptor as continuum_descriptor
    from .surgical_vascular_graph import enabled as graph_enabled
    from .surgical_bipolar_thermal import descriptor as thermal_descriptor
    return {'enabled': True, 'profile': copy.deepcopy(engine.scene.properties['tissuePhysicsV2']),
        'capabilities': {'patientCalibratedPhysiology': False, 'globalPerfusion': False,
            'damageTriggeredVesselWounds': True, 'coupledDeformation': True, 'workLimitedDrilling': True},
        'brain': continuum_descriptor(engine) or {'model': brain.profile()['model']}, 'bone': bone.descriptor(engine), 'vessel': vessel.descriptor(engine),
        **({'bipolarThermal': thermal_descriptor(engine)} if 'bipolarThermalV1' in engine.scene.properties else {}),
        'limits': ['Uncalibrated research mechanics; no clinical outcome prediction.',
            'Continuous local displacements rasterized to the common source grid for collision, surfaces and slices.',
            'Compressed soft cells retain all source identities; rendered occupancy is their union, not incompressible FEM.',
            ('Declared partial vascular graph; patient perfusion and outcomes unvalidated.' if graph_enabled(engine) else
             'Finite shared vascular compartment, not an anatomical perfusion network.'),
            'Drill heat has lumped cooling; no irrigation coupling or validated necrosis.']}

def metrics(engine):
    if not enabled(engine): return {}
    displacements = engine.state.get('deformation', {})
    from .surgical_vascular_graph import enabled as graph_enabled
    vascular = vessel.metrics(engine)
    from .surgical_bipolar_thermal import metrics as thermal_metrics
    return {**bone.metrics(engine), **thermal_metrics(engine), 'severedVesselConnections': (vascular['cutEdgeCount'] if graph_enabled(engine) else
        sum(not f['patent'] for f in engine.state['vesselsV2']['faces'].values())), 'vesselPhysicsV2': vascular,
        'brainActiveNodes': len(displacements),
        'brainMaxDisplacementMm': max((float(np.linalg.norm(v)) for v in displacements.values()), default=0.),
        'brainElasticEnergyNmm': engine.state['deformationModel'].get('elasticEnergyNmm', 0.),
        'brainMaxNeighborDisplacementRatio': engine.state['deformationModel'].get('maxNeighborDisplacementRatio', 0.)}
