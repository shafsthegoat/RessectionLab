"""Opt-in, explicitly assumed research laws; never patient physiology inference.

The vessel model uses local voxel adjacency, not an anatomical centerline graph.
A severed adjacency has no conductance across that connection. Each new cut
creates one equivalent wound outlet with declared Poiseuille assumptions. Global
perfusion remains unknown because inflow/territory anatomy is not supplied.
"""
from __future__ import annotations
import copy
import math

SCHEMA = 'resectionlab.tissue-physics.v1'
VESSEL_PARAMETERS = ('radiusMm', 'pressurePa', 'viscosityPaS', 'pathLengthMm',
                     'maxFlowMm3PerSecond', 'coagulationExposure')


def validate_tissue_physics(scene):
    profile = scene.properties.get('tissuePhysics')
    if profile is None: return
    if not isinstance(profile, dict) or profile.get('schema') != SCHEMA:
        raise ValueError('TISSUE_PHYSICS_SCHEMA_REQUIRED')
    if set(profile) - {'schema', 'provenance', 'vessel', 'bone'}:
        raise ValueError('UNKNOWN_TISSUE_PHYSICS_FIELD')
    provenance = profile.get('provenance')
    if not isinstance(provenance, dict) or provenance.get('kind') != 'assumed-research' or not isinstance(provenance.get('description'), str) or not provenance['description'].strip():
        raise ValueError('EXPLICIT_ASSUMED_RESEARCH_PROVENANCE_REQUIRED')
    if 'bone' in profile and profile['bone'] != {'model': 'rigid-cell-removal-v1'}:
        raise ValueError('UNSUPPORTED_BONE_PHYSICS_MODEL')
    if 'vessel' in profile:
        vessel = profile['vessel']
        if not isinstance(vessel, dict) or vessel.get('model') != 'local-poiseuille-v1' or set(vessel) != {'model', *VESSEL_PARAMETERS}:
            raise ValueError('COMPLETE_EXPLICIT_VESSEL_PROFILE_REQUIRED')
        for name in VESSEL_PARAMETERS:
            value = vessel[name]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or name != 'pressurePa' and value == 0:
                raise ValueError('INVALID_VESSEL_PARAMETER:' + name)
        # Compute in log space to bound all positive finite declarations without
        # overflow or a hidden parameter clamp.
        assumed_wound_flow(vessel)
    if not any(key in profile for key in ('bone', 'vessel')):
        raise ValueError('TISSUE_PHYSICS_PROFILE_HAS_NO_MODEL')


def vessel_profile(engine):
    return engine.scene.properties.get('tissuePhysics', {}).get('vessel')


def assumed_wound_flow(profile):
    """Equivalent outlet flow in mm³/s, capped by an explicit model parameter."""
    if profile['pressurePa'] == 0: return 0.
    # mm-based radius and length: SI Poiseuille conversion leaves 1e0 overall
    # (r_mm^4 / L_mm * 1e-9 m³/s, then * 1e9 mm³/m³).
    log_flow = (math.log(math.pi / 8.) + 4 * math.log(profile['radiusMm']) + math.log(profile['pressurePa'])
                - math.log(profile['viscosityPaS']) - math.log(profile['pathLengthMm']))
    return math.exp(min(log_flow, math.log(profile['maxFlowMm3PerSecond'])))


def initialize_physics(engine):
    validate_tissue_physics(engine.scene)
    if vessel_profile(engine) is not None:
        engine.state.setdefault('vesselConnections', {})
        engine.state.setdefault('vesselWounds', {})


def record_vessel_transections(engine, edges):
    profile = vessel_profile(engine)
    if profile is None: return []
    labels = engine.scene.labels.ravel(); created = []
    for a, b in sorted(set(tuple(sorted(map(int, edge))) for edge in edges)):
        if any(not labels[cell] or engine.by_label[int(labels[cell])]['material'] != 'vessel' for cell in (a, b)): continue
        key = f'{a}:{b}'
        if key in engine.state['vesselConnections']: continue
        source_id = 'vessel-wound:' + key
        engine.state['vesselConnections'][key] = {'cells': [a, b], 'continuity': 'severed', 'patent': False, 'perfusion': 'unknown'}
        engine.state['vesselWounds'][key] = {'cells': [a, b], 'sourceId': source_id,
            'model': profile['model'], 'flowInterpretation': 'one equivalent outlet per severed voxel adjacency'}
        engine.state['sources'].append({'id': source_id, 'cell': a, 'kind': 'modeled-vessel-wound',
            'connectionId': key, 'rateMm3PerSecond': assumed_wound_flow(profile),
            'coagulationExposure': profile['coagulationExposure'], 'patent': False,
            'bleedingActive': True, 'exposure': 0., 'reservoirMm3': 0., 'emittedMm3': 0.})
        created.append(key)
    return created


def physics_descriptor(engine):
    profile = engine.scene.properties.get('tissuePhysics')
    return {'enabled': profile is not None, 'profile': copy.deepcopy(profile),
            'capabilities': {'rigidBoneRemoval': True, 'damageTriggeredVesselWounds': vessel_profile(engine) is not None,
                             'patientCalibratedPhysiology': False, 'globalPerfusion': False},
            'limits': ['local voxel adjacency, not anatomical vessel centerlines',
                      'assumed equivalent outlet flow, not patient blood loss prediction',
                      'one modeled outlet per cut adjacency; total flow depends on grid resolution',
                      'no pressure network, pulsatility, collateral territories, or tissue perfusion']}


def physics_metrics(engine):
    wounds = [source for source in engine.state['sources'] if source.get('kind') == 'modeled-vessel-wound']
    return {'severedVesselConnections': len(engine.state.get('vesselConnections', {})),
            'activeVesselWounds': sum(bool(source['bleedingActive']) for source in wounds),
            'modeledVesselFluidGeneratedMm3': sum(source.get('emittedMm3', 0.) for source in wounds),
            'modeledVesselReservoirMm3': sum(source.get('reservoirMm3', 0.) for source in wounds),
            'globalPerfusion': 'unknown'}
