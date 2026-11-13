"""One complete-sequence ledger for native planning, comparison and display.

Only committed native actions contribute. Preview/search cost belongs to the
planner's TransitionMeter, never the patient's treatment ledger. Unsupported
consequences remain null; no scalar return can override engine constraints.
"""
from __future__ import annotations

import numpy as np

SCHEMA = 'resectionlab.sequence-outcome.v1'


def _quantity(value, baseline, unit, *, aggregation='cumulative', status='source_count'):
    return {'unit': unit, 'status': status, 'aggregation': aggregation,
            'cumulative': float(value), 'baseline': float(baseline),
            'incremental': float(value-baseline) if aggregation == 'cumulative' else None}


def _unknown(reason, unit):
    return {'unit': unit, 'status': 'unavailable', 'cumulative': None,
            'baseline': None, 'incremental': None, 'reason': reason}


def _exchanges(engine):
    return sum(action['operation'] == 'exchange' for action in engine.actions)


def _original_target(engine, baseline):
    from .surgical_scene import original_target_memberships
    memberships = original_target_memberships(engine.scene)
    if memberships is None:
        return {'status': 'unavailable', 'sourceVolumeMm3': None, 'removedMm3': None,
            'residualMm3': None, 'fractionRemoved': None,
            'reason': 'Original target annotation memberships are unavailable in this categorical-only scene; owned-label target is reported separately.'}
    cells = np.unique(np.concatenate(list(memberships.values()))) if memberships else np.empty(0, np.int64)
    removed = np.isin(cells, engine.state['removed'])
    before = np.isin(cells, baseline.state['removed'])
    volume = engine.scene.voxel_mm3
    residual = cells[~removed]
    return {'status': 'source_annotation_membership',
        'sourceVolumeMm3': float(len(cells)*volume),
        'removedMm3': float(removed.sum()*volume),
        'incrementalRemovedMm3': float((removed.sum()-before.sum())*volume),
        'residualMm3': float(len(residual)*volume),
        'fractionRemoved': float(removed.mean()) if len(cells) else None,
        'residualCentroidRasMm': engine.scene.points(residual).mean(axis=0).tolist() if len(residual) else None,
        'meaning': 'Union of original target memberships, including later-owned or protected overlaps. Physical source removal is reported here; treatment credit uses the separate owned-target removal dimension.'}


def sequence_ledger(engine, baseline, *, target, metrics, baseline_metrics, mapped_materials):
    """Build from an already validated native prefix and source summary.

    Call summarize_engine_outcomes for the public checked entry point. This
    adapter performs no full-grid scans or simulator calls.
    """
    dimensions = {name: _quantity(metrics[name], baseline_metrics[name], unit)
                  for name, unit in (('targetRemovedMm3', 'mm3'),
                                     ('nonTargetRemovedMm3', 'mm3'),
                                     ('elapsedSeconds', 's'))}
    dimensions['elapsedSeconds']['status'] = 'modeled_unvalidated'
    dimensions['toolExchanges'] = _quantity(_exchanges(engine), _exchanges(baseline), 'count')
    dimensions['acceptedActions'] = _quantity(len(engine.actions), len(baseline.actions), 'count')
    dimensions['workingMaterialExposureMm3S'] = {
        name: _quantity(value*engine.scene.voxel_mm3,
            baseline_metrics['exposureByMaterialCellSeconds'].get(name, 0.)*engine.scene.voxel_mm3,
            'mm3*s', status='modeled_unvalidated')
        for name, value in metrics['exposureByMaterialCellSeconds'].items()}
    # Continuum and heat models already retain cumulative source exposure and
    # peaks in native state. Instantaneous lattice displacement is not recast
    # as a historical maximum or a cumulative injury measure.
    for name, unit in (('brainStrainSquaredMm3S', 'mm3*s'),
                       ('brainStressSquaredPa2Mm3S', 'Pa2*mm3*s'),
                       ('thermalPeakC', 'degC'), ('thermalExposureMm3KS', 'mm3*K*s'),
                       ('bloodGeneratedMm3', 'mm3'), ('motorEncounterMm3', 'mm3'),
                       ('languageEncounterMm3', 'mm3')):
        dimensions[name] = _unknown('Required declared model or source-bound evidence is unavailable.', unit)
    continuum = engine.state.get('brainContinuumV1')
    before_continuum = baseline.state.get('brainContinuumV1')
    if continuum is not None and before_continuum is not None:
        brain = {str(row['id']) for row in engine.scene.regions if row['material'] == 'soft_tissue'}
        for name, field, unit in (
            ('brainStrainSquaredMm3S', 'strainSquaredVolumeTimeMm3Seconds', 'mm3*s'),
            ('brainStressSquaredPa2Mm3S', 'stressSquaredVolumeTimePa2Mm3Seconds', 'Pa2*mm3*s')):
            def total(state):
                return sum(state['exposureBySourceLabel'].get(label, {}).get(field, 0.) for label in brain)
            if brain:
                dimensions[name] = _quantity(total(continuum), total(before_continuum), unit,
                                              status='modeled_unvalidated')
    if 'bipolarThermalV1' in engine.state and 'bipolarThermalV1' in baseline.state:
        for name, metric, unit, aggregation in (
            ('thermalPeakC', 'bipolarPeakTemperatureC', 'degC', 'peak'),
            ('thermalExposureMm3KS', 'bipolarExcessTemperatureVolumeTimeMm3KSeconds', 'mm3*K*s', 'cumulative')):
            dimensions[name] = _quantity(metrics[metric], baseline_metrics[metric], unit,
                aggregation=aggregation, status='modeled_unvalidated')
    mapped_vessels = 'vessel' in mapped_materials
    if mapped_vessels and ('vascularGraphV1' in engine.state or 'vesselsV2' in engine.state):
        dimensions['bloodGeneratedMm3'] = _quantity(metrics['vesselPhysicsV2']['generatedMm3'],
            baseline_metrics['vesselPhysicsV2']['generatedMm3'], 'mm3', status='modeled_unvalidated')
    withdrawn = not engine.state['holds'] and not any(
        slot['inserted'] or slot['active'] or slot['grip'] for slot in engine.state['slots'].values())
    guard = engine.guard_report()
    missing = [name for name, value in dimensions.items()
               if isinstance(value, dict) and value.get('status') == 'unavailable']
    return {'schema': SCHEMA, 'sceneId': engine.scene.identity,
        'stateId': engine.state_id, 'baselineStateId': baseline.state_id,
        'target': target, 'targetScope': 'owned categorical target labels',
        'originalTarget': _original_target(engine, baseline), 'dimensions': dimensions,
        'withdrawalComplete': withdrawn,
        'protection': {'contractId': guard.get('contractId'), 'status': guard.get('status', 'unknown-coverage'),
                       'missingRequired': list(guard['missingRequired'])},
        'missingEvidence': missing,
        'accounting': 'Unique source-cell removal and cumulative native fields over the complete accepted history, including access, preparation, exchanges and exit. Shared prefix counted once.',
        'limitations': ['Working-material exposure omits comprehensive swept-shaft contact; it is not injury.',
            'Modeled consequences are uncalibrated. Missing labels and clinical outcomes remain unknown.',
            'Rejected proposals and discarded previews contribute no treatment; their computation is charged separately.']}


def sequence_outcome(engine, *, baseline=None, use_geometry_cache=True):
    """Public checked contract shared by planners and the desktop bridge."""
    from .surgical_modeled_outcomes import summarize_engine_outcomes
    return summarize_engine_outcomes(engine, baseline=baseline,
        use_geometry_cache=use_geometry_cache)['sequence']
