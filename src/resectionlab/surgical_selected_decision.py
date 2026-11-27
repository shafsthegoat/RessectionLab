"""Receipt of a selected native route at the intact state of its derivative.

The forecast belongs to the exporter and is kept verbatim as its recorded
provenance. Owner decision 2026-11-27: the planned derivative is an ordinary
operable scene, so the receipt neither refuses nor locks editing; it is shown
only while the engine is still intact. The outcome is measured again from the
restored engine, including the original target membership.
"""
from .core import semantic_digest
from .surgical_sequence_outcomes import original_target_residual_mm3
from .surgical_source_inspection import PROPERTY


def selected_route_forecast():
    return {'schema': 'resectionlab.native-route-forecast.v1', 'decision': 'STOP',
            'stage': 'before-access', 'preparationActions': 0, 'withdrawalActions': 0,
            'targetRemovedMm3': 0., 'nonTargetRemovedMm3': 0., 'elapsedSeconds': 0.,
            'physicalCandidatesExamined': 0,
            'reason': 'No qualified treatment and withdrawal sequence is available for this native inspection source.',
            'requiredEvidence': [
                'Independent review of patient, timepoint, registration, target and observed coverage, including brainstem and cerebellum.',
                'Reviewed entry, permitted compartments, initial scalp/access stage and swept instrument work domain.',
                'Source-bound tissue permissions, harm budgets, and a validated admission record for these exact inputs.']}


def selected_plan_decision(engine, admission, *, branch_id):
    """Return a receipt for a forecast-bearing planned source while it is intact."""
    adapter = engine.scene.source.get('adapter', {})
    route = adapter.get('selectedApproach', {})
    forecast = route.get('forecast')
    if forecast is None:
        return None
    # A legacy saved derivative also carries the retired marker; its copy of
    # the route binding must still agree. New derivatives bind through the route.
    marker = engine.scene.properties.get(PROPERTY)
    if (forecast != selected_route_forecast()
            or route.get('sourceSnapshotId') != adapter.get('sourceCollectionHash')
            or adapter['sourceLayers']['tumor']['layerId'] != route.get('targetLayerId')
            or marker is not None and (not isinstance(marker, dict)
                or marker.get('sourceSnapshotId') != route.get('sourceSnapshotId')
                or marker.get('targetBinding') != route.get('targetBinding')
                or marker.get('approachId') != route.get('id')
                or marker.get('planRevision') != route.get('planRevision'))):
        raise ValueError('Selected plan forecast differs from its source binding')
    metrics = engine.metrics()
    if (engine.actions or engine.history or engine.state['removed'] or metrics['elapsedSeconds']
            or any(slot['inserted'] or slot['active'] for slot in engine.state['slots'].values())):
        return None
    residual = original_target_residual_mm3(engine)
    if residual is None:
        raise ValueError('Selected plan requires the original target membership on the engine grid')
    preparation = {'gridSpacingMm': adapter['resolutionMm'],
                   'allowEstimatedAlignment': adapter['allowUnreviewedRegistration'],
                   'layers': {role: {'layerId': row['layerId'], 'sampledMaskHash': row['sampledMaskHash']}
                              for role, row in adapter['sourceLayers'].items()}}
    result = {'schema': 'resectionlab.native-route-decision.v1', 'patientId': adapter['sourcePatient'],
              'sourceSnapshotId': route['sourceSnapshotId'], 'planRevision': route['planRevision'],
              'routeId': route['id'], 'routeName': route['name'], 'parameterHash': route['parameterHash'],
              'measurementHash': route['measurementHash'], 'targetLayerId': route['targetLayerId'],
              'targetBinding': route['targetBinding'], 'geometry': route['geometry'], 'preparation': preparation,
              'caseHash': engine.scene.source['caseHash'], 'sceneId': engine.scene.identity,
              'branchId': branch_id, 'stateId': engine.state_id, 'forecast': forecast,
              'admissionCodes': [row['code'] for row in admission['blockers']],
              'measured': {'actions': len(engine.actions), 'targetRemovedMm3': metrics['targetRemovedMm3'],
                           'nonTargetRemovedMm3': metrics['nonTargetRemovedMm3'],
                           'originalTargetResidualMm3': residual, 'elapsedSeconds': metrics['elapsedSeconds'],
                           'insertedTools': sum(slot['inserted'] for slot in engine.state['slots'].values()),
                           'activeTools': sum(slot['active'] for slot in engine.state['slots'].values())}}
    result['decisionId'] = semantic_digest(result)
    return result
