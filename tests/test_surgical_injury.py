"""Opt-in injury mode on a small synthetic fixture (engineering geometry, not anatomy).

Guarded mode must keep today's refusals and state identities exactly; injury
mode must record what an unsafe move does and let the motion continue.
"""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from resectionlab.surgical_engine import SurgicalEngine, _microstep_checkpoint
from resectionlab.surgical_injury import PROPERTY, default_profile
from resectionlab.surgical_rl_protection import GuardConfig, ProtectionAnnotation, ProtectionSet
from resectionlab.surgical_scene import SurgicalScene

Q = [0., 0., 0., 1.]


def scene(mode=None, **overrides):
    labels = np.zeros((24, 24, 28), np.uint16)
    labels[0:8, :, 8:10] = 1      # dura over the drill site only
    labels[:, :, 10:24] = 2       # cortex
    labels[14:17, 4:20, 13] = 3   # one vessel strand under the CUSA site
    labels[20:23, 2:6, 10:20] = 4  # rigid bone pillar
    labels[10:13, 15:19, 10:13] = 5  # exposed lesion
    regions = [dict(id=1, name='dura', material='membrane'), dict(id=2, name='cortex', material='soft_tissue'),
               dict(id=3, name='artery', material='vessel'), dict(id=4, name='bone pillar', material='bone'),
               dict(id=5, name='lesion', material='tumor', target=True)]
    lesion = [int(c) for c in np.flatnonzero(labels.ravel() == 5)]
    properties = {'attachments': [{'id': 'lesion-pia', 'kind': 'tumor', 'cells': lesion[:6], 'breakable': False}]}
    if mode is not None:
        properties[PROPERTY] = {**default_profile(), 'mode': mode, **overrides}
    return SurgicalScene(labels, np.eye(4), regions, {'caseId': 'injury-fixture',
                         'provenance': 'generated engineering geometry; not patient anatomy'}, properties)


def pose(position, q=Q):
    return {'position': [float(v) for v in position], 'quaternion': [float(v) for v in q]}


def move(slot, position, duration=1., q=Q, op='move'):
    return {'slot': slot, 'operation': op, 'duration': duration, 'path': [pose(position, q)]}


def tool(slot, name, at):
    return [{'slot': slot, 'operation': 'exchange', 'instrument': name, 'duration': 1.},
            {'slot': slot, 'operation': 'place', 'duration': 1., 'path': [pose(at)]}]


def on(slot):
    return {'slot': slot, 'operation': 'activate', 'duration': .1}


def off(slot):
    return {'slot': slot, 'operation': 'deactivate', 'duration': .1}


SCENARIOS = {
    # CUSA aspirating onto a vessel: UNSUPPORTED_MATERIAL.
    'cusa_vessel': [*tool('right', 'cusa', (15.5, 12, 4)), on('right'), move('right', (15.5, 12, 15))],
    # A passive dissector pushed through dura into cortex: RETAINED_TISSUE_COLLISION.
    'dissector_push': [*tool('left', 'dissector', (4, 12, 4)), move('left', (4, 12, 13))],
    # An active CUSA levered sideways in a deep corridor: SHAFT_COLLISION.
    'cusa_lever': [*tool('right', 'cusa', (19, 13, 4)), on('right'), move('right', (19, 13, 22), 3.),
                   move('right', (19, 13, 22), q=Rotation.from_euler('y', -20, degrees=True).as_quat())],
    # A cranial drill run onto dura and cortex: UNSUPPORTED_MATERIAL (plunge).
    'drill_plunge': [*tool('left', 'cranial_drill', (4, 12, 4)), on('left'), move('left', (4, 12, 14), 2.)],
    # A passive CUSA swept into a bone pillar: blunt cortex contact, then rigid bone.
    'bone_sweep': [*tool('right', 'cusa', (21.5, 12, 4)), on('right'), move('right', (21.5, 12, 16), 2.),
                   off('right'), move('right', (21.5, 3, 16), 2.)],
    # A forceps pull against an intact pial attachment: ATTACHMENT_PREVENTS_DISPLACEMENT.
    'lesion_pull': [*tool('right', 'tumor_forceps', (11.5, 17, 4)),
                    {'slot': 'right', 'operation': 'grasp', 'duration': 1., 'parameters': {'jawOpeningMm': 0.},
                     'path': [pose((11.5, 17, 11))]}, move('right', (11.5, 17, 6))],
}

# Captured from the pre-injury engine (HEAD d1e5f678, reference backend): the
# final (status, reasonCode) and state id of each guarded scenario.
GUARDED = {
    'cusa_vessel': ('partial', 'UNSUPPORTED_MATERIAL'),
    'dissector_push': ('partial', 'RETAINED_TISSUE_COLLISION'),
    'cusa_lever': ('partial', 'SHAFT_COLLISION'),
    'drill_plunge': ('partial', 'UNSUPPORTED_MATERIAL'),
    'lesion_pull': ('rejected', 'ATTACHMENT_PREVENTS_DISPLACEMENT'),
}
GUARDED_STATE_IDS = {
    'cusa_vessel': 'sha256:5d62720a648f375eee5e4a11c0cf591b6838d798d2792b28ac7f048b7699ad99',
    'dissector_push': 'sha256:470e49f666ce840e3220edb46a5431b8da9b537fbeae299b7a3ff5d3adb05962',
    'cusa_lever': 'sha256:c260cf3d9ef4f08056d05c9a0506508d911f76d3792648cbd1b5185ce31c7e86',
    'drill_plunge': 'sha256:1c9e33c98cf084aa056a2fe82dbd26ea96c2db0b0da1b604cf1bdd276c1d2b88',
    'lesion_pull': 'sha256:269ce0dccfd2cc239b850dcde56f1cbf698e7d1e17e11c228d152d68dbe49b16',
}


def run(engine, actions):
    results = [engine.step(action) for action in actions]
    return results


def injury_state(engine):
    return engine.state['injuryV1']


def test_no_property_keeps_refusals_state_ids_and_metrics():
    for name, expected in GUARDED.items():
        engine = SurgicalEngine(scene())
        results = run(engine, SCENARIOS[name])
        assert (results[-1]['status'], results[-1]['reasonCode']) == expected, name
        assert results[-1]['stateId'] == GUARDED_STATE_IDS[name], name
        assert 'injuryV1' not in engine.state
        assert not any(key.startswith('injury') for key in engine.metrics())


def test_guarded_mode_refuses_like_no_property():
    for name, expected in GUARDED.items():
        engine = SurgicalEngine(scene('guarded'))
        results = run(engine, SCENARIOS[name])
        assert (results[-1]['status'], results[-1]['reasonCode']) == expected, name
        assert injury_state(engine)['events'] == [] and engine.metrics()['injuryEvents'] == 0


def test_invalid_profile_is_refused():
    with pytest.raises(ValueError, match='SURGICAL_INJURY_PROFILE_INVALID'):
        SurgicalEngine(scene('injury', lacerationTravelMm=0.))
    with pytest.raises(ValueError, match='SURGICAL_INJURY_PROFILE_INVALID'):
        SurgicalEngine(scene('reckless'))


def test_shaft_into_brain_contuses_then_lacerates():
    engine = SurgicalEngine(scene('injury'))
    results = run(engine, SCENARIOS['cusa_lever'])
    assert results[-1]['status'] == 'accepted'
    kinds = [e['kind'] for e in results[-1]['events']]
    assert 'contusion' in kinds and 'laceration' in kinds
    assert kinds.index('contusion') < kinds.index('laceration')
    record = injury_state(engine)
    assert all(e['code'] == 'SHAFT_COLLISION' for e in record['events'] if e['mechanism'] == 'blunt')
    assert record['contused'] and record['lacerated']
    assert set(record['lacerated']) <= set(engine.state['removed'])
    assert not set(record['contused']) & set(engine.state['removed'])
    metrics = engine.metrics()
    assert metrics['injuryContusedMm3'] == len(record['contused']) and metrics['injuryLaceratedMm3'] == len(record['lacerated'])
    assert 'cortex' in metrics['injuryByRegionMm3']
    json.dumps(record)


def test_short_blunt_push_contuses_without_laceration():
    engine = SurgicalEngine(scene('injury', lacerationTravelMm=50.))
    results = run(engine, SCENARIOS['dissector_push'])
    assert results[-1]['status'] == 'accepted'
    record = injury_state(engine)
    assert record['contused'] and not record['lacerated']
    assert not set(record['contused']) & set(engine.state['removed'])


def test_cusa_on_vessel_records_vessel_injury_and_removes_cell():
    engine = SurgicalEngine(scene('injury'))
    results = run(engine, SCENARIOS['cusa_vessel'])
    assert results[-1]['status'] == 'accepted'
    record = injury_state(engine)
    vessel = [e for e in record['events'] if e['kind'] == 'vessel_injury']
    assert vessel and vessel[0]['instrument'] == 'cusa' and vessel[0]['regions'] == ['artery']
    ids = {c for e in vessel for c in e['cellIds']}
    assert ids == set(record['vesselInjured']) and ids <= set(engine.state['removed'])
    assert all(engine.scene.labels.flat[c] == 3 for c in ids)
    assert engine.metrics()['injuryVesselCells'] == len(ids)


def test_drill_plunge_lacerates_dura_and_cortex():
    engine = SurgicalEngine(scene('injury'))
    results = run(engine, SCENARIOS['drill_plunge'])
    assert results[-1]['status'] == 'accepted'
    record = injury_state(engine)
    plunges = [e for e in record['events'] if e['kind'] == 'plunge']
    assert plunges and {r for e in plunges for r in e['regions']} == {'dura', 'cortex'}
    assert set(record['lacerated']) <= set(engine.state['removed'])
    assert engine.metrics()['injuryByRegionMm3']['dura']['lacerated'] > 0


def test_rigid_bone_still_refuses_and_failed_microstep_rolls_back_injury():
    engine = SurgicalEngine(scene('injury'))
    results = run(engine, SCENARIOS['bone_sweep'])
    assert results[-1]['status'] == 'partial' and results[-1]['reasonCode'] in ('UNSUPPORTED_MATERIAL', 'RETAINED_TISSUE_COLLISION', 'SHAFT_COLLISION')
    record = injury_state(engine)
    bone = {int(c) for c in np.flatnonzero(engine.scene.labels.ravel() == 4)}
    touched = set(record['contused']) | set(record['lacerated']) | set(map(int, record['contusionDoseNs']))
    assert not bone & (touched | set(engine.state['removed']))
    assert record['contused'] or record['lacerated']
    # Injury recorded inside the refused microstep was rolled back with it:
    # the ledger holds exactly the injury events the committed substeps returned.
    returned = [e for r in results for e in r['events'] if e.get('kind') in
                ('contusion', 'laceration', 'vessel_injury', 'plunge', 'avulsion')]
    assert returned == record['events']


def test_microstep_checkpoint_detaches_injury_state():
    engine = SurgicalEngine(scene('injury'))
    run(engine, SCENARIOS['dissector_push'])
    before = json.dumps(injury_state(engine), sort_keys=True)
    copied = _microstep_checkpoint(engine.state)
    copied['injuryV1']['contused'].append(-1)
    copied['injuryV1']['contusionDoseNs']['x'] = 1.
    copied['injuryV1']['events'].clear()
    assert json.dumps(injury_state(engine), sort_keys=True) == before


def test_forceps_pull_avulses_intact_attachment():
    guarded = SurgicalEngine(scene())
    refused = run(guarded, SCENARIOS['lesion_pull'])[-1]
    assert refused['status'] in ('partial', 'rejected') and refused['reasonCode'] == 'ATTACHMENT_PREVENTS_DISPLACEMENT'
    engine = SurgicalEngine(scene('injury'))
    results = run(engine, SCENARIOS['lesion_pull'])
    assert results[-1]['reasonCode'] != 'ATTACHMENT_PREVENTS_DISPLACEMENT'
    record = injury_state(engine)
    assert record['avulsions'] == ['lesion-pia'] and 'lesion-pia' in engine.state['separatedAttachments']
    assert [e['attachments'] for e in record['events'] if e['kind'] == 'avulsion'] == [['lesion-pia']]
    assert engine.metrics()['injuryAvulsions'] == 1


def test_protected_structure_contact_is_recorded_by_name():
    def guarded_engine(mode):
        s = scene(mode)
        mask = s.labels == 3
        protection = ProtectionSet(s, [ProtectionAnnotation.from_mask(s, 'artery-a', mask,
            provenance='synthetic test membership')], clearance_mm=0., required=())
        return SurgicalEngine(s, protection=protection, guard_config=GuardConfig(require_critical_coverage=False))
    refused = run(guarded_engine(None), SCENARIOS['cusa_vessel'])[-1]
    assert refused['status'] == 'rejected' and refused['reasonCode'] == 'PROTECTED_STRUCTURE'
    engine = guarded_engine('injury')
    results = run(engine, SCENARIOS['cusa_vessel'])
    assert results[-1]['status'] == 'accepted'
    named = [e for e in injury_state(engine)['events'] if 'artery-a' in e['structures']]
    assert {e['kind'] for e in named} >= {'vessel_injury', 'protected_structure_injury'}
    assert engine.metrics()['injuryProtectedStructures'] == ['artery-a']


def test_replay_and_snapshot_round_trip_are_deterministic():
    actions = [*SCENARIOS['cusa_lever'], off('right')]
    first = SurgicalEngine(scene('injury')); run(first, actions)
    second = SurgicalEngine(scene('injury')); run(second, actions)
    assert first.state_id == second.state_id and injury_state(first) == injury_state(second)
    replayed = first.replay(first.actions)
    assert replayed.state_id == first.state_id
    restored = SurgicalEngine.restore(first.scene, first.snapshot())
    assert restored.state_id == first.state_id and injury_state(restored) == injury_state(first)



def test_outcome_and_consequence_ledgers_add_injury_only_with_property():
    from resectionlab.surgical_consequences import INJURY_METRICS, _vector_metrics, consequence_vector
    from resectionlab.surgical_modeled_outcomes import summarize_engine_outcomes
    plain = SurgicalEngine(scene()); run(plain, SCENARIOS['drill_plunge'])
    assert 'injury' not in summarize_engine_outcomes(plain)
    vector = consequence_vector(plain, plain.fresh())
    assert 'injuryMetricIds' not in vector and not any(k.startswith('injury:') for k in vector['dimensions'])
    engine = SurgicalEngine(scene('injury')); run(engine, SCENARIOS['drill_plunge'])
    summary = summarize_engine_outcomes(engine)['injury']
    assert summary['injuryLaceratedMm3']['delta']['value'] == engine.metrics()['injuryLaceratedMm3'] > 0
    vector = consequence_vector(engine, engine.fresh())
    assert set(_vector_metrics(vector)) == set(vector['dimensions']) and set(INJURY_METRICS) <= set(vector['dimensions'])
    assert vector['dimensions']['injury:laceratedMm3']['incremental'] == engine.metrics()['injuryLaceratedMm3']


def test_lacerated_tumour_is_removed_but_earns_no_treatment_credit():
    from resectionlab.surgical_consequences import consequence_vector
    # A passive dissector forced down through the lesion tears it; it does not treat it.
    actions = [*tool('left', 'dissector', (11.5, 16.5, 4)), move('left', (11.5, 16.5, 12.5), 3.), move('left', (11.5, 16.5, 4))]
    engine = SurgicalEngine(scene('injury')); run(engine, actions)
    labels = engine.scene.labels.ravel()
    lacerated = set(injury_state(engine)['lacerated'])
    torn_tumour = sum(labels[c] == 5 for c in lacerated)
    assert torn_tumour > 0 and lacerated <= set(engine.state['removed'])
    metrics = engine.metrics()
    assert metrics['targetRemovedMm3'] == 0. and metrics['injuryLaceratedTargetMm3'] == torn_tumour
    # The cavity keeps the torn cells; the ledgers add up with an explicit injury share.
    assert metrics['removedByLabelMm3']['lesion'] == torn_tumour == metrics['injuryLaceratedByLabelMm3']['lesion']
    total = len(engine.state['removed']) * engine.scene.voxel_mm3
    assert metrics['targetRemovedMm3'] + metrics['injuryLaceratedTargetMm3'] + metrics['nonTargetRemovedMm3'] == total
    assert sum(metrics['removedByLabelMm3'].values()) == total
    assert sum(metrics['injuryLaceratedByLabelMm3'].values()) == metrics['injuryLaceratedMm3']
    vector = consequence_vector(engine, engine.fresh())
    assert vector['dimensions']['targetRemovedMm3']['cumulative'] == 0.
    # Guarded mode keeps the plain ledger and adds no injury share.
    guarded = SurgicalEngine(scene('guarded')); run(guarded, actions)
    assert guarded.metrics()['injuryLaceratedTargetMm3'] == 0. and guarded.metrics()['injuryLaceratedByLabelMm3'] == {}


def test_traction_tear_helper_raises_in_guarded_mode_and_records_in_injury_mode():
    from resectionlab.surgical_injury import tear
    with pytest.raises(ValueError, match='OPERATIVE_REFLECTION_TRACTION_LIMIT'):
        tear(SurgicalEngine(scene('guarded')), 'OPERATIVE_REFLECTION_TRACTION_LIMIT', 'left', [0], 'dura-flap')
    engine = SurgicalEngine(scene('injury'))
    cell = int(np.flatnonzero(engine.scene.labels.ravel() == 1)[0])
    tear(engine, 'OPERATIVE_REFLECTION_TRACTION_LIMIT', 'left', [cell], 'dura-flap')
    assert injury_state(engine)['avulsions'] == ['dura-flap'] and injury_state(engine)['contused'] == [cell]
    assert engine.metrics()['injuryAvulsions'] == 1
