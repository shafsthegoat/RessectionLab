"""Source, observation and authoritative-transition checks for anatomy A/B IPC."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from resectionlab.core import CaseData, thaw_json
from resectionlab.desktop_bridge import BridgeSession, _Request
from resectionlab.imaging import load_case, read_case_artifacts
from resectionlab.patient_anatomy import PatientImagingCollection, PatientAnatomySnapshot
from resectionlab.surgical_patient import prepare_surgical_environment
from resectionlab.surgical_anatomy_comparison import compare_anatomy

pytest.importorskip('gymnasium')


def make_pair(tmp_path):
    shape = (24, 24, 24)
    image = nib.Nifti1Image(np.ones(shape, np.float32), np.eye(4))
    image.header.set_xyzt_units('mm')
    nib.save(image, tmp_path / 'scan.nii.gz')
    collection = PatientImagingCollection.create(tmp_path / 'native', 'paired-patient')
    collection.add_series(tmp_path / 'scan.nii.gz', series_id='mri', modality='T1')
    brain = np.zeros(shape, np.uint8); brain[6:18, 6:18, 5:18] = 1
    tumor = np.zeros(shape, np.uint8); tumor[10:14, 10:14, 10:14] = 1
    vessels = np.zeros(shape, np.uint8); vessels[10, 8, 2:14] = 1
    for name, array, kind in [('brain', brain, 'brain'), ('tumor', tumor, 'tumor'), ('vessels', vessels, 'vessels')]:
        collection.add_layer(array, np.eye(4), layer_id=name, source_series='mri', kind=kind, role='target' if name == 'tumor' else 'context', color='#aaaaaa')
    collection.save()
    layers = {k: k for k in ('brain', 'tumor', 'vessels')}
    before = prepare_surgical_environment(collection.root, tmp_path / 'before', layer_ids=layers)
    mask = collection.mask(collection.layer('vessels')).copy(); mask[10, 9, 2:14] = True
    collection.commit_edit('vessels', mask, author='test reviewer', expected=collection.identity(), operations=[{'operation': 'explicit fixture change'}])
    after = prepare_surgical_environment(collection.root, tmp_path / 'after', layer_ids=layers)
    def entry(report):
        return SimpleNamespace(case=load_case(report['casePath']), artifacts=read_case_artifacts(report['casePath']))
    return collection, entry(before), entry(after)


def call(session, operation, args):
    return session.execute(operation, args, _Request(operation), lambda *_: None)


def test_comparison_uses_observation_only_policy_and_shared_geometry(tmp_path, monkeypatch):
    collection, before, after = make_pair(tmp_path)
    import resectionlab.surgical_anatomy_comparison as module
    original = module.candidates
    seen = []
    def boundary(observation):
        assert set(observation) == {'local', 'slots', 'tool_parameters', 'metrics', 'goal', 'coverage'}
        seen.append(observation['local'].copy())
        return original(observation)
    monkeypatch.setattr(module, 'candidates', boundary)
    # Fixed exterior prefix brings the same observed workspace next to the
    # changed vascular margin; evaluator geometry never chooses policy actions.
    prefix = [{'slot': 'left', 'operation': 'exchange', 'instrument': 'cusa', 'duration': .1},
              {'slot': 'left', 'operation': 'place', 'duration': .1, 'parameters': {'radiusMm': .6},
               'pose': {'position': [10., 10., -2.], 'quaternion': [0., 0., 0., 1.]}}]
    report = compare_anatomy(before, after, prefix=prefix)
    assert len(seen) == 2
    assert report['controls']['training'] is False
    assert report['maskChanges']['vessels']['addedCells'] == 12
    assert report['maskChanges']['brain']['changedCells'] == 0
    assert report['delta']['matchedPrefixLocalChannels']['material'] > 0
    assert report['before']['targetVolumeMm3'] == report['after']['targetVolumeMm3']
    assert report['after']['sourceLayers']['vessels']['reviewStatus'] != 'reviewed'
    assert len(report['before']['policy'][0]['candidateOutcomes']) == 9
    assert report['controls']['prefix'] == prefix
    json.dumps(report, allow_nan=False)


def test_ipc_comparison_does_not_mutate_saved_desktop_timeline(tmp_path):
    _, before, after = make_pair(tmp_path)
    session = BridgeSession(tmp_path / 'transfers')
    try:
        hashes = [session._install_case(entry.case, entry.artifacts, _Request('loadCase'))['caseHash'] for entry in (before, after)]
        original = [thaw_json(session._get_case(key).artifacts) for key in hashes]
        report = call(session, 'compareSurgicalAnatomy', {'beforeCaseHash': hashes[0], 'afterCaseHash': hashes[1]})
        assert report['before']['caseHash'] == hashes[0]
        assert report['after']['caseHash'] == hashes[1]
        assert [thaw_json(session._get_case(key).artifacts) for key in hashes] == original
    finally:
        session.close()


def test_other_acquisition_and_undeclared_mask_changes_are_rejected(tmp_path):
    _, before, after = make_pair(tmp_path)
    metadata = thaw_json(after.case.metadata)
    metadata['surgical_adapter']['sourceManifest']['series'][0]['original']['sha256'] = '0' * 64
    altered = CaseData(case_id=after.case.case_id, mri=after.case.mri, compartments=after.case.compartments,
                       affine=after.case.affine, source_refs=after.case.source_refs, metadata=metadata)
    with pytest.raises(ValueError, match='Acquired source hashes'):
        compare_anatomy(before, SimpleNamespace(case=altered, artifacts=after.artifacts))
    with pytest.raises(ValueError, match='Undeclared anatomy changed: vessels'):
        compare_anatomy(before, after, changed_layers=['tumor'])


def test_extra_scene_assumptions_cannot_change_under_a_fixed_profile(tmp_path):
    _, before, after = make_pair(tmp_path)
    metadata = thaw_json(after.case.metadata)
    metadata['surgical_properties']['boneAnchors'] = [0]
    altered = CaseData(case_id=after.case.case_id, mri=after.case.mri, compartments=after.case.compartments,
                       affine=after.case.affine, source_refs=after.case.source_refs, metadata=metadata)
    with pytest.raises(ValueError, match='scene assumptions differ'):
        compare_anatomy(before, SimpleNamespace(case=altered, artifacts=after.artifacts))


def test_segmentation_revision_rejects_stale_planning_and_source_rehearsal(tmp_path):
    collection, _, _ = make_pair(tmp_path)
    session = BridgeSession(tmp_path / 'transfers')
    try:
        opened = call(session, 'anatomyOpen', {'path': str(collection.root / 'patient-anatomy.json')})
        source = {'collectionId': opened['collectionId'], 'snapshotId': opened['document']['snapshot_id']}
        plan_args = {**source, 'start': [10., 10., -2.], 'end': [10., 10., 12.], 'radiusMm': .6}
        plan = call(session, 'anatomyPlan', plan_args)
        old_identity = collection.identity()
        mask = collection.mask(collection.layer('vessels')).copy(); mask[11, 9, 8] = True
        collection.commit_edit('vessels', mask, author='revision test', expected=old_identity, operations=[])
        with pytest.raises(ValueError, match='changed|stale'):
            call(session, 'anatomyPlan', plan_args)
        from resectionlab.anatomy_planning import planning_case
        revised = planning_case(PatientAnatomySnapshot.capture(collection))
        assert revised.metadata['patient_anatomy_snapshot'] != old_identity
        with pytest.raises(ValueError, match='changed'):
            call(session, 'prepareSurgicalEnvironment', {**source, 'layers': {'brain':'brain','tumor':'tumor','vessels':'vessels'}})
    finally:
        session.close()
