"""Real native collection/import path, including existing review refusals."""
import json

import numpy as np
import pytest

from resectionlab.imaging import load_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_patient import prepare_surgical_patient, ORDER
from resectionlab.surgical_rl_protection import annotation_from_collection
from test_surgical_patient import fixture


STRESS = {'assumed_translation_mm': .4, 'assumed_rotation_degrees': 0.,
          'rotation_pivot_target_ras_mm': None}


def prepared(tmp_path, *, reviewed=True, registration='reviewed'):
    c = fixture(tmp_path, status=registration)
    slab = np.zeros((12, 12, 12), np.uint8); slab[:, :, :4] = 1
    row = c.add_layer(slab, c.series('mri')['affine'], layer_id='container',
        source_series='mri', kind='critical', role='critical', color='#aa2222',
        label='Conservative inferior compartment, not anatomical segmentation',
        provenance={'kind': 'generated-control', 'clinicalValidation': False})
    if reviewed: row['review_status'] = 'reviewed'
    c.save()
    report = prepare_surgical_patient(c.root, tmp_path/'derived',
        layer_ids={k: k for k in ORDER}, allow_unreviewed_registration=True)
    case = load_case(report['casePath'])
    return c, case, from_case(case)


def test_actual_collection_slab_import_avoids_source_cloud_and_preserves_source(tmp_path, monkeypatch):
    import resectionlab.surgical_rl_protection as module
    c, case, scene = prepared(tmp_path)
    original = c.identity()
    with monkeypatch.context() as patch:
        patch.setattr(module, 'cKDTree', lambda *_a, **_kw: pytest.fail('Source point cloud allocated'))
        annotation = annotation_from_collection(scene, c.root, 'container', 'brainstem', slab_transfer=STRESS)
    assert annotation.cells and annotation.source_binding
    assert annotation.review_status == 'reviewed'
    assert 'not anatomical segmentation' in annotation.display_name
    facts = json.loads(annotation.provenance)
    assert facts['transfer']['registrationOrAnatomyQualified'] is False
    assert facts['transfer']['assumedStress']['translationL2Mm'] == .4
    assert c.identity() == original
    from resectionlab.surgical_rl_protection import ProtectionSet, GuardConfig
    from resectionlab.surgical_engine import SurgicalEngine
    engine = SurgicalEngine(scene, protection=ProtectionSet(scene, [annotation], strict=True),
        guard_config=GuardConfig())
    point = scene.points([annotation.cells[len(annotation.cells)//2]])[0].tolist()
    placed = engine.step({'slot': 'left', 'operation': 'place', 'duration': .1,
        'pose': {'position': [30., 30., 30.], 'quaternion': [0., 0., 0., 1.]}})
    assert placed['status'] == 'accepted'
    before = engine.snapshot()
    result = engine.step({'slot': 'left', 'operation': 'apply', 'duration': .1,
                         'pose': {'position': point, 'quaternion': [0., 0., 0., 1.]}})
    assert result['status'] == 'rejected' and 'PROTECTED' in result['reasonCode']
    assert engine.snapshot() == before


def test_unreviewed_slab_still_refuses_and_explicit_opt_in_does_not_promote(tmp_path):
    c, _, scene = prepared(tmp_path, reviewed=False)
    with pytest.raises(ValueError, match='Unreviewed anatomy'):
        annotation_from_collection(scene, c.root, 'container', 'brainstem', slab_transfer=STRESS)
    annotation = annotation_from_collection(scene, c.root, 'container', 'brainstem',
        slab_transfer=STRESS, allow_unreviewed=True)
    assert annotation.review_status == 'unreviewed'
    assert c.layer('container')['review_status'] == 'unreviewed'


def test_native_allocation_cap_precedes_annotation_tuple_allocation(tmp_path, monkeypatch):
    from resectionlab.surgical_rl_protection import ProtectionAnnotation
    c, _, scene = prepared(tmp_path)
    monkeypatch.setattr(ProtectionAnnotation, 'from_mask', lambda *_a, **_kw: pytest.fail('Tuple allocation before cap'))
    with pytest.raises(ValueError, match='before point/tuple'):
        annotation_from_collection(scene, c.root, 'container', 'brainstem',
            slab_transfer=STRESS, max_native_protection_cells=1)


def test_changed_collection_snapshot_cannot_rebind_existing_scene(tmp_path):
    c, _, scene = prepared(tmp_path)
    c.layer('container')['coverage'] = 'changed evidence'; c.save()
    with pytest.raises(ValueError, match='snapshot'):
        annotation_from_collection(scene, c.root, 'container', 'brainstem', slab_transfer=STRESS)
