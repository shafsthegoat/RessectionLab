from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from resectionlab.core import SourceRef, thaw_json, semantic_digest
from resectionlab.imaging import save_case, load_case
from resectionlab.surgical_recorded_sequential import display_case, PROPERTY
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_sequential_task import generated_task


@pytest.fixture
def source():
    task = generated_task(200)
    return task, display_case(task)


def test_exact_native_source_identity_survives_case_save(source, tmp_path):
    task, case = source
    path = tmp_path/'source.ressectionlab'
    save_case(case, path)
    restored = from_case(load_case(path), task.scene.regions)
    assert restored.identity == task.scene.identity
    assert restored.source == task.scene.source
    assert restored.properties == task.scene.properties
    assert restored.regions == task.scene.regions
    np.testing.assert_array_equal(restored.labels, task.scene.labels)


@pytest.mark.parametrize('location,key,value', [
    ('adapter','patientId','patient'), ('adapter','patient_id','patient'),
    ('adapter','source_patient','patient'), ('adapter','sourceManifest','patient.json'),
    ('adapter','type','nifti-custom'), ('adapter','type','mesh-custom'),
    ('metadata','patientId','patient'), ('metadata','source_patient','patient'),
    ('declaration-source','patientId','patient'), ('declaration-source','source_patient','patient'),
])
def test_contradictory_markers_refused_before_generator(source, location, key, value):
    task, case = source; metadata = thaw_json(case.metadata)
    if location == 'adapter': metadata['surgical_adapter'][key] = value
    elif location == 'metadata': metadata[key] = value
    else:
        metadata[PROPERTY]['source'][key] = value
        raw = metadata[PROPERTY]
        raw['contentHash'] = semantic_digest({k:v for k,v in raw.items() if k != 'contentHash'})
    changed = replace(case, metadata=metadata)
    with patch('resectionlab.surgical_sequential_task.generated_task') as generator:
        with pytest.raises(ValueError, match='RECORDED_SEQUENTIAL'):
            from_case(changed, task.scene.regions)
        generator.assert_not_called()


def test_observed_reference_cannot_select_synthetic_adapter(source):
    task, case = source
    changed = replace(case, source_refs=(SourceRef('MRI', 'synthetic://sequential-task/200', provenance='observed'),))
    with patch('resectionlab.surgical_sequential_task.generated_task') as generator:
        with pytest.raises(ValueError, match='GENERATED_SOURCE_REQUIRED'):
            from_case(changed, task.scene.regions)
        generator.assert_not_called()


@pytest.mark.parametrize('edit', ['mri','mask','affine','regions','properties','native-source','extra-metadata','task-hash'])
def test_source_transport_tampering_cannot_rebind_native_episode(source, edit):
    task, case = source; metadata = thaw_json(case.metadata); kwargs = {}
    mapping = task.scene.regions
    if edit == 'mri':
        pixels = case.mri.copy(); pixels[0,0,0] = 1; kwargs['mri'] = pixels
    elif edit == 'mask':
        masks = {k:v.copy() for k,v in case.compartments.items()}
        masks['Generated target'][0,0,0] = True; kwargs['compartments'] = masks
    elif edit == 'affine':
        affine = case.affine.copy(); affine[0,3] += 1; kwargs['affine'] = affine
    elif edit == 'regions': mapping = [{**r,'name':r['name']} for r in mapping]; mapping[0]['color'] = '#ffffff'
    elif edit == 'properties': metadata['surgical_properties']['unexpected'] = True
    elif edit == 'native-source': metadata[PROPERTY]['source']['provenance'] = 'edited'
    elif edit == 'extra-metadata': metadata['looksGenerated'] = True
    elif edit == 'task-hash': metadata[PROPERTY]['generatorDefinitionSha256'] = '0'*64
    raw = metadata[PROPERTY]; raw['contentHash'] = semantic_digest({k:v for k,v in raw.items() if k != 'contentHash'})
    changed = replace(case, metadata=metadata, **kwargs)
    with pytest.raises(ValueError, match='RECORDED_SEQUENTIAL'):
        from_case(changed, mapping)


def test_unrelated_task_file_change_preserves_exact_generator_transport(source):
    task, case = source
    with patch('resectionlab.surgical_recorded_sequential.task_implementation_sha256', return_value='1'*64):
        restored = from_case(case, task.scene.regions)
    assert restored.identity == task.scene.identity


@pytest.mark.parametrize('parent', [-1, 1_000_000, True, 200.0, '200'])
def test_parent_bounds_refused_before_generator(source, parent):
    task, case = source; metadata=thaw_json(case.metadata); raw=metadata[PROPERTY]
    raw['parent']=parent;raw['contentHash']=semantic_digest({k:v for k,v in raw.items() if k!='contentHash'})
    changed=replace(case,metadata=metadata)
    with patch('resectionlab.surgical_sequential_task.generated_task') as generator:
        with pytest.raises(ValueError,match='SOURCE_BINDING_CHANGED'): from_case(changed,task.scene.regions)
        generator.assert_not_called()


def test_original_source_masks_cannot_be_silently_dropped(source):
    task,case=source;masks={k:v.copy() for k,v in case.source_compartments.items()}
    masks['Generated target'][0,0,0]=True
    with pytest.raises(ValueError,match='EXACT_SYNTHETIC_TRANSPORT_REQUIRED'):
        from_case(replace(case,source_compartments=masks),task.scene.regions)


@pytest.mark.parametrize('field', ['critical_evidence','functional_evidence','structural_evidence','prior_proposals'])
def test_extra_evidence_refused_before_generator(source, field):
    # Exercise the adapter guard directly; CaseData additionally validates the
    # typed records before this boundary can be reached by a saved case.
    from types import SimpleNamespace
    from resectionlab.surgical_recorded_sequential import restore_recorded_source
    task,case=source
    with patch('resectionlab.surgical_recorded_sequential.restore_recorded_source',side_effect=lambda c,s,m:s):
        ordinary=from_case(case,task.scene.regions)
    proxy=SimpleNamespace(**{name:getattr(case,name) for name in case.__dataclass_fields__ if not name.startswith('_')})
    setattr(proxy,field,{'unexpected':'evidence'})
    with patch('resectionlab.surgical_sequential_task.generated_task') as generator:
        with pytest.raises(ValueError,match='EXACT_SYNTHETIC_TRANSPORT_REQUIRED'):
            restore_recorded_source(proxy,ordinary,task.scene.regions)
        generator.assert_not_called()
