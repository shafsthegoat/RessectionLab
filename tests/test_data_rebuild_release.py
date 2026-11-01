"""A republish must preserve production receipts and recheck canonical bytes."""
import json
from dataclasses import replace
from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import data_rebuild_release as release
from resectionlab.data_root import sha256_file
from resectionlab.imaging import load_nifti_case
from resectionlab.workspace_bundle import save_workspace


def test_republication_reuses_only_unchanged_production_reopen(tmp_path,monkeypatch):
    for part in ['qc/independent','catalog','cases/real','cases/controls','cases/synthetic','staging']:
        (tmp_path/part).mkdir(parents=True)
    image=nib.Nifti1Image(np.arange(60,dtype=np.float32).reshape(3,4,5),np.eye(4));image.header.set_xyzt_units('mm')
    nifti=tmp_path/'staging/source.nii.gz';nib.save(image,nifti)
    case=load_nifti_case(nifti)
    case=replace(case,metadata={**dict(case.metadata),'data_case':{'schema':'ressectionlab-data-case-v1',
        'person_id':'TEST:001','role':'TRAIN','classification':'real_control'}})
    package=tmp_path/'staging/one.ressectionlab'
    save_workspace(case,package,sources=[],imaging=None,episode=None,selection=None,artifacts={})
    decision={'schema':'ressectionlab-independent-release-decisions-v1','reviewer':'offline regression fixture',
        'cases':[{'case_id':'test-one','path':str(package),'sha256':sha256_file(package),'family_id':'TEST:001',
            'role':'TRAIN','decision':'accepted_with_explicit_limitations','capabilities':['source_inspection'],'limitations':['test fixture']}]}
    (tmp_path/'qc/independent/release-decisions-v1.json').write_text(json.dumps(decision))
    monkeypatch.setattr(release,'checked_root',lambda *_a,**_kw:tmp_path)
    first=release.publish(tmp_path);original=tmp_path/first['cases'][0]['review_path'];before=original.read_bytes()
    monkeypatch.setattr(release,'load_workspace',lambda *_:pytest.fail('Identical published bytes already reopened through production'))
    second=release.publish(tmp_path)
    review=json.loads((tmp_path/second['cases'][0]['review_path']).read_text())
    assert review['reused_production_reopen_receipt']['path']==str(original.relative_to(tmp_path))
    assert original.read_bytes()==before
    package.unlink() # Canonical self-contained bytes are independently usable.
    release.publish(tmp_path) # Repeated snapshots cannot introduce a self-reference.
    (tmp_path/second['cases'][0]['path']).write_bytes(b'changed canonical package')
    with pytest.raises(ValueError,match='different existing release'):
        release.publish(tmp_path)


def test_append_preserves_frozen_authority_and_rejects_duplicate_or_changed_receipt(tmp_path,monkeypatch):
    for part in ['qc/independent','catalog','cases/real','cases/controls','cases/synthetic','staging']:
        (tmp_path/part).mkdir(parents=True)
    monkeypatch.setattr(release,'checked_root',lambda *_a,**_kw:tmp_path)
    def candidate(number):
        nifti=tmp_path/f'staging/source-{number}.nii.gz'
        image=nib.Nifti1Image(np.arange(60,dtype=np.float32).reshape(3,4,5)+number,np.eye(4))
        image.header.set_xyzt_units('mm');nib.save(image,nifti)
        case=load_nifti_case(nifti)
        case=replace(case,metadata={**dict(case.metadata),'data_case':{'schema':'ressectionlab-data-case-v1',
            'person_id':f'TEST:{number}','role':'TRAIN','classification':'synthetic_scenario_on_acquired_host'}})
        package=tmp_path/f'staging/{number}.ressectionlab'
        save_workspace(case,package,sources=[],imaging=None,episode=None,selection=None,artifacts={})
        return {'case_id':f'test-{number}','path':str(package.relative_to(tmp_path)),'sha256':sha256_file(package),
            'family_id':f'TEST:{number}','role':'TRAIN','decision':'accepted_with_explicit_limitations',
            'capabilities':['bounded_negative_test'],'limitations':['fixture; no target reach']}
    old=candidate(1);new=candidate(2)
    authority={'schema':'ressectionlab-independent-release-decisions-v1','reviewer':'offline independent fixture','cases':[old]}
    old_decision=tmp_path/'qc/independent/release-decisions-v1.json';old_decision.write_text(json.dumps(authority))
    base=release.publish(tmp_path,freeze='base');base_file=tmp_path/'catalog/releases/base.json'
    old_bytes=base_file.read_bytes();old_authority=old_decision.read_bytes()
    # Same physical case with a different task declaration adds a package, not
    # another person or synthetic target variant.
    host,_,_=release.load_workspace(tmp_path/old['path'])
    metadata=dict(host.metadata);metadata['data_case']={**dict(host.metadata['data_case']),'synthetic_variant_count':0}
    copied=tmp_path/'staging/task-copy.ressectionlab'
    save_workspace(replace(host,case_id='task-copy',metadata=metadata),copied,sources=[],imaging=None,episode=None,selection=None,artifacts={})
    copy_decision={**old,'case_id':'test-task-copy','path':str(copied.relative_to(tmp_path)),'sha256':sha256_file(copied)}
    new_decision=tmp_path/'qc/independent/additions.json';authority['cases']=[new,copy_decision];new_decision.write_text(json.dumps(authority))
    added=release.publish(tmp_path,freeze='added',decision_relative='qc/independent/additions.json',inherit_relative='catalog/releases/base.json')
    assert added['case_count']==3 and added['person_family_count']==2 and added['synthetic_variant_count']==2
    assert added['cases'][2]['synthetic_variant_count']==0
    assert added['cases'][0]==base['cases'][0] and base_file.read_bytes()==old_bytes and old_decision.read_bytes()==old_authority
    assert added['inherited_release']['sha256']==sha256_file(base_file)
    authority['cases']=[old];new_decision.write_text(json.dumps(authority))
    with pytest.raises(ValueError,match='duplicate stable case'):
        release.publish(tmp_path,decision_relative='qc/independent/additions.json',inherit_relative='catalog/releases/base.json')
    authority['cases']=[];new_decision.write_text(json.dumps(authority))
    (tmp_path/base['cases'][0]['review_path']).write_text('{}')
    with pytest.raises(ValueError,match='(?:acceptance|receipt) changed'):
        release.publish(tmp_path,decision_relative='qc/independent/additions.json',inherit_relative='catalog/releases/base.json')
