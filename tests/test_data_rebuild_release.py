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
