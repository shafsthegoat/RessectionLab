"""Source-substitution controls on generated data; no patient assets needed."""
from pathlib import Path
import sys
import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from prepare_btc_recovery_workspace import verify_source_arrays
from resectionlab.core import CaseData, SourceRef


@pytest.fixture
def source(tmp_path):
    image = np.arange(512,dtype=np.float32).reshape(8,8,8)
    annotation = np.zeros((8,8,8),np.float32);annotation[2:4,2:4,2:4]=.75
    affine=np.eye(4);affine[:3,3]=[12,13,14]
    a,b=tmp_path/'image.nii.gz',tmp_path/'annotation.nii.gz'
    for data,destination in ((image,a),(annotation,b)):
        native=nib.Nifti1Image(data,affine);native.header.set_xyzt_units('mm');nib.save(native,destination)
    case=CaseData(case_id='generated-source',mri=image,affine=affine,
        compartments={'target':annotation>=.5},source_refs=(SourceRef('generated','synthetic://test',provenance='simulated'),),
        metadata={'receipt':'same-metadata-does-not-attest-data'})
    return case,a,b


def test_original_arrays_match(source):
    verify_source_arrays(*source)


def test_annotation_axis_reversal_is_checked_in_physical_coordinates(source):
    case,a,b=source
    original=nib.load(b)
    flip=np.eye(4);flip[0,0]=-1;flip[0,3]=case.mri.shape[0]-1
    nib.save(nib.Nifti1Image(np.asarray(original.dataobj)[::-1],original.affine@flip,original.header),b)
    verify_source_arrays(case,a,b)


@pytest.mark.parametrize('field',['image','affine','target'])
def test_changed_arrays_with_unchanged_receipt_are_rejected(source,field):
    case,a,b=source
    if field=='image':
        image=case.mri.copy();image[0,0,0]+=1;case=case.revised(mri=image)
    elif field=='affine':
        affine=case.affine.copy();affine[0,3]+=.001;case=case.revised(affine=affine)
    else:
        target=case.compartments['target'].copy();target[0,0,0]=True;case=case.revised(compartments={'target':target})
    with pytest.raises(ValueError,match='differs'):
        verify_source_arrays(case,a,b)
