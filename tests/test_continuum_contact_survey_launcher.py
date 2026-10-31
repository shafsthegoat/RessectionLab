"""Support reconstruction and endpoint joins for the export-only launcher."""
import importlib.util
from pathlib import Path
import copy
import sys
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('survey_continuum_contacts',Path(__file__).parents[1]/'scripts/survey_continuum_contacts.py')
module=importlib.util.module_from_spec(spec);scripts=str(Path(__file__).parents[1]/'scripts')
sys.path.insert(0,scripts)
try:spec.loader.exec_module(module)
finally:sys.path.remove(scripts)


def test_full_corner_reconstruction_keeps_constrained_coordinates_zero():
    g={'cells':np.array([10]),'nodes':np.zeros((8,3)),'elementNodes':np.arange(8)[None,:],
        'freeDofs':np.tile([True,False,True],8)}
    ids=np.flatnonzero(g['freeDofs']);d={'freeIndices':ids,'displacement':np.arange(len(ids),dtype=float)}
    field=module.corner_fields(g,d)
    expected=np.zeros(24);expected[ids]=d['displacement']
    assert np.array_equal(field,expected.reshape((1,8,3)))
    d['freeIndices']=ids[::-1]
    with pytest.raises(ValueError,match='FIELD_SUPPORT'):module.corner_fields(g,d)


def test_native_alignment_uses_source_owners_and_refuses_outside_or_duplicate_owner():
    native={'sourceCells':np.array([8,2]),'cornerDisplacementMm':np.stack([np.ones((8,3)),np.full((8,3),2.)])}
    result=module.aligned_native_field(np.array([2,5,8]),native)
    assert np.array_equal(result[:,0,0],[2.,0.,1.])
    for ids in ([8,1],[8,8]):
        native['sourceCells']=np.array(ids)
        with pytest.raises(ValueError,match='NATIVE_SUPPORT'):module.aligned_native_field(np.array([2,5,8]),native)


def test_comparison_refuses_pose_mismatch_and_preserves_ambiguity_differences():
    cap={'slot':'left','part':'shaft','envelopeIndex':0,'startMm':[0.,0.,0.],
        'endMm':[0.,0.,1.],'radiusMm':.5,'rasterSourceIds':[],
        'witnessedSourceIds':[3],'indeterminateSourceIds':[4],'forbiddenWitnessedSourceIds':[3],
        'forbiddenIndeterminateSourceIds':[4],'rasterForbiddenSourceIds':[]}
    a={'capsules':[cap],'instantaneousPostFieldCusaCandidateOrders':{'left':[3,4]}};b=copy.deepcopy(a)
    b['capsules'][0]['indeterminateSourceIds']=[5];b['instantaneousPostFieldCusaCandidateOrders']['left']=[4,3]
    diff=module.compare_worlds(a,b)
    assert diff['capsules'][0]['indeterminateSourceIdsAdded']==[5] and diff['capsules'][0]['indeterminateSourceIdsLost']==[4]
    assert diff['postFieldCusaCandidateOrderChanged']
    b['capsules'][0]['startMm']=[0.,0.,.01]
    with pytest.raises(ValueError,match='POSE_JOIN'):module.compare_worlds(a,b)
