"""Exact pre-physical phase joins and honest counterfactual differences."""
import copy
import importlib.util
from pathlib import Path
import sys
import pytest

from resectionlab.surgical_pre_effect_capture import capture_pre_effects
from test_surgical_pre_effect_capture import prepared

scripts=str(Path(__file__).parents[1]/'scripts');sys.path.insert(0,scripts)
spec=importlib.util.spec_from_file_location('survey_continuum_pre_effects',Path(scripts)/'survey_continuum_pre_effects.py')
module=importlib.util.module_from_spec(spec)
try:spec.loader.exec_module(module)
finally:sys.path.remove(scripts)


def event_fixture():
    _,engine=prepared()
    with capture_pre_effects():
        for _ in range(5):engine.step({'slot':'right','operation':'apply','duration':.1})
    frame=engine.pre_effect_frames()[-1];native=engine.mechanics_inputs()
    frame.update(branch='short',coupledStep=1,tissueStateAfter=1,tissueStateBefore=0,previousArrayHash='previous',
        sourceOperatorIdentityAfter='operator',sampleFieldsFile='event.npz',sampleFieldsSha256='fieldhash')
    removed=sorted(set(frame['removedSourceIdsAfter'])-set(frame['removedSourceIds']))
    assert len(removed)==1
    row={'branch':'short','step':1,'tissueState':1,'withinTick':0,'suffixTick':frame['nativeTickIndex'],
        'event':{'removedSourceIds':removed},'previousArrayHash':'previous','sourceOperatorIdentity':'operator',
        **{key:frame[key] for key in ('actionIndex','slot','operation')}}
    sample={'file':'event.npz','sha256':'fieldhash'}
    return frame,row,sample,native


@pytest.mark.parametrize('key,value',[('nativeTickIndex',0),('coupledStep',2),('previousArrayHash','wrong'),
    ('tissueStateBefore',1),('elapsedSecondsBefore',0.),('sampleFieldsSha256','wrong'),
    ('removedSourceIdsAfter',[]),('sourceOperatorIdentityAfter','wrong')])
def test_native_source_event_join_refuses_looking_like_valid_metadata(key,value):
    frame,row,sample,native=event_fixture();module.require_join(frame,row,sample,native,0)
    frame[key]=value
    with pytest.raises(ValueError):module.require_join(frame,row,sample,native,0)


def test_a_counterfactual_difference_is_evidence_and_does_not_become_an_accepted_sequence():
    native={'limiter':{'resolvedMm':[0.,0.,0.]},'orderedSourceIds':[3,4],'candidateSourceIds':[3,4],
        'cusaAfter':{'progressExact':{}},'removedSourceIdsAfter':[3],'events':[{'removedCells':[3]}]}
    coupled=copy.deepcopy(native);coupled['orderedSourceIds']=[4,3];coupled['removedSourceIdsAfter']=[4]
    coupled['events']=[{'removedCells':[4]}]
    difference=module.compare(native,coupled)
    assert not difference['allCusaDecisionsExact'] and difference['limiter'] and not difference['events']
    assert difference['removedSourceIdsAdded']==[4] and difference['removedSourceIdsLost']==[3]
    assert native['removedSourceIdsAfter']==[3]
