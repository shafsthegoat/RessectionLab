"""Source-only admission without eager geometry; generated arrays only."""
import pytest

from test_patient_planning_admission import fixture_contract, rebind
from resectionlab import patient_planning_admission as candidate
from resectionlab.native_resection import NativeResectionEngine


def test_source_only_validation_has_zero_constructor_or_preview(monkeypatch):
    case,args=fixture_contract()
    def forbidden(*a,**k):raise AssertionError('eager inventory')
    monkeypatch.setattr(candidate,'NativeSpatialTask',forbidden)
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    validated=candidate.validate_patient_planning_source(case,**args)
    assert validated[2]==args['protocol'] and validated[3]=='generated_interface_control'


@pytest.mark.parametrize('subject,role',[('ReMIND-008','TRAIN'),('ReMIND-013','SELECT')])
def test_eager_factory_uses_source_validator_once(subject,role,monkeypatch):
    case,args=fixture_contract(subject,role)
    calls=[]
    original=candidate.validate_patient_planning_source
    def tracked(*a,**kw):
        calls.append((a,kw))
        return original(*a,**kw)
    monkeypatch.setattr(candidate,'validate_patient_planning_source',tracked)
    task,context=candidate.make_patient_planning_task(case,**args)
    assert len(calls)==1 and calls[0][0][0] is case
    context.require_task(task)
    assert task.step('STOP').terminated


@pytest.mark.parametrize('mutation',['cohort','header_only','private_field','private_reward','bool_budget','held_out'])
def test_existing_refusals_before_inventory(mutation,monkeypatch):
    case,args=fixture_contract()
    if mutation=='cohort':args['cohort_bytes']+=b' '
    if mutation=='header_only':args['qc_receipt']['scope']='header_only'
    if mutation=='private_field':args['source_binding']['private_ventricular_hash']='7'*64
    if mutation=='private_reward':args['protocol']['private_reference_used']=True
    if mutation=='bool_budget':args['protocol']['search']['max_calls']=True
    if mutation=='held_out':
        case,args=fixture_contract('ReMIND-067','TRAIN')
    rebind(args)
    def forbidden(*a,**k):raise AssertionError('eager inventory')
    monkeypatch.setattr(candidate,'NativeSpatialTask',forbidden)
    for call in (candidate.validate_patient_planning_source,candidate.make_patient_planning_task):
        with pytest.raises(ValueError):call(case,**args)
