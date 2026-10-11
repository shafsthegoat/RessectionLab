"""Source-only admission extraction parity; generated arrays only."""
from pathlib import Path
import importlib.util
import sys

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT/'tests'))
from test_patient_planning_admission import fixture_contract, rebind
from resectionlab import patient_planning_admission as baseline
from resectionlab.native_resection import NativeResectionEngine


def staged():
    path=Path(__file__).parent/'stage/resectionlab/patient_planning_admission.py'
    spec=importlib.util.spec_from_file_location('resectionlab._pilot_source_admission',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module


def test_source_only_validation_has_zero_constructor_or_preview(monkeypatch):
    case,args=fixture_contract();candidate=staged()
    def forbidden(*a,**k):raise AssertionError('eager inventory')
    monkeypatch.setattr(candidate,'NativeSpatialTask',forbidden)
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    validated=candidate.validate_patient_planning_source(case,**args)
    assert validated[2]==args['protocol'] and validated[3]=='generated_interface_control'


@pytest.mark.parametrize('subject,role',[('ReMIND-008','TRAIN'),('ReMIND-013','SELECT')])
def test_default_eager_output_and_native_history_exact(subject,role):
    case,args=fixture_contract(subject,role);candidate=staged()
    old,oc=baseline.make_patient_planning_task(case,**args)
    new,nc=candidate.make_patient_planning_task(case,**args)
    assert oc.record()==nc.record() and oc.fingerprint==nc.fingerprint
    assert old.observation().fingerprint==new.observation().fingerprint
    assert old.candidate_inventory()==new.candidate_inventory()
    assert old.step('STOP').info==new.step('STOP').info
    assert old.metrics()==new.metrics()


@pytest.mark.parametrize('mutation',['cohort','header_only','private_field','private_reward','bool_budget','held_out'])
def test_existing_refusals_before_inventory(mutation,monkeypatch):
    case,args=fixture_contract();candidate=staged()
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
    with pytest.raises(ValueError):baseline.make_patient_planning_task(case,**args)
