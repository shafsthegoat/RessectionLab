"""Metadata gates for the fixed real demonstration; no patient image is read."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).parents[1]
spec=importlib.util.spec_from_file_location('patient_bc',ROOT/'scripts/run_real_patient_imitation.py')
bc=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bc)


@pytest.fixture(autouse=True)
def anchored(tmp_path,monkeypatch):
    # Finite metadata doubles keep software tests independent of ignored real
    # checkpoints and runtime artifacts. No anatomy or success is simulated.
    definition=bc.rl.declaration()
    previous={'status':'complete','optimizer_updates':2}
    teacher={'status':'complete','independent_evaluation':{'accepted':True},
        'decisions':[{'status':'returned','terminated':False},
                     {'status':'returned','terminated':False},
                     {'status':'returned','terminated':True}]}
    for name,value in [('declaration-input.json',definition),('receipt.json',previous),('greedy_search.json',teacher)]:
        (tmp_path/name).write_text(json.dumps(value))
    (tmp_path/'initial.pt').write_bytes(b'finite metadata checkpoint double')
    (tmp_path/'output-sha256.json').write_text(json.dumps({name:bc.sha256(tmp_path/name)
        for name in bc.ANCHOR_FILES if name!='output-sha256.json'}))
    monkeypatch.setattr(bc,'ANCHOR',tmp_path)
    return tmp_path


def reseal(anchor,filename,payload):
    (anchor/filename).write_text(json.dumps(payload))
    index=json.loads((anchor/'output-sha256.json').read_text())
    index[filename]=bc.sha256(anchor/filename)
    (anchor/'output-sha256.json').write_text(json.dumps(index))


def test_completed_teacher_metadata_validates_without_case_load():
    definition,cohort,previous,teacher=bc.validate(bc.declaration())
    assert definition['member']['subject']=='sub-PAT05'
    assert previous['optimizer_updates']==2 and teacher['independent_evaluation']['accepted']


@pytest.mark.parametrize('field,value',[('status','awaiting_independent_check'),('terminated',False),('accepted',False)])
def test_coherently_resealed_partial_or_rejected_teacher_cannot_train(anchored,field,value):
    teacher=json.loads((anchored/'greedy_search.json').read_text())
    if field=='terminated':teacher['decisions'][-1][field]=value
    elif field=='accepted':teacher['independent_evaluation'][field]=value
    else:teacher[field]=value
    reseal(anchored,'greedy_search.json',teacher)
    with pytest.raises(ValueError,match='completed independently'):
        bc.validate(bc.declaration())


def test_checkpoint_bytes_must_match_original_execution_inventory(anchored):
    with (anchored/'initial.pt').open('ab')as handle:handle.write(b'changed')
    with pytest.raises(ValueError,match='execution-time'):
        bc.validate(bc.declaration())


def test_no_hidden_budget_extension():
    record=deepcopy(bc.declaration());record['settings']['optimizer_updates']=9
    with pytest.raises(ValueError,match='fixed eight'):
        bc.validate(record)
