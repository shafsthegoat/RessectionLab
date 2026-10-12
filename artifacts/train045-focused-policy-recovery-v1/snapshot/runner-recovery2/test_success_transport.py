"""Completed frozen-packet transport, including real generated native exports."""
import importlib.util,json,os,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'model-recovery'),str(HERE.parent/'model')]
from common import semantic,write_json
from comparison import plain,validate_plan
from plan_transport import persist_completed_rollout,seal_plan_pair
from recovery_parity import require_retained_greedy_prefix
from compact_output import encode_gzip,decode_gzip
from resectionlab.core import freeze_json,semantic_digest
from resectionlab.patient_planning_cohort_io import OutputBudget


def output(name):
    path=Path(os.environ['TMPDIR']).resolve()/name;path.mkdir(exist_ok=False)
    return path,OutputBudget(path,2*1024**2)


def test_actual_native_frozen_exports_validate_both_plan_seals_and_report():
    path=HERE.parent/'model-recovery'/'test_focused_policy.py'
    spec=importlib.util.spec_from_file_location('_actual_native_transport_fixture',path)
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    f=fixture.generated_fixture();adapter=fixture.adapter
    greedy,a=adapter.rollout(f['focused'],f['admission'],f['packets'],f['frontier'],method='GREEDY',seconds=20)
    scorer=adapter.FrozenFocusedTail(fixture.generated_tail(f),f['admission'],seconds=60)
    tail,b=adapter.rollout(f['focused'],f['admission'],f['packets'],f['frontier'],method='TAIL',scorer=scorer,seconds=20)
    # Reproduce the real failure type using ACTUAL canonical export_action packets.
    with pytest.raises(TypeError,match='mappingproxy'):json.dumps(f['packets'])
    out,sink=output('actual-frozen-success');report={'status':'generated-control','selections':{}}
    selected={};accounts={}
    for method,packets,account in (('GREEDY',greedy,a),('TAIL',tail,b)):
        p,c=persist_completed_rollout(out,sink,report,method,packets,account)
        assert report['selections'][method]['validation']=='pending'
        assert (out/(method+'-selection.json.gz')).is_file()
        # The frozen actual prefix is deliberately preserved at this call boundary.
        assert validate_plan(packets,account,f['packets'],f['frontier'],24)
        assert semantic(p)==semantic_digest(packets)
        report['selections'][method]['validation']='accepted'
        selected[method]=packets;accounts[method]=freeze_json(account)
    sealed=seal_plan_pair(out,sink,report,selected,accounts,f['frontier'],native_transition_attempts=0)
    assert set(sealed)=={'GREEDY','TAIL'} and report['plans_sealed_before_replay']
    write_json(out/'report.json',plain(report))
    stored=json.loads((out/'plans.json').read_bytes())
    for method in ('GREEDY','TAIL'):
        assert stored[method]['seal']==semantic({k:v for k,v in stored[method].items() if k!='seal'})
        assert stored[method]['planned_history_hash']==semantic_digest(accounts[method]['explored_history'])
    assert json.loads((out/'report.json').read_bytes())['plans_sealed_before_replay']['seals']=={
        m:sealed[m]['seal'] for m in ('GREEDY','TAIL')}


def frozen_full_transport_fixture():
    # Full23/11 shape exercises the acquired parity contract. These records are
    # explicitly scalar transport fixtures, never accepted physical histories.
    spec=importlib.util.spec_from_file_location('_full_parity_transport_fixture',HERE/'test_recovery_parity.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    retained,recovered=module.fixture()
    for account in (retained['policy'],recovered):
        for packet in account['explored_action_records']:
            packet.update(source_hash='generated-source',decision_model_hash='generated-model')
            packet['seal']=semantic({k:v for k,v in packet.items() if k!='seal'})
    return freeze_json(retained),freeze_json(recovered)


def test_success_full_frozen_pair_then_postseal_parity_and_normal_report_writer():
    retained,recovered=frozen_full_transport_fixture();out,sink=output('full-frozen-parity-success')
    prefix=recovered['explored_action_records'][:12]
    frontier=freeze_json({'steps':12,'source_hash':'generated-source','decision_model_hash':'generated-model','seal':semantic('generated-frontier')})
    report={'status':'generated-control','selections':{}};selected={};accounts={}
    for method in ('GREEDY','TAIL'):
        value={**plain(recovered),'method':method};packets=freeze_json(value['explored_action_records']);account=freeze_json(value)
        p,c=persist_completed_rollout(out,sink,report,method,packets,account)
        assert validate_plan(packets,account,prefix,frontier,24)
        selected[method]=packets;accounts[method]=account
        if method=='GREEDY':
            with pytest.raises(ValueError):seal_plan_pair(out,sink,report,selected,accounts,frontier,native_transition_attempts=0)
        report['selections'][method]['validation']='accepted'
    seal_plan_pair(out,sink,report,selected,accounts,frontier,native_transition_attempts=0)
    assert (out/'plans.json').is_file()
    # Decode/parity occurs only AFTER both immutable plans are persisted.
    compressed,binding=encode_gzip(plain(retained));decoded=decode_gzip(compressed,binding)
    report['retained_greedy_prefix_parity']=require_retained_greedy_prefix(freeze_json(decoded),accounts['GREEDY'])
    write_json(out/'report.json',plain(report))
    saved=json.loads((out/'report.json').read_bytes())
    assert saved['retained_greedy_prefix_parity']['accepted']
    assert saved['retained_greedy_prefix_parity']['exact_full_prefix_actions']==23
    assert saved['retained_greedy_prefix_parity']['exact_committed_suffix_decisions']==11


def test_returned_complete_account_survives_later_validation_failure():
    _,recovered=frozen_full_transport_fixture();out,sink=output('persist-before-failure')
    report={'status':'generated-control','selections':{}}
    packets,account=persist_completed_rollout(out,sink,report,'GREEDY',recovered['explored_action_records'],recovered)
    with pytest.raises(ValueError):validate_plan(packets,account,packets[:11],{'steps':12},24)
    saved=json.loads((out/'report.json').read_bytes());binding=saved['selections']['GREEDY']
    decoded=decode_gzip((out/binding['path']).read_bytes(),binding)
    assert decoded['complete'] is True and len(decoded['explored_history'])==24
    assert binding['validation']=='pending' and not saved.get('plans_sealed_before_replay')
