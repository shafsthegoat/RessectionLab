"""Generated serialization stress, not a native feasibility or patient test."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frontier_compact_output',HERE/'compact_output.py')
codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)


def digest(value):return 'sha256:'+hashlib.sha256(codec.canonical(value)).hexdigest()


def representative_selection():
    """24 complete packets/history rows; all detailed fields remain in payload.

    Repeated microstep cells model the observed serialization failure without
    executing geometry. They are deliberately not an accepted physical trace.
    """
    packets=[];history=[];states=[];sidecars={}
    cells=[[i,2*i,3*i] for i in range(64)]
    micro={'interval':0,'removed_indices_native':cells,'contact_indices_native':cells,
        'shaft_cells_native':cells,'tip_cells_native':cells,'entry_mm':[1.,2.,3.],
        'tip_mm':[1.,2.,4.],'whole_native_model_semantics':'synthetic transport data only'}
    for step in range(24):
        state_hash=digest({'state':step})
        action={'tool_id':'generated_tool','entry_mm':[1.,2.,3.],'tip_mm':[1.,2.,float(step+4)]}
        packet={'schema':'focused-public-goal-action-v1','source_hash':digest('source'),
            'decision_model_hash':digest('model'),'source_state_hash':state_hash,'steps_taken':step,
            'current_tool':None if step==0 else 'generated_tool','focus':None if step<12 else
            {'source_state_hash':state_hash,'public_goal':action},'query_catalog_digest':digest({'catalog':step}),
            'action_id':'action-'+str(step),'physical_action':action}
        packet['seal']=digest(packet);packets.append(packet)
        history.append({'action_id':packet['action_id'],'reward':float(step)-4.,'target_removed_mm3':float(step),
            'normal_removed_mm3':4.,**action,'microsteps':[{**micro,'interval':j} for j in range(160)],
            'removed_indices_native':cells,'contact_indices_native':cells,'source_state_hash':state_hash})
        evidence={'source_state_hash':state_hash,'tool':action['tool_id'],'entry_mm':action['entry_mm'],
            'requested_tip_mm':action['tip_mm'],'blocked_indices_native':cells[:4],
            'complete_first_failure_set':True,'truncated':False,'cell_limit':16,
            'reason':'SHAFT_BLOCKED_BY_REMAINING_NATIVE_TISSUE'}
        evidence['fingerprint']=digest(evidence);sidecars[evidence['fingerprint']]=evidence
        ledger=[{'action_id':'candidate-'+str(i),'family':'nominal' if i<120 else 'obstruction',
            'tool_id':'generated_tool','voxel':[i,2,3],'verification':'NATIVE_VERIFIED',
            'feasible':i%3==0,'reason':'FEASIBLE' if i%3==0 else evidence['reason']} for i in range(152)]
        ledger.extend({'action_id':None,'family':'obstruction','tool_id':'generated_tool','voxel':[i,7,9],
            'verification':'PROVIDER_DISPOSITION','feasible':False,'reason':'OBSTRUCTION_SELECTION_CAP'} for i in range(32))
        states.append({'depth':step+1,'source_state_hash':state_hash,'goal_focus':packet['focus'],
            'complete_page_enumeration':True,'pages':[{'page_index':i,'ledger':ledger} for i in range(5)],
            'selected':{'action_id':packet['action_id'],'action_geometry':action,'actual_removed_cells':cells},
            'goals':[{'current_catalog_action_id':'candidate-'+str(i),'sidecar_fingerprint':evidence['fingerprint'],
                'retained_blocker_cells':cells[:4],'first_failure_set_complete':True} for i in range(120)],
            'dependency_diagnostics':{'active_goal':action,'pair_limit':32,'pair_cap_reached':False,
                'probes':[{'current_catalog_action_id':'prep-'+str(i),'sidecar':evidence,
                    'retained_prerequisite_cells':cells[:4]} for i in range(32)]},
            'obstruction_accounting':{'selected_cells':cells[:16],'omitted_cells':cells[16:]}})
    return {'complete':True,'model_transition_calls':12,'fixed_prefix_length':12,'fixed_prefix_return':18.,
        'selected_prefix_length':24,'selected_total_return':180.,'selected_net_suffix_return':162.,
        'explored_total_return':180.,'explored_net_suffix_return':162.,'states':states,
        'selected_action_records':packets,'explored_action_records':copy.deepcopy(packets),
        'explored_actions':[p['action_id'] for p in packets],'explored_history':history,
        'terminal':'HORIZON','time_cap_reached':False,'call_cap_reached':False},sidecars


def test_complete_24_action_lossless_bound_and_independent_join_fields():
    original,sidecars=representative_selection()
    before=digest(original);raw=codec.canonical(original)
    assert 6*1024**2<len(raw)<codec.MAX_RAW  # reproduces the old uncompressed failure scale
    encoded,binding=codec.encode_gzip(original)
    decoded=codec.decode_gzip(encoded,binding)
    assert decoded==original and digest(decoded)==before
    assert codec.encode_gzip(original)==(encoded,binding)  # fixed time, no filenames
    assert len(decoded['explored_history'])==len(decoded['explored_action_records'])==24
    assert decoded['states']==original['states']  # full ledgers/omissions/focus/sidecars intact
    assert decoded['selected_action_records']==original['selected_action_records']
    for packet in decoded['explored_action_records']:
        assert packet['seal']==digest({k:v for k,v in packet.items() if k!='seal'})
    exact_sidecars={'records':sidecars,'phase_counts':{'focused_suffix_planning':{k:1 for k in sidecars}},
        'scope':'complete first-failure sets, not later corridor proof'}
    sidebytes,sidemeta=codec.encode_gzip(exact_sidecars)
    assert codec.decode_gzip(sidebytes,sidemeta)==exact_sidecars
    for fingerprint,row in sidecars.items():assert fingerprint==digest({k:v for k,v in row.items() if k!='fingerprint'})
    # Reserve1MiB for original frontier/source metadata, plans/endpoints/report/supervision.
    assert len(encoded)+len(sidebytes)+1024**2<3*1024**2
    cfg=json.loads((HERE/'config.json').read_text())
    assert cfg['caps']['output_bytes']==3*1024**2 and cfg['caps']['whole_build_bytes']==8*1024**2
    assert cfg['caps']['native_previews']==12000 and cfg['caps']['native_transitions']==75
    assert cfg['max_steps']==24 and cfg['selector_seconds']==180
    for name in ('frontier_selector.py','frontier_protocol.py','common.py','run_owned.py','bootstrap_smoke.py','freeze_sources.py'):
        assert (HERE/name).read_bytes()==(HERE.parent/'recovery'/name).read_bytes()


def test_binding_and_decompressed_bound_reject_without_silent_truncation(monkeypatch):
    encoded,binding=codec.encode_gzip({'unchanged_complete_record':'x'*1000})
    with pytest.raises(ValueError):codec.decode_gzip(encoded[:-1],binding)
    wrong={**binding,'decompressed_sha256':'0'*64}
    with pytest.raises(ValueError):codec.decode_gzip(encoded,wrong)
    wrong={**binding,'decompressed_bytes':codec.MAX_RAW+1}
    with pytest.raises(ValueError):codec.decode_gzip(encoded,wrong)
    monkeypatch.setattr(codec,'MAX_RAW',10)
    with pytest.raises(InterruptedError,match='Decompressed'):codec.encode_gzip({'too_large':'x'*11})
    monkeypatch.setattr(codec,'MAX_RAW',32*1024**2)
    monkeypatch.setattr(codec,'MAX_GZIP',10)
    with pytest.raises(InterruptedError,match='Compressed'):codec.encode_gzip({'still_too_large':'abcdef'})
