"""Saved small evidence/source audit; never open predecessor payloads or execute validation."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'build/hbe-v5-n12-nocache-50pct-probe-v1'
ATTEMPT=BASE/'attempt-01'
BINDINGS={}
READ_BYTES=0

def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path,expected=None):
    global READ_BYTES
    path=Path(path)
    rel=path.relative_to(ROOT)
    assert rel.parts[0] != 'outputs', 'Predecessor payload reads forbidden'
    before=path.lstat()
    assert stat.S_ISREG(before.st_mode) and before.st_size <= 1024**2, str(path)
    raw=path.read_bytes()
    after=path.lstat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    observed=sha(raw)
    if expected is not None: assert observed==expected, str(path)
    BINDINGS[str(rel)]={'sha256':observed,'bytes':len(raw)}
    READ_BYTES+=len(raw)
    return raw

def document(path,expected=None): return json.loads(read(path,expected))
def git_blob(commit,path):
    return subprocess.check_output(['git','show',commit+':'+str(path)],cwd=ROOT)

head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
authorization=document(BASE/'root-authorization.json')
selected=authorization['source_head']
assert selected=='abbc3e31ef2dd9123e470a31b3701eb375f85424'
read(ROOT/'build/hbe-v5-n12-nocache-50pct-independent-v1/REPORT.md',authorization['report_sha256'])
read(BASE/'probe.py',authorization['probe_sha256'])
prior_review=document(ROOT/'build/hbe-v5-n12-nocache-50pct-independent-v1/audit.json','52769c4a7cccdaa765dc814234fb41c8d97cbfd014d9683abadeb47eac11ff7c')
for item in prior_review['candidate_files'].values(): read(ROOT/item['path'],item['sha256'])
for item in prior_review['pinned_dependencies']: read(ROOT/item['path'],item['sha256'])
for item in prior_review['old20_working_and_git_bindings']+prior_review['unchanged_committed_v1_launcher_sources']:
    raw=read(ROOT/item['path'],item['sha256'])
    assert sha(git_blob(selected,item['path']))==item['sha256']
    assert sha(git_blob(head,item['path']))==item['sha256']

expected_names={'receipt.json','worker-result.json','validation-identity.json','hint-audit.json',
                'host-time-series.json','worker-samples.json','readout-console.txt'}
assert {p.name for p in ATTEMPT.iterdir()}==expected_names
saved={name:read(ATTEMPT/name) for name in sorted(expected_names)}
assert saved['readout-console.txt']==b''
r=json.loads(saved['receipt.json'])
w=json.loads(saved['worker-result.json'])
i=json.loads(saved['validation-identity.json'])
h=json.loads(saved['hint-audit.json'])
series=json.loads(saved['host-time-series.json'])
snapshots=json.loads(saved['worker-samples.json'])
assert r['status']=='completed_read_only_50pct_probe'
assert r['source_commit']==r['observed_head_postguard']==selected
assert r['probe_source_sha256']==r['probe_source_sha256_postguard']==authorization['probe_sha256']
assert r['original_source_closure']==r['original_source_closure_postguard']==prior_review['old_source_closure']
assert r['native_calls']==r['hbe_readout_calls']==0
assert r['release_written'] is False and r['hbe_output_reserved'] is False
assert r['readout_calls_attempted']==1 and r['supervisor_stage_label']=='readout_api_only_not_hbe_readout'
assert r['caps']=={'aggregate_output_bytes':4194304,'attempts':1,'numerical_threads':1,
 'outer_wall_seconds':150,'sampled_process_group_rss_bytes':3221225472,'worker_wall_seconds':147}
assert r['host_policy']=={'initial_available_percent_floor':50,'during_validation_available_percent_floor':45,'normal_kernel_pressure_mask':1}
stage=r['readout_stage']
assert stage['status']=='completed_within_caps' and stage['exit_code']==0 and stage['kill_reason'] is None
assert stage['command']==[str(ROOT/'.venv/bin/python'),'-B',str(BASE/'probe.py'),'worker',selected,authorization['probe_sha256']]
assert 0 < stage['elapsed_seconds'] < 147
assert 0 < r['outer_elapsed_seconds'] < 150 and stage['elapsed_seconds'] < r['outer_elapsed_seconds']
assert 0 < stage['peak_sampled_process_group_rss_bytes'] < 3221225472
assert stage['peak_sampled_active_output_bytes'] < 4194304
assert stage['wall_cap_seconds']==147 and stage['sampled_process_group_rss_cap_bytes']==3221225472 and stage['active_output_cap_bytes']==4194304
assert r['owned_worker_cleanup']=={'contained':True,'direct_child_reaped':True,'errors':[],
 'exit_code':0,'fallback_used':False,'remaining_members':[]}
assert w['status']==r['worker_result_status']=='read_only_validation_memory_diagnostic_complete'
assert w['source_commit']==selected and w['native_calls']==0 and w['release_written'] is False and w['target_reserved'] is False
assert w['postvalidation_below_45_percent'] is False
assert 0 < w['elapsed_seconds'] < stage['elapsed_seconds']
closure=r['original_source_closure']
assert w['adapted_deck_sha256']==w['expected_adapted_deck_sha256']==i['deck_sha256']==closure['expected_adapted_deck_sha256']
assert i['run_id']==closure['run_id']=='tension:N12:S60:reference' and i['index']==8
assert i['source_hashes']==closure['old_source_hashes'] and len(i['source_hashes'])==20
assert i['adapter_receipt']==closure['adapter_receipt'] and i['observed_head_preflight']==selected
original_template=document(ROOT/'build/hbe-v5-tension-n12-s60-release-prep/RELEASE_TEMPLATE.json',closure['original_template_sha256'])
assert original_template['status']=='TEMPLATE_NOT_RELEASED'
assert i['previous']['sha256']==[x['sha256'] for x in original_template['prior_receipts']]
prior_identity=document(ROOT/'build/hbe-v5-n12-nocache-comparison-v1/hinted-01/validation-identity.json','2314bb377008cbf11c0b683ea5af3305e29c0b4d4a6a7e869ee5a4939d90da57')
assert {k:v for k,v in i.items() if k!='observed_head_preflight'}=={k:v for k,v in prior_identity.items() if k!='observed_head_preflight'}
for name in ['backend_profile','runtime_identity','preparation','source_map','v5_declaration']:
    item=original_template[name]
    read(ROOT/item['path'],item['sha256'])

assert r['host_before']['available_percent']>=50 and r['host_before']['kernel_pressure_mask']==1
assert r['host_breach'] is None and series['breach'] is None
hs=series['samples']
assert len(hs)==r['host_sample_count']==27
assert hs[0]['label']=='before_worker_supervision' and hs[-1]['label']=='after_owned_cleanup'
assert hs[0]['available_percent']>=50
assert all(x['label']=='supervised_rss_tick' for x in hs[1:-1])
assert all(x['available_percent']>=45 and x['kernel_pressure_mask']==1 for x in hs)
assert all(0<=x['elapsed_seconds']<r['outer_elapsed_seconds'] for x in hs)
assert all(a['elapsed_seconds']<b['elapsed_seconds'] for a,b in zip(hs,hs[1:]))
assert len(snapshots)==w['samples']==24
assert snapshots[0]['label']=='stdlib_host_before_hbe_import' and snapshots[0]['direct_kernel_host']['available_percent']>=50
assert all(x['direct_kernel_host']['available_percent']>=45 and x['direct_kernel_host']['kernel_pressure_mask']==1 for x in snapshots)
assert all(x['memory_pressure']['exit_code']==0 and x['vm_stat']['exit_code']==0 for x in snapshots)
assert all(0<=x['elapsed_seconds']<w['elapsed_seconds'] for x in snapshots)
assert all(a['elapsed_seconds']<b['elapsed_seconds'] for a,b in zip(snapshots,snapshots[1:]))
all_swap={r['host_before']['swap_used_bytes']}|{x['swap_used_bytes'] for x in hs}|{x['direct_kernel_host']['swap_used_bytes'] for x in snapshots}
assert len(all_swap)==1
prior_hints=document(ROOT/'build/hbe-v5-n12-nocache-comparison-v1/hinted-01/hint-audit.json','ddcd8ed51626ebb1c707024eb77a7bb716d0d9791f268a12a1f761e513e3f8b4')
assert h==prior_hints
assert h['eligible_files']==h['hinted_file_opens']==len(h['hinted_file_paths'])==14
counts=Counter(h['hinted_file_paths'])
assert len(counts)==12
metadata={}
allowed={str((ROOT/x['path']).parent) for x in original_template['prior_receipts']}
for path,count in counts.items():
    p=Path(path)
    entry=p.lstat()  # metadata only; never open large files.
    assert stat.S_ISREG(entry.st_mode) and str(p.parent) in allowed and entry.st_size>=16*1024**2
    assert p.name in {'nodes.log','elements.log'}
    assert count==(2 if '01-compression-N12-S60-reference' in str(p) else 1)
    metadata[str(p.relative_to(ROOT))]={'bytes':entry.st_size,'hinted_open_count':count}
assert sum(x['bytes']*x['hinted_open_count'] for x in metadata.values())==h['eligible_bytes']==2801621755
assert sum(x['bytes'] for x in metadata.values())==2753200498

closed=sum(len(v) for v in saved.values())
assert closed < 4194304
snapshot=r['final_retained_output_bytes_before_final_receipt']
pre_final=dict(r)
del pre_final['final_retained_output_bytes_before_final_receipt']
# Receipt outer timing string can grow/shrink; independently report, do not relabel a snapshot as final.

def vm(sample):
    raw=sample['vm_stat']['stdout']
    assert 'page size of 16384 bytes' in raw
    return {k.strip('" '):int(v) for k,v in re.findall(r'^([^:]+):\s*([0-9]+)\.$',raw,re.M)}
outer_before=snapshots[7]; outer_after=snapshots[16]
assert outer_before['label']=='before_predecessor_chain_hashing' and outer_after['label']=='after_predecessor_chain_hashing'
b,a=vm(outer_before),vm(outer_after)
selected_vm=['Pageins','File-backed pages','Anonymous pages','Pages stored in compressor',
 'Pages occupied by compressor','Pageouts','Swapins','Swapouts']
vm_deltas={key:a[key]-b[key] for key in selected_vm}
native=ROOT/original_template['output_directory']
sidecar=ROOT/'outputs/mechanics/hbe-v5-nocache-policy-v1/08-tension-N12-S60-reference/attempt-01'
assert not native.exists() and not native.is_symlink() and not sidecar.exists() and not sidecar.is_symlink()
post=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert post==head
for name,raw in saved.items(): assert read(ATTEMPT/name)==raw
record={'decision':'GO_saved_read_only_evidence_HOLD_native_policy_change', 'source_commit':selected,'audit_current_head':head,
 'source_guard_working_historical_current_committed':True,'input_bindings':BINDINGS,
 'outer_seconds':r['outer_elapsed_seconds'],'worker_stage_seconds':stage['elapsed_seconds'],
 'worker_seconds':w['elapsed_seconds'],'peak_sampled_worker_group_rss_bytes':stage['peak_sampled_process_group_rss_bytes'],
 'peak_sampled_output_bytes':stage['peak_sampled_active_output_bytes'],'receipt_output_snapshot_bytes':snapshot,
 'final_closed_output_bytes':closed,'final_receipt_growth_bytes':closed-snapshot,
 'host_supervisor':{'before':r['host_before']['available_percent'],'initial':hs[0]['available_percent'],
  'final':hs[-1]['available_percent'],'minimum':min(x['available_percent'] for x in hs),
  'maximum':max(x['available_percent'] for x in hs),'samples':len(hs),'rss_ticks':len(hs)-2,
  'masks':sorted({x['kernel_pressure_mask'] for x in hs}),
  'maximum_sample_gap_seconds':max(b['elapsed_seconds']-a['elapsed_seconds'] for a,b in zip(hs,hs[1:]))},
 'host_worker':{'initial':snapshots[0]['direct_kernel_host']['available_percent'],
  'final':snapshots[-1]['direct_kernel_host']['available_percent'],
  'minimum':min(x['direct_kernel_host']['available_percent'] for x in snapshots),
  'maximum':max(x['direct_kernel_host']['available_percent'] for x in snapshots),
  'samples':len(snapshots),'masks':sorted({x['direct_kernel_host']['kernel_pressure_mask'] for x in snapshots})},
 'swap_used_bytes_constant':next(iter(all_swap)),
 'outer_chain_seconds':outer_after['elapsed_seconds']-outer_before['elapsed_seconds'],
 'outer_chain_rss_delta_bytes':outer_after['process_rss_bytes']-outer_before['process_rss_bytes'],
 'outer_chain_page_size_bytes':16384,'outer_chain_vm_delta_pages':vm_deltas,
 'hinted_opens':14,'unique_hinted_paths':12,'eligible_call_bytes':h['eligible_bytes'],
 'unique_eligible_bytes':sum(x['bytes'] for x in metadata.values()),'hinted_path_metadata_only':metadata,
 'identity_equals_prior_hinted_arm_except_head':True,'validated_previous':i['previous'],
 'owned_cleanup':r['owned_worker_cleanup'],'ordinal8_native_and_policy_directories_absent':True,
 'audit_full_predecessor_bytes_read':0,'audit_native_calls':0,'audit_hbe_replays':0,
 'audit_patient_or_measured_data_reads':0,'audit_small_file_bytes_read_including_repeats':READ_BYTES}
(HERE/'audit.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in record.items() if k not in ['input_bindings','hinted_path_metadata_only','validated_previous']},indent=2,sort_keys=True))
print('audit_sha256',sha((HERE/'audit.json').read_bytes()))
