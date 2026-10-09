"""Bounded saved-arm audit: small files/source and predecessor metadata only."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent
BASE=ROOT/'build/hbe-v5-n12-nocache-comparison-v1'
ARM=BASE/'hinted-01'
PREP=ROOT/'build/hbe-v5-tension-n12-s60-release-prep'
inputs={}
bytes_read=0
git_bytes=0

def bounded(path,cap=128*1024):
    global bytes_read
    if not path.is_absolute(): path=ROOT/path
    before=path.lstat()
    assert stat.S_ISREG(before.st_mode) and before.st_size<=cap,path
    raw=path.read_bytes()
    after=path.lstat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    inputs[str(path.relative_to(ROOT))]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    bytes_read+=len(raw)
    return raw

def doc(path,cap=128*1024): return json.loads(bounded(path,cap))
def sha(path,cap=128*1024): return hashlib.sha256(bounded(path,cap)).hexdigest()
def closed(path):
    total=0
    for p in path.iterdir():
        st=p.lstat()
        assert stat.S_ISREG(st.st_mode),p
        total+=st.st_size
    return total

receipt=doc(ARM/'receipt.json')
worker=doc(ARM/'worker-result.json')
identity=doc(ARM/'validation-identity.json')
hint=doc(ARM/'hint-audit.json')
samples=doc(ARM/'worker-samples.json')
assert bounded(ARM/'readout-console.txt')==b''
assert receipt['status']=='completed_read_only_arm'
assert receipt['arm']==hint['mode']=='hinted'
commit=receipt['source_commit']
assert commit==worker['source_commit']==identity['observed_head_preflight']=='d2b540f842ef1074db1e9b9c3ca7a767ea150f89'
assert sha(BASE/'compare.py')==receipt['comparison_source_sha256']==receipt['comparison_source_sha256_postguard']=='2113791dc16d7191b7c9c24f4148fb820450ecfff4c2d2f5c0af8ad8488d0b45'
assert sha(BASE/'hash_hint.py')==receipt['hint_source_sha256']=='96c71d0d9b8d896b713aef7305e3e86d5b12b9d2dca5fcc55515f6b2eb9cf3fc'
assert sha(ROOT/'build/hbe-v5-n12-preflight-memory-diagnostic-v1/diagnose.py')==receipt['diagnostic_source_sha256']=='8684aa673e8e0189bd6c5fdfab144e7ff6d8a33af5e16047caa5a88b196dce16'
assert receipt['readout_stage']['command'][-3:]==['hinted',commit,receipt['comparison_source_sha256']]
assert receipt['readout_stage']['status']=='completed_within_caps' and receipt['readout_stage']['exit_code']==0 and receipt['readout_stage']['kill_reason'] is None
assert receipt['caps']=={'aggregate_output_bytes':4194304,'attempts':1,'numerical_threads':1,'outer_wall_seconds':150,'sampled_process_group_rss_bytes':3221225472,'worker_wall_seconds':147}
assert receipt['outer_elapsed_seconds']<150 and receipt['readout_stage']['elapsed_seconds']<147
assert receipt['readout_stage']['peak_sampled_process_group_rss_bytes']==177553408<3221225472
assert receipt['readout_stage']['peak_sampled_active_output_bytes']<4194304
assert receipt['owned_worker_cleanup']=={'contained':True,'direct_child_reaped':True,'errors':[],'exit_code':0,'fallback_used':False,'remaining_members':[]}
assert receipt['native_calls']==receipt['hbe_readout_calls']==worker['native_calls']==0
assert receipt['release_written']==receipt['hbe_output_reserved']==worker['release_written']==worker['target_reserved']==False
assert receipt['readout_calls_attempted']==1 and receipt['supervisor_stage_label']=='readout_api_only_not_hbe_readout'
assert worker['status']=='read_only_validation_memory_diagnostic_complete'
assert closed(ARM)==76797<4194304
assert receipt['final_retained_output_bytes']==76758
assert sha(PREP/'RELEASE_TEMPLATE.json')=='32d7566d10d1a671fe1d7fcd2cb3bfc73dc70c3632d286480fd860e2e15d2feb'
template=doc(PREP/'RELEASE_TEMPLATE.json')
assert template['status']=='TEMPLATE_NOT_RELEASED'
assert identity['index']==template['ordinal']==8
assert identity['run_id']==template['run_id']=='tension:N12:S60:reference'
assert identity['adapter_receipt']==template['adapter_receipt']==doc(PREP/'adapter-receipt.json')
assert identity['deck_sha256']==worker['adapted_deck_sha256']==worker['expected_adapted_deck_sha256']==template['adapted_deck_sha256']==sha(PREP/'candidate-specimen.feb',2*1024**2)
assert not (ROOT/template['output_directory']).exists()
assert not (BASE/'baseline-01').exists()
for key in ['runtime_identity','backend_profile','preparation','v5_declaration','source_map']:
    binding=template[key]
    assert sha(ROOT/binding['path'],2*1024**2)==binding['sha256'],key
assert len(identity['source_hashes'])==len(template['source_bindings'])==20
for path,expected in identity['source_hashes'].items():
    current=bounded(ROOT/path)
    historic=subprocess.run(['git','show',commit+':'+path],cwd=ROOT,capture_output=True,check=True,timeout=5).stdout
    git_bytes+=len(historic)
    assert hashlib.sha256(current).hexdigest()==hashlib.sha256(historic).hexdigest()==template['source_bindings'][path]['sha256']==expected,path

receipts=[]
native=readout=prep=0.
primary=0
eligible={}
for index,binding in enumerate(template['prior_receipts']):
    assert sha(ROOT/binding['path'],1024**2)==binding['sha256']==identity['previous']['sha256'][index]
    saved=doc(ROOT/binding['path'],1024**2)
    assert saved['run_id']==binding['run_id']
    assert saved['status']==('failed_or_incomplete' if index==1 else 'passed_numerical_software_only')
    assert saved['native_calls_attempted']==1 and saved['no_retry'] is True
    assert sha(ROOT/saved['release_path'])==saved['release_sha256']
    historical_release=doc(ROOT/saved['release_path'])
    assert historical_release['source_commit']==saved['source_commit']
    assert historical_release['adapted_deck_sha256']==saved['adapted_deck_sha256']
    for key in ['runtime_identity','backend_profile']:
        assert historical_release[key]['sha256']==template[key]['sha256']==saved[key+'_sha256']
    native += saved['elapsed_seconds'] if index==0 else saved['native_stage']['elapsed_seconds']
    if index:
        readout += saved['readout_stage']['elapsed_seconds']
        prep += saved['prep_elapsed_seconds']
    directory=(ROOT/binding['path']).parent
    size=closed(directory)
    primary+=size
    for p in directory.iterdir():
        st=p.lstat()
        if st.st_size>=16*1024**2:
            eligible[str(p)]={'bytes':st.st_size,'ordinal':index}
    receipts.append({'ordinal':index,'receipt':binding,'release_sha256':saved['release_sha256'],'closed_output_bytes_metadata':size,'status':saved['status']})
previous=identity['previous']
assert previous['native_calls']==8
assert native==previous['native_seconds'] and readout==previous['readout_seconds'] and prep==previous['prep_seconds']
assert primary==previous['output_bytes']==2822230478
assert previous['combined_output_bytes']==primary+previous['supplement_output_bytes']==2822787870
assert previous['combined_wall_seconds']==native+readout+prep+previous['supplement_readout_seconds']+previous['supplement_prep_seconds']
assert previous['supplement_replay_calls']==1
admission_path=ROOT/'manifests/experiments/hbe-v5-n12-exact-supplement-admission-v1.json'
assert sha(admission_path)=='84cd645a10d70c07613e12d9548076336326b8e97510cfe83e39e261ec0eaa5e'
admission=doc(admission_path)
for key in ['supplement_release','supplement_receipt']:
    assert sha(ROOT/admission[key]['path'])==admission[key]['sha256']
supplement=doc(ROOT/admission['supplement_receipt']['path'])
assert supplement['status']=='supplemental_saved_attempt_numerical_pass_only'
assert supplement['original_status_preserved']=='failed_or_incomplete'
assert supplement['readout_stage']['elapsed_seconds']==previous['supplement_readout_seconds']
assert supplement['preparation_elapsed_seconds']==previous['supplement_prep_seconds']
assert closed((ROOT/admission['supplement_receipt']['path']).parent)==previous['supplement_output_bytes']==557392
assert len(eligible)==12 and sum(x['bytes'] for x in eligible.values())==2753200498
counts=Counter(hint['hinted_file_paths'])
assert set(counts)==set(eligible)
assert hint['eligible_files']==hint['hinted_file_opens']==sum(counts.values())==14
assert hint['eligible_bytes']==sum(eligible[p]['bytes']*n for p,n in counts.items())==2801621755
for p,n in counts.items(): assert n==(2 if eligible[p]['ordinal']==1 else 1)

assert len(samples)==worker['samples']==24
assert all(s['direct_kernel_host']['available_percent']==55 and s['direct_kernel_host']['kernel_pressure_mask']==1 for s in samples)
assert receipt['host_before']['available_percent']==55 and receipt['host_before']['kernel_pressure_mask']==1
assert worker['postvalidation_below_45_percent'] is False
keys={'pageins':'Pageins','file_backed_pages':'File-backed pages','anonymous_pages':'Anonymous pages','compressor_occupied_pages':'Pages occupied by compressor','compressor_stored_pages':'Pages stored in compressor','pageouts':'Pageouts','swapins':'Swapins','swapouts':'Swapouts'}
def vm(s):
    assert s['vm_stat']['exit_code']==0
    text=s['vm_stat']['stdout']
    assert int(re.search(r'page size of (\d+) bytes',text)[1])==16384
    return {k:int(re.search('^'+re.escape(label)+r':\s+(\d+)\.',text,re.M)[1]) for k,label in keys.items()}
before,after=samples[7],samples[16]
assert before['label']=='before_predecessor_chain_hashing' and after['label']=='after_predecessor_chain_hashing'
deltas={k:vm(after)[k]-vm(before)[k] for k in keys}
assert after['direct_kernel_host']['swap_used_bytes']==before['direct_kernel_host']['swap_used_bytes']
selected=[]
for index in [0,2,7,16,21,23]:
    s=samples[index]
    selected.append({'index':index,**{k:s[k] for k in ['label','elapsed_seconds','direct_kernel_host','process_rss_bytes']},'vm_pages':vm(s)})
package=BASE/'integration/artifacts/hbe-v5-n12-nocache-hinted-feasibility-v1'
manifest=doc(package/'MANIFEST.json')
assert len(manifest['files'])==13
package_bytes=0
for entry in manifest['files']:
    value=bounded(package/entry['path'])
    assert len(value)==entry['bytes'] and hashlib.sha256(value).hexdigest()==entry['sha256']
    package_bytes+=len(value)
assert package_bytes==123938
for p in ARM.iterdir(): assert bounded(package/'attempt-01'/p.name)==bounded(p)
for name in ['compare.py','hash_hint.py','PROTOCOL.md']:
    assert bounded(package/'source'/name)==bounded(BASE/name)
for name in ['test_generated_compare.py','test_generated_hash_hint.py']:
    assert bounded(package/'tests'/name)==bounded(BASE/name)
summary=doc(package/'summary.json')
assert summary['source_commit']==commit and summary['adapted_deck_sha256']==identity['deck_sha256']
for k in ['comparison_source_sha256','hint_source_sha256','diagnostic_source_sha256']: assert summary[k]==receipt[k]
assert summary['template_sha256']==sha(PREP/'RELEASE_TEMPLATE.json')
expected_result={'full_validate_release_completed':True,'outer_wall_seconds':receipt['outer_elapsed_seconds'],
    'peak_sampled_process_group_rss_bytes':177553408,'receipt_recorded_output_snapshot_bytes':76758,
    'independently_recounted_closed_output_bytes':76797,'supervisor_exit_code':0,'owned_worker_contained':True,
    'hinted_file_opens':14,'unique_hinted_files':12,'hinted_bytes_counting_nested_n12_repeat':2801621755,
    'hinted_unique_file_bytes':2753200498,'outer_chain_elapsed_seconds':after['elapsed_seconds']-before['elapsed_seconds'],
    'outer_chain_host_available_percent_before':55,'outer_chain_host_available_percent_after':55,
    'outer_chain_kernel_pressure_mask_before':1,'outer_chain_kernel_pressure_mask_after':1,
    'outer_chain_vm_pageins_delta_pages':deltas['pageins'],'outer_chain_vm_file_backed_delta_pages':deltas['file_backed_pages'],
    'outer_chain_vm_anonymous_delta_pages':deltas['anonymous_pages'],'outer_chain_vm_compressor_occupied_delta_pages':deltas['compressor_occupied_pages'],
    'outer_chain_swap_used_delta_bytes':0,'vm_page_size_bytes':16384}
assert summary['result']==expected_result
assert summary['calls']=={'native':0,'hbe_saved_readout':0,'baseline_comparison_arm':0,'hinted_full_validation_arm':1}
prior=doc(ROOT/'artifacts/hbe-v5-n12-preflight-memory-v1/summary.json')['read_only_diagnostic']
orient=summary['historical_unhinted_orientation']
assert orient['outer_chain_host_available_percent_before']==prior['outer_predecessor_before_direct_available_percent']
assert orient['outer_chain_host_available_percent_after']==prior['outer_predecessor_after_direct_available_percent']
assert orient['outer_chain_vm_pageins_delta_pages']==prior['pageins_during_outer_predecessor_pages']
assert orient['outer_chain_vm_file_backed_delta_pages']==prior['file_backed_delta_during_outer_predecessor_pages']
result={'status':'passed_independent_saved_arm_audit','decision':'GO_single_arm_feasibility_evidence_HOLD_automatic_native_release',
        'source_commit':commit,'current_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'outer_wall_seconds':receipt['outer_elapsed_seconds'],'peak_sampled_rss_bytes':receipt['readout_stage']['peak_sampled_process_group_rss_bytes'],
        'recorded_output_snapshot_bytes':receipt['final_retained_output_bytes'],'independent_closed_output_bytes':closed(ARM),
        'outer_predecessor_seconds':after['elapsed_seconds']-before['elapsed_seconds'],'outer_vm_delta_pages':deltas,
        'outer_rss_delta_bytes':after['process_rss_bytes']-before['process_rss_bytes'],
        'page_size_bytes':16384,'host_all_24_samples':{'available_percent':55,'kernel_pressure_mask':1},
        'hinted_opens':14,'unique_hinted_paths':12,'unique_eligible_bytes':2753200498,'actual_hinted_call_bytes':2801621755,
        'path_call_counts':dict(counts),'selected_snapshots':selected,'validated_previous':previous,'receipts':receipts,
        'package_manifest_entries_verified':13,'package_listed_bytes_verified':123938,'package_raw_and_source_files_exact':11,
        'input_bindings':inputs,'small_file_bytes_read_including_repeats':bytes_read,'historical_git_source_bytes_read':git_bytes,
        'full_predecessor_output_bytes_read':0,'native_calls':0,'hbe_replays':0,'patient_or_measured_data_reads':0}
(OUT/'audit.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['input_bindings','selected_snapshots','validated_previous','receipts','path_call_counts']},sort_keys=True))
