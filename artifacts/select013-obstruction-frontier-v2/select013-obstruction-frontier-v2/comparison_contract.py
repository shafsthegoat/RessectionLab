"""Same-cap public obstruction proposal contrast; historical source admission only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'build/select013-lazy-beam-recovery-v1'
OLD_CONTRACT_SHA='5f0e15aac211bce9e7817c11dcc2cf435de1d5c04d41e839a79a7e0b41cf2119'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(OLD/'comparison_contract.py')!=OLD_CONTRACT_SHA:raise ValueError('Historical metadata helper changed')
spec=importlib.util.spec_from_file_location('dead_frontier_historical_contract',OLD/'comparison_contract.py')
prior_contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior_contract)
read_pinned=prior_contract.read_pinned
OUTPUT='build/select013-obstruction-frontier-v2/attempt-01'
ARMS=('original_fixed_frontier','obstruction_opening_fixed_frontier')
WORKER_SECONDS,PARENT_SECONDS=180,210
MEMORY_BYTES,OUTPUT_BYTES,SUPERVISION_BYTES=3*1024**3,32*1024**2,16*1024**2
LIMITS={'native_previews':4000,'fixed_prefix_commits':22,'total_commits':30,'closure_STOP_calls':5,'diagnostic_previews':360,'max_added_branches':3,'cells_per_sidecar':32,'threads':1,'model_forwards':0,'updates':0}
# Two12-state prefixes (2880) +3 successor inventories (360) +3x120 direct re-previews (360)=3600; ceiling4000.
PINS={
 'result':{'path':'build/select013-lazy-beam-recovery-v1/attempt-01/result.json','sha256':'96e0b98069689a7ad6f1b986423f527ce931e04034b33605d3ebaf48e9e9d761'},
 'selection':{'path':'build/select013-lazy-beam-recovery-v1/attempt-01/observed_beam/selection.json','sha256':'8f62656d1081ea247499cb1ff7ed8b803bf2249d3e700703d3806506a151c8f4'},
 'receipt':{'path':'build/select013-lazy-beam-recovery-v1/attempt-01.supervision/receipt.json','sha256':'83031cb5bdeb9c9ee6bd47dcf53f28e3c7ea7f83207bc94b174d8501b994f232'},
 'release':{'path':'build/select013-lazy-beam-recovery-v1/root-release.json','sha256':'43e7cdf0e43a014154cc15236bbb4df1818ed1577b3cee5b8d75ac368e9aa3d9'},
}

PINS.update({'world': {'path': 'build/select013-lazy-beam-recovery-v1/attempt-01/observed_beam/comparison-world.json', 'sha256': '2dc67df3216033c72540cf2d83e3c614f3a5e62a2fb31ad3cf8c653925213995'}, 'lineage': {'path': 'build/select013-lazy-beam-recovery-v1/attempt-01/IL64-lineage.json', 'sha256': 'e1fe8425d052af596f3a830645e306c47b85f135264e39f81a45f632c3ae3172'}, 'context': {'path': 'build/select013-lazy-beam-recovery-v1/attempt-01/observed_beam/context.json', 'sha256': '178b81f87c8363a3f0c55be411b2085739e349415bbacc73a0fe3e90ad124887'}})

def baseline_historical():
    rows={k:read_pinned(v) for k,v in PINS.items()}
    r,p=rows['result'],rows['receipt'];sel=rows['selection']['accounting']
    if (not prior_contract.complete_result(r) or p['status']!='complete' or p['exit_code']!=0
        or p['cleanup_errors'] or p['final_owned_pids'] or not p['worker_termination_confirmed']
        or p['result_sha256']!=PINS['result']['sha256'] or p['release_sha256']!=PINS['release']['sha256']
        or sel['completed_layers']!=24 or sel['model_transition_calls']!=725
        or sel['call_cap_reached'] or sel['time_cap_reached']):raise ValueError('Exact clean completed predecessor required')
    ref=rows['release']['source_index'];idx=read_pinned(ref)
    rows['training']=prior_contract.authenticate_comparison(rows['release'])['IL64']['protocol']
    if rows['lineage']!=rows['context']['checkpoint_lineage']:raise ValueError('Saved lineage/context mismatch')
    layer=sel['layers'][10]
    opening=next(x for x in layer['opening_depth_retention'] if x['reason']=='opening')
    prefix=tuple(opening['actions'])
    checkpoints=[]
    for depth in range(1,12):
        match=[x for x in sel['layers'][depth-1]['retained_prefix_diagnostics']['retained_nonterminal_prefixes'] if tuple(x['actions'])==prefix[:depth]]
        if len(match)!=1:raise ValueError('Exact consecutive saved prefix missing')
        checkpoints.append(match[0])
    if checkpoints[-1]['legal_nonstop_actions']!=0 or len(prefix)!=11:raise ValueError('Fixed first dead frontier required')
    rows.update(prefix=prefix,prefix_checkpoints=checkpoints,prior_index=idx)
    return rows

WITNESS_PINS={
 'witness_result':{'path': 'build/select013-dead-frontier-witness-v1/attempt-01/result.json', 'sha256': 'e23d75b1171936af24ff0834e37a46e8426bd8b28a0a5f5017848cd78e65c0a2'},
 'witness_receipt':{'path': 'build/select013-dead-frontier-witness-v1/attempt-01.supervision/receipt.json', 'sha256': '085f40abf2ccad5133e8ddd283c97a80ec6a57d1d2acf240e52b3cec7da96cb0'},
 'witness_release':{'path': 'build/select013-dead-frontier-witness-v1/root-release.json', 'sha256': '8866a10e14574b79c46f93be9f5f4f4f8f3e827e9f82c94a5742a18d988a1477'},
 'witness_index':{'path': 'build/select013-dead-frontier-witness-v1/source-index.json', 'sha256': '71879696c1194859f8280b3865804c18a545d09011929fb0a8c39a7a4274a008'},
 'coverage':{'path': 'build/select013-dead-frontier-witness-v1/attempt-01/actual-blocker-capacity.json', 'sha256': 'e6fe994f45834be83740743dc9c9f198bcba55b55cc47083b0e183f7966b7961'},
 'witness_inventory':{'path': 'build/select013-dead-frontier-witness-v1/attempt-01/dead-frontier-inventory.json', 'sha256': '3957c5c7af6d239e9f939b9cf1d007e97136bdc8d44eff013fd98ef9dc25bd88'},
 'witness_history':{'path': 'build/select013-dead-frontier-witness-v1/attempt-01/diagnostic-prefix-history.json', 'sha256': '0eb2e6f46d135bb63e3d703931dc51bdd7b2f4d131e249b3ec833b0178261efa'},
}
PINS.update(WITNESS_PINS)
PRIOR_FAILURE_PINS={'failed_v1_result': {'path': 'build/select013-obstruction-frontier-v1/attempt-01/result.json', 'sha256': 'd940367188ad8a57c103f04f2e22556cb51592eb5adca3322cc23a156931ddcf'}, 'failed_v1_receipt': {'path': 'build/select013-obstruction-frontier-v1/attempt-01.supervision/receipt.json', 'sha256': '34827539399176277bc56177623657863ca1dd54899f4f564d09d42e03d8c9f8'}, 'failed_v1_release': {'path': 'build/select013-obstruction-frontier-v1/root-release.json', 'sha256': 'e7e3e38c58e76549983155c3d9d5d78ffc732254e1f1eee6d825d0141a4cb8c8'}, 'failed_v1_source_index': {'path': 'build/select013-obstruction-frontier-v1/source-index.json', 'sha256': '1fedb6edbf23742894c681a348398c08e927602c9c68603b5dfa18e9a73e1741'}}
PINS.update(PRIOR_FAILURE_PINS)

def historical():
    saved=baseline_historical();w={k:read_pinned(ref) for k,ref in WITNESS_PINS.items()}
    r,p=w['witness_result'],w['witness_receipt']
    if (r['status']!='complete_fixed_dead_frontier_diagnostic' or r['committed_actions']!=11
        or not r['prefix_exact'] or not r['world_equal'] or r['final_legal_motions']!=0
        or p['status']!='complete' or p['exit_code']!=0 or p['cleanup_errors'] or p['final_owned_pids']
        or not p['worker_termination_confirmed'] or p['result_sha256']!=WITNESS_PINS['witness_result']['sha256']
        or p['release_sha256']!=WITNESS_PINS['witness_release']['sha256']):raise ValueError('Exact completed blocker witness required')
    inv,cov=w['witness_inventory'],w['coverage']
    if (inv['emitted_count']!=117 or inv['omitted_count'] or inv['accepted_count'] or inv['candidate_cap']!=120
        or len(cov['outside_every_fixed_axis_tip_envelope'])!=13 or cov['classification_truncated']
        or cov['sidecar_evidence_truncated'] or cov['roundoff_ambiguous_cells']):raise ValueError('Fixed complete frontier evidence differs')
    if [r['action_id'] for r in w['witness_history']]!=list(saved['prefix']):raise ValueError('Witness prefix binding differs')
    failed={k:read_pinned(v) for k,v in PRIOR_FAILURE_PINS.items()}
    f,p=failed['failed_v1_result'],failed['failed_v1_receipt']
    if (f['status']!='failed_or_unresolved' or f['failure']['type']!='TypeError'
        or f['failure']['message']!='Unaccounted source value object' or f['committed_actions']!=0
        or f['native_budget']['native_preview_entries']!=117 or f['checkpoint_loads'] or f['model_forwards'] or f['updates']
        or p['status']!='failed_or_unresolved' or p['exit_code']!=1 or p['cleanup_errors'] or p['final_owned_pids']
        or not p['worker_termination_confirmed'] or p['result_sha256']!=PRIOR_FAILURE_PINS['failed_v1_result']['sha256']
        or p['release_sha256']!=PRIOR_FAILURE_PINS['failed_v1_release']['sha256']):raise ValueError('Preserved exact failedv1 required')
    saved.update(w);return saved

def complete_result(r):
    return (r.get('status')=='complete_fixed_obstruction_frontier_diagnostic'
        and r.get('world_equal') is True and r.get('context_equal') is True and r.get('prefix_exact') is True
        and r.get('committed_prefix_actions')==22 and r.get('complete_episode') is True
        and r.get('accepted_plan_claim') is False and r.get('configuration_change_declared') is True
        and r.get('baseline_frontier_exact') is True and r.get('branches_complete') is True
        and r.get('committed_actions')==24+2*r.get('branch_count',-100)
        and r.get('diagnostic_STOP_calls')==2+r.get('branch_count',-100)
        and 0<=r.get('branch_count',-1)<=3 and r.get('diagnostic_previews')==117*r.get('branch_count',-100)
        and r.get('checkpoint_loads')==r.get('model_forwards')==r.get('updates')==0
        and r.get('optimizer_attempts')==r.get('gradient_attempts')==0
        and r.get('private_reference_reads')==0 and r.get('held_inputs_opened') is False)

def source_guard(release,path,digest):
    if (sha(path)!=digest or release.get('status')!='released_one_attempt' or release.get('output')!=OUTPUT
        or release.get('limits')!=LIMITS or release.get('parent_seconds')!=PARENT_SECONDS
        or release.get('attempts')!=1 or release.get('automatic_retry') is not False
        or release.get('prior_pins')!=PINS):raise ValueError('Exact single diagnostic release required')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    index=read_pinned(release['source_index'])
    if head!=release['expected_head'] or index['head']!=head:raise ValueError('HEAD changed')
    for section in ('source_files','metadata_files'):
        for p,d in index[section].items():
            rel=Path(p)
            if rel.is_absolute() or '..' in rel.parts or (ROOT/rel).is_symlink() or sha(ROOT/rel)!=d:raise ValueError('Bound file changed: '+p)
    historical()
    return index
