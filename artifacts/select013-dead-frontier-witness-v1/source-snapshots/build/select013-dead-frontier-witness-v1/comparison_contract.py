"""One fixed public prefix witness; historical lineage is metadata, never a model capability."""
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
OUTPUT='build/select013-dead-frontier-witness-v1/attempt-01'
ARMS=('fixed_dead_frontier',)
WORKER_SECONDS,PARENT_SECONDS=180,210
MEMORY_BYTES,OUTPUT_BYTES,SUPERVISION_BYTES=3*1024**3,32*1024**2,16*1024**2
LIMITS={'native_previews':2000,'fixed_prefix_commits':11,'closure_STOP_calls':1,'diagnostic_previews':120,
    'cells_per_sidecar':32,'classified_unique_cells':128,'threads':1,'model_forwards':0,'updates':0}
PINS={
 'result':{'path':'build/select013-lazy-beam-recovery-v1/attempt-01/result.json','sha256':'96e0b98069689a7ad6f1b986423f527ce931e04034b33605d3ebaf48e9e9d761'},
 'selection':{'path':'build/select013-lazy-beam-recovery-v1/attempt-01/observed_beam/selection.json','sha256':'8f62656d1081ea247499cb1ff7ed8b803bf2249d3e700703d3806506a151c8f4'},
 'receipt':{'path':'build/select013-lazy-beam-recovery-v1/attempt-01.supervision/receipt.json','sha256':'83031cb5bdeb9c9ee6bd47dcf53f28e3c7ea7f83207bc94b174d8501b994f232'},
 'release':{'path':'build/select013-lazy-beam-recovery-v1/root-release.json','sha256':'43e7cdf0e43a014154cc15236bbb4df1818ed1577b3cee5b8d75ac368e9aa3d9'},
}

PINS.update({'world': {'path': 'build/select013-lazy-beam-recovery-v1/attempt-01/observed_beam/comparison-world.json', 'sha256': '2dc67df3216033c72540cf2d83e3c614f3a5e62a2fb31ad3cf8c653925213995'}, 'lineage': {'path': 'build/select013-lazy-beam-recovery-v1/attempt-01/IL64-lineage.json', 'sha256': 'e1fe8425d052af596f3a830645e306c47b85f135264e39f81a45f632c3ae3172'}, 'context': {'path': 'build/select013-lazy-beam-recovery-v1/attempt-01/observed_beam/context.json', 'sha256': '178b81f87c8363a3f0c55be411b2085739e349415bbacc73a0fe3e90ad124887'}})

def historical():
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

def complete_result(r):
    return (r.get('status')=='complete_fixed_dead_frontier_diagnostic' and r.get('world_equal') is True
        and r.get('prefix_exact') is True and r.get('committed_actions')==11
        and r.get('diagnostic_STOP_calls')==1 and r.get('complete_episode') is True
        and r.get('final_legal_motions')==0 and r.get('accepted_plan_claim') is False
        and r.get('checkpoint_loads')==r.get('model_forwards')==r.get('updates')==0
        and r.get('private_reference_reads')==0 and r.get('held_inputs_opened') is False
        and r.get('state_unchanged_by_diagnostics') is True)

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
