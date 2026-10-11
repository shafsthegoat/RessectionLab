"""Terminal-only stdlib scalar projection and lean file list; not a second audit.

No scientific imports, payloads, models, native calls, or copies of inventories.
Call only after root supplies actual terminal result/receipt byte hashes.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
ARMS=('original_lazy_beam','paged_obstruction_lazy_beam')
RELEASE='a957077b04785355b3fd1c8354743526e4e70ae6087de249e325386badad8e3e'
INDEX='23bf8d7bb299a05f7f24dfa5d372b6aec6ee4f0f47a88e1f007a8109d458526f'
HEAD='7cd2d61aaf49bcc89de12cd0e957b7cdab00efb6'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p,expected=None):
    if not p.is_file() or p.is_symlink():raise ValueError('Regular closed metadata required: '+str(p))
    if expected is not None and sha(p)!=expected:raise ValueError('Exact hash mismatch: '+str(p))
    return json.loads(p.read_text())
def ref(p):return {'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--receipt-sha256',required=True)
    ap.add_argument('--result-sha256');ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    attempt=HERE/'attempt-01';sup=HERE/'attempt-01.supervision'
    receipt=read(sup/'receipt.json',args.receipt_sha256)
    if not receipt.get('worker_termination_confirmed') or receipt.get('final_owned_pids') or receipt.get('cleanup_errors'):
        raise ValueError('Root must resolve incomplete process cleanup before projection')
    if receipt.get('release_sha256')!=RELEASE or receipt.get('source_index',{}).get('sha256')!=INDEX:
        raise ValueError('Unmatched released attempt')
    read(HERE/'root-release.json',RELEASE);read(HERE/'source-index.json',INDEX)
    result_path=attempt/'result.json';result=None
    if result_path.exists():
        if not args.result_sha256 or receipt.get('result_sha256')!=args.result_sha256:raise ValueError('Existing result must be pinned by root and terminal receipt')
        result=read(result_path,args.result_sha256)
    elif args.result_sha256 or receipt.get('result_sha256'):raise ValueError('Pinned result absent')
    r=result or {};out={'scope':'saved scalar projection, not a new independent geometry audit',
        'result_available':result is not None,'status':r.get('status'),'parent_status':receipt['status'],
        'result_sha256':args.result_sha256,'receipt_sha256':args.receipt_sha256,'executed_head':HEAD,
        'release_sha256':RELEASE,'source_index_sha256':INDEX,'held037':True,'EVAL067_closed':True,
        'parent_wall_seconds':receipt.get('elapsed_seconds'),'sampled_peak_rss_bytes':receipt.get('sampled_peak_rss_bytes'),
        'arms':{},'comparison':r.get('comparison'),'failure':r.get('failure'),
        'limits':'depth/frontier summaries cover recorded retained/terminal prefixes only; not every evaluated physical outcome'}
    all_inventory_refs=[]
    for arm in ARMS:
        row=r.get('arms',{}).get(arm,{});a=row.get('selector_accounting',{});layers=a.get('layers',[])
        trace=[]
        for layer in layers:
            diagnostics=layer.get('retained_prefix_diagnostics',{})
            trace.append({'depth':layer['depth'],'completed':layer['completed'],
                'evaluated_actions':layer['model_transition_calls'],
                'retention':layer.get('opening_depth_retention',[]),
                'retained':diagnostics.get('retained_nonterminal_prefixes',[]),
                'best_terminal':diagnostics.get('best_terminal_prefixes',[])})
        counters=Counter();omissions=Counter();catalogs=[]
        # Only entered after authenticated terminal receipt. Full ledgers remain
        # local; their exact byte hashes and small scalar facts survive archive.
        paths=sorted((attempt/arm/'pages').glob('*/*.json')) if arm==ARMS[1] else sorted((attempt/arm/'inventories').glob('*.json'))
        for path in paths:
            inv=read(path);all_inventory_refs.append(ref(path))
            for cell in inv.get('ledger',[]):
                verified=(cell.get('verification')=='NATIVE_VERIFIED' if arm==ARMS[1] else cell.get('proposal_reason')=='PROPOSED_UNCERTIFIED')
                if verified:counters[(cell.get('family'),cell.get('feasible'),cell.get('reason'))]+=1
                if inv.get('page_index',0)==0 and cell.get('proposal_reason') in ('CANDIDATE_CAP','OBSTRUCTION_SELECTION_CAP'):
                    omissions[cell['proposal_reason']]+=1
            if inv.get('page_index',0)==0:
                catalogs.append({'file':ref(path),'source_state_hash':inv.get('source_state_hash'),
                    'steps_taken':inv.get('steps_taken'),'obstruction_accounting':inv.get('obstruction_accounting')})
        out['arms'][arm]={'status':row.get('status','unavailable'),'outcomes':row.get('outcomes'),
            'actions':row.get('actions'),'native_transitions':row.get('native_transitions'),
            'full_previews':r.get('native_previews_by_arm',{}).get(arm),'completed_layers':a.get('completed_layers'),
            'expanded_nodes':a.get('expanded_nodes'),'planning_seconds':a.get('planning_seconds'),
            'call_cap':a.get('call_cap_reached'),'time_cap':a.get('time_cap_reached'),
            'negative_evaluated':a.get('negative_prefixes_evaluated'),'evaluated':a.get('model_transition_calls'),
            'h24_prefix_control':row.get('h24_prefix_control'),'trace':trace,
            'native_dispositions':[{'family':k[0],'feasible':k[1],'reason':k[2],'count':n} for k,n in sorted(counters.items(),key=lambda x:str(x[0]))],
            'omissions':dict(omissions),'catalogs':catalogs}
    names=['result.json','costs.json']
    files=[attempt/n for n in names]+[sup/n for n in ['receipt.json','declaration.json','worker-final.json']]
    files += [HERE/n for n in ['root-release.json','source-index.json','comparison_contract.py','select_worker.py','run_owned.py','freeze-runtime.py','test_owned_generated.py','controls-01.json','controls-01.log','controls-02.json','controls-02.log','extract_terminal.py','TERMINAL_RECIPE.txt']]
    for arm in ARMS:
        files += [attempt/arm/n for n in ['original-world-control.json','declared-rule-change.json','horizon-change.json','h24-prefix-control.json','selected-action-plan.json','complete-history.json','rollout-metrics.json','replay-metrics.json','independent-episode.json','search-unresolved.json']]
    out['lean_archive']={'files':[ref(p) for p in files if p.is_file()],
        'inventory_manifest':all_inventory_refs,'omitted':'full inventory/page JSON; duplicate selection accounting; per-step copies where complete history retained; canonical source already in Git',
        'failure_rule':'if a rollout/replay fails before complete-history exists, additionally retain that arm attempt/step JSON and exact failure certificate/log; never discard unfinished evidence'}
    args.output.parent.mkdir(exist_ok=True,parents=True)
    with args.output.open('x') as stream:json.dump(out,stream,indent=2,sort_keys=True);stream.write('\n')
    print(json.dumps({'output':str(args.output),'sha256':sha(args.output),'arms':{k:v['status'] for k,v in out['arms'].items()},'inventory_files':len(all_inventory_refs)}))
if __name__=='__main__':main()
