"""Terminal-only stdlib projection; never opens patient arrays or weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
EXPECTED_RELEASE='7c0d91b63c5922285976660ffc4d9ac2d7377ab8a8147ec39b021eb64fc80db7'
EXPECTED_INDEX='83afce441ab2fd109e7e7e83b76d6ff20450f719838bba03096a4352622cc54e'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def need(value,message):
    if not value:raise ValueError(message)
def ref(path):return {'path':path.relative_to(ROOT).as_posix(),'sha256':sha(path),'bytes':path.stat().st_size}
def counts(rows,key):return dict(sorted(Counter(row[key] for row in rows).items()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-sha256',required=True);parser.add_argument('--receipt-sha256',required=True)
    args=parser.parse_args();out=HERE/'attempt-01';sup=HERE/'attempt-01.supervision';post=HERE/'postrun'
    need(not post.exists(),'Preserve prior extraction; no overwrite')
    receipt=read(sup/'receipt.json')
    need(sha(sup/'receipt.json')==args.receipt_sha256,'Receipt pin differs')
    need(receipt['exit_code'] is not None and receipt['worker_termination_confirmed'] and not receipt['cleanup_errors'] and not receipt['final_owned_pids'],'No clean terminal child evidence')
    release=read(HERE/'root-release.json')
    need(sha(HERE/'root-release.json')==EXPECTED_RELEASE==receipt['release_sha256'],'Root release pin differs')
    need(release['source_index']['sha256']==EXPECTED_INDEX and sha(ROOT/release['source_index']['path'])==EXPECTED_INDEX,'Executed index differs')
    need(sha(out/'result.json')==args.result_sha256==receipt['result_sha256'],'Result pin differs')
    result=read(out/'result.json');costs=read(out/'costs.json')
    projection={'result':ref(out/'result.json'),'receipt':ref(sup/'receipt.json'),
        'release':ref(HERE/'root-release.json'),'source_index':ref(ROOT/release['source_index']['path']),
        'status':result['status'],'parent_status':receipt['status'],'failure':result.get('failure'),
        'complete_route_claim':False,'new_rule_generated_prefix':False,
        'scope':'old47 negative prefix with separately declared one-prep prospective catalog branches; preview feasibility is not executed route utility',
        'costs':{'parent_seconds':receipt['elapsed_seconds'],'sampled_peak_rss_bytes':receipt['sampled_peak_rss_bytes'],
            'worker_seconds':result['wall_seconds'],'native_previews':result['native_budget']['native_preview_entries'],
            'native_commits':result['native_commits'],'native_commit_attempts':result['native_commit_attempts'],
            'STOP_calls':result['STOP_calls'],'checkpoint_loads':result['checkpoint_loads'],'model_forwards':result['model_forwards'],
            'updates':result['updates'],'phases':costs['phases'],'output_bytes':receipt['output_bytes'],
            'supervision_bytes_before_receipt':receipt['supervision_bytes_before_receipt']},'branches':[]}
    if result['status']=='complete_prospective_residual_frontier_diagnostic':
        need(receipt['status']=='complete','Complete result without accepted owned receipt')
        old=read(out/'old-frontier/page00.json');catalog=read(out/'prospective/prospective-catalog.json')
        extra=[read(p) for p in sorted((out/'prospective').glob('extra-*.json'))]
        old_rows=[r for r in old['ledger'] if r['family']=='obstruction_opening']
        new_rows=[r for r in catalog['ledger'] if r['family']=='obstruction_opening']
        old_dispositions={(tuple(r['blocker_voxel']),r['tool_id']):r['proposal_reason'] for r in old_rows}
        changed=[]
        for row in new_rows:
            key=(tuple(row['blocker_voxel']),row['tool_id']);before=old_dispositions.get(key)
            if before!=row['reason']:changed.append({'voxel':key[0],'tool':key[1],'old':before,'new':row['reason']})
        old_verified={}
        for p in sorted((out/'old-frontier').glob('page*.json')):
            for row in read(p)['ledger']:
                if row['family']=='obstruction_opening' and row['verification']=='NATIVE_VERIFIED':old_verified[row['action_id']]=row
        need(len(extra)==result['extra_count'] and result['all_extra_pairs_verified'],'Extra verification denominator mismatch')
        projection['catalog']={'old':{'accounting':old['obstruction_accounting'],'dispositions':counts(old_rows,'proposal_reason'),
                'verified_extra_reasons':counts(list(old_verified.values()),'reason'),'legal_extras':sum(r['feasible'] for r in old_verified.values())},
            'prospective':{'accounting':catalog['accounting'],'dispositions':counts(new_rows,'reason'),
                'verified_extra_reasons':counts(extra,'reason'),'legal_extras':sum(r['feasible'] for r in extra),
                'all_extra_pairs_verified':True,'omitted_pairs':result['omitted_pairs']},'changed_eligibility':changed,
            'first_failure_scope':'bounded current first rejected interval cells, not complete corridor or minimal required set'}
        projection['old_prefix']={'actions':47,'return':result['prefix_return'],
            'closed_with_STOP':result['original_negative_prefix_closed'],'root_STOP_return':0.,
            'all_checkpoint_progress_exact':result['prefix_exact'],'final_old_page_exact':result['frontier_page_exact']}
        for i in range(result['branch_count']):
            branch=read(out/('prospective/branch-%02d/result.json'%i))
            audit=read(out/('independent/branch-%02d.json'%i))
            scored=branch['nominal_prep'];rays=[read(p) for p in sorted((out/('prospective/branch-%02d'%i)).glob('ray-*.json'))]
            need(len(rays)==result['base_count'],'Original ray re-preview denominator differs')
            projection['branches'].append({'index':i,'ordinal':branch['extra_ordinal'],
                'actual_first_blockers_removed':branch['first_blockers_removed'],
                'remaining_sampled_first_blockers':branch['remaining_sampled_first_blockers'],
                'prep_target_mm3':scored['target_removed_mm3'],'prep_outside_target_mm3':scored['normal_removed_mm3'],
                'prep_reward':scored['reward'],'prefix_plus_prep_STOP_return':branch['prefix_plus_prep_STOP_return'],
                'newly_feasible_original_rays':branch['newly_feasible_original_rays'],
                'reason_changes':branch['reason_changes'],'after_reasons':counts(rays,'reason'),
                'repreviewed_original_rays':len(rays),'native_replay_exact':branch['deterministic_native_commit_replay_exact'],
                'independent':{'file':ref(out/('independent/branch-%02d.json'%i)),**audit},'accepted_plan_claim':False})
        projection['independent_geometry']={'checks':result['independent_geometry_checks'],
            'all_feasible':result['independent_geometry_all_feasible'],'original_prefix':read(out/'independent/original-prefix.json')}
        need(result['native_commits']==47+2*len(projection['branches']) and result['STOP_calls']==1,'Commit counts differ')
        need(result['independent_geometry_checks']==1+len(projection['branches']),'Independent check denominator differs')
    # Lean archive preserves all decisive sidecars/certificates and source pins,
    # avoids duplicating per-step attempt/return files already in full native history.
    output_json=sorted(p for p in out.rglob('*.json') if 'prefix' not in p.relative_to(out).parts)
    metadata=[HERE/name for name in ('comparison_contract.py','residual_frontier.py','select_worker.py','run_owned.py',
        'freeze-runtime.py','extract_terminal.py','root-release.json','runtime-source-index.json','runner-pins.json','HANDOFF.txt',
        'canonical-controls-01.json','canonical-controls-01.log','helper-controls-01.json','helper-controls-01.log',
        'controls-01.json','controls-01.log','controls-02.json','controls-02.log','promotion-index.json','source.patch')]
    supervision=[sup/name for name in ('receipt.json','declaration.json','worker-final.json','worker.log')]
    archived=sorted(set(output_json+metadata+supervision))
    archived=[p for p in archived if p.is_file()]
    projection['lean_archive']={'files':[ref(p) for p in archived],'file_count':len(archived),'bytes':sum(p.stat().st_size for p in archived),
        'excluded_prefix_files':[ref(p) for p in sorted((out/'prefix').glob('*.json'))],
        'arrays_or_weights':False,'copies_made':False}
    post.mkdir()
    (post/'terminal-projection.json').write_text(json.dumps(projection,indent=2,sort_keys=True)+'\n')
    compact={'status':projection['status'],'parent':projection['parent_status'],'costs':projection['costs'],
        'old_dispositions':projection.get('catalog',{}).get('old',{}).get('dispositions'),
        'new_dispositions':projection.get('catalog',{}).get('prospective',{}).get('dispositions'),
        'branches':[{k:v for k,v in b.items() if k not in ('independent','remaining_sampled_first_blockers','newly_feasible_original_rays','reason_changes')}
            | {'newly_feasible_count':len(b['newly_feasible_original_rays'])} for b in projection['branches']],
        'archive_files':len(archived),'archive_bytes':projection['lean_archive']['bytes'],'accepted_plan_claim':False}
    (post/'compact-scalars.json').write_text(json.dumps(compact,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'projection':ref(post/'terminal-projection.json'),'compact':ref(post/'compact-scalars.json'),**compact}))
if __name__=='__main__':main()
