"""Terminal-only scalar projection. Never imports a runtime/model/array reader.

Invoke only after root reports terminal and supplies exact result/receipt pins.
This author projection is not a substitute for the separate independent audit.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat

STAGE=Path(__file__).resolve().parents[1]
ROOT=STAGE.parents[1]
SUBJECTS=['ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045','ReMIND-050']
METHODS=['switch_guard','public_bound']
OUTCOME_KEYS=['target_removed_mm3','normal_removed_mm3','total_reward','nonstop_actions',
    'tool_changes','complete_tool_path_length_mm','total_reference_target_mm3',
    'cumulative_contacted_tissue_upper_bound_mm3','currently_retained_contacted_tissue_upper_bound_mm3',
    'clinical_deficit_probability','partial_contact_reward_weight']

def require(value,message):
    if not value:raise ValueError(message)

def digest(value):
    require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'exact SHA required')
    return value

def read(path,expected=None):
    path=Path(path)
    fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
    try:
        s=os.fstat(fd);require(stat.S_ISREG(s.st_mode) and 0<s.st_size<=8*1024**2,'bounded regular saved JSON only')
        with os.fdopen(fd,'rb',closefd=False) as f:raw=f.read(8*1024**2+1)
        require(len(raw)==s.st_size,'saved file changed')
    finally:os.close(fd)
    sha=hashlib.sha256(raw).hexdigest()
    if expected is not None:require(sha==digest(expected),'saved pin mismatch: '+str(path))
    value=json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    return value,{'path':str(path.relative_to(ROOT)),'sha256':sha,'bytes':len(raw)}

def output_path(relative):
    p=Path(relative)
    require(not p.is_absolute() and '..' not in p.parts,'output reference must stay under attempt')
    return STAGE/'attempt-01'/p

def costs(row,phase):
    c=row.get('online_costs' if phase=='planning' else 'replay_costs',{})
    count=c.get('counts',{});seconds=c.get('seconds',{})
    return {'counts':count,'seconds':seconds,
        'bound_inclusive_setup_and_queries_seconds':seconds.get('public_bound_setup',0.)+seconds.get('public_bound_queries',0.),
        'inclusive_inventory_seconds':seconds.get('inventory_inclusive',0.),
        'public_snapshot':{'states':count.get('public_bound_states',0),
            'cumulative_bytes_copied':count.get('public_bound_snapshot_bytes',0),
            'cumulative_positive_cells':count.get('public_bound_snapshot_target_cells',0),
            'cumulative_unknown_cells':count.get('public_bound_snapshot_unknown_cells',0),
            'per_snapshot_cap_bytes':64*1024**2,
            'scope':'cumulative repeated snapshots, not simultaneous resident memory'},
        'timing_scope':'inventory includes setup/query/scoring; do not add inclusive and component durations'}

def inventories(row,phase):
    folder=STAGE/'attempt-01'/row['method']/row['subject']
    if phase=='replay':folder/='replay'
    records=[];pins=[];reasons=Counter();pruned=bounded=unbounded=queries=0
    for path in sorted((folder/'inventories').glob('state-*.json')):
        event,pin=read(path);pins.append(pin);q=event['shortlist'];bounds=q.get('public_reward_bounds',[])
        selected=q['selected_ids'];raw=q['raw_ids']
        require(len(selected)==len(set(selected))==q['queried_count'],'unique paid queries')
        require(len(raw)==len(set(raw))==q['raw_count'] and set(selected)<=set(raw),'raw catalog/query join')
        seen=set();state_reasons=Counter();state_pruned=0
        for b in bounds:
            identifier=b['proposal_id'];require(identifier in raw and identifier not in seen,'one bound per raw candidate')
            seen.add(identifier);u=b['reward_upper'];incumbent=b['incumbent_reward']
            require(math.isfinite(incumbent) and incumbent>=0,'finite known incumbent')
            if u is None:
                unbounded+=1;require(not b['pruned'] and identifier in selected,'unbounded retains exact preview')
            else:
                require(math.isfinite(u),'finite upper bound');bounded+=1
            if b['pruned']:
                require(u is not None and identifier not in selected and (u<=0 if incumbent==0 else u<incumbent),
                        'prune must preserve STOP/positive ties')
                state_pruned+=1
            else:require(identifier in selected,'unpruned conditional action queried')
            reasons[b['reason']]+=1;state_reasons[b['reason']]+=1
        require(state_pruned==q.get('bound_pruned_count',0),'state prune accounting')
        pruned+=state_pruned;queries+=len(selected)
        records.append({'step':event['step'],'raw_count':len(raw),'queried_count':len(selected),
            'bound_count':len(bounds),'bound_pruned_count':state_pruned,'bound_reasons':dict(state_reasons),
            'initial_best_positive_action':q.get('initial_best_positive_action'),
            'fallback_triggered':q.get('fallback_triggered'),
            'current_tool_comparison_triggered':q.get('current_tool_comparison_triggered')})
    require(len(records)<=24,'h24 inventory denominator')
    c=costs(row,phase)['counts']
    require(queries==c.get('native_previews',0),'inventory paid queries match local costs')
    require(pruned==c.get('public_bound_pruned',0),'prunes match local costs')
    require(unbounded==c.get('public_bound_unavailable',0),'unbounded match local costs')
    require(bounded+unbounded==c.get('public_bound_queries',0),'all bounds match local costs')
    return {'states':records,'native_queries':queries,'bound_queries':bounded+unbounded,
        'bounded':bounded,'unbounded':unbounded,'pruned':pruned,'reasons':dict(reasons)},pins

def main():
    p=argparse.ArgumentParser();p.add_argument('--result-sha256',required=True);p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--release-sha256',required=True);p.add_argument('--index-sha256',required=True)
    p.add_argument('--head',required=True)
    p.add_argument('--out',default='extract-attempt-01');a=p.parse_args()
    RELEASE_SHA=digest(a.release_sha256);INDEX_SHA=digest(a.index_sha256)
    require(len(a.head)==40 and all(c in '0123456789abcdef' for c in a.head),'exact root terminal HEAD required')
    require(Path(a.out).name==a.out,'single output directory name')
    out=Path(__file__).resolve().parent/a.out;out.mkdir(exist_ok=False)
    pins=[]
    def get(path,expected=None):
        value,pin=read(path,expected);pins.append(pin);return value
    try:
        result=get(STAGE/'attempt-01/result.json',a.result_sha256)
        receipt=get(STAGE/'attempt-01.supervision/receipt.json',a.receipt_sha256)
        release=get(STAGE/'root-release.json',RELEASE_SHA)
        index=get(STAGE/'source-index.json',INDEX_SHA)
        require(receipt['result_sha256']==a.result_sha256 and receipt['release_sha256']==RELEASE_SHA,'terminal joins')
        require(receipt['worker_termination_confirmed'] and not receipt['remaining_owned_pids'] and not receipt['cleanup_errors'],
                'owned terminal cleanup must be explicit')
        require(receipt['source_index']['sha256']==INDEX_SHA and release['source_index']['sha256']==INDEX_SHA,'frozen index joins')
        require(index['head']==release['expected_head']==a.head,'source HEAD join')
        require(release['version']=='five-TRAIN-public-reward-window-comparison-v2'
                and release['policy_version']=='nominal-top8-current-tool-public-window-v2','explicit local-window successor')
        require(result['subjects']==SUBJECTS and [(r['subject'],r['method']) for r in result['arms']]
                ==[(s,m) for s in SUBJECTS for m in METHODS],'complete declared five-pair denominator')
        summary={'status':result['status'],'parent_status':receipt['status'],'result_pin':pins[0],
            'scope':'author scalar projection; separately independently audited; TRAIN development only',
            'release_sha256':RELEASE_SHA,'source_index_sha256':INDEX_SHA,'head':index['head'],
            'parent':receipt,'all_plans_sealed_before_evaluation':result['all_plans_sealed_before_evaluation'],
            'fits':result['fits'],'optimizer_calls':result['optimizer_calls'],'neural_calls':result['model_calls'],
            'checkpoint_loads':result['checkpoint_loads'],'ridge_loads':result['ridge_loads'],
            'ridge_unchanged':result['ridge_unchanged'],'cache_restored':result['cache_restored'],
            'source_visits':result['source_visits'],'native_transitions':result['native_transitions'],
            'native_previews':result['native_previews'],'ridge_calls':result.get('ridge_calls'),
            'worker_seconds':result['elapsed_seconds'],'pairs':[],'clinical_validation':None,
            'limitations':['fixed-order one-pair-per-subject timing, not a causal or population speed claim',
                'bound is optimistic nominal target potential, not feasibility, physical safety or learned prediction',
                'unknown/unsupported/workspace-refused cases retain exact previews',
                'global unknown-cell count does not imply unknown coverage in a local query window',
                'snapshot byte counts accumulate across states, while parent RSS is whole-process sampled peak',
                'all full arrays, model parameters and inventory payloads stay local; only hashes/scalars projected']}
        require(summary['fits']==summary['optimizer_calls']==summary['neural_calls']==summary['checkpoint_loads']==0,
                'no learning or neural execution')
        for subject in SUBJECTS:
            pair={'subject':subject,'arms':[]};saved=[]
            for method in METHODS:
                row=next(r for r in result['arms'] if r['subject']==subject and r['method']==method)
                projection={'method':method,'status':row['status'],'complete':row['complete'],
                    'independent_accepted':row['independent_accepted'],'files':row['files'],
                    'costs':{phase:costs(row,phase) for phase in ('planning','replay')}}
                for k in ('source_preparation_seconds','cache_construction_seconds','source_through_plan_seconds',
                          'replay_source_preparation_seconds','replay_and_audit_seconds','online_previews','replay_previews'):
                    projection[k]=row.get(k)
                if row['complete']:
                    paths=row['files']
                    def joined(suffix):
                        key=next(k for k in paths if k.endswith(suffix));return get(output_path(key),paths[key])
                    plan=joined('/plan.json');metrics=joined('/episode-metrics.json');cert=joined('/independent-episode.json')
                    require(cert['accepted']==row['independent_accepted'] and cert['outcomes']==row['outcomes'],'accepted certificate/outcome joins')
                    projection['outcomes']={k:row['outcomes'][k] for k in OUTCOME_KEYS}
                    projection['actions']=plan['actions'];projection['plan_seal']=plan['seal']
                    projection['inventories']={}
                    for phase in ('planning','replay'):
                        record,invpins=inventories(row,phase);pins.extend(invpins);projection['inventories'][phase]=record
                    saved.append((plan,metrics,row['outcomes']))
                pair['arms'].append(projection)
            if len(saved)==2:
                bp,bm,bo=saved[0];up,um,uo=saved[1]
                normalized={**um,'decision_model_hash':bm['decision_model_hash']}
                pair['exact_physical_equivalence']={'actions':bp['actions']==up['actions'],
                    'history':bm['history']==um['history'],'metrics_except_outer_policy_identity':bm==normalized,'outcomes':bo==uo}
                b,u=pair['arms']
                pair['online_previews_saved']=b['online_previews']-u['online_previews']
                pair['source_through_plan_ratio']=u['source_through_plan_seconds']/b['source_through_plan_seconds']
            summary['pairs'].append(pair)
        totals={}
        for method in METHODS:
            rows=[r for pair in summary['pairs'] for r in pair['arms'] if r['method']==method]
            phases={}
            for phase in ('planning','replay'):
                inv=[r.get('inventories',{}).get(phase,{}) for r in rows]
                reasons=Counter()
                for v in inv:reasons.update(v.get('reasons',{}))
                phases[phase]={k:sum(v.get(k,0) for v in inv) for k in
                    ('native_queries','bound_queries','bounded','unbounded','pruned')}
                phases[phase]['reasons']=dict(reasons)
                phases[phase]['bound_setup_seconds']=sum(r['costs'][phase]['seconds'].get('public_bound_setup',0.) for r in rows)
                phases[phase]['bound_query_seconds']=sum(r['costs'][phase]['seconds'].get('public_bound_queries',0.) for r in rows)
                phases[phase]['snapshot_bytes_cumulative']=sum(r['costs'][phase]['public_snapshot']['cumulative_bytes_copied'] for r in rows)
            totals[method]={'completed_arms':sum(bool(r['complete']) for r in rows),
                'source_through_plan_seconds':sum(r.get('source_through_plan_seconds') or 0. for r in rows),
                'replay_and_audit_seconds':sum(r.get('replay_and_audit_seconds') or 0. for r in rows),
                'phases':phases}
        summary['totals']=totals
        # Independent arithmetic from each saved pair; do not trust result's flag.
        summary['all_pairs_exact']=all(all(p.get('exact_physical_equivalence',{}).values())
            and len(p.get('exact_physical_equivalence',{}))==4 for p in summary['pairs'])
        summary['complete_ten_arms']=sum(x['completed_arms'] for x in totals.values())==10
        write=lambda path,value:path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
        write(out/'summary.json',summary);write(out/'input-pins.json',pins)
        archive=[{'path':str((out/name).relative_to(ROOT)),'sha256':hashlib.sha256((out/name).read_bytes()).hexdigest()}
                 for name in ('summary.json','input-pins.json')]
        archive += [pin for pin in pins[:4]]
        # Keep complete plans/metrics/certificates; bulky inventory ledgers stay
        # local with exact pins. No parameter, image, source-array or cache copy.
        archive += [pin for pin in pins[4:] if Path(pin['path']).name in
                    ('plan.json','episode-metrics.json','independent-episode.json')]
        for name in ('sealed-plans.json','worker-final.json'):
            path=STAGE/'attempt-01'/name
            if path.is_file():
                _,pin=read(path);pins.append(pin);archive.append(pin)
        write(out/'input-pins.json',pins)
        archive[1]['sha256']=hashlib.sha256((out/'input-pins.json').read_bytes()).hexdigest()
        archive.append({'path':str(Path(__file__).resolve().relative_to(ROOT)),
                        'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        write(out/'compact-archive-list.json',archive)
        print(json.dumps({'status':summary['status'],'summary':str(out/'summary.json'),
                          'pairs':[(x['subject'],x.get('exact_physical_equivalence'),x.get('online_previews_saved')) for x in summary['pairs']]}))
    except BaseException as error:
        (out/'extract-failure.json').write_text(json.dumps({'type':type(error).__name__,'message':str(error),'pins_read':pins},indent=2)+'\n')
        raise

if __name__=='__main__':
    def timeout(*_):raise TimeoutError('30-second saved-JSON projection limit')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,30)
    try:main()
    finally:signal.setitimer(signal.ITIMER_REAL,0)
