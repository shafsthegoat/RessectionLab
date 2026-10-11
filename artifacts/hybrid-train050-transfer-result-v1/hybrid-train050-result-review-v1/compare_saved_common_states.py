"""Compare saved certified scores only where exact native state hashes match."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
BASE=ROOT/'build/hybrid-train050-transfer-v1/attempt-01'
PINS={}
def read(path):
    assert path.suffix=='.json' and path.stat().st_size<16*1024**2
    raw=path.read_bytes();PINS[str(path.relative_to(ROOT))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    return json.loads(raw)
def main():
    plans={m:read(BASE/m/'plan.json') for m in ('balanced_ridge_top8','exhaustive')}
    rows=[]
    for i,decision in enumerate(plans['balanced_ridge_top8']['accounting']['decisions']):
        inv=read(BASE/'balanced_ridge_top8/inventories'/f'state-{i:02d}.json');s=inv['shortlist']
        assert inv['source_state_hash']==decision['source_state_hash']
        ids=s['raw_ids'];scores=s['priority_scores'];order=sorted(range(len(ids)),key=lambda k:(-scores[k],k))
        row={'ridge_step':i,'state_hash':decision['source_state_hash'],'ridge_action':decision['selected_action_id'],
            'ridge_best_verified':max(decision['scores'],key=lambda x:x['reward']),
            'queried':s['queried_count'],'raw':s['raw_count'],'legal_queried':s['legal_count'],'same_state_exhaustive':[]}
        for j,other in enumerate(plans['exhaustive']['accounting']['decisions']):
            if other['source_state_hash']!=decision['source_state_hash']:continue
            other_inv=read(BASE/'exhaustive/inventories'/f'state-{j:02d}.json')
            assert other_inv['shortlist']['raw_ids']==ids and other_inv['shortlist']['input_binding']==s['input_binding']
            positives=[x for x in other['scores'] if x['reward']>0]
            choice=other['selected_action_id']
            row['same_state_exhaustive'].append({'step':j,'best_verified':max(other['scores'],key=lambda x:x['reward']),
                'positive_legal_actions':len(positives),'positive_legal_actions_not_queried_by_ridge':sum(x['action_id'] not in s['selected_ids'] for x in positives),
                'ridge_priority_rank_of_exhaustive_choice':None if choice=='STOP' else order.index(ids.index(choice))+1,
                'exhaustive_choice_in_ridge_shortlist':choice in s['selected_ids']})
        rows.append(row)
    report={'status':'SAVED_COMMON_STATE_SHORTLIST_RECALL_DIAGNOSIS','rows':rows,
        'finding':'Both first actions match. At next identical state ridge misses31 of32 positive legal actions, ranking the exhaustive seven-cell choice46th. It takes one target cell then STOPs after zero legal among8 queried.',
        'limit':'No exhaustive counterpart at the final ridge state. No assertion of remaining full-catalog feasibility there; no action replay, score/model recomputation or patient array read.',
        'read_pins':PINS}
    path=HERE/'common-state-diagnosis.json'
    with path.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
