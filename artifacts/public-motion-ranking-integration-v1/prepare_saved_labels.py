"""Read only the fixed saved TRAIN JSON; no project/model/array imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SUBJECTS = ('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045')
STEPS = (1,14,1,13)

def digest(value):
    return 'sha256:'+hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def main():
    pins = {}; cases = []; summaries = []
    for subject,count in zip(SUBJECTS,STEPS):
        old = ROOT/'build/post-exposure-learning-v1/IL64/attempt-01/teachers'/subject
        paths = {'scores':ROOT/'build/remind-post-exposure-feasibility-v1/attempt-01'/subject/'greedy-plan.json',
                 'trace':old/'complete-trace.json','plan':old/'plan.json','replay':old/'native-replay.json'}
        data = {}; hashes = {}
        for key,path in paths.items():
            raw=path.read_bytes(); hashes[key]=hashlib.sha256(raw).hexdigest();data[key]=json.loads(raw)
            pins[str(path.relative_to(ROOT))]={'sha256':hashes[key],'bytes':len(raw)}
        search=data['scores']; trace=data['trace']; plan=data['plan']['plan']; replay=data['replay']
        assert data['plan']['plan_seal']==digest(plan)
        assert replay['independent_geometry']['accepted'] is True and replay['independent_geometry']['complete_episode'] is True
        # The standard native replay uses the evaluator label for its equal
        # reference slot. Authenticate every field except this explicit scope
        # label; ranking rewards come ONLY from the public greedy score record.
        for public,checked in zip(plan['history'],replay['metrics']['history']):
            if public['action_id']!='STOP':
                assert public['outcome_scope']=='permitted_nominal_model'
                assert checked['outcome_scope']=='separate_evaluator_reference'
            assert {k:v for k,v in public.items() if k!='outcome_scope'}=={k:v for k,v in checked.items() if k!='outcome_scope'}
        assert len(replay['metrics']['history'])==len(plan['history'])
        assert replay['metrics']['source_hash']==plan['source_hash']
        assert replay['metrics']['decision_model_hash']==plan['decision_model_hash']
        a=search['accounting']; rows=[]; pairs=ties=0
        assert a['complete'] is True and a['method']=='observed_greedy'
        assert a['objective_source']=='permitted_nominal_target_and_frozen_geometric_costs'
        assert search['actions']==plan['actions'] and len(search['actions'])==count
        assert len(trace['decisions'])==len(a['decisions'])==len(plan['history'])==count
        for step,(decision,observed,native) in enumerate(zip(a['decisions'],trace['decisions'],plan['history'])):
            assert decision['step']==observed['step']==step
            assert decision['all_current_legal_actions_scored'] is True
            scores=decision['scores']; ids=observed['action_ids'];mask=observed['action_mask']
            assert [s['action_id'] for s in scores]==[a for a,m in zip(ids,mask) if m]
            assert decision['legal_nonstop_actions']==decision['scored_nonstop_actions']==len(scores)-1
            winner=max(scores,key=lambda x:x['reward'])
            assert winner['action_id']==decision['selected_action_id']==observed['action_id']==native['action_id']==plan['actions'][step]
            assert winner['reward']==observed['reward']==native['reward']
            assert observed['behavior_parameter_hash'] is None
            if native['action_id']!='STOP':
                assert native['source_state_hash']==decision['source_state_hash']
                assert native['source_hash']==plan['source_hash']
                # Stroke decision_model_hash is the engine config, while the
                # sealed plan header is the broader task model identity.
                assert native['decision_model_hash']==plan['history'][0]['decision_model_hash']
                assert native['outcome_scope']=='permitted_nominal_model'
            reward={s['action_id']:s['reward'] for s in scores}
            rows.append({'step':step,'observation_hash':observed['observation_hash'],
                'source_state_hash':decision['source_state_hash'],'action_ids':ids,'action_mask':mask,
                'teacher_action':observed['action_id'],'rewards':[reward[x] if m else None for x,m in zip(ids,mask)]})
            if native['action_id']!='STOP':
                r=[s['reward'] for s in scores if s['action_id']!='STOP']
                pairs+=sum(x>y for x in r for y in r)
                ties+=sum(r[i]==r[j] for i in range(len(r)) for j in range(i))
        cases.append({'subject':subject,'role':'TRAIN','source_hash':plan['source_hash'],
            'decision_model_hash':plan['decision_model_hash'],'input_sha256':hashes,'decisions':rows})
        summaries.append({'subject':subject,'states':count,'motion_states':count-1,
            'legal_score_rows':sum(sum(r['action_mask']) for r in rows),
            'strict_motion_pairs_per_update':pairs,'exact_tied_motion_pairs':ties})
    corpus={'version':'public_nominal_motion_gap_ranking_v1',
        'scope':'same_fixed_TRAIN_teacher_states_public_nominal_immediate_rewards',
        'objective_source':'permitted_nominal_target_and_frozen_geometric_costs',
        'private_reference_used':False,'subjects':cases}
    summary={'scope':'metadata joins only; no new scientific execution','corpus_hash':digest(corpus),
        'cases':summaries,'states':sum(x['states'] for x in summaries),
        'strict_motion_pairs_per_update':sum(x['strict_motion_pairs_per_update'] for x in summaries),
        'loss_forwards_per_update':29,'fixed_updates':64,'new_native_previews_for_labels':0,
        'added_arrays_or_policy_input_channels':0}
    for name,value in [('input-pins.json',pins),('public-score-corpus.json',corpus),('label-summary.json',summary)]:
        with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(summary,sort_keys=True))

if __name__=='__main__':main()
