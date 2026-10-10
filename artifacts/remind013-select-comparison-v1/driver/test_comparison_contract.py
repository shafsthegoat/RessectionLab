"""Tiny generated metadata only. No NumPy/Torch, acquired files or model loads."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import comparison_contract as c
from select_worker import greedy_accounting,returned_search_is_capped

class ContractTests(unittest.TestCase):
    def fixture(self, root, name='IL8_unweighted'):
        method='RL' if name=='RL8' else 'IL';updates=64 if name=='IL64_balanced' else 8
        balanced=name in ('IL8_balanced','IL64_balanced')
        directory=Path(name);refs={}
        def save(key,row):
            path=root/directory/(key+'.json');path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(row));ref={'path':path.relative_to(root).as_posix(),'sha256':c.sha(path)}
            refs[key]=ref;return ref
        execution={'patient_order':list(c.TRAIN),'heldout_execution':False,'max_steps':24,
            'proposal_config':{'max_candidates':120},'proposal_rule_hash':'sha256:'+'a'*64,
            'search':c.LIMITS['search'],'retention_mode':c.MODE,'task_condition':'PARTIAL_TARGET_PROGRESS'}
        if balanced:execution['il_teacher_weighting']='balanced_STOP_motion_CE_v1'
        protocol={'cohort_execution':execution,'updates_per_method':updates,'public_target_context_variant':c.TARGET_CONTEXT}
        release={'TRAIN':list(c.TRAIN),'SELECT_EVAL_execution':False,'status':'released_one_attempt',
            'source_index':{'path':'sources.json','sha256':'b'*64},'learning_protocol':protocol,
            'learning_protocol_hash':c.semantic(protocol),'output':directory.as_posix()}
        save('release',release)
        checkpoint={'path':method+'-final.psckpt','bytes':100,'sha256':'c'*64,'parameter_hash':'sha256:'+'d'*64}
        status='complete_balanced_IL64_TRAIN_only' if updates==64 else ('complete_balanced_IL_TRAIN_only' if balanced else 'complete_TRAIN_only_not_heldout_performance')
        result={'status':status,'SELECT_EVAL_opened':False,'private_reference_reads':0,
            'optimizer_updates':{method:updates},'teacher_statuses':{s:'complete_replayed' for s in c.TRAIN},
            'selection_readiness':{'ready':True},'checkpoints':{method:checkpoint}}
        save('result',result)
        worker={'status':'complete_owned_generated','result_sha256':refs['result']['sha256'],
            'canonical_result_sha256':refs['result']['sha256']}
        save('worker_final',worker)
        parent={'status':'complete','exit_code':0,'stop_reason':None,'cleanup_errors':[],
            'final_owned_pids':[],'worker_termination_confirmed':True,'release_sha256':refs['release']['sha256'],
            'result_sha256':refs['result']['sha256'],'worker_final_sha256':refs['worker_final']['sha256'],
            'source_index':release['source_index'],'checkpoints':{method:checkpoint}}
        save('parent',parent)
        contexts={}
        for subject in c.TRAIN:
            row={'subject':subject,'role':'TRAIN','patient_group':'ReMIND:'+subject[-3:],
                'cohort_sha256':c.COHORT_SHA,'learning_protocol_hash':c.semantic(protocol),'max_steps':24,
                'private_reference_in_task':False,'public_target_context_variant':c.TARGET_CONTEXT}
            contexts[subject]=save(subject,row)
            refs.pop(subject)
        refs.update(status='terminal_bound',contexts=contexts,
            checkpoint={'path':str(directory/checkpoint['path']),'sha256':checkpoint['sha256']})
        return refs

    def mutate(self,root,refs,key,change):
        path=root/refs[key]['path'];row=json.loads(path.read_text());change(row)
        path.write_text(json.dumps(row));refs[key]['sha256']=c.sha(path)

    def test_all_predetermined_metadata_chains_and_shared_world(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);release={'endpoints':{n:self.fixture(root,n) for n in c.ENDPOINTS}}
            rows=c.authenticate_comparison(release,root=root)
            self.assertEqual([r['updates'] for r in rows.values()],[8,8,8,64])
            self.assertEqual(len(rows['IL8_balanced']['context_hashes']),4)

    def test_corrupt_or_unclean_terminal_chain_refused(self):
        mutations=[('parent',lambda r:r.update(exit_code=1)),('parent',lambda r:r.update(final_owned_pids=[123])),
            ('worker_final',lambda r:r.update(result_sha256='0'*64)),
            ('result',lambda r:r.update(SELECT_EVAL_opened=True)),
            ('release',lambda r:r['learning_protocol'].update(updates_per_method=7))]
        for key,change in mutations:
            with self.subTest(key=key),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);refs=self.fixture(root);self.mutate(root,refs,key,change)
                with self.assertRaises(ValueError):c.authenticate_endpoint('IL8_unweighted',refs,root=root)

    def test_foreign_context_and_checkpoint_refused(self):
        for kind in ('context','checkpoint'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);refs=self.fixture(root)
                if kind=='checkpoint':refs['checkpoint']['path']='another/IL-final.psckpt'
                else:
                    record=refs['contexts'][c.TRAIN[0]];path=root/record['path']
                    row=json.loads(path.read_text());row['role']='SELECT';path.write_text(json.dumps(row));record['sha256']=c.sha(path)
                with self.assertRaises(ValueError):c.authenticate_endpoint('IL8_unweighted',refs,root=root)

    def test_pending_endpoint_is_not_skipped(self):
        with self.assertRaisesRegex(ValueError,'terminal provenance'):
            c.authenticate_endpoint('IL64_balanced',{'status':'pending_terminal_TRAIN_evidence'})

    def test_world_rejects_proposal_change_even_if_metadata_rebound(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows={n:self.fixture(root,n) for n in c.ENDPOINTS}
            # All internal hashes are regenerated: this is a valid distinct
            # protocol, but cannot enter a same-world comparison.
            changed=c.authenticate_endpoint('IL64_balanced',rows['IL64_balanced'],root=root)['protocol']
            original=copy.deepcopy(changed);changed['cohort_execution']['proposal_config']['max_candidates']=96
            self.assertNotEqual(c.comparison_world(changed),c.comparison_world(original))

    def test_held037_has_no_runnable_path_and_denominator_is_fixed(self):
        active={'patient_id':'ReMIND-013','role':'SELECT','patient_group':'ReMIND:013',
            'status':'PUBLIC_PARTIAL_TARGET_CONDITION_QUALIFIED','full_target_denominator_voxels':35260,
            'supported_target_positive_voxels':1579,'path':'public013.json','sha256':'a'*64}
        held={'patient_id':'ReMIND-037','role':'SELECT','status':'HOLD_ENCODED_SOURCE_SUPPORT_ARTIFACT',
            'path':None,'sha256':None,'replacement':None,'kept_in_planned_denominator':True}
        index={'cases':[active,held],'role':'SELECT','max_optimizer_updates':0,'planned_case_denominator':2}
        self.assertIs(c.select_public(index),active)
        for key,value in [('path','public037.json'),('sha256','a'*64),('replacement','ReMIND-067')]:
            altered=copy.deepcopy(index);altered['cases'][1][key]=value
            with self.assertRaises(ValueError):c.select_public(altered)

    def test_cheap_greedy_completion_must_be_explicit(self):
        raw={'method':'observed_greedy','complete':True,'model_transition_calls':1}
        self.assertEqual(greedy_accounting(raw)['existing_accounting'],raw)
        with self.assertRaises(ValueError):greedy_accounting({**raw,'complete':False})

    def test_returned_beam_call_cap_is_unresolved_before_collector(self):
        self.assertTrue(returned_search_is_capped({'call_cap_reached':True,'time_cap_reached':False}))
        self.assertTrue(returned_search_is_capped({'call_cap_reached':False,'time_cap_reached':True}))
        self.assertFalse(returned_search_is_capped({'call_cap_reached':False,'time_cap_reached':False}))

    def test_incomplete_arm_or_reordered_schedule_is_not_complete_comparison(self):
        row={'status':'complete_public_SELECT013_comparison','planned_SELECT_denominator':2,
            'held_cases':['ReMIND-037'],'optimizer_updates_on_SELECT':0,'private_reference_reads':0,
            'held_inputs_opened':False,'checkpoint_loads':4,'optimizer_attempts':0,'gradient_attempts':0,
            'total_policy_forward_calls':4,
            'EVAL_opened':False,'arm_order':list(c.ARMS),'arms':{a:{'status':'complete_replayed','steps':1} for a in c.ARMS}}
        self.assertTrue(c.complete_result(row))
        for key,value in [('checkpoint_loads',3),('held_inputs_opened',True),('gradient_attempts',1),
                ('optimizer_attempts',1),('total_policy_forward_calls',5)]:
            altered=copy.deepcopy(row);altered[key]=value;self.assertFalse(c.complete_result(altered))
        row['arms']['observed_beam']['status']='search_unresolved';self.assertFalse(c.complete_result(row))

if __name__=='__main__':unittest.main()
