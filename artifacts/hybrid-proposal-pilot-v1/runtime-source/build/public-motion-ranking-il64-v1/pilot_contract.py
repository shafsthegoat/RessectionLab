"""One fixed ranking IL64 contrast; metadata guards and unchanged owned limits."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
TRAIN=('ReMIND-002','ReMIND-015','ReMIND-018','ReMIND-045')
TEACHER_STEPS={'ReMIND-002':1,'ReMIND-015':14,'ReMIND-018':1,'ReMIND-045':13}
TEACHER_STATES=29
EXPOSURE='public-union-post-exposure-axis0-v1'
CLOSED={'SELECT':['ReMIND-013','ReMIND-037'],'EVAL':['ReMIND-067']}
PUBLIC_INDEX='build/remind-partial-domain-public-preparation-v1/manifests-v2/public-index.json'
PUBLIC_SHA='c5138559e45fd67b44b047e8ea6d800b56b8e63ae3371a5ff5c9d83e60b25fe8'
COHORT='manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA='326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
OCCUPANCY='cerebrum_plus_supplied_tumor_with_preserved_partial_source_domain'
STORAGE='cache_complete_replayed_TRAIN_teacher_traces_v1'
WEIGHTING='balanced_STOP_motion_CE_v1'
CACHE_BYTES=256*1024**2
WORKER_SECONDS=3540;PARENT_SECONDS=3600;MEMORY_BYTES=3*1024**3
OUTPUT_BYTES=128*1024**2;SUPERVISION_BYTES=16*1024**2
METHODS={'IL':{'updates':64,'worker_seconds':900,'parent_seconds':960,'source_visits':8,'loss_forward_cap':1856,'policy_forward_cap':1981,
    'native_preview_cap':70080,'teacher_trace_reuses':256,'teacher_cache_payload_bytes':CACHE_BYTES,
    'ranking_pair_terms_per_update':4970,'ranking_pair_terms_total':318080,'endpoint_pair_evaluations':9940}}

LABEL_REFS={'public-score-corpus.json': {'path': 'build/public-motion-ranking-v1/public-score-corpus.json', 'sha256': 'd4c9ade78e50171544ebc9076a6332d864586bdf3cfdd334944be058486781a4'}, 'input-pins.json': {'path': 'build/public-motion-ranking-v1/input-pins.json', 'sha256': '7048bbb3e64b5d36e3429da690e37093bf141b9f49625f4b5f500f0f36a50665'}, 'label-summary.json': {'path': 'build/public-motion-ranking-v1/label-summary.json', 'sha256': '8571e0f16520fa72662a4d66d7f9ac41cb08939cd360f17542e0798428a0013e'}}
BASELINE_REFS={'result': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/result.json', 'sha256': 'eedc4857c184d94913976cb5b59e01341e1b3a8bb0f001af43f8d079e3d87975'}, 'parent': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01.supervision/receipt.json', 'sha256': 'c9702883bcbdb9fb524a99b88fde1cac464116d399fce8aa63d866b57b4a1c82'}, 'release': {'path': 'build/post-exposure-learning-v1/IL64/root-release.json', 'sha256': 'f0df97b2e0222b3656d755ce1ebdaee388b381ae8e2616e5a67e9c50dfdc888b'}, 'recovery_result': {'path': 'build/post-exposure-IL045-evaluation-recovery-v1/attempt-01/result.json', 'sha256': '432535613f699604a4b0f3b0aaa0fedebbe932e5ca6b7006beedd66ff823a7ce'}, 'recovery_parent': {'path': 'build/post-exposure-IL045-evaluation-recovery-v1/attempt-01.supervision/receipt.json', 'sha256': 'c159b91ab2015dfd54dee677eb683b27e335a62657ba51838c398ed2d391a98e'}, 'outcomes': {'path': 'build/post-exposure-il045-recovery-result-v1/four-TRAIN-outcomes.json', 'sha256': '8b7334f9975dde70677403a2215e5a962b2ac0f6a4d9d87d9362bdc6bdafaa5c'}, 'plan_ReMIND-002': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-002/plan.json', 'sha256': '1614f9d2206038c3bda4f85eb0906c31464715b305d11a03827d61b00396d09b'}, 'ReMIND-002:0': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-002/state-00.json', 'sha256': 'a218135fe9bf2f6965aad8e3e73d8942f8844156dbb9e183394ca993649d124d'}, 'plan_ReMIND-015': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-015/plan.json', 'sha256': '66dadfe3b434996fb03284757b6b76d32baf310d7154f122172e79008402bb58'}, 'ReMIND-015:0': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-00.json', 'sha256': '13f20551c7029dbf6ff801ef43cf6a9da945ae5f5c2a702b832a6c620d224e4b'}, 'ReMIND-015:1': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-01.json', 'sha256': 'dfcb9f1d3cae2e18b048e9a9e1601d85d155ad77fe31fbe130561bcf033c463e'}, 'ReMIND-015:2': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-02.json', 'sha256': '145f3194db204c2f4e59b8a42be2ef5c083792bdf604d2ccaa46a6893979bde6'}, 'ReMIND-015:3': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-03.json', 'sha256': '822f0f9cdcd330a8b533312266fcadc28d3a779f107566c80b71bdda242b0186'}, 'ReMIND-015:4': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-04.json', 'sha256': '12ae8364301221eaad26e01c4bbd573956145d8ee42119a2674f6a08424d7871'}, 'ReMIND-015:5': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-05.json', 'sha256': '3d8a532b2aac95eec87e75eeefdcd16237d6cbc00cd3213bb51f1a1954331258'}, 'ReMIND-015:6': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-06.json', 'sha256': '73ba770eb1ec63e14638cce36737f95229136bf7ba03146b42fa4c9e2d85126a'}, 'ReMIND-015:7': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-07.json', 'sha256': '1b88805e0769574ba8b55800b4796a1ef81168d1a18f21a6a1cad63ce4209817'}, 'ReMIND-015:8': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-08.json', 'sha256': '76e17557f9040b1cff2503a10e09014d6ad6f3c4e6492582ee6e9abdd360b60f'}, 'ReMIND-015:9': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-09.json', 'sha256': '4316fda50f4c94b4cbcaacf2f23afb89faa1a2361ffcbca9cf425a7ba2876610'}, 'ReMIND-015:10': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-10.json', 'sha256': 'e9b0b0045d46cb919a22efee1eaf6573c5644036dc3074e66417fb2d0a8938ce'}, 'ReMIND-015:11': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-11.json', 'sha256': '38e6b0dacc0c123ab9a151ae40b28dcb2683821c6b03348a8d225ef99ed887e0'}, 'ReMIND-015:12': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-12.json', 'sha256': '5f352f6c675a54217acb88701fcdc55398d7d0088b0a1fff6872a7e77da2e209'}, 'ReMIND-015:13': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-015/state-13.json', 'sha256': '6f57d8d0d9b40bfdcf4daf14e8f288fc08fc3176097d53cd7ad075768ade7a5d'}, 'plan_ReMIND-018': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-018/plan.json', 'sha256': '9c1b11d559158387dc0e92760973e825f437526274677b85ed83f6d24aca2000'}, 'ReMIND-018:0': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-018/state-00.json', 'sha256': '24fd4d4c6f09b0560c6b3d33d6fe698dbf917c7d607b3dbcbc8254f5d2e34875'}, 'plan_ReMIND-045': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/TRAIN-greedy/IL/ReMIND-045/plan.json', 'sha256': '7411153dabdb28c5d1a85c9d278db9671080231329eab434182dff677c417b84'}, 'ReMIND-045:0': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-00.json', 'sha256': '7b001dbfddcaf12c39a496d777be74ce7fd342d887df7f635f120703d73732cb'}, 'ReMIND-045:1': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-01.json', 'sha256': 'f9d1a377d2bf468daabb78d71e2c96985853f04d4f11df1634c0815bfbe85e0e'}, 'ReMIND-045:2': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-02.json', 'sha256': '0e53cc265424c8719aaeb3500e68b2ba814b3345228116201d63744ba16cad1b'}, 'ReMIND-045:3': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-03.json', 'sha256': 'bbd786afb124d74bf393342902d2cb1be6e2f16b2832adf672bfbb1804297245'}, 'ReMIND-045:4': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-04.json', 'sha256': '922971a85a078f66a91c0f995e5ab699281ac8055e3ffc4094886d8014cfa3d4'}, 'ReMIND-045:5': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-05.json', 'sha256': '5c15501d64390522d623ffbd72413edcde5d48fbe7da281f03d19c95d7581955'}, 'ReMIND-045:6': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-06.json', 'sha256': 'fd9c567228b294642aa15e48ad22c37e1c59f569c6652b0ecff9d2c23acaf9f5'}, 'ReMIND-045:7': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-07.json', 'sha256': '39fb7c2cc2b62e8b0d414d60cb46771f9188a7e50d41dc59a323657e3c0a64c3'}, 'ReMIND-045:8': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-08.json', 'sha256': 'a334bfcf6fcfb72b96712ef385bf0f13b717d1f59e50f37b14d0fa4079422e8d'}, 'ReMIND-045:9': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-09.json', 'sha256': 'aed4cd1e13d4b15d38b768feff237acf78d9ecba48e182dc3ef0049ae6b1346c'}, 'ReMIND-045:10': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-10.json', 'sha256': '56f2dff748f0b5cc10590bd76a767339fe874b597424da55183b73c48c5c3b7d'}, 'ReMIND-045:11': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-11.json', 'sha256': '574475418fbe5a87c3c25e285aa7e97edfe1606ea65863ac08cdc0192f699807'}, 'ReMIND-045:12': {'path': 'build/post-exposure-learning-v1/IL64/attempt-01/teacher-readout/ReMIND-045/state-12.json', 'sha256': '010e8e9780a767e6a3dab09b802ea507923c22c371e248defd8860344383f336'}}
RANKING_SOURCE_PINS={'src/resectionlab/patient_planning_accumulation.py': '2fc5d6a14f9de3176d4ef75d5a58f5b075b15f861d6282ed2e97705da1f17307', 'src/resectionlab/patient_planning_cohort_spec.py': '9ae21d7bb7e437464e46e5e805ef7ea84e9675e5dd90e7b8149821c06f92a3a4', 'src/resectionlab/patient_planning_learning.py': 'e4d73d3286799cfb78bf415f85b2275ab7dbb60c051fbf683c2e96ca4967e90f', 'src/resectionlab/public_motion_ranking.py': '46ff5677029683c9cc52aebabebc9ab49c594afe5bf3aa0550ecaf0a98f53085'}
RANKING_HASH='sha256:5147c784a1e4aee38ffd2cf95881c40914f9807727d54eeaa4dce618100da0be'
RANKING_SPEC={'version':'public_nominal_motion_gap_ranking_v1','corpus_hash':RANKING_HASH,
    'state_count':29,'motion_states':25,'STOP_states':4}
PAIR_COUNTS={'ReMIND-002':0,'ReMIND-015':2516,'ReMIND-018':0,'ReMIND-045':2454}
PAIR_TERMS_PER_UPDATE=4970
PAIR_TERMS_TOTAL=318080

INPUTS={'result': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/result.json', 'sha256': '7a57fd74f58b15d6975be4459fe4f46e6396dfa9e2f3c0fec0ef68cf5e550358'}, 'parent': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01.supervision/receipt.json', 'sha256': '664b7ac101e8c79e512d52f971f424b5e2b1bbeaa7d7b1502c110afb4364f233'}, 'release': {'path': 'build/remind-post-exposure-feasibility-v1/root-release.json', 'sha256': '8315099d95b16a9dacac54a65c4d84d6385e458534b6dab26b73a37555d4af1a'}, 'source_index': {'path': 'build/remind-post-exposure-feasibility-v1/source-index.json', 'sha256': '4ac60917a28368056b1b0d07481a88f447be4116ec677fed1bb31dd2766c7b50'}, 'initial_release': {'path': 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1/root-release.json', 'sha256': 'c1cde8eb92f9df5d85ccb91ed0e591adb1da70e011582f3be116d237fecf72db'}, 'plan_ReMIND-002': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-002/greedy-plan.json', 'sha256': '757449e8ac744daee89dd085924a27c8a7a4fc571a1c906994188bf08c4989bc'}, 'metrics_ReMIND-002': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-002/episode-metrics.json', 'sha256': '59a3e11e3bd662063df245ba485ff36b4001c0445c688f572421ea42e1a83789'}, 'replay_ReMIND-002': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-002/independent-episode.json', 'sha256': '40d784c101280b97bc902fc77acdf363e5243b1bfab3cfeeb4feb276713acae8'}, 'plan_ReMIND-015': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-015/greedy-plan.json', 'sha256': '954ec5204482ba6c91867e34e61700470f060c74b702711e57414a77cb2cfcf1'}, 'metrics_ReMIND-015': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-015/episode-metrics.json', 'sha256': '3ff9a551a4cad52e26936c797cdaf0b6923a6ddf5ae894867c8dd517a99359d9'}, 'replay_ReMIND-015': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-015/independent-episode.json', 'sha256': 'bcaf1303133576b7a281eec3915a3088f95306b17ef655041222f85272387547'}, 'plan_ReMIND-018': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-018/greedy-plan.json', 'sha256': '4d0b9a0029293e787a99c276bd6bca9430bc0722c2eaaecd88bb193d4e1e5040'}, 'metrics_ReMIND-018': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-018/episode-metrics.json', 'sha256': '911e0d6060b4e16188a054ff2f736be90f897796db48b22c2b85a3faf16cb012'}, 'replay_ReMIND-018': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-018/independent-episode.json', 'sha256': '20d0f0be9e4aa990136d47e5e2c91c2df140492b130d850fcb641e737bf7e6cb'}, 'plan_ReMIND-045': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-045/greedy-plan.json', 'sha256': 'da7b49d6fde2c97c9f6911f42475afb7a4cc946e8c894e8267cc4a82679c58c1'}, 'metrics_ReMIND-045': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-045/episode-metrics.json', 'sha256': 'fa67b7ee6862222d480d60a570161e7990082e854e0327814191a629a48f6e21'}, 'replay_ReMIND-045': {'path': 'build/remind-post-exposure-feasibility-v1/attempt-01/ReMIND-045/independent-episode.json', 'sha256': 'a18c03193e4968bf2f2caa44257117c51107091ee0bdfe18083b7f5edd1b0db9'}}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def semantic(value):return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def output_for(method):
    if method not in METHODS:raise ValueError('Fixed IL64 or RL8 method required')
    return 'build/public-motion-ranking-il64-v1/'+method+str(METHODS[method]['updates'])+'/attempt-01'
def small(ref):
    path=ROOT/ref['path']
    if path.is_symlink() or not path.is_file() or not 0<path.stat().st_size<=4*1024**2 or sha(path)!=ref['sha256']:
        raise ValueError('Bounded pinned input changed: '+ref['path'])
    return json.loads(path.read_text())

def inputs():
    rows={k:small(v) for k,v in INPUTS.items()}
    result,parent,release=rows['result'],rows['parent'],rows['release']
    if (result['status']!='post_exposure_batch_complete' or result['training_admitted'] is not False
            or result['occupancy_condition']!=OCCUPANCY or result['post_exposure_condition']!=EXPOSURE
            or any(result[k]!=0 for k in ('model_calls','optimizer_calls','checkpoint_loads','source_arrays_written'))
            or parent['status']!='complete' or parent['exit_code']!=0 or parent['cleanup_errors']
            or parent['remaining_owned_pids'] or parent['worker_termination_confirmed'] is not True
            or parent['result_sha256']!=INPUTS['result']['sha256']
            or parent['release_sha256']!=INPUTS['release']['sha256']
            or parent['source_index']!=release['source_index']):
        raise ValueError('Clean completed fixed-four post-exposure measurement required')
    selected=result['cases']
    if [a['patient_id'] for a in selected]!=list(TRAIN):raise ValueError('Exact new four TRAIN cases required')
    config=result['proposal_config']
    if (config.get('obstruction_opening',False) is not False or config['max_candidates']!=120
            or config['tool_footprint_opening'] is not True or len(config['offsets_source_voxels'])!=13):
        raise ValueError('Tested historical13-axis/cap120 proposal world required')
    for arm in selected:
        subject=arm['patient_id'];saved=rows['plan_'+subject];metrics=rows['metrics_'+subject];audit=rows['replay_'+subject]
        if (arm['status']!='completed_fixed_post_exposure_route' or arm['role']!='TRAIN'
                or arm['route_complete'] is not True or arm['independent_episode_accepted'] is not True
                or arm['full_target_preserved'] is not True or arm['source_domain_extended'] is not False
                or saved['actions']!=arm['actions'] or len(saved['actions'])!=TEACHER_STEPS[subject]
                or saved['actions'][-1]!='STOP' or saved['accounting']['complete'] is not True
                or [h['action_id'] for h in metrics['history']]!=saved['actions']
                or metrics['terminated'] is not True or metrics['steps']!=TEACHER_STEPS[subject]
                or audit['accepted'] is not True or audit['complete_episode'] is not True
                or audit['committed_history_hash']!=semantic(metrics['history'])
                or audit['outcomes']!=arm['full_route_outcomes']
                or metrics['source_hash']!=arm['source_hash']
                or metrics['decision_model_hash']!=audit['decision_model_hash']
                or metrics['post_exposure']!=arm['post_exposure_start']):
            raise ValueError('Complete saved greedy teacher/world changed: '+subject)
        # This projection creates no training authority; every trace is recollected
        # and independently replayed under its new explicit learning context.
        saved['plan']={'source_hash':metrics['source_hash'],'decision_model_hash':metrics['decision_model_hash'],
            'initial_observation_hash':arm['initial_observation_hash'],'max_steps':24,'actions':saved['actions'],
            'history':metrics['history'],'terminal_reason':'STOP','parameter_hash':None,'architecture_hash':None,'learning_updates':0}
    rows['arms']={a['patient_id']:a for a in selected}
    return rows

def expected_configuration(method):
    data=inputs();protocol=json.loads(canonical(data['initial_release']['learning_protocol']))
    execution=protocol['cohort_execution'];arm=data['arms'][TRAIN[0]]
    protocol['updates_per_method']=METHODS[method]['updates']
    execution.update(proposal_config=data['result']['proposal_config'],proposal_rule_hash=data['metrics_'+TRAIN[0]]['proposal_rule_hash'],occupancy_condition=OCCUPANCY,post_exposure_condition=EXPOSURE,patient_order=list(TRAIN))
    protocol['version']='fixed-four-TRAIN-post-exposure-learning-v1'
    protocol['scope']='fixed_four_TRAIN_shared_population_pilot'
    execution.update(teacher_source='fixed_saved_greedy_complete_histories_v1',teacher_decisions=29,teacher_steps=[1,14,1,13])
    if method=='IL':execution.update(il_teacher_weighting=WEIGHTING,teacher_observations=STORAGE,teacher_cache_payload_bytes=CACHE_BYTES,il_motion_supervision=RANKING_SPEC)
    limits={'max_steps':24,'max_optimizer_updates':128 if method=='IL' else 16,
        'max_native_previews':7297920 if method=='IL' else 3373440,
        'max_policy_forwards':18624 if method=='IL' else 2496,'worker_seconds':METHODS[method]['worker_seconds'],
        'memory_bytes':MEMORY_BYTES,'threads':1,'search':execution['search'],
        'output_bytes':OUTPUT_BYTES,'checkpoint_bytes':8*1024**2}
    original=small(BASELINE_REFS['release'])['learning_protocol']
    projected=json.loads(canonical(protocol));projected['cohort_execution'].pop('il_motion_supervision')
    if canonical(projected)!=canonical(original):raise ValueError('Only the declared ranking supervision may change baseline protocol')
    return protocol,limits

def canonical_configuration(method):
    from resectionlab.core import thaw_json
    from resectionlab.native_proposals import NominalCavityProposalConfig
    from resectionlab.patient_planning_cohort_spec import sequential_learning_protocol,sequential_limits
    expected,_=expected_configuration(method);execution=expected['cohort_execution']
    options={'il_teacher_weighting':WEIGHTING,'teacher_observations':STORAGE,'teacher_cache_payload_bytes':CACHE_BYTES,'il_motion_supervision':RANKING_SPEC}
    protocol=sequential_learning_protocol(updates=METHODS[method]['updates'],max_steps=24,
        search=execution['search'],proposal_config=NominalCavityProposalConfig(**execution['proposal_config']),
        retention_mode=execution['retention_mode'],occupancy_condition=OCCUPANCY,post_exposure_condition=EXPOSURE,**options)
    limits=sequential_limits(protocol,worker_seconds=METHODS[method]['worker_seconds'],memory_bytes=MEMORY_BYTES,output_bytes=OUTPUT_BYTES)
    return thaw_json(protocol),thaw_json(limits)

def validate_release(release):
    method=release.get('method')
    if method not in METHODS:raise ValueError('Fixed declared method required')
    protocol,limits=expected_configuration(method)
    keys={'version','status','method','expected_head','output','TRAIN','closed_roles','SELECT_EVAL_execution',
        'ranking_inputs','baseline_inputs','new_four_training','attempts','automatic_retry','parent_seconds','supervision_bytes','cohort_limits',
        'limits','execution_limits','learning_protocol','learning_protocol_hash','inputs','source_index',
        'public_manifest_index','completion','claim'} | ({'matched_IL_completion'} if method=='RL' else set())
    if set(release)!=keys:raise ValueError('Exact endpoint declaration keys required')
    if release['public_manifest_index']!={'path':PUBLIC_INDEX,'sha256':PUBLIC_SHA}:raise ValueError('Exact public manifest index required')
    if (release.get('version')!='public-motion-ranking-IL64-v1'
            or release.get('status')!='released_one_attempt' or release.get('output')!=output_for(method)
            or release.get('TRAIN')!=list(TRAIN) or release.get('closed_roles')!=CLOSED
            or release.get('SELECT_EVAL_execution') is not False or release.get('new_four_training') is not True
            or release.get('attempts')!=1 or type(release.get('attempts')) is not int
            or release.get('automatic_retry') is not False or release.get('inputs')!=INPUTS
            or release.get('ranking_inputs')!=LABEL_REFS or release.get('baseline_inputs')!=BASELINE_REFS
            or canonical(release.get('execution_limits'))!=canonical(METHODS[method])
            or canonical(release.get('learning_protocol'))!=canonical(protocol)
            or canonical(release.get('cohort_limits'))!=canonical(limits)
            or release.get('learning_protocol_hash')!=semantic(protocol)
            or canonical(release.get('limits'))!=canonical({k:v for k,v in limits.items() if k not in ('output_bytes','checkpoint_bytes')})
            or type(release.get('parent_seconds')) is not int or type(release.get('supervision_bytes')) is not int
            or release.get('parent_seconds')!=METHODS[method]['parent_seconds'] or release.get('supervision_bytes')!=SUPERVISION_BYTES):
        raise ValueError('Exact separately bounded fixed endpoint release required')
    ranking_inputs()

def source_guard(release,release_path,release_sha):
    validate_release(release)
    if sha(release_path)!=release_sha:raise ValueError('Release changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,timeout=5).strip()
    if head!=release['expected_head']:raise ValueError('Canonical HEAD changed')
    ref=release['source_index'];index=small(ref)
    if index['head']!=head:raise ValueError('Source index HEAD differs')
    for section in ('source_files','metadata_files'):
        for path,digest in index[section].items():
            p=Path(path)
            if p.is_absolute() or '..' in p.parts or (ROOT/p).is_symlink() or sha(ROOT/p)!=digest:
                raise ValueError('Source/input changed: '+path)
    return index

def complete_result(result):
    method=result.get('method');cfg=METHODS.get(method,{})
    if not cfg:return False
    return (result.get('status')=='complete_matched_TRAIN_endpoint'
        and result.get('TRAIN')==list(TRAIN) and result.get('post_exposure_condition')==EXPOSURE
        and result.get('fresh_common_initialization_verified') is True
        and result.get('same_initial_parameters_as_original_IL') is True
        and result.get('public_motion_ranking_corpus_hash')==RANKING_HASH
        and result.get('confirmed_ranking_pair_terms')==PAIR_TERMS_TOTAL
        and result.get('endpoint_pair_evaluations')==9940
        and result.get('optimizer_updates')=={m:cfg['updates'] if m==method else 0 for m in ('IL','RL')}
        and (result.get('loss_forward_calls')==1856 if method=='IL' else 32<=result.get('loss_forward_calls',0)<=768)
        and result.get('teacher_logit_forwards')==TEACHER_STATES and result.get('checkpoint_loads')==1
        and result.get('completed_source_visits')==cfg['source_visits']
        and result.get('teacher_trace_reuses')==cfg['teacher_trace_reuses']
        and result.get('search_calls')==0 and result.get('private_reference_reads')==0 and result.get('SELECT_EVAL_opened') is False
        and result.get('teacher_decisions')==TEACHER_STATES
        and result.get('teacher_steps')==TEACHER_STEPS
        and 0<=result.get('native_preview_entries',-1)<=cfg['native_preview_cap']
        and result.get('total_policy_forward_calls')==result.get('loss_forward_calls',0)*(1 if method=='IL' else 2)+TEACHER_STATES+sum(r.get('steps',0) for r in result.get('TRAIN_greedy',{}).values())
        and result.get('total_policy_forward_calls',cfg['policy_forward_cap']+1)<=cfg['policy_forward_cap']
        and result.get('teacher_statuses')=={s:'complete_replayed' for s in TRAIN}
        and set(result.get('checkpoints',{}))=={method} and set(result.get('TRAIN_greedy',{}))==set(TRAIN)
        and all(result['TRAIN_greedy'][s].get('complete') is True for s in TRAIN)
        and result.get('selection_readiness',{}).get('ready') is True
        and result.get('selection_readiness',{}).get('execution_admitted') is False)

def endpoint_control(output,release):
    output=Path(output);result=json.loads((output/'result.json').read_text())
    if not complete_result(result):raise ValueError('Complete fixed ranking endpoint required')
    dynamics=json.loads((output/'training-dynamics.json').read_text())
    if (sha(output/'training-dynamics.json')!=result['training_dynamics_sha256']
            or len(dynamics['updates'])!=64
            or sum(r['confirmed_ranking_pair_terms'] for r in dynamics['updates'])!=PAIR_TERMS_TOTAL):
        raise ValueError('Exact64 updates and bounded pair work required')
    baseline=small(BASELINE_REFS['result'])
    if result['initial_parameter_hash']!=baseline['initial_parameter_hash']:
        raise ValueError('Original IL initial tensors changed')
    comparison=output/'original-IL-comparison.json'
    if sha(comparison)!=result['original_IL_comparison_sha256']:raise ValueError('Comparison changed')
    return {'status':'complete_fixed_endpoint','method':'IL',
        'initial_parameter_hash':result['initial_parameter_hash'],'same_initial_parameters_as_original_IL':True,
        'twenty_nine_saved_greedy_teacher_states':True,'checkpoint_reloaded':True,
        'TRAIN_greedy':result['TRAIN_greedy'],'teacher_metrics':result['endpoint_teacher_metrics'],
        'ranking_corpus_hash':RANKING_HASH,'confirmed_ranking_pair_terms':PAIR_TERMS_TOTAL,
        'original_IL_comparison_sha256':result['original_IL_comparison_sha256'],
        'training_dynamics_sha256':result['training_dynamics_sha256'],'heldout_or_clinical_claim':False}



def ranking_inputs():
    """Pin existing public labels and their provenance; no label query or model."""
    corpus=small(LABEL_REFS['public-score-corpus.json'])
    pins=small(LABEL_REFS['input-pins.json']);summary=small(LABEL_REFS['label-summary.json'])
    if (semantic(corpus)!=RANKING_HASH or summary['corpus_hash']!=RANKING_HASH
            or summary['states']!=29 or summary['strict_motion_pairs_per_update']!=4970):
        raise ValueError('Exact saved public score corpus required')
    for path,row in pins.items():
        if (ROOT/path).stat().st_size!=row['bytes']:raise ValueError('Label provenance size changed')
        small({'path':path,'sha256':row['sha256']})
    for case in corpus['subjects']:
        s=case['subject'];old='build/post-exposure-learning-v1/IL64/attempt-01/teachers/'+s
        names={'scores':'build/remind-post-exposure-feasibility-v1/attempt-01/'+s+'/greedy-plan.json',
            'trace':old+'/complete-trace.json','plan':old+'/plan.json','replay':old+'/native-replay.json'}
        if case['input_sha256']!={k:pins[p]['sha256'] for k,p in names.items()}:
            raise ValueError('Public labels differ from pinned score/trace/plan/replay lineage')
        pairs=0
        for row in case['decisions']:
            if row['teacher_action']=='STOP':continue
            rewards=[v for i,(v,m) in enumerate(zip(row['rewards'],row['action_mask'])) if i and m]
            pairs+=sum(a>b for a in rewards for b in rewards)
        if pairs!=PAIR_COUNTS[s]:raise ValueError('Exact per-patient ordered pair work changed')
    baseline={k:small(ref) for k,ref in BASELINE_REFS.items()}
    prior=baseline['result'];recovery=baseline['recovery_result']
    if (prior['status']!='failed_or_unresolved' or prior['optimizer_updates']!={'IL':64,'RL':0}
            or baseline['parent']['result_sha256']!=BASELINE_REFS['result']['sha256']
            or baseline['parent']['status']!='failed_or_unresolved'
            or baseline['recovery_parent']['result_sha256']!=BASELINE_REFS['recovery_result']['sha256']
            or baseline['recovery_parent']['status']!='complete'
            or recovery['independent_accepted'] is not True or recovery['full_history_equal'] is not True
            or recovery['original_IL_result_sha256']!=BASELINE_REFS['result']['sha256']
            or [r['subject'] for r in baseline['outcomes']]!=list(TRAIN)
            or not all(r['independent_accepted'] for r in baseline['outcomes'])):
        raise ValueError('Original failed IL plus separately accepted recovery must stay explicit')
    return corpus,baseline
