"""Saved JSON/byte audit only. No project imports, tensor decoding or evaluation."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / 'outputs/learning/public-contact-v1/attempt-01'
SUP = RUN.with_name(RUN.name + '.supervision')
bindings = {}

def read(path):
    path = Path(path)
    raw = path.read_bytes()
    bindings[str(path.relative_to(ROOT))] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return raw

def js(path):
    return json.loads(read(path))

def digest(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

receipt = js(SUP / 'receipt.json')
decl = js(SUP / 'declaration.json')
result = js(RUN / 'result.json')
freeze = js(RUN / 'experiment-freeze.json')
final_freeze = js(RUN / 'final-checkpoint-freeze.json')
assert receipt['result_sha256'] == bindings[str((RUN/'result.json').relative_to(ROOT))]['sha256']
assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
assert not receipt['cleanup_errors'] and not receipt['final_owned_pids'] and receipt['worker_termination_confirmed']
assert receipt['caps'] == decl['caps'] and not receipt['automatic_retry']
assert receipt['elapsed_seconds'] < receipt['caps']['hard_total_wall_seconds']
assert receipt['sampled_peak_rss_bytes'] < receipt['caps']['sampled_owned_tree_rss_bytes']
assert result['status'] == 'complete_with_all_failures_preserved'
assert result['patient_reads'] == 0
assert result['measurement_status'] == 'single_frozen_pass_finished_no_checkpoint_selection'
assert result['experiment_hash'] == digest(freeze) == decl['released_experiment_hash'] == final_freeze['experiment_hash']

source_matches = {}
for relative, expected in decl['source_bindings'].items():
    raw = read(ROOT / relative)
    committed = subprocess.check_output(['git', 'show', decl['head'] + ':' + relative], cwd=ROOT)
    assert hashlib.sha256(raw).hexdigest() == expected == hashlib.sha256(committed).hexdigest(), relative
    source_matches[relative] = expected
for name, field in [('contact_owned_worker.py', 'worker_sha256'), ('run_contact_owned.py', 'supervisor_sha256'),
                    ('darwin_fast_sampler.py', 'darwin_sampler_sha256')]:
    assert hashlib.sha256(read(ROOT/'build/goal-conditioned-policy-v1'/name)).hexdigest() == decl[field]
assert hashlib.sha256(read(Path(decl['source_index_path']))).hexdigest() == decl['source_index_sha256']

# Parse the immutable protocol literal without importing/executing project code.
tree = ast.parse((ROOT/'src/resectionlab/contact_learning_contract.py').read_text())
version = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='VERSION' for t in n.targets))
protocol_node = next(n.value.args[0] for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id=='PROTOCOL' for t in n.targets))
class Version(ast.NodeTransformer):
    def visit_Name(self, node):
        assert node.id == 'VERSION'
        return ast.Constant(version)
protocol = ast.literal_eval(Version().visit(protocol_node))
assert protocol == freeze['protocol']

family = freeze['family_manifest']
roles = {role: [r['layout_id'] for r in family['source_bindings'] if r['role']==role]
         for role in ('TRAIN','SELECT','MEASUREMENT_EVAL')}
assert {k:len(v) for k,v in roles.items()} == {'TRAIN':12,'SELECT':4,'MEASUREMENT_EVAL':8}
assert len(set(sum(roles.values(), []))) == 24
split_roles = {}
for depth, selected in ((2,2),(3,1),(4,1)):
    group = sorted((r for r in family['layouts'] if r['recipe']['front_depths'][2]==depth),
                   key=lambda r:digest({'split': family['split_version'], 'recipe':r['recipe']}))
    for i, row in enumerate(group):
        assert digest(row['recipe']) == row['recipe_hash']
        split_roles[row['recipe']['layout_id']] = 'TRAIN' if i<4 else 'SELECT' if i<4+selected else 'MEASUREMENT_EVAL'
        assert split_roles[row['recipe']['layout_id']] == row['role']
assert all(split_roles[r['layout_id']] == r['role'] for r in family['source_bindings'])
source_rows = {r['layout_id']:r for r in family['source_bindings']}
train_keys = {(layout,goal) for layout in roles['TRAIN'] for goal in ('surface','deep')}

checkpoints = {}
initial = result['initial_checkpoint']
for method, rec in [('COMMON',initial), *[(m,result['training'][m]['checkpoint']) for m in ('IL','RL')]]:
    path = Path(rec['path']); raw = read(path)
    assert len(raw) == rec['bytes'] and hashlib.sha256(raw).hexdigest() == rec['sha256']
    # Inspect manifest only; tensor payloads are not decoded, imported or loaded.
    with zipfile.ZipFile(path) as archive:
        assert archive.getinfo('manifest.json').file_size <= 512*1024
        metadata = json.loads(archive.read('manifest.json'))
    assert metadata['lineage'] == rec['lineage']
    assert metadata['parameter_hash'] == rec['parameter_hash'] == rec['lineage']['parameter_hash']
    assert metadata['architecture_hash'] == freeze['architecture_hash']
    assert rec['lineage']['initial_parameter_hash'] == initial['parameter_hash']
    assert rec['lineage']['experiment_hash'] == result['experiment_hash']
    if method != 'COMMON':
        assert result['training'][method]['updates'] == 32
        assert result['training'][method]['status'] == 'completed_fixed_endpoint'
        assert rec['lineage']['optimizer_updates'] == 32
        assert rec['lineage']['kind'] == 'final'
        assert {(b['layout_id'],b['goal_id']) for b in rec['lineage']['training_bindings']} == train_keys
        assert all(b['role']=='TRAIN' for b in rec['lineage']['training_bindings'])
        assert final_freeze['checkpoints'][method] == {'file_sha256':rec['sha256'], 'parameter_hash':rec['parameter_hash'], 'lineage_hash':digest(rec['lineage'])}
    checkpoints[method] = {k:rec[k] for k in ('sha256','bytes','parameter_hash')}

teachers = result['teacher_slots']
assert len(teachers)==24 and {(t['layout_id'],t['goal_id']) for t in teachers}==train_keys
teacher_actions = Counter()
teacher_bindings = {}
for i,t in enumerate(teachers):
    assert js(RUN/f'teacher-{i:02d}.json') == t
    assert t['status']=='complete_bounded_teacher' and t['role']=='TRAIN'
    assert not t['search']['call_cap_reached'] and not t['search']['time_cap_reached']
    assert t['label_count'] == len(t['strategy']['strategy']['actions'])
    assert t['strategy']['strategySeal'] == digest(t['strategy']['strategy'])
    teacher_actions.update(t['strategy']['strategy']['actions'])
    teacher_bindings[digest(t['binding'])] = t['binding']

training = {}
for method in ('IL','RL'):
    previous = initial['parameter_hash']; sample_count=0; rl_transitions=0; rl_success=0; changed=0
    sample_bindings=set(); rl_keys=set(); unique_samples=set()
    assert len(list(RUN.glob(method+'-update-*.json'))) == 32
    for index in range(1,33):
        update = js(RUN/f'{method}-update-{index:02d}.json')
        ur = update['update_receipt']
        assert update['method']==method and update['update']==index and ur['cumulative_updates']==index
        assert ur['initial_parameter_hash']==previous and ur['optimizer_steps']==1
        previous=ur['updated_parameter_hash']; changed+=ur['parameters_changed']
        assert all(s['binding_hash'] in teacher_bindings for s in update['samples'])
        sample_bindings.update(s['binding_hash'] for s in update['samples'])
        unique_samples.update((s['binding_hash'],s['observation_hash'],s['action_id']) for s in update['samples'])
        sample_count+=len(update['samples'])
        if method=='IL':
            assert len(update['samples'])==4 and update['native_trajectories'] is None
            assert update['loss']['supervised_actions']==4
        else:
            episodes=update['native_trajectories']; assert len(episodes)==4
            lengths=0
            for ep in episodes:
                assert (ep['layout_id'],ep['goal_id']) in train_keys
                assert ep['binding']==teacher_bindings[digest(ep['binding'])]
                rl_keys.add((ep['layout_id'],ep['goal_id']))
                lengths+=len(ep['strategy']['strategy']['actions'])
                assert ep['strategy']['strategySeal']==digest(ep['strategy']['strategy'])
                rl_success+=ep['strategy']['metrics']['goal_contacted_and_retained']
            assert lengths==len(update['samples'])
            rl_transitions+=lengths
            assert update['rollout_transitions_cumulative']==rl_transitions<=256
        assert update['elapsed_seconds']<60
    assert previous==checkpoints[method]['parameter_hash']
    assert sample_bindings==set(teacher_bindings)
    training[method]={'updates':32,'changed_parameter_receipts':changed,'samples':sample_count,'unique_sample_tuples':len(unique_samples),
                      'last_update_elapsed_seconds':update['elapsed_seconds'],'final_parameter_hash':previous,
                      'rl_episodes':128 if method=='RL' else 0,'rl_contact_successes':rl_success,
                      'rl_transition_count':rl_transitions}

summary=[]; episode_count=0; initial_obs={}; row_short=[]
for role in ('SELECT','MEASUREMENT_EVAL'):
    keys=[(r['layout_id'],g) for r in family['source_bindings'] if r['role']==role for g in ('surface','deep')]
    rows=result[role]
    assert len(rows)==4*len(keys)
    expected=[]
    for index,(layout,goal) in enumerate(keys):
        methods=['STOP','SEARCH','IL','RL']; methods=methods[index%4:]+methods[:index%4]
        for method in methods:
            expected.append((layout,goal,method))
            matches=[r for r in rows if (r['layout_id'],r['goal_id'],r['method'])==(layout,goal,method)]
            assert len(matches)==1
            row=matches[0]
            assert row['role']==role and row['status']=='complete'
            disk=js(RUN/f'{role}-{index:02d}-{method}-result.json')
            assert disk=={k:v for k,v in row.items() if k not in ('complete_online_wall_seconds','timing_scope')}
            envelope=js(RUN/f'{role}-{index:02d}-{method}-episode.json')
            ep=envelope['episode']; body={k:v for k,v in ep.items() if k!='episodeId'}
            assert json.loads(envelope['episodeCanonicalJson'])==body
            assert digest(body)==ep['episodeId']==row['episode_id']
            assert ep['metrics']==row['metrics'] and ep['planning']==row['planning']
            assert ep['layoutId']==layout and ep['publicGoal']['goalId']==goal and ep['splitRole']==role
            assert ep['sourceHash']==source_rows[layout]['source_hash']
            assert ep['publicGoal']['objectiveHash']==source_rows[layout]['goals'][goal]['objective_hash']
            assert ep['publicGoal']['goalGridHash']==source_rows[layout]['goals'][goal]['goal_grid_hash']
            assert not ep['patientAdmission'] and not ep['clinicalValidation']
            assert ep['planning']['optimizer_updates']==0
            assert ep['metrics']['terminated'] and ep['metrics']['steps']<=2
            assert abs(sum(h['reward'] for h in ep['history'])-ep['metrics']['total_reward'])<1e-12
            assert ep['planning']['strategy']['history']==ep['history']==ep['metrics']['history']
            assert digest(ep['planning']['strategy'])==ep['planning']['strategySeal']
            assert ep['metrics']['goal_contacted_and_retained']==ep['sequentialEffect']['publicGoalContactedAndRetained']
            key=(role,layout,goal)
            if key in initial_obs: assert initial_obs[key]==ep['initialObservationBinding']
            else: initial_obs[key]=ep['initialObservationBinding']
            if role=='MEASUREMENT_EVAL':
                assert ep['taskContract']['declaration']['checkpoint_freeze_hash']==digest(final_freeze)
            if method in ('IL','RL'):
                author=ep['learnedAuthorship']; cp=final_freeze['checkpoints'][method]
                assert author['checkpointFileSha256']==cp['file_sha256'] and author['parameterHash']==cp['parameter_hash']
                assert author['trainingLineageHash']==cp['lineage_hash'] and author['completedUpdates']==32
                assert author['inferenceOptimizerUpdates']==0 and author['checkpointKind']=='final'
                assert ep['planning']['strategy']['actions']==['STOP']
            else: assert ep['learnedAuthorship'] is None
            assert row['complete_online_wall_seconds']<12
            assert not row['planning'].get('call_cap_reached') and not row['planning'].get('time_cap_reached')
            episode_count+=1
            row_short.append({'role':role,'layout':layout,'goal':goal,'method':method,'success':ep['metrics']['goal_contacted_and_retained'],
                              'reward':ep['metrics']['total_reward'],'removed_mm3':ep['metrics']['removed_volume_mm3'],
                              'planning_calls':ep['planning']['model_transition_calls'], 'online_seconds':row['complete_online_wall_seconds']})
    assert [(r['layout_id'],r['goal_id'],r['method']) for r in rows]==expected
    for method in ('STOP','SEARCH','IL','RL'):
        for goal in ('surface','deep','all'):
            subset=[r for r in rows if r['method']==method and (goal=='all' or r['goal_id']==goal)]
            summary.append({'role':role,'method':method,'goal':goal,'n':len(subset),
                            'successes':sum(r['metrics']['goal_contacted_and_retained'] for r in subset),
                            'mean_reward':statistics.mean(r['metrics']['total_reward'] for r in subset),
                            'mean_removal_mm3':statistics.mean(r['metrics']['removed_volume_mm3'] for r in subset),
                            'online_seconds':sum(r['complete_online_wall_seconds'] for r in subset),
                            'actor_forwards':sum(r['planning'].get('actor_forward_calls',0) for r in subset),
                            'planning_model_calls':sum(r['planning']['model_transition_calls'] for r in subset)})

# Trace sample accounting is separate from planning, inventory, seal and export replays.
assert result['costs']['IL.loss_forwards']['actor_forward_calls']==training['IL']['samples']==128
assert result['costs']['RL.loss_forwards']['actor_forward_calls']==training['RL']['samples']==164
assert result['costs']['RL.collection']['native_transition_calls']==164
assert result['costs']['RL.collection_native_replay']['native_transition_calls']==328
assert result['costs']['teacher.trace']['native_transition_calls']==sum(t['label_count'] for t in teachers)==40
assert result['costs']['teacher.native_replay']['native_transition_calls']==80
assert result['costs']['teacher.search']['native_transition_calls']==sum(t['search']['model_transition_calls'] for t in teachers)
cost_phases=list(result['costs'])
assert cost_phases.index('checkpoint_freeze_verification')<cost_phases.index('SELECT.checkpoint_load')<cost_phases.index('MEASUREMENT_EVAL.checkpoint_load')
for role in ('SELECT','MEASUREMENT_EVAL'):
    for row in result[role]:
        key=f"{role}.{row['method']}.{row['layout_id']}.{row['goal_id']}.planning"
        if row['method'] in ('IL','RL'):
            assert result['costs'][key]['actor_forward_calls']==row['planning']['actor_forward_calls']

audit={'status':'PASS_SAVED_RESULT_CONSISTENCY_NEGATIVE_LEARNED_OUTCOME','audit_scope':'stdlib saved JSON, byte hashes, checkpoint manifest only; zero model construction/load, training, inference or native replay',
       'execution_head':decl['head'],'head_at_audit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'experiment_hash':result['experiment_hash'],'source_worktree_and_execution_commit_matches':source_matches,
       'roles':roles,'checkpoints':checkpoints,'training':training,'teachers':{'slots':24,'complete':24,'labels':40,'STOP_trajectories':8,
       'contact_successes':sum(t['strategy']['metrics']['goal_contacted_and_retained'] for t in teachers),'search_calls':sum(t['search']['model_transition_calls'] for t in teachers)},
       'online_episodes_checked':episode_count,'summary':summary,'rows':row_short,'supervision':receipt,
       'worker_complete_wall_seconds':result['complete_wall_seconds'],'worker_peak_rss_bytes':result['process_peak_rss_bytes'],
       'result_cost_phase_count':len(cost_phases),'all_planned_rows_complete_and_uncapped':True,
       'timing_fields_absent_in_individual_row_files':['complete_online_wall_seconds','timing_scope'],
       'limitations':['No native replay or fresh policy evaluation; saved consistency and source-path audit only.',
       'Eight held-out layouts with two correlated goals each, one seed, single fixed 32-update endpoint.',
       'Four methods are paired on the same tasks, not independent denominator expansion.',
       'Source declaration pins 14 explicit files, not the full transitive Python/environment closure; no supervisor terminal source postguard.',
       'All IL/RL online actions STOP; successful experiment completion does not mean successful contact or clinical validation.'],
       'bindings':bindings}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:audit[k] for k in ('status','execution_head','roles','training','teachers','online_episodes_checked')},indent=2))
print('audit_sha256',hashlib.sha256((OUT/'audit.json').read_bytes()).hexdigest())
