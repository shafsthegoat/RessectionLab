"""Root-released positive factory/context/STOP control; no learning or search."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FAMILY = ROOT/'build/public-contact-family-v1/structural-manifest.json'
FILES = ['public_contact_family.py','public_surface_contact.py','native_spatial_task.py',
    'development_episode.py','native_resection.py','core.py','geometry.py',
    'spatial_observations.py','sequential_spatial_observation.py','evaluation.py']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pins():
    return {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve(),
        *[ROOT/'src/resectionlab'/name for name in FILES]]}

if '--child' in sys.argv:
    declaration = json.loads((OUT/'declaration.json').read_text())
    sys.path.insert(0, str(ROOT/'src'))
    blocked = []
    def guard(event, args):
        if event in ('socket.connect','socket.bind','subprocess.Popen'):
            blocked.append(event)
            raise RuntimeError('No network/subprocess in generated STOP control')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = str(args[0])
            if '/data/' in path or path.endswith(('.pt','.pth','.ckpt','.safetensors','.nii','.nii.gz','.ressectionlab')):
                blocked.append(path)
                raise RuntimeError('No patient/model/case payload access')
    sys.addaudithook(guard)
    import numpy as np
    from resectionlab import public_contact_family as family
    from resectionlab import public_surface_contact as contact
    from resectionlab import development_episode, observed_search
    from resectionlab.core import semantic_digest, thaw_json, array_digest
    from resectionlab.native_spatial_task import NativeSpatialTask
    from resectionlab.native_resection import NativeResectionEngine
    assert Path(family.__file__).resolve() == ROOT/'src/resectionlab/public_contact_family.py'
    assert family.family_digest() == declaration['family_hash']
    def forbidden(*args, **kwargs):
        raise AssertionError('Search, planning transitions and training not authorized')
    observed_search.observed_beam_search = forbidden
    development_episode.observed_beam_search = forbidden
    NativeSpatialTask.observed_one_step_search = forbidden
    NativeSpatialTask._observed_greedy_search = forbidden
    NativeSpatialTask.advance_planning = forbidden
    stage = 'initialization'
    counts = {}
    def counter(name):
        values = counts.setdefault(stage, {})
        values[name] = values.get(name, 0)+1
    original_prepare = NativeSpatialTask._prepare_inventory
    def prepare(self):
        counter('inventory_requests')
        if self._inventory is None:
            counter('terminal_inventory_builds' if self.terminated else 'active_inventory_builds')
        return original_prepare(self)
    NativeSpatialTask._prepare_inventory = prepare
    original_preview = NativeResectionEngine.preview_stroke
    def preview(self, *args, **kwargs):
        counter('geometry_previews')
        return original_preview(self, *args, **kwargs)
    NativeResectionEngine.preview_stroke = preview
    original_transition = NativeSpatialTask._transition
    def transition(self, action, **kwargs):
        assert action == 'STOP', 'Only explicit STOP is authorized in this control'
        counter('stop_transitions')
        return original_transition(self, action, **kwargs)
    NativeSpatialTask._transition = transition
    rows=[]
    started=time.monotonic()
    for selection in declaration['ordered_selections']:
        layout_id=selection['layout_id']; goal_id=selection['goal_id']
        assert selection['role'] in ('TRAIN','SELECT')
        assert family.layout_metadata(layout_id)['role'] == selection['role']
        stage=layout_id+':factory'
        task=family.make_family_task(layout_id, goal_id, cancelled=lambda: time.monotonic()-started>17.)
        original_state=task._engine.state_hash
        stage=layout_id+':context_and_inventory'
        context,binding=family.bind_family_context(task, layout_id=layout_id, goal_id=goal_id,
            experiment_hash=semantic_digest(declaration))
        assert binding['role']==selection['role'] and not binding['training_admission']
        assert binding['family_hash']==declaration['family_hash']
        assert context.declaration_hash==semantic_digest(binding)
        context.require_task(task)
        initial=task.observation()
        context.require_observation(initial)
        inventory=task.candidate_inventory()
        assert initial.action_ids[0]=='STOP' and initial.action_modes[0]=='stop'
        assert bool(initial.action_mask[0])
        stage=layout_id+':seal_stop'
        package=contact.seal_complete_strategy(task,['STOP'])
        assert task._engine.state_hash==original_state and task.metrics()['steps']==0
        stage=layout_id+':execute_stop'
        result=task.step('STOP')
        context.require_task(task)
        context.require_observation(result.observation)
        assert result.terminated and result.reward==0.
        assert task.metrics()['history']==package['strategy']['history']
        assert semantic_digest(development_episode.replay_frames(task, task.metrics()['history']))==semantic_digest(package['replayFrames'])
        stage=layout_id+':verify_replay'
        assert contact.verify_complete_strategy(task, json.loads(json.dumps(package)))
        assert [f['phase'] for f in package['replayFrames']]==['initial','stop']
        assert package['geometryAudit']['feasible']
        assert task.metrics()['goal_retained'] and not task.metrics()['goal_contacted_and_retained']
        assert task.metrics()['removed_volume_mm3']==0 and task.metrics()['total_reward']==0.
        assert np.array_equal(task._engine.remaining_mask, task.case.observed_support)
        assert not task._engine.removed_mask.any() and not task._engine.contact_mask.any()
        assert not task._engine.probe_contact_mask.any()
        assert 'reference_hash' not in task.metrics()
        rows.append({'selection':selection,'binding':thaw_json(binding),
            'context':thaw_json(context._record()), 'initial_observation_hash':initial.fingerprint,
            'initial_inventory':inventory, 'package':package,
            'final_physical_masks':{name:array_digest(getattr(task._engine,name)) for name in
                ('remaining_mask','removed_mask','contact_mask','probe_contact_mask')},
            'assertions_passed':True})
    assert not blocked and 'torch' not in sys.modules and 'nibabel' not in sys.modules
    assert sum(v.get('stop_transitions',0) for v in counts.values())==6
    assert sum(v.get('geometry_previews',0) for v in counts.values())==90
    assert sum(v.get('active_inventory_builds',0) for v in counts.values())==6
    record={'schema':'family-positive-stop-control-v1','status':'PASS','rows':rows,
        'counts_by_stage':counts,'elapsed_seconds':time.monotonic()-started,
        'teacher_or_search_calls':0,'actor_forwards':0,'optimizer_updates':0,
        'held_out_task_executions':0,'patient_reads':0,
        'scope':'factory/context/STOP correctness only; no goal-achievement evidence'}
    (OUT/'result.json').write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'PASS','selections':declaration['ordered_selections'],
        'counts_by_stage':counts,'elapsed_seconds':record['elapsed_seconds']}))
    raise SystemExit(0)

assert not (OUT/'declaration.json').exists(), 'This fixed control is one-shot; preserve original result'
manifest=json.loads(FAMILY.read_text())
selections=[{'role':role,'layout_id':sorted(r['layout_id'] for r in manifest['source_bindings'] if r['role']==role)[0],
    'goal_id':'surface','actions':['STOP']} for role in ('TRAIN','SELECT')]
before=pins()
declaration={'schema':'family-positive-stop-control-declaration-v1',
    'family_hash':manifest['family_hash'],'manifest_sha256':sha(FAMILY),
    'source_hashes':before,'ordered_selections':selections,
    'selection_rule':'lexicographically first canonical TRAIN then first SELECT; fixed before execution',
    'wall_cap_seconds':20,'thread_count':1,'max_steps_per_control_execution':1,
    'seal_executions_per_task':1,'direct_executions_per_task':1,'replay_executions_per_task':1,
    'search':False,'training':False,'held_out_execution':False,'model_access':False,'patient_access':False}
(OUT/'declaration.json').write_text(json.dumps(declaration,indent=2,sort_keys=True)+'\n')
env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1',**{k:'1' for k in
    ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}
started=time.monotonic()
with (OUT/'terminal.log').open('w') as log:
    try:
        child=subprocess.run([str(ROOT/'.venv/bin/python'),str(Path(__file__).resolve()),'--child'],
            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=20)
        exit_code=child.returncode
    except subprocess.TimeoutExpired:
        exit_code=124
elapsed=time.monotonic()-started
after=pins()
receipt={'status':'PASS' if exit_code==0 and before==after and elapsed<20 else 'FAIL',
    'exit_code':exit_code,'elapsed_seconds':elapsed,'wall_cap_seconds':20,'thread_count':1,
    'source_hashes_before':before,'source_hashes_after':after,
    'declaration_sha256':sha(OUT/'declaration.json'),
    'result_sha256':sha(OUT/'result.json') if (OUT/'result.json').exists() else None}
(OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
print(json.dumps(receipt))
raise SystemExit(0 if receipt['status']=='PASS' else 1)
