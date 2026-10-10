"""Fixed saved-teacher readout and existing native greedy baseline; no fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = 'build/five-state-policy-diagnostic-v1/attempt-01'
TRAIN = ('ReMIND-008', 'ReMIND-010', 'ReMIND-020', 'ReMIND-025')
RUNS = {1: 'build/cross-patient-planning-v1/fixed-four-train-pilot-v1',
        8: 'build/cross-patient-planning-v1/fixed-four-train-eight-updates-v1'}
CAPS = dict(worker_seconds=240, parent_seconds=270, memory_bytes=3*1024**3,
            native_previews=16000, policy_forwards=25, checkpoint_loads=4,
            optimizer_updates=0, greedy_seconds_per_case=60, threads=1,
            output_bytes=32*1024**2, supervision_bytes=16*1024**2)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())
def canonical(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def semantic(value): return 'sha256:'+hashlib.sha256(canonical(value).encode()).hexdigest()


def source_guard(release, path, digest):
    if (sha(path) != digest or release.get('status') != 'released_one_attempt'
            or release.get('output') != OUTPUT or release.get('caps') != CAPS
            or release.get('TRAIN') != list(TRAIN) or release.get('attempts') != 1
            or release.get('SELECT_EVAL_execution') is not False
            or release.get('private_reference_execution') is not False):
        raise ValueError('Exact diagnostic release required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip()
    if head != release['expected_head']: raise ValueError('Source HEAD changed')
    ref = release['source_index']; index_path = ROOT/ref['path']
    if sha(index_path) != ref['sha256']: raise ValueError('Source/input index changed')
    index = read(index_path)
    if index['head'] != head: raise ValueError('Index HEAD differs')
    for section in ('source_files', 'metadata_files'):
        for relative, expected in index[section].items():
            item = Path(relative)
            if item.is_absolute() or '..' in item.parts or (ROOT/item).is_symlink():
                raise ValueError('Bound regular local source/metadata required')
            if sha(ROOT/item) != expected: raise ValueError('Changed source/input: '+relative)
    return index


def completed_training():
    records = {}
    for updates, directory in RUNS.items():
        parent = ROOT/directory; result = read(parent/'attempt-01/result.json')
        receipt = read(parent/'attempt-01.supervision/receipt.json')
        release = read(parent/'root-release.json')
        if (receipt['status'] != 'complete' or receipt['exit_code'] != 0
                or receipt['cleanup_errors'] or receipt['final_owned_pids']
                or receipt['worker_termination_confirmed'] is not True
                or receipt['release_sha256'] != sha(parent/'root-release.json')
                or receipt['result_sha256'] != sha(parent/'attempt-01/result.json')
                or result['status'] != 'complete_TRAIN_only_not_heldout_performance'
                or result['optimizer_updates'] != {'IL': updates, 'RL': updates}
                or result['SELECT_EVAL_opened'] is not False):
            raise ValueError('Clean fixed endpoint required before checkpoint decoding')
        records[updates] = dict(root=parent, result=result, release=release,
            contexts={s: read(parent/('attempt-01/'+s+'-context.json')) for s in TRAIN})
    if records[1]['result']['initial_parameter_hash'] != records[8]['result']['initial_parameter_hash']:
        raise ValueError('Prior runs do not share initial tensors')
    expected = dict(records[1]['release']['learning_protocol']); expected['updates_per_method'] = 8
    if canonical(expected) != canonical(records[8]['release']['learning_protocol']):
        raise ValueError('Historical protocols differ beyond declared updates')
    return records


def complete_result(result):
    return (result.get('status') == 'complete_fixed_five_state_readout_and_native_greedy'
        and result.get('policy_forwards') == 25 and result.get('checkpoint_loads') == 4
        and result.get('optimizer_updates') == 0 and result.get('teacher_states') == 5
        and result.get('complete_greedy_replays') == 4 and set(result.get('patients',{})) == set(TRAIN)
        and result.get('SELECT_EVAL_opened') is False and result.get('private_reference_reads') == 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path, required=True)
    parser.add_argument('--release-sha256', required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT/'src'))
    from resectionlab.legacy_transfer_worker import _require_parent_lease
    _require_parent_lease()  # Before torch, models, checkpoint or patient input.
    release = read(args.release); source_guard(release, args.release, args.release_sha256)
    prior = completed_training()
    output = ROOT/OUTPUT; output.mkdir(exist_ok=False)
    supervision = output.with_name(output.name+'.supervision')
    if not supervision.is_dir(): raise ValueError('Owned parent reservation required')
    started = time.perf_counter()
    def progress(phase, **details):
        p = supervision/'worker-progress.tmp'
        p.write_text(json.dumps(dict(phase=phase, seconds=time.perf_counter()-started, **details))+'\n')
        p.replace(supervision/'worker-progress.json')
    def deadline(*unused): raise TimeoutError('Readout/baseline worker deadline')
    previous = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, CAPS['worker_seconds'])
    result = dict(status='started', policy_forwards=0, checkpoint_loads=0,
        optimizer_updates=0, teacher_states=0, complete_greedy_replays=0,
        SELECT_EVAL_opened=False, private_reference_reads=0, patients={})
    budget = costs = sink = None
    try:
        import gc
        import weakref
        import numpy as np
        import torch
        from resectionlab.core import thaw_json
        from resectionlab.native_resection import NativeResectionEngine
        from resectionlab.planning_budget import PlanningBudget
        from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash, world_to_sample_grid
        from resectionlab.public_target_context import public_target_features
        from resectionlab.patient_planning_cohort_io import load_cohort_checkpoint, OutputBudget
        from resectionlab.patient_planning_cohort_visits import make_train_visit_factories
        from resectionlab.patient_planning_cohort_sequential import _bounded_writes
        from resectionlab.patient_planning_preflight import _CallCosts, _collect, _seal_and_replay
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        result['runtime'] = dict(python=sys.version, executable=sys.executable,
            torch=torch.__version__, numpy=np.__version__, threads=1, interop_threads=1,
            device='cpu', dtype='float32')
        sink = OutputBudget(output, CAPS['output_bytes'])
        protocol = prior[8]['release']['learning_protocol']
        models = {}; metadata = {}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(protocol['seed'])
            models['INITIAL'] = SpatialPolicy(SpatialPolicyConfig(**protocol['architecture']),
                public_target_context_variant=protocol['public_target_context_variant'])
        if parameter_hash(models['INITIAL']) != prior[8]['result']['initial_parameter_hash']:
            raise ValueError('Exact common initial parameters did not reconstruct')
        for updates in (1, 8):
            record = prior[updates]
            contexts = {'ReMIND:'+s.rsplit('-', 1)[1]: semantic(record['contexts'][s]) for s in TRAIN}
            for method in ('IL', 'RL'):
                pin = record['result']['checkpoints'][method]
                if pin['path'] != method+'-final.psckpt': raise ValueError('Unexpected checkpoint path')
                name = method+'_'+str(updates)
                load_started = time.perf_counter()
                models[name], meta = load_cohort_checkpoint(record['root']/'attempt-01'/pin['path'],
                    expected_sha256=pin['sha256'], expected_learning_protocol=record['release']['learning_protocol'],
                    expected_context_hashes=contexts, expected_method=method)
                result['checkpoint_loads'] += 1
                if parameter_hash(models[name]) != pin['parameter_hash']: raise ValueError('Checkpoint parameters differ')
                metadata[name] = dict(file_sha256=pin['sha256'], parameter_hash=pin['parameter_hash'],
                    completed_updates=meta['completed_updates'], learning_protocol_hash=meta['learning_protocol_hash'],
                    load_seconds=time.perf_counter()-load_started)
        frozen = {k: parameter_hash(p) for k,p in models.items()}
        for model in models.values(): model.eval(); model.requires_grad_(False)
        sink.write(output/'model-identities.json', dict(parameters=frozen, checkpoints=metadata))
        old = prior[8]['release']; public = old['public_manifest_index']
        factories = make_train_visit_factories(manifest_index_path=ROOT/public['path'],
            manifest_index_sha256=public['sha256'], cohort_bytes=(ROOT/'manifests/experiments/remind-component-cohort-v1.json').read_bytes(),
            learning_protocol=protocol, limits=old['cohort_limits'], released_record=old,
            released_sha256=sha(prior[8]['root']/'root-release.json'), progress=progress)
        budget = PlanningBudget(NativeResectionEngine, max_native_previews=CAPS['native_previews'], seconds=CAPS['worker_seconds'])
        costs = _CallCosts(budget, CAPS['policy_forwards'])
        def guard(): budget.check(); sink.check()
        with budget, costs, _bounded_writes(sink):
            for subject in TRAIN:
                progress('public_source', subject=subject)
                dest = output/subject; dest.mkdir(); source = dest/'source'; source.mkdir()
                with costs.scope(subject+'.public_reconstruction'):
                    base, context = factories[subject](output=source)
                    if canonical(context.record()) != canonical(prior[8]['contexts'][subject]):
                        raise ValueError('Reconstructed exact historical context differs')
                    context.require_task(base); case_ref = weakref.ref(base.case)
                saved = {u: read(prior[u]['root']/('attempt-01/teachers/'+subject+'/complete-trace.json')) for u in (1,8)}
                actions = [d['action_id'] for d in saved[8]['decisions']]
                if any(len(saved[u]['decisions']) != (2 if subject=='ReMIND-025' else 1) for u in (1,8)):
                    raise ValueError('The fixed five teacher states changed')
                model_task = base.planning_clone()
                with costs.scope(subject+'.fixed_teacher_states'):
                    for step, action in enumerate(actions):
                        guard(); observation = model_task.observation(); context.require_observations((observation,))
                        for u in (1,8):
                            expected = saved[u]['decisions'][step]
                            if (expected['action_id'] != action or expected['observation_hash'] != observation.fingerprint
                                    or expected['action_ids'] != list(observation.action_ids)
                                    or expected['action_mask'] != observation.action_mask.tolist()):
                                raise ValueError('Exact frozen teacher state/action/inventory differs')
                        geom = observation.action_geometry
                        points = geom[:,None,1:4]+np.linspace(0,1,protocol['architecture']['ray_samples'])[None,:,None]*(geom[:,None,4:7]-geom[:,None,1:4])
                        grids = world_to_sample_grid(points, observation.affine_ras_mm, observation.image_channels.shape[1:])
                        inbounds = (np.abs(grids)<=1).all(-1); inbounds[0] = False
                        public, relation = public_target_features(observation, reference_mm=protocol['architecture']['physical_reference_mm'])
                        inputs = dict(observation_hash=observation.fingerprint,
                            channel_available=observation.channel_available.tolist(),
                            target_positive_covered_crop_cells=int(np.count_nonzero((observation.image_channels[2]>0)&observation.coverage[2])),
                            target_coverage_cells=int(np.count_nonzero(observation.coverage[2])),
                            ray_sample_inbounds=inbounds.tolist(), public_target_features=public.tolist(),
                            public_target_relation=relation.tolist(), action_geometry=geom.tolist(),
                            scope='public input presence and geometry, not causal feature attribution')
                        scores = {}; label = observation.action_ids.index(action)
                        for name, model in models.items():
                            guard()
                            with torch.no_grad(): logits, value = model(observation)
                            legal = np.flatnonzero(observation.action_mask)
                            logp = logits.log_softmax(-1); probs = logits.softmax(-1)
                            raw = logits.cpu().numpy(); rank = sorted(legal, key=lambda i:(-float(raw[i]),int(i)))
                            movement = [int(i) for i in rank if i]
                            best = movement[0] if movement else None
                            scores[name] = dict(logits=[float(raw[i]) if observation.action_mask[i] else None for i in range(len(raw))],
                                probabilities=probs.tolist(), teacher_probability=float(probs[label]),
                                teacher_CE=float(-logp[label]), teacher_rank=rank.index(label)+1,
                                teacher_movement_rank=None if label==0 else movement.index(label)+1,
                                greedy_action=observation.action_ids[rank[0]], stop_probability=float(probs[0]),
                                teacher_minus_STOP=float(raw[label]-raw[0]),
                                best_movement_minus_STOP=None if best is None else float(raw[best]-raw[0]), value=float(value))
                        sink.write(dest/f'state-{step:02d}.json', dict(subject=subject, step=step,
                            teacher_action=action, action_ids=list(observation.action_ids), action_mask=observation.action_mask.tolist(),
                            inputs=inputs, scores=scores))
                        outcome = model_task.advance_planning(action)
                        if (outcome.reward != saved[8]['decisions'][step]['reward']
                                or outcome.terminated != saved[8]['decisions'][step]['terminated']):
                            raise ValueError('Teacher committed outcome changed')
                        result['teacher_states'] += 1
                    if (not model_task.terminated
                            or semantic(model_task.metrics()['history']) != semantic(saved[8]['metrics']['history'])):
                        raise ValueError('Frozen complete teacher history differs')
                del model_task, observation, outcome
                progress('native_greedy', subject=subject)
                greedy_dir = dest/'GREEDY'; greedy_dir.mkdir()
                with costs.scope(subject+'.native_greedy_selection'):
                    try: selected, accounting = base.observed_greedy_search(seconds=CAPS['greedy_seconds_per_case'])
                    except BaseException as error:
                        sink.write(greedy_dir/'selection-failure.json', dict(type=type(error).__name__, message=str(error), accounting=getattr(error,'accounting',None)))
                        raise
                    sink.write(greedy_dir/'selection.json', dict(actions=selected, accounting=accounting))
                    if accounting.get('complete') is not True: raise ValueError('Incomplete native greedy selection')
                with costs.scope(subject+'.native_greedy_collection_replay'):
                    trace = _collect(base, context, actions=selected, output=greedy_dir, guard=guard)
                    sealed = _seal_and_replay(base, context, trace, method='NATIVE_GREEDY', policy=None, updates=0, output=greedy_dir, guard=guard)
                    actual_return = sum(t.reward for t in trace.transitions)
                    if actual_return != accounting['estimated_incremental_return']: raise ValueError('Greedy replay reward differs')
                result['patients'][subject] = dict(actions=list(selected), return_value=actual_return,
                    saved_SEARCH_actions=actions, actions_equal_SEARCH=list(selected)==actions,
                    candidate_scores=accounting['evaluated_nonstop_actions'], selection_commits=accounting['model_transition_calls'],
                    plan_seal=sealed['plan_seal'])
                result['complete_greedy_replays'] += 1
                del trace, sealed, base; gc.collect()
                if case_ref() is not None: raise ValueError('Patient source retained across visits')
            if any(parameter_hash(p)!=frozen[k] or p.training or any(q.requires_grad or q.grad is not None for q in p.parameters()) for k,p in models.items()):
                raise ValueError('Readout changed frozen weights/gradient state')
            result['policy_forwards'] = costs.forwards
            if (result['teacher_states'],result['policy_forwards'],result['checkpoint_loads'],result['complete_greedy_replays']) != (5,25,4,4):
                raise ValueError('Fixed diagnostic counts differ')
            guard(); budget.complete(history_complete=True)
        source_guard(release, args.release, args.release_sha256)
        result['status'] = 'complete_fixed_five_state_readout_and_native_greedy'
    except BaseException as error:
        result.update(status='failed_or_unresolved', failure=dict(type=type(error).__name__,message=str(error)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        result['wall_seconds'] = time.perf_counter()-started
        if costs is not None:
            result['policy_forwards'] = costs.forwards
            result['costs'] = costs.rows
        if budget is not None: result['native_budget'] = budget.snapshot()
        result['cost_scope'] = 'historical training costs separate; current checkpoint/readout, source, selection and complete replay measured; nested costs are inclusive'
        result['historical_context_scope'] = 'exact prior eight-update context replayed for identity; current owned execution authorizes zero updates only'
        with (output/'result.json').open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')


if __name__ == '__main__': main()
