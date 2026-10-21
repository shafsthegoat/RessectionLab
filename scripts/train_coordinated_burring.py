#!/usr/bin/env python3
"""Reproduce a frozen shallow-burring task, three Q learners and native replay.

All generated datasets and demonstration bundles stay in ignored testcases/.
No training on patients beyond this single declared development source.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.coordinated_burring import ACTIONS, BurringTask, QPolicy, TaskConfig
from resectionlab.core import array_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_rl_protection import GuardConfig, load_protection
from resectionlab.surgical_telemetry import capture_episode, export_telemetry
from resectionlab.workspace_bundle import load_workspace, save_workspace

SETTINGS = dict(seeds=[17, 29, 43], episodes=48, alpha=.25, gamma=.95,
                epsilon=.35, evaluation_random_seeds=list(range(100, 130)),
                wall_budget_seconds=900, transition_budget=1600)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def run(output, finalize_existing=False):
    started = time.perf_counter()
    if not finalize_existing and output.exists() and (output / 'protocol.json').exists():
        raise ValueError('Refusing to overwrite an existing attempt; use a distinct output directory')
    output.mkdir(parents=True, exist_ok=True)
    recipe = json.loads((ROOT / 'artifacts/integrated-rehearsal-v1/protocol.json').read_text())['cases'][0]
    source = ROOT / recipe['case']
    if digest(source) != recipe['sha256'] or recipe['role'] != 'development':
        raise ValueError('Pinned development source changed')
    scene = load_environment(source)
    protection = load_protection(scene, ROOT / recipe['protection'], strict=True)
    entry = recipe['actions'][0]['pose']
    pocket = copy.deepcopy(entry)
    pocket['position'][2] = -404.79
    config = TaskConfig()
    protocol = dict(schema='resectionlab.coordinated-burring.v1', source=str(source),
        source_sha256=digest(source), protection_sha256=digest(ROOT / recipe['protection']),
        scene_id=scene.identity, label_hash=array_digest(scene.labels),
        config=asdict(config), settings=SETTINGS, entry_pose=entry, pocket_pose=pocket,
        burr_pose=recipe['actions'][1]['pose'],
        actions=ACTIONS, code_hashes={str(p.relative_to(ROOT)): digest(p) for p in
        [ROOT / 'scripts/train_coordinated_burring.py', *sorted((ROOT / 'src/resectionlab').glob('surgical*.py')),
         ROOT / 'src/resectionlab/coordinated_burring.py']},
        scope='one development anatomy; fixed spatial macros; learned instrument scheduling only',
        reward='0.2 capped bone progress + 0.2 capped aspirated debris + 1 success - 0.05 per decision - 0.001 per simulated second - 1 failure',
        clinical_validation='none')
    if finalize_existing:
        frozen = json.loads((output / 'protocol.json').read_text())
        for path, expected in frozen['code_hashes'].items():
            if path != 'scripts/train_coordinated_burring.py' and digest(ROOT / path) != expected:
                raise ValueError('Frozen simulator/task changed: ' + path)
        assert frozen['config'] == asdict(config) and frozen['settings'] == SETTINGS
        assert frozen['source_sha256'] == digest(source)
        protocol = frozen
        write(output / 'finalization-repair.json', dict(
            original_runner_sha256=frozen['code_hashes']['scripts/train_coordinated_burring.py'],
            repaired_runner_sha256=digest(Path(__file__)), extra_optimizer_updates=0,
            reason='Handle source bundles without a workspace or prior surgical_toolbox artifact; resume export only.'))
    else:
        write(output / 'protocol.json', protocol)  # Freeze BEFORE baseline and fitting.
    costs = dict(engine_calls=0, cache_hits=0, engine_seconds=0., max_action_seconds=0.)

    def command(engine, action):
        t = time.perf_counter()
        result = engine.step(action)
        elapsed = time.perf_counter() - t
        costs['engine_calls'] += 1
        costs['engine_seconds'] += elapsed
        costs['max_action_seconds'] = max(costs['max_action_seconds'], elapsed)
        return result

    engine = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig())
    if command(engine, dict(slot='right', operation='exchange', instrument='frazier_suction', duration=.2))['status'] != 'accepted':
        raise AssertionError('Setup failed')
    initial = BurringTask(engine, entry, pocket, config, burr_pose=recipe['actions'][1]['pose'])
    cache = {(): initial}

    def transition(prefix, choice):
        if time.perf_counter() - started > SETTINGS['wall_budget_seconds'] or costs['engine_calls'] >= SETTINGS['transition_budget']:
            raise RuntimeError('Frozen resource budget exhausted')
        key = (*prefix, choice)
        if key in cache:
            costs['cache_hits'] += 1
        else:
            task = cache[prefix].clone()
            # Instrument every real public call without changing its engine class.
            original_step = task.engine.step
            def measured(action):
                t = time.perf_counter()
                result = original_step(action)
                elapsed = time.perf_counter() - t
                costs['engine_calls'] += 1
                costs['engine_seconds'] += elapsed
                costs['max_action_seconds'] = max(costs['max_action_seconds'], elapsed)
                return result
            task.engine.step = measured
            try:
                task.step(choice)
            finally:
                # Never copy a bound method or closure to another cached engine.
                del task.engine.step
            cache[key] = task
        task = cache[key]
        return key, task

    def episode(choose):
        prefix, task = (), initial
        while not task.done:
            prefix, task = transition(prefix, int(choose(task.observe())))
        return prefix, task

    def report(task):
        return dict(success=task.success(), reason=task.reason, return_=task.total_reward,
            actions=[d['action'] for d in task.decisions], decisions=task.decisions,
            measurements=task.measurements, debris_mm3=task.engine.state['debrisMm3'],
            aspirated_mm3=task.engine.state.get('debrisRemovedMm3', 0.),
            state_id=task.engine.state_id, guard=task.engine.guard_report())

    if not finalize_existing:
        baselines = {}
        for name, sequence in dict(scripted=[1, 2], drill_only=[1]*4, suction_only=[2]*4,
                                   reversed_once=[2, 1, 0], stop=[0]).items():
            choices = iter(sequence)
            _, task = episode(lambda obs: next(choices, 0))
            baselines[name] = report(task)
        write(output / 'baselines.json', baselines)
        if not baselines['scripted']['success'] or any(baselines[k]['success'] for k in ('drill_only', 'suction_only', 'reversed_once', 'stop')):
            raise AssertionError('Two-instrument/order necessity gate failed; no fitting admitted')
        print('Scripted and instrument-ablation gate passed', flush=True)

        learners, curves = [], []
        for seed in SETTINGS['seeds']:
            rng = np.random.default_rng(seed)
            policy = QPolicy()
            for index in range(SETTINGS['episodes']):
                prefix, task = (), initial
                errors = []
                while not task.done:
                    obs = task.observe()
                    choice = int(rng.integers(3)) if rng.random() < SETTINGS['epsilon'] else policy.choose(obs)
                    prefix, following = transition(prefix, choice)
                    errors.append(policy.update(obs, choice, following.decisions[-1]['reward'],
                        following.observe(), following.done, SETTINGS['alpha'], SETTINGS['gamma']))
                    task = following
                curves.append(dict(seed=seed, episode=index, success=task.success(),
                                   return_=task.total_reward, td_errors=errors, updates=policy.updates,
                                   trajectory=task.decisions))
            learners.append(policy)
            write(output / f'q-seed-{seed}.json', policy.record())
            print('Fit seed', seed, 'updates', policy.updates, flush=True)
        # Freeze ALL policies before evaluation. No seed or checkpoint selection.
        write(output / 'learning-curve.json', curves)
        checkpoints = {seed: digest(output / f'q-seed-{seed}.json') for seed in SETTINGS['seeds']}
        evaluation = []
        finals = []
        for seed, policy in zip(SETTINGS['seeds'], learners):
            _, task = episode(policy.choose)
            finals.append(task)
            evaluation.append(dict(method='Q_learning', seed=seed, **report(task)))
            _, untrained = episode(QPolicy().choose)
            evaluation.append(dict(method='untrained', seed=seed, **report(untrained)))
        for seed in SETTINGS['evaluation_random_seeds']:
            rng = np.random.default_rng(seed)
            _, task = episode(lambda obs: int(rng.integers(3)))
            evaluation.append(dict(method='random', seed=seed, **report(task)))
        for seed in SETTINGS['seeds']:
            assert digest(output / f'q-seed-{seed}.json') == checkpoints[seed]
        write(output / 'evaluation.json', evaluation)
        write(output / 'fit-cost.json', {**costs, 'wall_seconds': time.perf_counter()-started})
    else:
        baselines = json.loads((output / 'baselines.json').read_text())
        evaluation = json.loads((output / 'evaluation.json').read_text())
        learners, finals = [], []
        checkpoints = {}
        for seed in SETTINGS['seeds']:
            saved = json.loads((output / f'q-seed-{seed}.json').read_text())
            policy = QPolicy()
            policy.table = {tuple(row['state']): np.asarray(row['values']) for row in saved['table']}
            policy.updates = saved['updates']
            learners.append(policy)
            checkpoints[seed] = digest(output / f'q-seed-{seed}.json')
            _, task = episode(policy.choose)
            original = next(row for row in evaluation if row['method']=='Q_learning' and row['seed']==seed)
            assert report(task) == {k:v for k,v in original.items() if k not in ('method','seed')}
            finals.append(task)
        print('Resumed finalization of frozen checkpoints; zero optimizer updates', flush=True)

    # Every seed gets a fresh engine replay, independent source-cell accounting,
    # and a native bundle. No checkpoint is promoted or hidden by best-seed choice.
    case, artifacts, workspace = load_workspace(source)
    image_hash = array_digest(case.mri)
    replays, telemetry = [], []
    auxiliary_replay_calls = 0
    for seed, task in zip(SETTINGS['seeds'], finals):
        snapshot = task.engine.snapshot()
        fresh = SurgicalEngine(scene, protection=protection, guard_config=GuardConfig())
        for action in snapshot['actions']:
            if command(fresh, action)['status'] == 'rejected':
                raise AssertionError('Fresh public replay rejected')
        assert fresh.state_id == snapshot['stateId']
        removed = np.asarray(fresh.state['removed'], dtype=int)
        material = {r['id']: r['material'] for r in scene.regions}
        assert all(material[int(scene.labels.flat[c])] == 'bone' for c in removed)
        assert len(removed) * scene.voxel_mm3 == task.measurements['bone_removed_mm3']
        assert abs(len(removed) * scene.voxel_mm3 - fresh.state['debrisMm3'] -
                   fresh.state.get('debrisRemovedMm3', 0.) - fresh.state.get('debrisEscapedMm3', 0.)) < 1e-8
        record = copy.deepcopy(artifacts)
        record.setdefault('surgical_toolbox', {}).update(mapping=scene.regions,
            branches={'main': snapshot}, branchId='main',
            position=len(snapshot['actions']), parents={'main': None}, timeline={'main': fresh.history})
        destination = output / f'learned-seed-{seed}.ressectionlab'
        if workspace is None:
            save_case(case, destination, artifacts=record)
        else:
            save_workspace(case, destination, sources=workspace['sources'], imaging=workspace['imagingState'],
                           episode=workspace['episode'], selection=workspace['selection'],
                           originals=workspace['originals'], artifacts=record)
        check = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_integrated_rehearsal.py'),
            '--check-saved', str(destination)], cwd=ROOT, capture_output=True, text=True, check=True)
        reopened = json.loads(check.stdout)
        auxiliary_replay_calls += len(snapshot['actions'])
        assert reopened['stateId'] == fresh.state_id
        replays.append(dict(seed=seed, state_id=fresh.state_id, native=str(destination),
            fresh_process=reopened, source_cell_and_mass_balance=True))
        write(output / f'replay-seed-{seed}.json', snapshot)
        telemetry.append(capture_episode(fresh, label='RL', episode_id=f'Q-seed-{seed}'))
        auxiliary_replay_calls += len(snapshot['actions'])
    export_telemetry(telemetry, output / 'telemetry')
    assert array_digest(scene.labels) == protocol['label_hash'] and array_digest(case.mri) == image_hash
    result = dict(status='complete', baselines=baselines, evaluation=evaluation,
        learned_successes=sum(t.success() for t in finals), learned_seeds=len(finals),
        random_successes=sum(r['success'] for r in evaluation if r['method']=='random'),
        random_episodes=len(SETTINGS['evaluation_random_seeds']),
        updates=[p.updates for p in learners], checkpoints=checkpoints, replays=replays,
        source_arrays_unchanged=True, source_image_hash=image_hash,
        costs={**costs, 'wall_seconds':time.perf_counter()-started,
               'auxiliary_fresh_process_and_telemetry_replay_calls': auxiliary_replay_calls,
               'total_public_transition_calls': costs['engine_calls'] + auxiliary_replay_calls,
               'max_action_seconds_scope': 'in-process counted engine calls; excludes child and telemetry calls',
               'peak_process_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},
        limitations=['one fixed development anatomy, no patient generalization',
            'learned discrete scheduling, scripted spatial motion',
            'exact deterministic prefix cache reused for sampled Q updates; cache hits are not new simulator calls',
            'fixed task evaluation repeats do not constitute independent patient trials',
            'shallow pocket, no through-skull opening or clinical validation'])
    if finalize_existing:
        result['previous_fit_cost'] = json.loads((output / 'fit-cost.json').read_text())
        result['costs']['scope'] = 'resumed export only; previous_fit_cost is separate; interrupted export cost unavailable'
        result['finalization_only_no_extra_updates'] = True
    write(output / 'result.json', result)
    print(json.dumps({k:result[k] for k in ('learned_successes','learned_seeds','random_successes','random_episodes','updates','costs')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'testcases/coordinated-burring-v1')
    parser.add_argument('--finalize-existing', action='store_true', help='Resume export of frozen policies without fitting')
    args = parser.parse_args()
    run(args.output.resolve(), args.finalize_existing)
