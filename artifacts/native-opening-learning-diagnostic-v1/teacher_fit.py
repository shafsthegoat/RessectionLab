"""One released 15-forward generated-fixture diagnostic; no training or search."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SAVED = ROOT / 'artifacts/native-opening-learning-v1'
SNAPSHOT = SAVED / 'source-snapshot'
SETTINGS = {'max_forwards': 15, 'wall_seconds': 30., 'rss_bytes': 2*1024**3,
            'cpu_threads': 1, 'optimizer_updates': 0, 'prefix_transitions': 4,
            'teacher_states': 5, 'retry': False}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def verify(settings):
    if settings['settings'] != SETTINGS or sha(__file__) != settings['script_sha256']:
        raise ValueError('Changed diagnostic source/settings')
    for name, digest in settings['input_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Changed immutable binding: '+name)
    return True


def worker(settings, output):
    started = time.perf_counter()
    record = {'status':'starting', 'settings':SETTINGS, 'forward_attempts':0, 'forward_completed':0,
              'prefix_transitions':0, 'states':[], 'models':{}, 'optimizer_updates':0}
    def preserve():
        record['elapsed_seconds'] = time.perf_counter()-started
        write(output/'readout.json', record)
    def guard():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform != 'darwin': rss *= 1024
        if time.perf_counter()-started >= 28. or rss > SETTINGS['rss_bytes']:
            raise InterruptedError('Diagnostic cooperative time/RSS limit')
    preserve()
    try:
        verify(settings)
        sys.path.insert(0, str(SNAPSHOT/'src'))
        import numpy as np
        import torch
        from resectionlab.spatial_policy import SpatialPolicy, SpatialPolicyConfig, parameter_hash
        from resectionlab.native_spatial_task import make_native_opening_task
        from resectionlab.data_policy import GeneratedDevelopmentContext
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        declaration = json.loads((SAVED/'declaration-input.json').read_text())
        preparation = json.loads((SAVED/'preparation.json').read_text())
        result = json.loads((SAVED/'result.json').read_text())
        teacher = json.loads((SAVED/'teacher.json').read_text())
        proofs = json.loads((SAVED/'teacher-demonstration-checks.json').read_text())
        expected_rows = {r['observation_hash']:r for r in teacher['state_rows']}
        if len(expected_rows) != 5 or len(proofs) != 5 or not all(p['accepted'] for p in proofs):
            raise ValueError('Original five independently validated teacher states required')
        base = make_native_opening_task(cancelled=lambda: bool(guard()))
        context = GeneratedDevelopmentContext(sha(SAVED/'declaration-input.json'),
                    (base.case.source_hash,), base.decision_model_hash)
        context.require_task(base)
        if (base.case.source_hash != preparation['source_hash']
                or base.decision_model_hash != preparation['decision_model_hash']
                or base.observation().fingerprint != preparation['initial_observation_hash']):
            raise ValueError('Original task/context differs')
        observations = []
        reconstruction_started = time.perf_counter()
        for proof in proofs:
            guard()
            if len(proof['prefix']) not in (0,1):
                raise ValueError('Only original nonterminal one-step prefixes admitted')
            task = base.planning_clone()
            for action in proof['prefix']:
                record['prefix_transitions'] += 1
                if record['prefix_transitions'] > 4: raise ValueError('Prefix call cap')
                task.step(action)
            observation = task.observation()
            context.require_observations((observation,))
            expected = expected_rows[proof['observation_hash']]
            if (observation.fingerprint != proof['observation_hash']
                    or list(observation.action_ids) != expected['action_ids']
                    or expected['selected_action_id'] != proof['selected_action_id']
                    or not all(observation.action_mask) or task.terminated):
                raise ValueError('Reconstructed observation/action differs from frozen teacher')
            observations.append((observation, expected['selected_action_id']))
            record['states'].append({'observation_hash':observation.fingerprint,
                                     'prefix':proof['prefix'], 'teacher_action':expected['selected_action_id']})
            preserve()
        record['reconstruction_seconds'] = time.perf_counter()-reconstruction_started
        record['context'] = asdict(context)
        for name, filename in [('initial','initial.pt'),('BC','BC-latest.pt'),('scratch-RL','scratch-RL-latest.pt')]:
            guard()
            # Original architecture metadata contains NumPy float64 scalars.
            # Same narrow, temporary allowlist as the saved-output auditor.
            unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(SAVED/filename)
            if set(unsafe) != {'numpy._core.multiarray.scalar', 'numpy.dtype'}:
                raise ValueError('Unexpected checkpoint metadata globals')
            with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, type(np.dtype('f8'))]):
                checkpoint = torch.load(SAVED/filename, map_location='cpu', weights_only=True)
            admitted = GeneratedDevelopmentContext(**checkpoint['context'])
            if admitted != context:
                raise ValueError('Checkpoint generated-development context mismatch')
            if name != 'initial' and (checkpoint['updates'] != 16
                    or checkpoint['initial_parameter_hash'] != result['initial_parameter_hash']):
                raise ValueError('Wrong fixed final checkpoint or ancestry')
            expected_hash = result['initial_parameter_hash'] if name == 'initial' else result['training'][name]['latest_parameter_hash']
            model = SpatialPolicy(SpatialPolicyConfig(**declaration['policy']))
            if model.architecture_record() != checkpoint['architecture']:
                raise ValueError('Checkpoint architecture mismatch')
            model.load_state_dict(checkpoint['policy'], strict=True)
            model.eval()
            before = parameter_hash(model)
            if before != checkpoint['parameter_hash'] or before != expected_hash:
                raise ValueError('Checkpoint parameter hash mismatch')
            row = {'status':'reading', 'initial_parameter_hash':before, 'context_fingerprint':context.fingerprint,
                   'architecture_hash':model.architecture_hash, 'checkpoint_sha256':sha(SAVED/filename), 'predictions':[]}
            record['models'][name] = row
            try:
                for observation, label in observations:
                    guard()
                    record['forward_attempts'] += 1
                    if record['forward_attempts'] > 15: raise ValueError('Forward call cap')
                    preserve()
                    begin = time.perf_counter()
                    with torch.inference_mode():
                        logits, value = model(observation)
                        probabilities = logits.softmax(-1)
                        label_index = observation.action_ids.index(label)
                        competitors = logits.clone(); competitors[label_index] = -torch.inf
                        top = int(logits.argmax())
                        non_stop_top = int(logits[1:].argmax())+1
                        selected_probability = float(probabilities[label_index])
                        prediction = {'observation_hash':observation.fingerprint,
                            'teacher_action':label, 'teacher_probability':selected_probability,
                            'teacher_cross_entropy':float(-logits.log_softmax(-1)[label_index]),
                            'teacher_rank':1+sum(float(v)>float(logits[label_index]) for v in logits),
                            'selected_action':observation.action_ids[top], 'correct':top==label_index,
                            'teacher_logit_margin':float(logits[label_index]-competitors.max()),
                            'teacher_probability_margin':selected_probability-float(probabilities[[j for j in range(len(logits)) if j != label_index]].max()),
                            'stop_probability':float(probabilities[0]),
                            'best_nonstop_action':observation.action_ids[non_stop_top],
                            'stop_minus_best_nonstop_logit':float(logits[0]-logits[non_stop_top]),
                            'probabilities':[float(v) for v in probabilities],
                            'logits':[float(v) for v in logits], 'value':float(value),
                            'forward_seconds':time.perf_counter()-begin}
                    record['forward_completed'] += 1
                    row['predictions'].append(prediction)
                    preserve()
            finally:
                row['final_parameter_hash'] = parameter_hash(model)
                row['weights_unchanged'] = row['final_parameter_hash'] == before
                preserve()
            if not row['weights_unchanged']: raise ValueError('Diagnostic mutated model weights')
            row['mean_cross_entropy'] = sum(x['teacher_cross_entropy'] for x in row['predictions'])/5
            row['correct_states'] = sum(x['correct'] for x in row['predictions'])
            row['status'] = 'complete'
            del checkpoint, model
        for name, module in tuple(sys.modules.items()):
            if name == 'resectionlab' or name.startswith('resectionlab.'):
                source = Path(module.__file__).resolve()
                if not source.is_relative_to(SNAPSHOT) or str(source.relative_to(SNAPSHOT)) not in declaration['source_sha256']:
                    raise ValueError('Local numerical module outside original source snapshot')
        verify(settings)
        record.update(status='complete', exact_input_source_hashes_unchanged=True,
                      note='Teacher-fit diagnostic only; no optimizer, new search or evaluation rollout.')
        preserve()
    except BaseException as error:
        record.update(status='failed', error={'type':type(error).__name__, 'message':str(error),
                                             'traceback':traceback.format_exc()})
        preserve()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if sha(args.settings) != args.expected_sha256: raise ValueError('Changed prospective settings')
    settings = json.loads(args.settings.read_text()); verify(settings)
    output = HERE/'teacher-fit-attempt-01'
    if args.worker:
        worker(settings, output)
        return
    output.mkdir(exist_ok=False)
    write(output/'release.json', settings)
    spec = importlib.util.spec_from_file_location('diagnostic_supervisor', ROOT/'scripts/febio_runtime.py')
    supervisor = importlib.util.module_from_spec(spec); spec.loader.exec_module(supervisor)
    environment = {k:v for k,v in os.environ.items() if not k.startswith(('DYLD_', 'PYTHON'))}
    environment.update({'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                        'MKL_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'})
    receipt = supervisor.supervise([sys.executable, str(Path(__file__).resolve()), '--worker', '--settings',
        str(args.settings.resolve()), '--expected-sha256', args.expected_sha256], output/'supervision',
        cwd=ROOT, environment=environment, seconds=30., rss_bytes=2*1024**3)
    verify(settings)
    print(json.dumps(receipt, indent=2))
    if receipt['status'] != 'completed': raise SystemExit(1)


if __name__ == '__main__': main()
