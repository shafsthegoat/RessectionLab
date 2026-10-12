"""Fixed runner contract; no task/model/array imports."""
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import sysconfig

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
TRAIN = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
IL = 'build/post-exposure-learning-v1/IL64/attempt-01'
RL = 'build/post-exposure-learning-rl8-recovery-v1/RL8/attempt-01'
OLD_INDEX = 'artifacts/post-exposure-rl8-result-v1/audit-input-index.json'
PUBLIC = 'build/remind-partial-domain-public-preparation-v1/manifests-v2/public-index.json'
PUBLIC_SHA = 'c5138559e45fd67b44b047e8ea6d800b56b8e63ae3371a5ff5c9d83e60b25fe8'
COHORT = 'manifests/experiments/remind-component-cohort-v1.json'
COHORT_SHA = '326b4ebb4a6e439e47fb166d8fcfeec5ff65798294820d5aac0ed240b21fdd05'
HELPER = 'build/native-cut-residual-diagnostic-v1/monitor-exit-repair-v1/owned_child_exit.py'
HELPER_SHA = 'aff4b197ea033aef8b160f43eabdd5a03394ae4024b6fc2f59f5241f2fcbfe14'
SAMPLER = 'build/goal-conditioned-policy-v1/darwin_fast_sampler.py'
SAMPLER_SHA = '906a70502e4f0a48fff925cca9fc1fe45882818558cc7ac95e9034e3d6910d2c'
CHECKPOINTS = {
    'IL64': {'path': IL+'/IL-final.psckpt', 'sha256': 'c174b92b09ed8eccd034a7bc8bc7cb9a4d64e09aa40d6f7a3d596ae88ea57c5c', 'bytes': 137636,
             'parameters': 'sha256:59506939917827f7555a61c930c33285dcb62feda52ec98576194a82e39587f3'},
    'RL8': {'path': RL+'/RL-final.psckpt', 'sha256': '87181c166a8c7255e11bcc57d362ec60edd822617786d2c5a729006a5037c34d', 'bytes': 137635,
            'parameters': 'sha256:39a44a88a19c634e938878e8029dcafc04a7669f1b7fa69422ab57d502015b4d'}}
ORIGINAL_RUN='build/train-continuation-value-v1/run-01'
ORIGINAL_REPORT_SHA='9c8c9534151e4287883f33dd530a9b64e02a1e70929a3d12e2bff4ae731e53ee'
ORIGINAL_TARGETS_SHA='a346840c7dcc360229f3a75643bff57f6d411de835e073a9b62f236adeec35f6'
ORIGINAL_AUDIT='build/train-continuation-value-v1/independent/audit-01.json'
ORIGINAL_AUDIT_SHA='40a69aecc0d25e3b454bf87074c7fff058d338b99d4652813e4952cf02758fe9'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def semantic(value):
    return 'sha256:'+hashlib.sha256(canonical(value)).hexdigest()


def read_bytes(path, expected=None, limit=8*1024**2):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError('Bounded regular input required: '+str(path))
        with os.fdopen(fd, 'rb', closefd=False) as stream: raw = stream.read(limit+1)
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or len(raw) != before.st_size):
            raise ValueError('Input changed during bounded read')
    finally: os.close(fd)
    if expected is not None and hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Pinned bytes changed: '+str(path))
    return raw


def sha(path): return hashlib.sha256(read_bytes(path)).hexdigest()
def read_json(path, expected=None): return json.loads(read_bytes(path, expected))


def runtime_roots():
    """Only interpreter-selected dependency directories, never dataset/build roots."""
    roots=sorted({str(Path(sysconfig.get_paths()[key]).resolve()) for key in ('purelib','platlib')})
    if any(not Path(root).is_relative_to((ROOT/'.venv').resolve()) or Path(root).name!='site-packages' for root in roots):
        raise ValueError('Existing repository venv dependency roots required')
    return roots


def write_json(path, value, limit=2*1024**2):
    raw = canonical(value)+b'\n'
    if len(raw) > limit: raise ValueError('Declared JSON output cap')
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def tree_bytes(root, live=False, live_output=None):
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs+files:
            path = Path(directory)/name
            try: value = path.lstat()
            except FileNotFoundError as error:
                if (live and error.errno == errno.ENOENT and Path(directory) == Path(live_output or root)
                        and name in files and name in {'report.json.tmp', 'progress.json.tmp'}): continue
                raise
            if stat.S_ISLNK(value.st_mode): raise ValueError('Linked stage entry refused')
            if stat.S_ISREG(value.st_mode): total += value.st_size
            elif not stat.S_ISDIR(value.st_mode): raise ValueError('Nonregular stage entry')
    return total


def guard(root, index):
    for name, digest in index['files'].items(): read_bytes(root/name, digest)


def useful_gate(cases):
    """All four outcomes preserved; one strict aggregate work/route gain, ties negative."""
    tol = 1e-8
    valid = set(cases) == set(TRAIN) and all(set(rows) == {'GREEDY','Q64','CALIBRATED'}
        and all(r.get('complete') is True and r.get('audit', {}).get('accepted') is True
                for r in rows.values()) for rows in cases.values())
    result = {'comparison_admissible': valid, 'useful_gain': False, 'tie_is_negative': True,
              'tolerance': tol, 'primary_comparison': 'fresh_full_inventory_GREEDY_vs_CALIBRATED'}
    if not valid: return result
    checks = {}
    for subject, rows in cases.items():
        a, b = rows['GREEDY'], rows['CALIBRATED']; x, y = a['audit']['outcomes'], b['audit']['outcomes']
        checks[subject] = {
            'same_world': a['world'] == b['world'],
            'target_no_loss': y['target_removed_mm3']+tol >= x['target_removed_mm3'],
            'outside_no_increase': y['normal_removed_mm3'] <= x['normal_removed_mm3']+tol,
            'contact_no_increase': y['cumulative_contacted_tissue_upper_bound_mm3'] <= x['cumulative_contacted_tissue_upper_bound_mm3']+tol,
            'return_no_loss': y['total_reward']+tol >= x['total_reward']}
    def sum_field(method, field): return sum(r[method]['audit']['outcomes'][field] for r in cases.values())
    preview = {m: sum(r[m]['planning_costs']['native_preview_entries'] for r in cases.values()) for m in ('GREEDY','CALIBRATED')}
    transitions = {m: sum(r[m]['planning_costs'].get('native_transition_calls', 0) for r in cases.values()) for m in ('GREEDY','CALIBRATED')}
    gains = {'fewer_planning_previews': preview['CALIBRATED'] < preview['GREEDY'],
             'fewer_planning_transitions': transitions['CALIBRATED'] < transitions['GREEDY'],
             'shorter_complete_tool_path': sum_field('CALIBRATED','complete_tool_path_length_mm') < sum_field('GREEDY','complete_tool_path_length_mm')-tol,
             'fewer_executed_nonstop_actions': sum_field('CALIBRATED','nonstop_actions') < sum_field('GREEDY','nonstop_actions')}
    result.update(checks=checks, strict_gains=gains, planning_previews=preview, planning_transitions=transitions,
                  useful_gain=all(all(r.values()) for r in checks.values()) and any(gains.values()),
                  setup_and_head_training_amortized=False)
    return result


def join_planning_replay(accounting, result, *, method):
    """Join retained candidate scoring to committed independent replay, not IDs alone."""
    decisions=accounting['decisions']; actual=result['decisions']
    if len(decisions)!=len(actual): raise ValueError('Planning/replay decision count differs')
    total=0.
    for old,new in zip(decisions,actual):
        selected=old.get('selected_action_id',old.get('action_id'))
        if selected!=new['action_id']:raise ValueError('Planning/replay action changed')
        if 'scores' in old:
            scores=[r for r in old['scores'] if r['action_id']==selected]
            if len(scores)!=1 or scores[0]['reward']!=new['reward']:raise ValueError('Actual nominal guard/replay reward changed')
        else:
            if old['reward']!=new['reward']:raise ValueError('Frozen endpoint replay reward changed')
        for key in ('source_state_hash','result_state_hash','observation_hash','terminated'):
            if key in old and old[key]!=new[key]:raise ValueError('Planning/replay '+key+' differs')
        if 'actual_reward' in old and old['actual_reward']!=new['reward']:raise ValueError('Committed planned reward differs')
        total+=new['reward']
    if 'estimated_incremental_return' in accounting and total!=accounting['estimated_incremental_return']:
        raise ValueError('Complete planned and replay return differ')
    if method in ('Q64','CALIBRATED'):
        if (accounting['nominal_history_hash']!=result['nominal_history_hash']
                or accounting['source_hash']!=result['world']['source_hash']
                or accounting['decision_model_hash']!=result['world']['decision_model_hash']):
            raise ValueError('Exact Q nominal history/source/model differs after independent replay')
    return True


def require_baseline_fit(actual, original):
    """Timing/corpus provenance may differ; identical math must reproduce exactly."""
    keys=('base_parameter_hash','architecture_hash','initial_head_hash','fitted_head_hash',
          'config','updates','final_weighted_mse','target','weighting','initialization')
    if any(actual[k]!=original[k] for k in keys):raise ValueError('Original fixed64 head did not reproduce')
    return True


def require_baseline_route(actual, original):
    keys=('actions','world','nominal_history_hash')
    if any(actual[k]!=original[k] for k in keys) or actual['audit']['outcomes']!=original['audit']['outcomes']:
        raise ValueError('Retained fixed baseline route did not reproduce')
    return True


CACHE_ATTEMPT='build/continuation-next-iteration-v1/run-01'
CACHE_SOURCE_INDEX='build/continuation-next-iteration-v1/runner/source-index.json'
CACHE_SOURCE_SHA='37c03f09c3754429ecb2b8fa24e5093a7efd11424ef33c1c093a43ce1b15a3b5'
CACHE_PINS={
    'report.json':'64bd8f8d46efdd0f85baac977de3e34b2105301ec48a4b1ac9ab9735dce81a32',
    'launcher.json':'613f388fda57d8963db7bd6b063988bfeab723223173fa575c66b99b52792b15',
    'cache-binding.json':'fd7dafde7547e73229cb4cb833e6be899a6fc5514837fa383a34bc5718f494ba',
    'cache-receipt.json':'2584f76c894f456d7a3151fd501f17e796250de31a0c92cb2e8ac21009fcef0d',
    'reconstruction-evidence.json':'f6ce724fb9b98c54c3f3121d843d8ee1cff885a271f74f14a53b4d286fbfb475',
    'authenticated-features.zip':'2ebd7bd2ad777492bec95fa910ee2fb1c4405da04a30f69219b448ae20d236f7',
    'logged-targets.json':'a346840c7dcc360229f3a75643bff57f6d411de835e073a9b62f236adeec35f6',
    'trace-manifest.json':'c53bde6cfece252006d6a3f39789a9b897abf938a104287cb3c7ebb485720476'}


def require_cache_origin(prior,parent,binding,receipt,evidence,source_index):
    """Pure metadata joins before any parameter/cache decode; no inferred success."""
    cleanup=parent['owned_lifecycle']
    if (prior['status']!='failed_or_incomplete' or prior['failure']['type']!='FileExistsError'
            or prior['failure']['phase']!='authenticated_cache_save_and_exact_reload'
            or prior['head_optimizer_updates']!=0 or prior['calibration_calls_attempted']!=0
            or prior['cases'] or prior['source_unchanged'] is not True
            or parent['status']!='incomplete_or_failed' or parent['worker_exit_code']!=1
            or not cleanup['verified'] or not cleanup['child_reaped'] or cleanup['exit_code']!=1
            or cleanup['errors'] or cleanup['remaining_owned_pids']
            or parent['report_sha256']!=CACHE_PINS['report.json']
            or parent['source_index_sha256']!=CACHE_SOURCE_SHA
            or prior['source_index_sha256']!=CACHE_SOURCE_SHA
            or source_index['execution_released'] is not True):
        raise ValueError('Exact cleanly contained pre-fit cache publication failure required')
    origin=binding['origin']; costs=origin['original_reconstruction_costs']
    if (receipt['binding']!=binding or receipt['sha256']!=CACHE_PINS['authenticated-features.zip']
            or receipt['bytes']!=670261 or receipt['original_reconstruction_costs']!=costs
            or origin['source_index_sha256']!=CACHE_SOURCE_SHA
            or origin['encoder_checkpoint_sha256']!=CHECKPOINTS['IL64']['sha256']
            or origin['reconstruction_evidence_sha256']!=CACHE_PINS['reconstruction-evidence.json']
            or evidence['source_index_sha256']!=CACHE_SOURCE_SHA
            or evidence['corpus_seal']!=binding['corpus_seal'] or prior['corpus_seal']!=binding['corpus_seal']
            or evidence['original_reconstruction_costs']!=costs
            or evidence['reconstructed_traces']!=prior['reconstructed_traces']
            or len(evidence['reconstructed_traces'])!=36
            or sum(r['states'] for r in evidence['reconstructed_traces'])!=497
            or costs['native_preview_entries']!=45560 or costs['native_transition_calls']!=497
            or costs['frozen_policy_forwards']!=497 or prior['total_frozen_policy_forwards']!=497
            or prior['native_budget']['native_preview_entries']!=45560
            or any(r['audit']['accepted'] is not True for r in evidence['reconstructed_traces'])):
        raise ValueError('Exact authenticated complete cache origin/costs differ')
    for name,digest in origin['feature_source_sha256'].items():
        if source_index['files']['src/'+name.replace('.','/')+'.py']!=digest:
            raise ValueError('Original feature producer source binding differs')
    return True
