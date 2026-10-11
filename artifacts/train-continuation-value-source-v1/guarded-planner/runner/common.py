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
    valid = set(cases) == set(TRAIN) and all(set(rows) == {'GREEDY','IL64','RL8','Q_GUIDED'}
        and all(r.get('complete') is True and r.get('audit', {}).get('accepted') is True
                for r in rows.values()) for rows in cases.values())
    result = {'comparison_admissible': valid, 'useful_gain': False, 'tie_is_negative': True,
              'tolerance': tol, 'primary_comparison': 'fresh_full_inventory_GREEDY_vs_guarded_Q_GUIDED'}
    if not valid: return result
    checks = {}
    for subject, rows in cases.items():
        a, b = rows['GREEDY'], rows['Q_GUIDED']; x, y = a['audit']['outcomes'], b['audit']['outcomes']
        checks[subject] = {
            'same_world': a['world'] == b['world'],
            'target_no_loss': y['target_removed_mm3']+tol >= x['target_removed_mm3'],
            'outside_no_increase': y['normal_removed_mm3'] <= x['normal_removed_mm3']+tol,
            'contact_no_increase': y['cumulative_contacted_tissue_upper_bound_mm3'] <= x['cumulative_contacted_tissue_upper_bound_mm3']+tol,
            'return_no_loss': y['total_reward']+tol >= x['total_reward']}
    def sum_field(method, field): return sum(r[method]['audit']['outcomes'][field] for r in cases.values())
    preview = {m: sum(r[m]['planning_costs']['native_preview_entries'] for r in cases.values()) for m in ('GREEDY','Q_GUIDED')}
    transitions = {m: sum(r[m]['planning_costs'].get('native_transition_calls', 0) for r in cases.values()) for m in ('GREEDY','Q_GUIDED')}
    gains = {'fewer_planning_previews': preview['Q_GUIDED'] < preview['GREEDY'],
             'fewer_planning_transitions': transitions['Q_GUIDED'] < transitions['GREEDY'],
             'shorter_complete_tool_path': sum_field('Q_GUIDED','complete_tool_path_length_mm') < sum_field('GREEDY','complete_tool_path_length_mm')-tol,
             'fewer_executed_nonstop_actions': sum_field('Q_GUIDED','nonstop_actions') < sum_field('GREEDY','nonstop_actions')}
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
    if method=='Q_GUIDED':
        if (accounting['nominal_history_hash']!=result['nominal_history_hash']
                or accounting['source_hash']!=result['world']['source_hash']
                or accounting['decision_model_hash']!=result['world']['decision_model_hash']):
            raise ValueError('Exact Q nominal history/source/model differs after independent replay')
    return True
