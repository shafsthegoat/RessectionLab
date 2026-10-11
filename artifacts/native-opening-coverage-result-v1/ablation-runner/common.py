"""Small scalar contract shared by the generated pair and owned launcher."""
import errno
import hashlib
import json
import os
from pathlib import Path
import stat

TOL = 1e-10  # fixed numerical comparison allowance, not a learned threshold
HELPER = 'build/native-cut-residual-diagnostic-v1/monitor-exit-repair-v1/owned_child_exit.py'
HELPER_SHA = 'aff4b197ea033aef8b160f43eabdd5a03394ae4024b6fc2f59f5241f2fcbfe14'
SAMPLER = 'build/goal-conditioned-policy-v1/darwin_fast_sampler.py'
SAMPLER_SHA = '906a70502e4f0a48fff925cca9fc1fe45882818558cc7ac95e9034e3d6910d2c'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value, limit=1536*1024):
    raw = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    if len(raw) > limit:
        raise ValueError('Declared JSON output bound exceeded')
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def tree_bytes(root, live=False, live_output=None):
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs+files:
            path = Path(directory)/name
            try:
                value = path.lstat()
            except FileNotFoundError as error:
                if (live and error.errno == errno.ENOENT and Path(directory) == Path(live_output or root)
                        and name in files and name in {'report.json.tmp', 'progress.json.tmp'}):
                    continue
                raise
            if stat.S_ISLNK(value.st_mode):
                raise ValueError('Linked source/output refused')
            if stat.S_ISREG(value.st_mode):
                total += value.st_size
            elif not stat.S_ISDIR(value.st_mode):
                raise ValueError('Nonregular source/output refused')
    return total


def guard(root, index):
    for name, digest in index['files'].items():
        path = root/name
        if path.is_symlink() or not path.is_file() or sha(path) != digest:
            raise ValueError('Frozen source changed: '+name)


def complete_search(record):
    layers = record.get('layers', [])
    return (not record.get('call_cap_reached', True) and not record.get('time_cap_reached', True)
            and record.get('actor_forward_calls') == 0 and bool(layers)
            and record.get('completed_layers') == len(layers)
            and all(row.get('completed') is True for row in layers))


def useful_gate(arms, positive):
    complete = (len(arms) == 2 and all(a.get('status') == 'completed' and a.get('terminated') is True
                and a.get('audit', {}).get('feasible') is True and complete_search(a.get('search', {})) for a in arms))
    result = {'comparison_admissible': complete and positive.get('passed') is True,
              'useful_gain': False, 'tie_is_negative': True, 'numerical_allowance': TOL}
    if not result['comparison_admissible']:
        result['reason'] = 'Incomplete search, route audit, or positive control'; return result
    old, new = arms
    equal = all(old['identity'][key] == new['identity'][key] for key in
                ('source_hash', 'decision_model_hash', 'initial_inventory_sha256', 'reward', 'model', 'volume'))
    checks = {
        'same_model_inputs_reward_candidates': equal,
        'both_selected_routes_positive': all(a['metrics']['target_removed_mm3'] > 0. and a['metrics']['total_reward'] > 0. for a in arms),
        'whole_target_no_loss': new['metrics']['target_removed_mm3']+TOL >= old['metrics']['target_removed_mm3'],
        'whole_outside_no_increase': new['metrics']['normal_removed_mm3'] <= old['metrics']['normal_removed_mm3']+TOL,
        'return_no_loss': new['metrics']['total_reward']+TOL >= old['metrics']['total_reward'],
        'certified_target_no_loss': new['physical']['target_mm3']+TOL >= old['physical']['target_mm3'],
        'final_certified_outside_no_increase': new['physical']['outside_supplied_target_mm3'] <= old['physical']['outside_supplied_target_mm3']+TOL,
        'strictly_fewer_search_transitions': new['search']['model_transition_calls'] < old['search']['model_transition_calls'],
        'strictly_lower_preparation_outside_volume': new['preparation_outside_mm3'] < old['preparation_outside_mm3']-TOL,
    }
    result.update(checks)
    result['useful_gain'] = (all(v for k, v in checks.items() if not k.startswith('strictly_'))
                            and (checks['strictly_fewer_search_transitions'] or checks['strictly_lower_preparation_outside_volume']))
    result['reason'] = 'Predeclared paired outcome and cost gate; ties are negative'
    return result
