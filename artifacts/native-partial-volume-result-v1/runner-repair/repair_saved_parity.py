"""Saved JSON comparison only; never imports a model or changes run-01."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'src/resectionlab').is_dir())
BINDINGS = {
    'runner': ('build/native-partial-volume-v1/runner/probe_partial.py', '23929e6b1d449153365761ea906640ec24a50fb762dc4488ed3b1fb7f2fae6bd'),
    'config': ('build/native-partial-volume-v1/runner/config.json', '4092c2524de267c0584c6eaf4c345645fcd95d1a0aefd879c04e9f330b672433'),
    'current': ('build/native-partial-volume-v1/run-01/report.json', 'c9265daaf186b025403ee499a333f0e2811feab66f65e10a98a3d97f17300ee9'),
    'prior': ('build/native-cut-union-v1/run-01/report.json', '3784d71445463bd97ef717ffb0a191852cab0cc75ae85bfe4d26b8c8dff36d42'),
    'default': ('artifacts/native-generic-discretization-v1/run-01/report.json', '8e40ef8f6c4504c8eedb020687250c59cb4e58f869fe02bf530a9a001761bc9f'),
    'launcher': ('build/native-partial-volume-v1/run-01/launcher.json', 'd9be4f777fdf07567c5aa9ce4ed45a1c4ac0c96ae5b071b089d4150fd4ec177a'),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def digest(value):
    # Same byte recipe as the executed BASE.digest, with no new tolerance.
    return 'sha256:' + hashlib.sha256(canonical(value)).hexdigest()


def functions_from_frozen_runner(raw, config):
    wanted = {'projection', 'criterion', 'equivalent'}
    functions = [node for node in ast.parse(raw).body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted]
    if {node.name for node in functions} != wanted:
        raise ValueError('Frozen comparison functions absent')
    namespace = {'config': lambda: config}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen-comparison-functions>', 'exec'), namespace)
    return namespace


def corrected_comparison(current, prior, default, functions):
    project = functions['projection']
    original = deepcopy(current['useful_criterion'])
    replayed = functions['criterion'](default, prior, current)
    if replayed != original:
        raise ValueError('Original criterion does not reproduce before correction')
    prior_rows = {row['row_id']: row for row in prior['rows']}
    if len(current['rows']) != 18 or len(prior_rows) != 18 or {r['row_id'] for r in current['rows']} != set(prior_rows):
        raise ValueError('Exact original 18-row inventory required')
    corrected = deepcopy(current)
    rows = []
    for row in corrected['rows']:
        old = project(prior_rows[row['row_id']])
        new = project(row)
        old_digest, new_digest = digest(old), digest(new)
        if row['retained_union_projection_sha256'] != old_digest or row['current_legacy_projection_sha256'] != new_digest:
            raise ValueError('Stored projection hash does not reproduce')
        equal = canonical(old) == canonical(new)
        rows.append({'row_id': row['row_id'], 'original_parity': row['legacy_projection_equal_within_tolerance'],
                     'exact_saved_projection_equal': equal, 'baseline_projection_sha256': old_digest,
                     'current_projection_sha256': new_digest,
                     'refused_actions': [{'action_index': a['index'], 'reason': a['reason'], 'failure_tip_mm': a['failure_tip_mm']}
                                         for a in row['actions'] if a['status'] == 'rejected']})
        row['legacy_projection_equal_within_tolerance'] = equal
    result = functions['criterion'](default, prior, corrected)
    allowed = {'legacy_action_whole_cell_reward_projection_unchanged', 'meets_predeclared_useful_criterion'}
    if any(result[key] != original[key] for key in original if key not in allowed) or set(result) != set(original):
        raise ValueError('Unrelated scientific criterion changed')
    return {'schema': 'native-partial-volume-saved-parity-correction-v1',
            'status': 'saved_projection_comparison_complete', 'original_criterion': original,
            'corrected_criterion': result, 'rows': rows,
            'all_18_saved_projections_exactly_equal': all(r['exact_saved_projection_equal'] for r in rows),
            'numeric_tolerance_used_for_saved_projection_comparison': 0,
            'cause': 'Original comparator rejects ordinary JSON float versus in-memory numpy.float64 failure-tip coordinates; native_resection.finish constructs tuple(failure_tip) from an ndarray.',
            'correction_scope': 'Exact saved projection equality only; original run/report/verdict preserved; no physical metric, completion, reward, refusal or success threshold changed.',
            'new_native_calls': 0, 'new_independent_geometry_replays': 0, 'new_plane_clipping_calls': 0}


def bound_inputs():
    result = {}
    for key, (name, expected) in BINDINGS.items():
        path = ROOT / name
        if path.is_symlink() or path.stat().st_size > 1024 * 1024:
            raise ValueError('Bound small source/report required')
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Original saved binding changed: ' + key)
        result[key] = raw
    return result


def calculate():
    raw = bound_inputs()
    def load(key):
        def refuse(token):
            raise ValueError('Nonfinite saved value: ' + token)
        return json.loads(raw[key], parse_constant=refuse)
    launcher = load('launcher')
    if launcher['status'] != 'completed' or launcher['report_sha256'] != BINDINGS['current'][1]:
        raise ValueError('Original completed process receipt required')
    functions = functions_from_frozen_runner(raw['runner'], load('config'))
    result = corrected_comparison(load('current'), load('prior'), load('default'), functions)
    result['input_bindings'] = {k: {'path': v[0], 'sha256': v[1]} for k, v in BINDINGS.items()}
    if bound_inputs() != raw:
        raise ValueError('Original inputs changed during saved comparison')
    return result


def main():
    started = time.monotonic()
    result = calculate()
    result['elapsed_seconds'] = time.monotonic() - started
    result['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode() + b'\n'
    if len(encoded) > 32 * 1024:
        raise ValueError('Compact correction cap')
    destination = HERE / 'saved-parity-correction.json'
    with destination.open('xb') as stream:
        stream.write(encoded)
    print(json.dumps({'path': str(destination), 'sha256': hashlib.sha256(encoded).hexdigest(),
                      'bytes': len(encoded), 'corrected_criterion': result['corrected_criterion']}))
    return 0 if result['all_18_saved_projections_exactly_equal'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
