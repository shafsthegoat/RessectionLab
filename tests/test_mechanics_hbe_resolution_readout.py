"""Small analytical/dispatch controls; no saved specimen outputs or solvers read."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import mechanics_hbe_outputs as output
from scripts import mechanics_hbe_readout as readout


ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT/'manifests/experiments/hbe-01-03-axial-resolution-v1.json'
TIMES = [i/60 for i in range(61)]


def declaration(root):
    # Only prospective declaration and original protocol text, no linked results.
    data = DECLARATION.read_bytes()
    path = root/'declaration.json'
    path.write_bytes(data)
    body = json.loads(data)
    original = body['original_protocol']['path']
    destination = root/original
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes((ROOT/original).read_bytes())
    return {'path': path.name, 'sha256': hashlib.sha256(data).hexdigest()}


class InputReached(Exception):
    """Sentinel proving dimension dispatch without constructing a large mesh."""


def untouched():
    raise InputReached
    yield ''


@pytest.mark.parametrize('count', [20001, 23101, 25000])
def test_larger_parser_is_explicit_and_bounded_without_large_arrays(count):
    args = dict(expected_times=TIMES, item_count=count, field_count=9, record_name='nodes')
    with pytest.raises(ValueError, match='bounded output dimensions'):
        next(output.iter_data_records(untouched(), **args))
    with pytest.raises(InputReached):
        next(output.iter_resolution_records(untouched(), declaration_sha256=readout.RESOLUTION_DECLARATION_SHA256, **args))
    with pytest.raises(ValueError, match='bounded output dimensions'):
        next(output.iter_resolution_records(untouched(), declaration_sha256=readout.RESOLUTION_DECLARATION_SHA256,
             **{**args, 'item_count': 25001}))


def test_resolution_parser_requires_explicit_hash_and_unchanged_S60_grid():
    args = dict(expected_times=TIMES, item_count=1, field_count=1, record_name='nodes')
    for bad in (None, 'future', 'A'*64):
        with pytest.raises(ValueError, match='SHA256'):
            output.iter_resolution_records(untouched(), declaration_sha256=bad, **args)
    with pytest.raises(ValueError, match='S60'):
        output.iter_resolution_records(untouched(), declaration_sha256='1'*64,
            **{**args, 'expected_times': np.linspace(0, 1, 121)})
    text = '\n'.join(f'*Step = {i}\n*Time = {t:.9g}\n*Data = nodes\n1,{i}'
                     for i, t in enumerate(TIMES))
    rows = list(output.iter_resolution_records(text.splitlines(), declaration_sha256='1'*64, **args))
    assert len(rows) == 61 and rows[-1]['values'].tolist() == [[60.]]
    with pytest.raises(ValueError, match='Missing, extra or unbound output state'):
        list(output.iter_resolution_records(text.replace('*Step = 10', '*Step = 9').splitlines(),
             declaration_sha256='1'*64, **args))


@pytest.mark.parametrize('N', [16, 24])
def test_legacy_reader_cannot_opt_into_new_meshes(N):
    with pytest.raises(ValueError, match='outside frozen specimen task'):
        readout.read_run('.', {}, protocol_sha256='1'*64, expected_branch='compression',
            expected_mesh_N=N, expected_steps=60, expected_mu_Pa=1000)


@pytest.mark.parametrize('branch,N,counts', [
    ('compression', 16, (7209, 6144)), ('compression', 24, (23101, 20736)),
    ('tension', 16, (7209, 6144)), ('tension', 24, (23101, 20736)),
])
def test_exact_four_case_dispatch_keeps_science_and_counts(tmp_path, monkeypatch, branch, N, counts):
    binding = declaration(tmp_path)
    seen = []
    def isolated_engine(root, inputs, **kwargs):
        # Dispatch test only; the existing numerical engine is tested separately.
        seen.append(kwargs)
        return {'test_only_dispatch_not_a_readout': True}, None
    monkeypatch.setattr(readout, '_read_run', isolated_engine)
    receipt, cache = readout.read_resolution_run(tmp_path, {}, declaration_binding=binding,
        expected_branch=branch, expected_mesh_N=N)
    args = seen[0]
    assert (args['expected_branch'], args['expected_mesh_N'], args['expected_steps'], args['expected_mu_Pa']) == (branch, N, 60, 1000.)
    assert args['protocol_sha256'] == readout.ORIGINAL_PROTOCOL_SHA256
    assert args['maximum_items'] == 25000 and args['expected_mesh_counts'] == counts
    assert args['retain_scale_primitives'] is False and cache is None
    assert args['required_loading_fields'] == {'resolution_declaration_sha256': binding['sha256']}
    assert receipt['schema'] == 'hbe-resolution-run-readout-v1'
    assert receipt['resolution_declaration'] == binding
    with pytest.raises(InputReached):
        next(args['record_reader'](untouched(), expected_times=TIMES, item_count=counts[0],
                                   field_count=9, record_name='nodes'))


@pytest.mark.parametrize('branch,N', [('torsion_pos', 16), ('compression', 12), ('tension', 32), ('tension', 16.)])
def test_other_cases_stop_before_numeric_engine(tmp_path, monkeypatch, branch, N):
    binding = declaration(tmp_path)
    monkeypatch.setattr(readout, '_read_run', lambda *args, **kwargs: pytest.fail('No primitive read allowed'))
    with pytest.raises(ValueError, match='Only declared axial'):
        readout.read_resolution_run(tmp_path, {}, declaration_binding=binding,
            expected_branch=branch, expected_mesh_N=N)


def test_declaration_drift_rehashed_change_and_post_read_drift_fail(tmp_path, monkeypatch):
    binding = declaration(tmp_path)
    original = (tmp_path/binding['path']).read_bytes()
    (tmp_path/binding['path']).write_bytes(original+b' ')
    with pytest.raises(ValueError, match='hash changed'):
        readout.read_resolution_run(tmp_path, {}, declaration_binding=binding, expected_branch='compression', expected_mesh_N=16)
    changed = dict(binding, sha256=hashlib.sha256(original+b' ').hexdigest())
    with pytest.raises(ValueError, match='Exact separately frozen'):
        readout.read_resolution_run(tmp_path, {}, declaration_binding=changed, expected_branch='compression', expected_mesh_N=16)
    (tmp_path/binding['path']).write_bytes(original)
    def drift(*args, **kwargs):
        (tmp_path/binding['path']).write_bytes(original+b' ')
        return {'test_only': True}, None
    monkeypatch.setattr(readout, '_read_run', drift)
    with pytest.raises(ValueError, match='hash changed'):
        readout.read_resolution_run(tmp_path, {}, declaration_binding=binding, expected_branch='compression', expected_mesh_N=16)


def test_missing_loading_study_binding_and_wrong_mesh_counts_reject_before_physics(tmp_path):
    binding = declaration(tmp_path)
    spec = importlib.util.spec_from_file_location('analytical_8_node_fixture',
        Path(__file__).with_name('test_mechanics_hbe_readout.py'))
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    inputs = fixture.fixture(tmp_path)  # Tiny analytical text only, never actual output.
    def replace_json(key, edits):
        path = tmp_path/inputs[key]['path']
        value = json.loads(path.read_text())
        value.update(edits)
        data = json.dumps(value).encode()
        path.write_bytes(data)
        inputs[key]['sha256'] = hashlib.sha256(data).hexdigest()
    replace_json('mesh', {'mesh_N': 24})
    replace_json('loading', {'protocol_sha256': readout.ORIGINAL_PROTOCOL_SHA256,
                             'mesh_sha256': inputs['mesh']['sha256']})
    with pytest.raises(ValueError, match='explicitly selected study binding'):
        readout.read_resolution_run(tmp_path, inputs, declaration_binding=binding, expected_branch='tension', expected_mesh_N=24)
    replace_json('loading', {'resolution_declaration_sha256': binding['sha256']})
    with pytest.raises(ValueError, match='node/cell counts'):
        readout.read_resolution_run(tmp_path, inputs, declaration_binding=binding, expected_branch='tension', expected_mesh_N=24)
