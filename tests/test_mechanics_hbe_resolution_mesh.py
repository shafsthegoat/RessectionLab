"""Metadata and mocked-worker controls; no native mesh or specimen decks."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'scripts/mechanics_hbe_mesh.py'
BASE = ROOT / 'manifests/experiments/hbe-01-03-mechanics-poc-v1.json'
EXTENSION = ROOT / 'manifests/experiments/hbe-01-03-axial-resolution-v1.json'
spec = importlib.util.spec_from_file_location('hbe_resolution_mesh_tests', SOURCE)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_every_legacy_function_keeps_exact_historical_source():
    before = subprocess.run(['git', '-C', str(ROOT), 'show',
        '715f0bf99fdd76eb984e5522aa2c7fa538498f63:scripts/mechanics_hbe_mesh.py'],
        capture_output=True, check=True).stdout
    assert hashlib.sha256(before).hexdigest() == '3cb4d99f5d2d7fa2f66fb5af56ebffcf19342b9bee0b4cb2c9acfe98aabdd892'
    old, current = before.decode(), SOURCE.read_text()
    functions = lambda text: {node.name: ast.get_source_segment(text, node)
                              for node in ast.parse(text).body if isinstance(node, ast.FunctionDef)}
    old_functions, new_functions = functions(old), functions(current)
    assert all(new_functions[name] == body for name, body in old_functions.items())
    assert set(new_functions) - set(old_functions) == {'read_resolution_protocol', 'prepare_resolution_level'}


def test_extension_changes_only_a_returned_copy_of_levels():
    original = M.read_protocol(BASE)
    resolved, extension = M.read_resolution_protocol(BASE, EXTENSION, M.RESOLUTION_SHA256)
    assert resolved['mesher']['levels'] == extension['mesh_levels']
    resolved['mesher']['levels'] = copy.deepcopy(original['mesher']['levels'])
    assert resolved == original and M.read_protocol(BASE) == original
    assert extension['co_primary_triplets'] == [[12, 16, 24], [8, 16, 24]]


@pytest.mark.parametrize('fault', ['declared_hash', 'extension_bytes', 'base_bytes'])
def test_changed_bindings_reject_before_any_runtime_access(tmp_path, fault):
    base, extension, expected = BASE, EXTENSION, M.RESOLUTION_SHA256
    if fault == 'declared_hash':
        expected = '0' * 64
    elif fault == 'extension_bytes':
        extension = tmp_path / 'extension.json'
        extension.write_bytes(EXTENSION.read_bytes() + b'\n')
    else:
        base = tmp_path / 'base.json'
        base.write_bytes(BASE.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        M.read_resolution_protocol(base, extension, expected)
    assert 'gmsh' not in sys.modules


def fake_worker(monkeypatch, tmp_path, N, *, bad_quality=None, fail_deck=False):
    calls = {'generate': 0, 'initialize': 0, 'finalize': 0, 'decks': [], 'runtime_checks': 0}
    runtime = {'module': {'path': str(tmp_path / 'private_gmsh.py')},
               'library': {'path': str(tmp_path / 'private_gmsh.dylib')}, 'version': '4.15.2'}

    def verify(*_):
        calls['runtime_checks'] += 1
        return runtime

    def initialize(args, **kwargs):
        assert args == [] and kwargs == {'readConfigFiles': False, 'run': False}
        calls['initialize'] += 1

    def generate(dimension):
        assert dimension == 3
        calls['generate'] += 1

    gmsh = SimpleNamespace(__file__=runtime['module']['path'], __version__='4.15.2',
        lib=SimpleNamespace(_name=runtime['library']['path']), initialize=initialize,
        finalize=lambda: calls.__setitem__('finalize', calls['finalize'] + 1),
        option=SimpleNamespace(getString=lambda _: '4.15.2', setNumber=lambda *_: None),
        model=SimpleNamespace(mesh=SimpleNamespace(generate=generate)),
        write=lambda path: Path(path).write_text('mock native output; not a mesh'))
    monkeypatch.setattr(M, 'verify_runtime', verify)
    monkeypatch.setattr(M.importlib.util, 'spec_from_file_location', lambda *_: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda _: None)))
    monkeypatch.setattr(M.importlib.util, 'module_from_spec', lambda _: gmsh)
    monkeypatch.setattr(M, 'build_five_block_geometry', lambda _g, n, protocol: {'N': n})
    count = {16: 7209, 24: 23101}.get(N, 1)
    if bad_quality == 'nodes':
        count -= 1
    mesh = {'mesh_N': N, 'node_ids': list(range(1, count + 1)), 'mock_only': True}
    monkeypatch.setattr(M, 'extract_mesh', lambda *_: mesh)
    quality = {'passed': True, 'checks': {'declared_cell_count': True, 'positive_rest_jacobians': True},
               'relative_volume_error': .001, 'maximum_radial_boundary_sag_over_R': .001}
    if bad_quality == 'volume':
        quality['relative_volume_error'] = 1.
    if bad_quality == 'sag':
        quality['maximum_radial_boundary_sag_over_R'] = 1.
    monkeypatch.setattr(M, 'mesh_quality', lambda *_: copy.deepcopy(quality))

    def deck(actual, branch, steps, modulus, protocol):
        assert actual is mesh and steps == 60 and modulus == 1000.
        assert protocol['geometry'] == M.read_protocol(BASE)['geometry']
        calls['decks'].append(branch)
        if fail_deck and branch == 'tension':
            raise RuntimeError('Injected deck publication failure')
        xml = 'mock unit deck: ' + branch
        return xml, {'branch': branch, 'steps': steps, 'mu_Pa': modulus,
                     'deck_sha256': hashlib.sha256(xml.encode()).hexdigest()}

    monkeypatch.setattr(M, 'specimen_deck', deck)
    return calls


def prepare(tmp_path, N):
    return M.prepare_resolution_level(N, BASE, EXTENSION, M.RESOLUTION_SHA256,
                                      tmp_path / 'runtime.json', 'mock-runtime-binding', tmp_path / 'out')


@pytest.mark.parametrize('N', [16, 24])
def test_exact_two_deck_schedule_and_original_loading_identity(monkeypatch, tmp_path, N):
    calls = fake_worker(monkeypatch, tmp_path, N)
    receipt = prepare(tmp_path, N)
    assert receipt['status'] == 'prepared_not_solved' and receipt['gmsh_generation_calls'] == 1
    assert receipt['solver_calls'] == 0 and not receipt['curve_values_opened']
    assert calls == {'generate': 1, 'initialize': 1, 'finalize': 1,
                     'decks': ['compression', 'tension'], 'runtime_checks': 2}
    assert set(receipt['decks']) == {'compression-60-reference', 'tension-60-reference'}
    assert all(receipt['quality']['checks'][key] for key in ('declared_node_count', 'finest_volume', 'finest_boundary_sag'))
    for name in receipt['decks']:
        loading = json.loads((tmp_path / 'out' / name / 'loading.json').read_text())
        assert loading['protocol_sha256'] == M.PROTOCOL_SHA256
        assert loading['resolution_declaration_sha256'] == M.RESOLUTION_SHA256
        assert loading['mesh_sha256'] == receipt['mesh_sha256'] and loading['role'] == 'reference'
    assert 'gmsh' not in sys.modules
    with pytest.raises(FileExistsError):
        prepare(tmp_path, N)
    assert calls['generate'] == 1


@pytest.mark.parametrize('N', [4, 12, 20, True, 16.])
def test_undeclared_level_cannot_import_or_generate(monkeypatch, tmp_path, N):
    calls = fake_worker(monkeypatch, tmp_path, N)
    with pytest.raises(ValueError):
        prepare(tmp_path, N)
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert receipt['status'] == 'failed' and receipt['gmsh_generation_calls'] == 0
    assert calls['runtime_checks'] == calls['initialize'] == calls['generate'] == 0


@pytest.mark.parametrize('fault', ['volume', 'sag', 'nodes'])
def test_n16_must_pass_original_finest_geometry_and_declared_count_gates(monkeypatch, tmp_path, fault):
    calls = fake_worker(monkeypatch, tmp_path, 16, bad_quality=fault)
    with pytest.raises(ValueError, match='quality gates'):
        prepare(tmp_path, 16)
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert receipt['status'] == 'failed' and not receipt['quality']['passed']
    assert receipt['gmsh_generation_calls'] == calls['generate'] == calls['finalize'] == 1
    assert calls['decks'] == [] and (tmp_path / 'out/mesh.json').is_file()


def test_partial_deck_failure_retains_first_deck_and_never_retries(monkeypatch, tmp_path):
    calls = fake_worker(monkeypatch, tmp_path, 24, fail_deck=True)
    with pytest.raises(RuntimeError, match='publication'):
        prepare(tmp_path, 24)
    receipt = json.loads((tmp_path / 'out/receipt.json').read_text())
    assert receipt['status'] == 'failed' and set(receipt['decks']) == {'compression-60-reference'}
    assert calls['generate'] == calls['finalize'] == 1
    assert (tmp_path / 'out/compression-60-reference/specimen.feb').is_file()


def test_rejected_ambient_module_is_preserved(monkeypatch, tmp_path):
    calls = fake_worker(monkeypatch, tmp_path, 16)
    ambient = SimpleNamespace(unrelated=True)
    monkeypatch.setitem(sys.modules, 'gmsh', ambient)
    with pytest.raises(RuntimeError, match='ambient'):
        prepare(tmp_path, 16)
    assert sys.modules['gmsh'] is ambient
    assert calls['generate'] == calls['initialize'] == 0
