"""Pure scalar algebra, declaration and refusal controls only.

No meshes, scientific response arrays, fabricated response logs, native calls,
study preparation, release creation or measured-data access are used.
"""
import ast
from copy import deepcopy
import hashlib
import inspect
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_halfheight_global as core
from scripts import mechanics_hbe_halfheight_global_readout as reader
from scripts import mechanics_hbe_halfheight_global_experiment as runner


@pytest.fixture(autouse=True)
def no_execution_or_response_parsing(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Native execution, preparation and response parsing forbidden in pure controls')
    for module, name in ((runner.subprocess, 'Popen'), (runner.old, 'solve'), (runner.runtime, 'supervise'),
                         (core, 'generate_full_mesh'), (reader, '_records'), (runner, 'replay_P1'),
                         (reader, 'replay_prior_runs'), (reader.original_readout, 'read_run'),
                         (reader.original_readout, 'read_resolution_run')):
        monkeypatch.setattr(module, name, forbidden)


@pytest.fixture(scope='module')
def study():
    return core.declaration(ROOT, {'path': core.DECLARATION_PATH, 'sha256': core.DECLARATION_SHA256})


def test_frozen_counts_limits_design_and_unchanged_sources(study):
    assert len(study['inherited_source_sha256']) == 14
    for key, binding in study['baseline_P1']['source_bindings'].items():
        source = ROOT/'scripts'/Path(binding['path']).name
        assert hashlib.sha256(source.read_bytes()).hexdigest() == study['inherited_source_sha256'][key]
    assert study['ordered_runs'] == [core.RUN_ID]
    assert core.COUNTS == {'full_nodes': 53329, 'full_elements': 49152, 'half_nodes': 28233,
                           'half_elements': 24576, 'midplane_nodes': 3137}
    n = 32
    cross = 3*n*n+2*n+1
    assert cross == core.COUNTS['midplane_nodes']
    assert cross*(n//2+1) == core.COUNTS['full_nodes']
    assert cross*(n//4+1) == core.COUNTS['half_nodes']
    assert 3*n*n*(n//2) == core.COUNTS['full_elements']
    assert 3*n*n*(n//4) == core.COUNTS['half_elements']
    assert study['budgets'] == {'aggregate_specimen_seconds': 1200, 'each_active_run_output_bytes': 768*1024**2,
        'each_solver_seconds': 750, 'generated_output_bytes': 2*1024**3, 'gmsh_generation_calls': 1,
        'maximum_specimen_solver_calls': 1, 'pure_preparation_seconds': 60, 'runtime_threads': 1,
        'sampled_process_family_rss_bytes': 3*1024**3}
    assert study['measured_data_access'] is study['calibration'] is study['patient_data_access'] is False
    proposal = core.access.verify_binding(ROOT, study['reviewed_proposal'], read_json=True)
    review = core.access.verify_binding(ROOT, study['independent_design_review'], read_json=True)
    assert review['blocking_defects'] == [] and review['proposal_sha256'] == study['reviewed_proposal']['sha256']
    assert proposal['case'] == study['case']
    assert all(proposal[k] == v for k, v in study['numerical_design'].items())


def test_only_mesh_level_selection_changes(study):
    protocol = core.access.verify_binding(ROOT, study['original_protocol'], read_json=True)
    adapted = core.geometry_protocol(protocol, study)
    assert adapted['mesher']['levels'] == [{'N': 32, 'expected_nodes': 53329, 'nominal_hex8_cells': 49152}]
    adapted['mesher']['levels'] = protocol['mesher']['levels']
    assert adapted == protocol
    altered = deepcopy(study); altered['budgets']['each_solver_seconds'] += 1
    with pytest.raises(ValueError, match='frozen'):
        core.require_study(altered)
    with pytest.raises(ValueError, match='Exact newly prepared'):
        core.extract_halfheight({'mesh_N': 24}, protocol, study)


@pytest.mark.parametrize('levels', reader.TRIPLETS)
@pytest.mark.parametrize('p', [.25, 1., 2., 4.])
@pytest.mark.parametrize('sign', [-1, 1])
def test_scalar_power_law_recovers_signed_limit(levels, p, sign):
    # Dimensionless algebraic sequence, not fabricated specimen observations.
    values = [3.+sign*1000./n**p for n in levels]
    result = reader.unequal_order(levels, values)
    assert result['status'] == 'eligible'
    assert result['order'] == pytest.approx(p, abs=2e-8)
    assert result['force_limit_N'] == pytest.approx(3., abs=1e-9)
    assert result['signed_remaining_indicator_N'] == pytest.approx(3.-values[-1], abs=1e-9)
    assert result['continuum_error_bound'] is False


@pytest.mark.parametrize('values,status', [
    ((0., 2., 1.), 'opposite_signs'),
    ((0., 1e-8, 2e-8), 'below_original_difference_floor'),
    ((0., 1e-8, 1.), 'insufficient_increment_resolution'),
    ((0., 1., 3.), 'no_admissible_positive_root'),
    ((math.nan, 0., 1.), 'invalid'),
    ((-1e308, 1e308, 1e308), 'invalid'),
])
def test_null_scalar_estimates_remain_unresolved(values, status):
    result = reader.unequal_order((16, 24, 32), values)
    assert result['status'] == status
    assert result['order'] is result['force_limit_N'] is result['signed_remaining_indicator_N'] is None


def test_root_domain_monotonicity_and_sublinear_increasing_raw_changes():
    a, b = math.log(24/16), math.log(32/24)
    ratios = [reader.order_ratio(i/100, a, b) for i in range(1601)]
    assert all(second > first for first, second in zip(ratios, ratios[1:]))
    for ratio in (a/b, ratios[-1]*1.01, reader.order_ratio(5e-7, a, b)):
        assert reader.unequal_order((16, 24, 32), (0., ratio, ratio+1.))['order'] is None
    values = [1000./n**.25 for n in (12, 16, 24)]
    assert abs(values[2]-values[1]) > abs(values[1]-values[0])
    assert reader.unequal_order((12, 16, 24), values)['order'] == pytest.approx(.25)


def test_scalar_envelope_cannot_hide_remaining_bias_or_unresolved_state():
    forces = {n: 3.+1000./n for n in (8, 12, 16, 24, 32)}
    state = reader.state_diagnostic(forces, index=60, force_limit_N=.01)
    assert state['status'] == 'resolved_conditional_model'
    assert state['two_latest_limit_envelope_N'] == pytest.approx(1000./32)
    assert state['remaining_within_allowance'] is False
    assert state['force_limit_instability'] is state['increasing_order_drift'] is False
    constant = {n: 1. for n in forces}
    state = reader.state_diagnostic(constant, index=1, force_limit_N=.01)
    assert state['status'] == 'unresolved' and state['two_latest_limit_envelope_N'] is None
    assert reader.state_diagnostic(constant, index=0, force_limit_N=.01)['status'] == 'rest/not_estimated'


def test_frozen_forecast_remains_forecast_and_fails_remaining_screen(study):
    # Scalar arithmetic already declared in the reviewed proposal; no saved
    # response series is opened, and the prospective N32 value is not observed.
    forces = {8: -.037106420547429, 12: -.036784800891650496,
              16: -.0365813469705313, 24: -.03633856601773044,
              32: study['numerical_design']['forecast_not_observation']['endpoint_force_N']}
    allowance = reader.FORCE_FLOOR_N+.02*abs(forces[32])
    result = reader.state_diagnostic(forces, index=60, force_limit_N=allowance)
    assert result['orders']['N8-N12-N16']['order'] == pytest.approx(.33003322965017934)
    assert result['orders']['N12-N16-N24']['order'] == pytest.approx(.4825833378127983)
    assert result['orders']['N16-N24-N32']['order'] == pytest.approx(.4825833378127983)
    assert abs(forces[32]-forces[24]) < allowance
    assert result['remaining_within_allowance'] is False
    assert result['two_latest_limit_envelope_N'] > allowance


@pytest.mark.parametrize('kind', ['nodes', 'elements'])
def test_exact_new_parser_call_contract_without_log_fixture(study, monkeypatch, kind):
    seen = {}
    def parser(lines, **kwargs):
        seen.update(kwargs)
        return iter(())
    monkeypatch.setattr(reader.outputs, '_iter_data_records', parser)
    spec = study['parser'][kind]
    args = dict(study=study, kind=kind, item_count=spec['items'], field_count=spec['fields'],
                record_name=spec['name'], expected_times=reader.TIMES, expected_run_id=core.RUN_ID)
    assert list(reader.iter_global_records((), **args)) == []
    assert seen['maximum_bytes'] == spec['maximum_bytes'] and seen['maximum_items'] == spec['items']
    for name, value in (('item_count', spec['items']+1), ('field_count', spec['fields']-1),
                        ('record_name', 'other'), ('expected_run_id', 'compression:N24:S60:reference'),
                        ('expected_times', (0., 1.))):
        with pytest.raises(ValueError, match='Exact N32'):
            reader.iter_global_records((), **dict(args, **{name: value}))


def test_node_authentication_uses_384_mib_on_both_sides(monkeypatch):
    seen = {}
    def verify(root, binding, *, maximum_bytes):
        seen[binding['path']] = maximum_bytes
    monkeypatch.setattr(reader.access, 'verify_binding', verify)
    reader.verify_native_bindings(ROOT, {name: {'path': name} for name in reader.inherited.PRIMITIVE_KEYS})
    assert seen['nodes'] == 384*1024**2 and seen['elements'] == 256*1024**2 and seen['solver'] == 16*1024**2
    tree = ast.parse(inspect.getsource(reader.read_global_run))
    assert sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'verify_native_bindings'
               for n in ast.walk(tree)) == 2


def test_legacy_parser_limits_remain_unchanged():
    with pytest.raises(ValueError, match='Invalid bounded output'):
        next(reader.outputs.iter_data_records((), expected_times=reader.TIMES, item_count=20001,
              field_count=9, record_name='x'))
    with pytest.raises(ValueError, match='Invalid bounded output'):
        next(reader.outputs.iter_resolution_records((), expected_times=reader.TIMES, item_count=25001,
              field_count=9, record_name='x', declaration_sha256=core.DECLARATION_SHA256))


def test_undeclared_phase_and_incomplete_native_metadata_are_refused():
    with pytest.raises(ValueError, match='Separate preparation'):
        runner.preflight(ROOT, {}, {}, None)
    with pytest.raises(ValueError, match='Complete bounded'):
        runner.require_completed_native({}, 750)
    compile(runner.P1_REPLAY_CODE, '<fixed-original-context-replay>', 'exec')


@pytest.mark.parametrize('phase,release', [('prepare', {}), ('solve', {'authorized': True}),
                                        ('prepare', {'schema': 'hbe-halfheight-global-release-v1', 'authorized': False})])
def test_no_release_inferred_from_declaration(study, monkeypatch, phase, release):
    binding = {'path': core.DECLARATION_PATH, 'sha256': core.DECLARATION_SHA256}
    monkeypatch.setattr(core, 'declaration', lambda *_: study)
    monkeypatch.setattr(runner.access, 'verify_binding', lambda root, record, **kw: study if record == binding else release)
    with pytest.raises(ValueError, match='Separate explicit source-bound'):
        runner.preflight(ROOT, binding, {'path': 'absent-release.json', 'sha256': '0'*64}, phase)
