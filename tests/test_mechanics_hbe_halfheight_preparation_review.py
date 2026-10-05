"""Independent analytical topology and prepared-input integration controls.

No saved specimen, native mesher, solver or measured response is opened. The
analytical box deliberately bypasses only the public specimen-count gate.
"""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from scripts import mechanics_hbe_halfheight_mesh as mesh
from scripts import mechanics_hbe_halfheight_experiment as runner

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('halfheight_review_box_factory',
    Path(__file__).with_name('test_mechanics_hbe_halfheight_mesh.py'))
fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)


def protocol():
    return mesh.BASE.read_protocol(ROOT/'manifests/experiments/hbe-01-03-mechanics-poc-v1.json')


def test_reflection_maps_work_with_reversed_original_node_and_cell_numbering():
    p = protocol(); full = fixtures.box(p)
    count = len(full['node_ids']); remap = lambda old: count+1-old
    full['rest_nodes_m'].reverse(); full['gmsh_node_ids'].reverse()
    full['elements_hex8'] = [[remap(i) for i in cell] for cell in reversed(full['elements_hex8'])]
    full['gmsh_element_ids'].reverse()
    for boundary in full['boundaries'].values():
        boundary['node_ids'] = sorted(map(remap, boundary['node_ids']))
        boundary['faces_quad4'] = [[remap(i) for i in face] for face in boundary['faces_quad4']]
    before = deepcopy(full); extracted = mesh._extract_verified(full, p)
    assert full == before
    mapping, half = extracted['mapping'], extracted['mesh']
    assert mapping['half_to_full_node_ids'] != list(range(1,19))
    assert mapping['half_to_full_element_ids'] == [5,6,7,8]
    X = np.asarray(full['rest_nodes_m']); x = np.asarray(half['rest_nodes_m'])
    np.testing.assert_array_equal(x, X[np.asarray(mapping['half_to_full_node_ids'])-1])
    covered = []
    for reflection in mapping['reflections']:
        expected = x*np.asarray(reflection['signs'])+np.asarray(reflection['rest_offset_m'])
        np.testing.assert_array_equal(expected, X[np.asarray(reflection['node_full_ids'])-1])
        covered.extend(reflection['element_full_ids'])
        for row, target, permutation in zip(half['elements_hex8'], reflection['element_full_ids'], reflection['element_half_to_full_local']):
            reflected = np.asarray(reflection['node_full_ids'])[np.asarray(row)-1]
            np.testing.assert_array_equal(reflected, np.asarray(full['elements_hex8'][target-1])[permutation])
    assert sorted(covered) == list(range(1,9))


@pytest.mark.parametrize('factor,allowed', [(.25, True), (2., False)])
def test_reflection_tolerance_never_snaps_or_changes_source(factor, allowed):
    p = protocol(); full = fixtures.box(p)
    tolerance = 1e-10*max(p['geometry']['radius_m'], p['geometry']['height_m'])
    full['rest_nodes_m'][-1][0] += factor*tolerance
    original = deepcopy(full)
    if allowed:
        result = mesh._extract_verified(full, p)
        assert 0 < result['verification']['maximum_coordinate_mismatch_m'] < tolerance
        assert result['mesh']['rest_nodes_m'] == original['rest_nodes_m'][:18]
    else:
        with pytest.raises(ValueError, match='counterparts'):
            mesh._extract_verified(full, p)
    assert full == original


def prepared_analytic(root, monkeypatch):
    declaration = json.loads((ROOT/runner.DECLARATION_PATH).read_text()); p = protocol()
    directory = root/'study'/'preparation'; directory.mkdir(parents=True)
    sources = {'halfheight_mesh': runner.old.saved(root, root/'analytical-source.json', {'fixture': True})}
    references = {}
    for N in (8,12):
        full = fixtures.box(p); full['mesh_N'] = N
        binding = runner.old.saved(root, root/f'full-{N}.json', full)
        for branch in ('compression','tension'):
            references[f'{branch}:N{N}:S60:reference'] = {'primitive_bindings': {'mesh': binding}}
    plan = {'study': declaration, 'study_binding': {'path': runner.DECLARATION_PATH, 'sha256': runner.DECLARATION_SHA},
        'directory': str(directory), 'output_root': str(directory.parent), 'protocol': p, 'inputs': {},
        'source_bindings': sources, 'backend_profile': declaration['original_backend_profile'],
        'runtime_identity': declaration['runtime_identity'], 'full_references': references}
    state = {'extraction_invocations':0,'solver_invocations':0,'gmsh_generation_calls':0,'measured_data_accessed':False,
        'cases':{},'levels':{},'runs':{key:{'status':'not_executed'} for key in declaration['ordered_runs']}}
    # Only public clinical-specimen counts are bypassed. Actual extraction, deck
    # generation, backend transformation and revalidation functions run unmocked.
    monkeypatch.setattr(runner.half_mesh, 'extract_halfheight', mesh._extract_verified)
    watch = SimpleNamespace(active=None)
    runner.prepare(root, plan, state, watch)
    plan.update(cases=state['cases'], levels=state['levels'])
    return plan, state, watch


def test_actual_helpers_roundtrip_frozen_preparation_and_cut_xml(tmp_path, monkeypatch):
    plan, state, _ = prepared_analytic(tmp_path, monkeypatch)
    before = {str(p):p.read_bytes() for p in Path(plan['directory']).rglob('*') if p.is_file()}
    runner.verify_prepared_geometry(tmp_path, plan)
    assert before == {name:Path(name).read_bytes() for name in before}
    H = plan['protocol']['geometry']['height_m']
    for key, row in plan['cases'].items():
        branch = key.split(':')[0]; sign = -1 if branch == 'compression' else 1
        tree = ET.fromstring((tmp_path/row['deck']['path']).read_bytes())
        original = ET.fromstring((tmp_path/row['backend_source_deck']['path']).read_bytes())
        assert tree.find('./Control/solver/linear_solver').attrib == {'type':'accelerate'}
        assert ET.tostring(tree.find('Material')) == ET.tostring(original.find('Material'))
        loading = runner.access.verify_binding(tmp_path,row['loading'],read_json=True)
        assert loading['full_height_m'] == H and loading['retained_height_m'] == H/2
        assert loading['load_coordinate'][-1] == pytest.approx(sign*.15*H)
        bcs = list(tree.find('Boundary')); cut = [bc for bc in bcs if bc.attrib['node_set'].startswith('top_node_')]
        assert len(bcs) == len(cut)+3 and len(cut) == 9
        for bc in cut:
            assert bc.findtext('dof') == 'z' and bc.findtext('relative') == '0'
            curve = tree.find("./LoadData/load_controller[@id='"+bc.find('value').attrib['lc']+"']/points")
            points = [tuple(map(float,pt.text.split(','))) for pt in curve]
            for j in (0,17,60):
                assert float(bc.findtext('value'))*points[j][1] == pytest.approx(sign*.15*H*j/60/2)
    assert state['extraction_invocations'] == 2 and state['solver_invocations'] == state['gmsh_generation_calls'] == 0


@pytest.mark.parametrize('fault', ['half_again_cut_amplitude','false_mapping','wrong_full_height'])
def test_semantically_rehashed_preparation_drift_rejects_before_solver(tmp_path, monkeypatch, fault):
    plan, state, watch = prepared_analytic(tmp_path, monkeypatch)
    key = plan['study']['ordered_runs'][0]; row = plan['cases'][key]
    name = {'half_again_cut_amplitude':'deck','false_mapping':'reconstruction','wrong_full_height':'loading'}[fault]
    path = tmp_path/row[name]['path']
    if fault == 'half_again_cut_amplitude':
        text = path.read_text(); assert '>0.5</value>' in text
        path.write_text(text.replace('>0.5</value>', '>0.25</value>'))
    else:
        value = json.loads(path.read_text())
        if fault == 'false_mapping':
            value['mapping']['reflections'][1]['node_full_ids'].reverse()
        else:
            value['full_height_m'] /= 2
        runner.runtime.write_json(path, value)
    row[name]['sha256'] = runner.runtime.sha(path)
    if name != 'loading':
        loading_path = tmp_path/row['loading']['path']; loading = json.loads(loading_path.read_text())
        loading[name+'_sha256'] = row[name]['sha256']; runner.runtime.write_json(loading_path,loading)
        row['loading']['sha256'] = runner.runtime.sha(loading_path)
    experiment = tmp_path/'study'/'experiment'; experiment.mkdir(); plan['directory'] = str(experiment)
    monkeypatch.setattr(runner.old, 'solve', lambda *a: pytest.fail('No solver may receive rehashed drift'))
    with pytest.raises(ValueError, match='differs'):
        runner.solve_cases(tmp_path, plan, state, watch, runner.time.monotonic()+420)
    assert state['solver_invocations'] == 0 and all(v['status']=='not_executed' for v in state['runs'].values())
    assert not (experiment/'runs').exists()
