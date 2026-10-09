"""Analytical preparation controls only; no patient arrays or native VTK/Gmsh."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT/relative)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


W = load('case4_v4_witness_pure', 'scripts/mechanics_patient_mesh_boundary6_witness.py')
D = load('case4_v1_diagnostic_pure', 'scripts/mechanics_patient_mesh_diagnostic.py')
BASE = load('case4_original_sample_pure', 'scripts/mechanics_patient_mesh.py')
CONFIG = json.loads((ROOT/W.MANIFEST).read_text())


def test_declaration_is_non_executable_and_query_count_is_exact():
    assert CONFIG['execution_authorized'] is False
    assert CONFIG['caps'] == W.CAPS
    assert CONFIG['original_fidelity']['limit_m'] == .002
    assert CONFIG['original_fidelity']['chunk_triangles'] == 256
    assert (CONFIG['original_fidelity']['source_to_mesh']['sample_count']
            +CONFIG['original_fidelity']['mesh_to_source']['sample_count']
            +2428+7284+4856+20) == 1193462
    assert W.CAPS['maximum_distance_queries'] > 1193462
    assert all('/landmark' not in row['path'] and '/during' not in row['path']
               and 'mask' not in row['path'] and 'MRI' not in row['path']
               for row in CONFIG['inputs'].values())
    assert set(CONFIG['inputs']) == {'native_surface','nodes_m','tet10_indices',
        'gmsh_node_ids','gmsh_element_ids','candidate_manifest','candidate_result',
        'candidate_acceptance','candidate_release','candidate_summary'}


def test_v4_directed_preserves_original_order_cover_and_global_bound():
    vertices = []
    for i in range(262):
        scale = 3 if i in (0,131,261) else 1
        vertices.extend(np.array([[0.,0.,0.],[.001,0.,0.],[0.,.001,.0003]])*scale
                        + [i*.02,0.,0.])
    vertices = np.array(vertices)
    faces = np.arange(len(vertices)).reshape(-1,3)
    distance = lambda points: np.abs(points[:,2])
    expected_chunks = list(BASE.triangle_samples(vertices,faces,.001,20000))
    expected_count = sum(len(points) for points,_ in expected_chunks)
    expected_bound = max(float(distance(points).max())+cover
                         for points,cover in expected_chunks)
    result,maxima,upper = W.directed(vertices,faces,distance,W.Budget(),D)
    prior,prior_maxima = D.directed(vertices,faces,distance,D.Budget())
    assert result['sample_count'] == expected_count == prior['sample_count']
    assert result['maximum_sample_distance_m'] == prior['maximum_sample_distance_m']
    assert result['full_surface_upper_bound_m'] == prior['full_surface_upper_bound_m'] == expected_bound
    assert np.array_equal(maxima,prior_maxima)
    assert upper.shape == (len(faces),)
    assert upper.max() == expected_bound
    assert np.all(upper>=maxima)
    assert result['worst_10'] == sorted(result['worst_10'],
        key=lambda row:(-row['distance_m'],row['ordinal']))
    assert all(row['original_chunk_cover_m'] > 0 for row in result['worst_10'])


def test_cube_is_concentrated_only_if_both_area_and_capture_thresholds_hold():
    q = CONFIG['localization_decision']
    positions = np.array([[.001,.001,.001],[.006,.001,.001]])
    rows = [dict(centroids_m=positions,areas_m2=np.array([1.,1.]),
                 witness_area_fraction=.04) for _ in range(2)]
    result = W.cube_decision(rows,q)
    assert result['status'] == 'concentrated_under_diagnostic'
    assert result['best_cube_capture_fractions'] == {
        'source_to_mesh':pytest.approx(1.),'mesh_to_source':pytest.approx(1.)}
    rows[1]['witness_area_fraction'] = .051
    assert 'falsified' in W.cube_decision(rows,q)['status']


def test_multifocal_cube_falsifies_and_uses_lexicographic_tie():
    q = CONFIG['localization_decision']
    points = np.array([[-.091,-.091,-.091],[.091,.091,.091]])
    rows = [dict(centroids_m=points,areas_m2=np.array([1.,1.]),
                 witness_area_fraction=.01) for _ in range(2)]
    result = W.cube_decision(rows,q)
    assert result['status'] == 'local_refinement_premise_falsified_at_fixed_scale'
    assert result['best_cube_capture_fractions']['source_to_mesh'] == pytest.approx(.5)
    assert result['best_cube_capture_fractions']['mesh_to_source'] == pytest.approx(.5)
    assert result['best_cube_origin_ras_m'] == pytest.approx([-.12,-.12,-.12])


def test_disjoint_direction_witnesses_ignore_empty_cubes_in_zero_score_tie():
    q = CONFIG['localization_decision']
    rows = [dict(centroids_m=np.array([[-.091,.091,.091]]),
                 areas_m2=np.array([1.]),witness_area_fraction=.01),
            dict(centroids_m=np.array([[.091,-.091,-.091]]),
                 areas_m2=np.array([1.]),witness_area_fraction=.01)]
    result = W.cube_decision(rows,q)
    assert result['status'] == 'local_refinement_premise_falsified_at_fixed_scale'
    # The lexicographically smallest grid origin (-.12,-.12,-.12) is empty.
    # The first eligible origin contains only the first direction's witness.
    assert result['best_cube_origin_ras_m'] == pytest.approx([-.12,.065,.065])
    assert result['best_cube_capture_fractions'] == {
        'source_to_mesh':pytest.approx(1.),'mesh_to_source':pytest.approx(0.)}


def test_invalid_or_empty_witnesses_do_not_get_diffuse_label():
    q = CONFIG['localization_decision']
    bad = dict(centroids_m=np.empty((0,3)),areas_m2=np.empty(0),witness_area_fraction=0.)
    good = dict(centroids_m=np.array([[0.,0.,0.]]),areas_m2=np.array([1.]),
                witness_area_fraction=.01)
    with pytest.raises(ValueError,match='Both original failed directions'):
        W.cube_decision([good,bad],q)
    bad = dict(centroids_m=np.array([[np.nan,0.,0.]]),areas_m2=np.array([1.]),
               witness_area_fraction=.01)
    with pytest.raises(ValueError,match='Invalid witness'):
        W.cube_decision([good,bad],q)


def test_no_release_means_no_saved_geometry_execution(tmp_path):
    release = tmp_path/'release.json'
    release.write_text('{}')
    with pytest.raises((KeyError,ValueError)):
        W.specification(release,tmp_path/'attempt')


def test_direct_worker_rejected_before_any_saved_input_or_helper(tmp_path,monkeypatch):
    monkeypatch.delenv(W.LAUNCH_TOKEN_ENV,raising=False)
    monkeypatch.setattr(W,'specification',lambda *_:pytest.fail('Geometry preflight must not run'))
    with pytest.raises(ValueError,match='one-shot supervised launch'):
        W.worker(tmp_path/'release.json',tmp_path/'attempt')
    assert not (tmp_path/'attempt').exists()


def test_parent_failure_after_mkdir_still_writes_terminal_acceptance(tmp_path,monkeypatch):
    release=tmp_path/'release.json'
    release.write_text('{}')
    monkeypatch.setattr(W,'specification',lambda *_:(tmp_path,{}, {}, {}, {},
        {str(release):W.sha(release)}))
    monkeypatch.setattr(W,'module',lambda *_:(_ for _ in ()).throw(RuntimeError('supervisor unavailable')))
    output=tmp_path/'attempt'
    assert W.run(release,output)==1
    receipt=json.loads((output/'acceptance.json').read_text())
    assert receipt['status']=='failed_or_incomplete'
    assert receipt['error']['message']=='supervisor unavailable'
    assert receipt['candidate_accepted'] is False and receipt['solver_calls']==0
    assert receipt['input_hashes_unchanged']=={str(release):True}


def test_parent_passes_one_shot_bound_token_to_supervised_child(tmp_path,monkeypatch):
    release=tmp_path/'release.json'
    release.write_text('{}')
    monkeypatch.setattr(W,'specification',lambda *_:(tmp_path,{}, {}, {}, {},
        {str(release):W.sha(release)}))
    runtime=SimpleNamespace(private_environment=lambda _:{},declaration=lambda:{})

    def supervised(_runtime,_command,output,environment,_caps,*,spec):
        assert environment[W.LAUNCH_TOKEN_ENV]
        monkeypatch.setenv(W.LAUNCH_TOKEN_ENV,environment[W.LAUNCH_TOKEN_ENV])
        W.require_supervised_launch(release,output)
        assert spec.output_bytes==W.CAPS['output_bytes']
        return ({'status':'failed_or_incomplete','exit_code':1},{'error':None})

    guard=SimpleNamespace(ExecutionSpec=lambda *args:SimpleNamespace(output_bytes=args[6]),
                          supervised=supervised,
                          output_usage=lambda *_args,**_kwargs:{'files':2,'bytes':100})
    monkeypatch.setattr(W,'module',lambda _path,name:
        runtime if name=='v4_witness_supervisor' else guard)
    output=tmp_path/'attempt'
    assert W.run(release,output)==1
    assert (output/'launch-context.json').is_file()
    receipt=json.loads((output/'acceptance.json').read_text())
    assert receipt['status']=='failed_or_incomplete'
    assert receipt['solver_calls']==0 and receipt['mesher_calls']==0
