"""Saved-only mesh/deck audit; never imports Gmsh or invokes a solver."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
import tarfile
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RAW = ROOT/'outputs/mechanics/hbe-01-03-resolution-v1/mesh-preparation'
FROZEN = ROOT/'outputs/validation/hbe-resolution-5b80e3f/frozen-source'
COMMIT = '5b80e3fee2349bc78d3f2ceb9b211efb305a1fc9'
checked = {}


def sha(path):
    path = Path(path)
    assert path.suffix not in {'.csv', '.zip', '.nii'}, 'No measured or image payload access'
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024**2):
            h.update(block)
    return h.hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    assert expected is None or expected == digest, str(path)
    checked[str(path)] = digest
    return digest


def read(binding):
    path = ROOT/binding['path']
    bind(path, binding['sha256'])
    return json.loads(path.read_text())


def load_module(name, filename):
    path = FROZEN/'scripts'/filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path.resolve()
    return module


def main():
    started = time.monotonic()
    result = json.loads((RAW/'result.json').read_text()); bind(RAW/'result.json')
    baseline = read(result['baseline']); state = read(result['state'])
    for path, digest in baseline['inputs'].items():
        bind(path, digest)
    release = read(baseline['release_binding']); study = read(baseline['study_binding'])
    protocol = read(study['original_protocol'])
    assert release['schema'] == 'hbe-resolution-release-v1' and release['authorized'] is True
    assert release['phase'] == baseline['phase'] == result['phase'] == state['phase'] == 'prepare'
    assert release['source_commit'] == COMMIT
    assert release['source_bindings'] == baseline['source_bindings']
    assert result['status'] == state['status'] == 'prepared_not_solved'
    assert state['gmsh_generation_calls'] == state['mesh_preparation_invocations'] == 2
    assert state['solver_invocations'] == 0 and state['measured_data_accessed'] is False
    assert set(state['runs']) == set(study['ordered_runs'])
    assert all(row['status'] == 'not_executed' for row in state['runs'].values())
    assert not (RAW.parent/'experiment').exists() and not (RAW.parent/'.solve-started.json').exists()
    archive = ROOT/release['source_archive']['path']; bind(archive, release['source_archive']['sha256'])
    assert release['source_archive']['sha256'] == '29f5866817a260e49275e51510a1937f432cc1bc40762009025748092e8eefdb'
    wanted = {f'scripts/{Path(v["path"]).name}': v['sha256'] for v in release['source_bindings'].values()}
    wanted.update({baseline['study_binding']['path']: baseline['study_binding']['sha256'],
                   study['original_protocol']['path']: study['original_protocol']['sha256']})
    with tarfile.open(archive) as tar:
        assert tar.pax_headers['comment'] == COMMIT
        members = [m for m in tar if not m.isdir()]
        assert len(members) == len(wanted) == 12 and {m.name for m in members} == set(wanted)
        for member in members:
            assert member.isfile()
            data = tar.extractfile(member).read()
            assert hashlib.sha256(data).hexdigest() == wanted[member.name]
            assert data == (FROZEN/member.name).read_bytes()
            assert data == subprocess.check_output(['git', 'show', f'{COMMIT}:{member.name}'], cwd=ROOT)
            bind(FROZEN/member.name, wanted[member.name])
    mesh_module = load_module('resolution_saved_mesh_audit', 'mechanics_hbe_mesh.py')
    backend = load_module('resolution_saved_backend_audit', 'mechanics_hbe_backend.py')
    assert 'gmsh' not in sys.modules
    extended = json.loads(json.dumps(protocol)); extended['mesher']['levels'] = study['mesh_levels']
    levels = {}
    for declared in study['mesh_levels']:
        N = declared['N']; base = RAW/f'N{N}'/'generated'
        receipt = read(state['levels'][str(N)])
        bind(base/'mesh.json', receipt['mesh_sha256']); bind(base/'specimen.msh', receipt['native_mesh_sha256'])
        saved_mesh = json.loads((base/'mesh.json').read_text())
        assert receipt['status'] == 'prepared_not_solved' and receipt['gmsh_generation_calls'] == 1
        assert receipt['solver_calls'] == 0 and receipt['curve_values_opened'] is False
        assert receipt['source_sha256'] == release['source_bindings']['mesh_deck']['sha256']
        assert receipt['protocol_sha256'] == study['original_protocol']['sha256']
        assert receipt['resolution_declaration_sha256'] == baseline['study_binding']['sha256']
        assert receipt['runtime_receipt_sha256'] == baseline['gmsh_runtime']['sha256']
        assert receipt['runtime_unchanged'] is True
        before = time.monotonic()
        quality = mesh_module.mesh_quality(saved_mesh, extended)
        limits = protocol['mesher']['mesh_quality']
        quality['checks'].update(
            declared_node_count=len(saved_mesh['node_ids']) == declared['expected_nodes'],
            finest_volume=quality['relative_volume_error'] <= limits['finest_relative_volume_error_max'],
            finest_boundary_sag=quality['maximum_radial_boundary_sag_over_R'] <= limits['finest_max_radial_sag_over_R'])
        quality['passed'] = all(quality['checks'].values())
        assert quality == receipt['quality'] and quality['passed'] is True
        recompute_seconds = time.monotonic()-before
        decks = {}
        for branch in study['branches']:
            key = f'{branch}:N{N}:S60:reference'; row = state['cases'][key]
            for binding in row.values():
                bind(ROOT/binding['path'], binding['sha256'])
            assert row['mesh']['sha256'] == receipt['mesh_sha256']
            original_path = base/f'{branch}-60-reference'/'specimen.feb'
            original_loading_path = original_path.with_name('loading.json')
            hashes = receipt['decks'][f'{branch}-60-reference']
            bind(original_path, hashes['deck_sha256']); bind(original_loading_path, hashes['loading_sha256'])
            xml, loading = mesh_module.specimen_deck(saved_mesh, branch, 60, 1000., extended)
            assert xml.encode() == original_path.read_bytes()
            loading.update(role='reference', mesh_sha256=receipt['mesh_sha256'],
                protocol_sha256=study['original_protocol']['sha256'],
                resolution_declaration_sha256=baseline['study_binding']['sha256'])
            assert loading == json.loads(original_loading_path.read_text())
            adapted = (ROOT/row['deck']['path']).read_bytes()
            assert backend.transform_deck(xml).encode() == adapted
            assert dict(loading, deck_sha256=row['deck']['sha256']) == read(row['loading'])
            previous = baseline['prior_runs'][f'{branch}:N12:S60:reference']['primitive_bindings']['deck']
            bind(ROOT/previous['path'], previous['sha256'])
            old_tree = ET.fromstring((ROOT/previous['path']).read_bytes()); new_tree = ET.fromstring(adapted)
            unchanged = ['Module', 'Control', 'Material', 'MeshDomains', 'LoadData']
            for tag in unchanged:
                assert ET.tostring(old_tree.find(tag)) == ET.tostring(new_tree.find(tag)), (N, branch, tag)
            # Boundary entries necessarily grow with the top-node inventory.
            expected_bc = {('bottom', axis): '1' for axis in 'xyz'}
            for node in saved_mesh['boundaries']['top']['node_ids']:
                node_set = f'top_node_{node}'
                assert new_tree.find(f'./Mesh/NodeSet[@name="{node_set}"]').text == str(node)
                for axis in 'xyz':
                    expected_bc[node_set, axis] = '2' if axis == 'z' else '1'
            assert set(map(int, new_tree.find('./Mesh/NodeSet[@name="bottom"]').text.split(','))) == set(saved_mesh['boundaries']['bottom']['node_ids'])
            observed_bc = {}
            for bc in new_tree.find('Boundary'):
                assert bc.tag == 'bc' and bc.attrib['type'] == 'prescribed displacement'
                key = (bc.attrib['node_set'], bc.find('dof').text)
                assert key not in observed_bc and bc.find('value').text == '1' and bc.find('relative').text == '0'
                observed_bc[key] = bc.find('value').attrib['lc']
            assert observed_bc == expected_bc
            decks[branch] = {'original_rebuilt_byte_exact': True, 'backend_only_transform_byte_exact': True,
                            'loading_exact': True, 'unchanged_N12_physics_subtrees': unchanged,
                            'every_top_node_and_bottom_set_prescription_checked': True}
        levels[str(N)] = {'quality_recomputed_exact': True, 'quality': quality,
                         'quality_recompute_seconds': recompute_seconds, 'decks': decks}
    supervision = result['supervision']; watch = json.loads((RAW/'output-watch.json').read_text())
    bind(RAW/'output-watch.json')
    assert supervision['status'] == 'completed' and supervision['exit_code'] == 0
    assert supervision['elapsed_seconds'] < 120 and supervision['wall_cap_seconds'] == 120
    assert supervision['kill_reason'] is None and supervision['cleanup_error'] is None
    assert supervision['sampled_peak_process_group_rss_bytes'] < supervision['rss_cap_bytes'] == 3*1024**3
    assert watch['maximum_active_bytes'] < 512*1024**2 and watch['maximum_total_bytes'] < 2*1024**3
    assert 'gmsh' not in sys.modules
    for path, digest in checked.items():
        assert sha(path) == digest, path
    report = {'schema': 'hbe-resolution-saved-mesh-review-v1', 'status': 'saved_preparation_verified_no_solve',
        'source_commit': COMMIT, 'archive': release['source_archive'], 'release': baseline['release_binding'],
        'source_archive_members': wanted, 'baseline_input_count': len(baseline['inputs']),
        'checked_files_unchanged': len(checked), 'checked_files': checked, 'levels': levels,
        'runtime_identity': baseline['runtime_identity'], 'gmsh_runtime': baseline['gmsh_runtime'],
        'calls': {'actual_preparation_gmsh_calls': 2, 'actual_specimen_solver_calls': 0,
                  'audit_gmsh_calls': 0, 'audit_solver_calls': 0},
        'resource_evidence': {'parent_seconds': supervision['elapsed_seconds'], 'worker_seconds': state['elapsed_seconds'],
            'sampled_peak_process_group_rss_bytes': supervision['sampled_peak_process_group_rss_bytes'],
            'output_watch': watch, 'timing_scope': 'Worker/level/generation intervals are nested inside parent, not additive.'},
        'audit_seconds': time.monotonic()-started,
        'limits': ['Saved mesh quality is recomputed with the exact frozen pure helper; this is not a separate meshing algorithm or certified geometry error bound.',
                   'Byte-exact deck regeneration is pure text/array work; neither Gmsh nor FEBio was imported or run by this audit.',
                   'No experimental response or patient data was accessed. No numerical mesh-convergence or physical-validation result exists for these new levels.',
                   'Sampled RSS/output peaks can miss between-sample peaks. Solver execution still requires a separate release.'],
        'blocking_findings': []}
    (OUT/'review.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'levels': list(levels), 'checked_files': len(checked),
                     'audit_seconds': report['audit_seconds'], 'review_sha256': sha(OUT/'review.json')}))


if __name__ == '__main__':
    main()
