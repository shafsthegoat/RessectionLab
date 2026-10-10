"""Source-only HBE v5 complete-stream reader; no execution or physical release.

The six bound files are source deck, adapted deck, mesh, node output, element
output and solver log. The last three are strictly parsed, never substituted.
"""
from __future__ import annotations

from contextlib import ExitStack
import itertools
import json
import math
from pathlib import Path

from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_frame as frame
from scripts import mechanics_hbe_v5_source_bindings as sources
from scripts.mechanics_hbe_access import local_path, verify_binding
from scripts.mechanics_hbe_outputs import _iter_data_records, check_solver_records
from scripts.mechanics_hbe_physics import energy_work_check


MAX_PRIMITIVE_BYTES = 2 * 1024**3
MAX_SOLVER_BYTES = 128 * 1024**2
MAX_ITEMS = 50_000
INPUT_KEYS = frozenset({'source_deck', 'adapted_deck', 'mesh', 'nodes', 'elements', 'solver'})


def _candidate_output_path(root, binding):
    relative = Path(binding['path'])
    if (relative.is_absolute() or '..' in relative.parts
            or relative.parts[:2] != ('outputs', 'mechanics')):
        raise ValueError('Candidate solver output must stay under outputs/mechanics')
    base = Path(root).resolve()
    candidate = base / relative
    if any(part.is_symlink() for part in (candidate, *candidate.parents)
           if part != base and part.is_relative_to(base)):
        raise ValueError('Symlinked candidate solver output rejected')
    return local_path(root, binding['path'])


def _bound_reconstruction(root, prior, row, native_mesh_binding, native_mesh):
    """Extract the mapping inside the hash-bound historical wrapper."""
    n = row['N']
    old = prior['meshes'][str(n)]
    expected_schema = ('hbe-halfheight-spatial-reconstruction-v1' if n in (16, 24)
                       else 'hbe-halfheight-global-reconstruction-v1' if n == 32
                       else 'hbe-halfheight-global-n36-reconstruction-v1' if n == 36
                       else None)
    if expected_schema is None or native_mesh_binding != old['half_mesh']:
        raise ValueError('Half-height mesh differs from frozen v4 ancestry')
    full = json.loads(sources._read_bound(root, old['full_mesh'], maximum=sources.MAX_MESH_BYTES))
    wrapper = json.loads(sources._read_bound(root, old['reconstruction'],
                                             maximum=sources.MAX_MESH_BYTES))
    if (wrapper.get('schema') != expected_schema
            or wrapper.get('full_mesh') != old['full_mesh']
            or wrapper.get('half_mesh') != old['half_mesh']
            or not isinstance(wrapper.get('mapping'), dict)
            or wrapper['mapping'].get('schema') != 'hbe-halfheight-mapping-v1'):
        raise ValueError('Reconstruction wrapper or embedded mapping differs')
    mapping = wrapper['mapping']
    # prepare_generated_frame validates complete fingerprints, oriented
    # permutations, coverage, midplane and rest-coordinate identity once.
    return full, mapping


def evaluate_stream(contract, mesh_manifest, node_lines, element_lines, solver_lines,
                    *, reconstruction=None):
    """Evaluate one complete generated-fixture or future hash-bound native stream.

    A returned numerical pass is *not* physical validation. The caller must
    authenticate the contract, source, mesh and outputs before native use.
    """
    stream = evaluated_frames(contract, mesh_manifest, node_lines, element_lines, solver_lines,
                              reconstruction=reconstruction)
    while True:
        try: next(stream)
        except StopIteration as complete: return complete.value


def evaluated_frames(contract, mesh_manifest, node_lines, element_lines, solver_lines,
                     *, reconstruction=None, fitted=False, primitive_caps=None):
    """Stream-local primitives; caller may reduce each yielded state then discard it.

    Reference numerical result is byte-equivalent before provenance to the old
    evaluate_stream. No raw state list is retained. fitted is not an IO release.
    """
    if fitted:
        if contract.get('fixed_fit_sha256') != frame.FIXED_FIT_SHA256 or contract.get('mu_Pa') != frame.FIXED_MU_PA:
            raise ValueError('Fixed fitted contract required')
    elif contract.get('mu_Pa') != 1000.:
        raise ValueError('Pinned reference material required')
    primitive_caps = primitive_caps or {'nodes': MAX_PRIMITIVE_BYTES, 'elements': MAX_PRIMITIVE_BYTES}
    if set(primitive_caps) != {'nodes', 'elements'} or any(type(v) is not int or not 0 < v <= MAX_PRIMITIVE_BYTES for v in primitive_caps.values()):
        raise ValueError('Bounded primitive caps required')
    steps = contract.get('steps')
    times = contract.get('times')
    if (type(steps) is not int or steps not in (60, 120)
            or type(times) not in (list, tuple) or tuple(times) != tuple(i / steps for i in range(steps + 1))
            or contract.get('native_execution_released') is not False
            or contract.get('source_binding_checked') is not False):
        raise ValueError('Closed exact v5 S60/S120 contract required')
    prepared = frame.prepare_generated_frame(contract, mesh_manifest,
                                             reconstruction=reconstruction)
    mesh = prepared.native_mesh
    if len(mesh.rest_nodes_m) > MAX_ITEMS or mesh.element_count > MAX_ITEMS:
        raise ValueError('Native primitive dimensions exceed bounded parser')
    solver = check_solver_records(solver_lines, expected_times=times,
                                  residual_floor_N2=(1e-10 * contract['mu_Pa'] * mesh.radius_m**2)**2,
                                  maximum_bytes=MAX_SOLVER_BYTES)
    nodes = _iter_data_records(node_lines, expected_times=times,
                               item_count=len(mesh.rest_nodes_m), field_count=9,
                               record_name='mechanics_nodes_si',
                               maximum_bytes=primitive_caps['nodes'], maximum_items=MAX_ITEMS)
    elements = _iter_data_records(element_lines, expected_times=times,
                                  item_count=mesh.element_count, field_count=8,
                                  record_name='mechanics_elements_si',
                                  maximum_bytes=primitive_caps['elements'], maximum_items=MAX_ITEMS)
    frames = []
    maximum_ratios = {}
    for index, (node, element) in enumerate(itertools.zip_longest(nodes, elements)):
        if node is None or element is None or index > steps:
            raise ValueError('Complete synchronized native primitive streams required')
        if node['step'] != index or element['step'] != index:
            raise ValueError('Native primitive frame order differs')
        primitive = []
        if fitted:
            result = frame.evaluate_fitted_frame(contract, prepared, node, element, primitive_sink=primitive.append)
        else:
            result = frame._evaluate_prepared_frame(contract, prepared, node, element, primitive_sink=primitive.append)
        yield primitive[0], prepared
        for key, value in result['criteria_ratios'].items():
            if not math.isfinite(value):
                raise ValueError('Nonfinite numerical criterion')
            maximum_ratios[key] = max(maximum_ratios.get(key, 0.), value)
        frames.append(result)
    if len(frames) != steps + 1:
        raise ValueError('Initial and every converged state required')
    full_force = [row['applied_force_N'] for row in frames]
    full_energy = [row['energy_J'] for row in frames]
    full_work = energy_work_check(contract['full_coordinates_m'], full_force, full_energy,
                                  mu_Pa=contract['mu_Pa'], radius_m=mesh.radius_m,
                                  height_m=.00489159)
    maximum_ratios['full_work_energy'] = full_work['maximum_error_J'] / full_work['limit_J']
    native_work = None
    if contract['native_domain'] == 'lower_half_reconstructed':
        native_force = [row['raw_top_or_midplane_force_N'] for row in frames]
        native_energy = [row['native_energy_J'] for row in frames]
        native_work = energy_work_check(contract['native_coordinates_m'], native_force, native_energy,
                                        mu_Pa=contract['mu_Pa'], radius_m=mesh.radius_m,
                                        height_m=mesh.height_m)
        maximum_ratios['native_work_energy'] = native_work['maximum_error_J'] / native_work['limit_J']
    maximum_ratios['solver_residual'] = max(
        row['actual_N2'] / row['limit_N2'] for row in solver['states'])
    passed = (solver['passed'] and full_work['passed']
              and (native_work is None or native_work['passed'])
              and all(row['fixture_passed'] for row in frames)
              and all(value <= 1 for value in maximum_ratios.values()))
    return {
        'schema': 'hbe-v5-complete-stream-v1', 'run_id': contract['run_id'],
        'frame_count': len(frames), 'steps': steps,
        'representation': frames[0]['representation'], 'numerical_passed': passed,
        'criteria_max_ratio': maximum_ratios, 'solver': solver,
        'full_energy_work': full_work, 'native_energy_work': native_work,
        'load_coordinate_full_m': list(contract['full_coordinates_m']),
        'applied_force_N': full_force, 'energy_J': full_energy,
        'probe_displacements_m': [row['probe_displacements_m'] for row in frames],
        'minimum_sampled_J': min(row['minimum_sampled_J'] for row in frames),
        'minimum_logged_J': min(row['minimum_logged_J'] for row in frames),
        'provenance': {'v5_declaration_sha256': contract['v5_declaration_sha256'],
                       'source_deck_sha256': contract['source_deck_sha256'],
                       'adapted_deck_sha256': contract['adapted_deck_sha256'],
                       'native_output_observed': False,
                       'generated_fixture_only': True,
                       'physical_validation_pass': None,
                       'measured_response_accessed': False,
                       'patient_data_accessed': False},
        'logged_sed_used_for_energy_gate': False,
        'sampled_J_positivity_is_not_everywhere_proof': True,
    }


def read_bound_run(root, run_id, bindings, preparation_receipt):
    """Verify one run's six file identities, then parse its complete native logs.

    Source-deck and mesh IDs must match the frozen twelve-row mapping. This is
    a read-only checker, not permission to run FEBio or inspect measured HBE.
    """
    root = Path(root)
    if set(bindings) != INPUT_KEYS:
        raise ValueError('Exactly six source/mesh/native-output bindings required')
    for name in ('adapted_deck', 'nodes', 'elements', 'solver'):
        _candidate_output_path(root, bindings[name])
    sources.validate_binding_manifest(root, inspect_sources=True)
    binding_manifest = json.loads(sources._read_bound(
        root, {'path': sources.BINDING_PATH, 'sha256': sources.BINDING_SHA256}, maximum=64*1024))
    study, prior = v5.validate_preparation(root)
    row = v5.run_spec(study, prior, run_id)
    key = binding_manifest['run_source_keys'][run_id]
    source_binding = binding_manifest['source_decks'][key]
    mesh_binding = binding_manifest['meshes'][source_binding['mesh']]
    if (bindings['source_deck'] != {k: source_binding[k] for k in ('path', 'sha256')}
            or bindings['mesh'] != mesh_binding):
        raise ValueError('Run source/mesh differs from frozen v5 mapping')
    source = sources._read_bound(root, bindings['source_deck'], maximum=sources.MAX_SOURCE_BYTES)
    adapted = sources._read_bound(root, bindings['adapted_deck'], maximum=sources.MAX_SOURCE_BYTES)
    mesh_data = json.loads(sources._read_bound(root, bindings['mesh'], maximum=sources.MAX_MESH_BYTES))
    contract = frame.verified_schedule((root / v5.DECLARATION_PATH).read_bytes(),
                                       (root / study['previous_v4']['path']).read_bytes(),
                                       run_id, source, adapted, preparation_receipt)
    if (contract['source_deck_sha256'] != bindings['source_deck']['sha256']
            or contract['adapted_deck_sha256'] != bindings['adapted_deck']['sha256']):
        raise ValueError('Bound deck identities differ from schedule receipt')
    reconstruction = None
    if row['native_domain'] == 'lower_half_reconstructed':
        reconstruction = _bound_reconstruction(root, prior, row, bindings['mesh'], mesh_data)
    for name in ('nodes', 'elements', 'solver'):
        verify_binding(root, bindings[name], maximum_bytes=(MAX_SOLVER_BYTES if name == 'solver'
                                                             else MAX_PRIMITIVE_BYTES))
    with ExitStack() as stack:
        opened = {name: stack.enter_context(_candidate_output_path(root, bindings[name]).open(
            encoding='utf-8', errors='strict')) for name in ('nodes', 'elements', 'solver')}
        result = evaluate_stream(contract, mesh_data, opened['nodes'], opened['elements'],
                                 opened['solver'], reconstruction=reconstruction)
    for name in INPUT_KEYS:
        verify_binding(root, bindings[name], maximum_bytes=(MAX_SOLVER_BYTES if name == 'solver'
                                                             else MAX_PRIMITIVE_BYTES))
    # Hash-bound saved text alone cannot prove it came from native FEBio.
    result['provenance'].update(native_output_observed=None, generated_fixture_only=None,
                                output_origin='unverified_saved_stream',
                                source_binding_checked=True,
                                primitive_bindings=bindings,
                                reconstruction_provenance=('reflected_native_half_not_native_full'
                                                           if reconstruction is not None else 'full_native'))
    return result
