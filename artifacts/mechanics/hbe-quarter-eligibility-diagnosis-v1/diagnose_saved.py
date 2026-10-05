"""Fixed saved-coordinate/topology diagnosis; never generate geometry or decks."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BINDINGS = {8: '3a6b5bb720bd3998f035abecf15d5c603d46c5ed281ce1d08c4aadec69c1dab9',
            12: '0ed154be0d17c8baaac6ee0288381a53ec100f93cd936a194202f69ef32ae033'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    results = {}
    for N, expected in BINDINGS.items():
        path = ROOT / f'outputs/mechanics/hbe-01-03-poc-v1/mesh-preparation/N{N}/generated/mesh.json'
        assert sha(path) == expected
        mesh = json.loads(path.read_text())
        X = np.asarray(mesh['rest_nodes_m'], dtype=np.float64)
        E = np.asarray(mesh['elements_hex8'], dtype=np.int64) - 1
        H, R = mesh['geometry']['height_m'], mesh['geometry']['radius_m']
        tolerance = 1e-10 * max(R, H)
        lookup = {tuple(sorted(row)): i for i, row in enumerate(E)}
        assert len(lookup) == len(E)
        tree = cKDTree(X)
        row = {'mesh_sha256': expected, 'node_count': len(X), 'element_count': len(E),
               'unchanged_coordinate_tolerance_m': tolerance, 'planes': {}, 'reflections': {}}
        for name, signs, offset, axis in [('x0', [-1, 1, 1], [0, 0, 0], 0),
                ('y0', [1, -1, 1], [0, 0, 0], 1), ('xy', [-1, -1, 1], [0, 0, 0], None),
                ('z_midheight', [1, 1, -1], [0, 0, H], 2)]:
            reflected = X * signs + offset
            distance, neighbors = tree.query(reflected, k=2, workers=1)
            paired = neighbors[:, 0]
            valid_nodes = (distance[:, 0] <= tolerance) & (distance[:, 1] > tolerance)
            cell_pairs = np.array([lookup.get(tuple(sorted(paired[cell])), -1) for cell in E])
            strict_cells = valid_nodes[E].all(axis=1) & (cell_pairs >= 0)
            row['reflections'][name] = {
                'maximum_nearest_node_mismatch_m': float(distance[:, 0].max()),
                'nodes_matching_original_tolerance': int(valid_nodes.sum()),
                'nearest_node_ids_bijective': len(np.unique(paired)) == len(X),
                'strict_reflected_cells_found': int(strict_cells.sum()),
                'nearest_id_cell_incidence_count_ignoring_coordinate_gate': int((cell_pairs >= 0).sum()),
                'nearest_id_cell_pairs_bijective': len(np.unique(cell_pairs)) == len(E) and bool((cell_pairs >= 0).all()),
                'all_nodes_and_cells_match_original_tolerance': bool(valid_nodes.all() and strict_cells.all()),
                'scope': 'Nearest-ID incidence without the coordinate gate is diagnostic only, never geometric acceptance.'}
            if axis is None:
                continue
            plane = H / 2 if axis == 2 else 0.
            coordinate = X[:, axis] - plane
            lo, hi = coordinate[E].min(axis=1), coordinate[E].max(axis=1)
            cross = np.flatnonzero((lo < -tolerance) & (hi > tolerance))
            lesser = np.minimum(-lo[cross], hi[cross])
            # Fixed nearest-ID nodes identify the nominal midline topology;
            # they need not satisfy the frozen coordinate criterion.
            nominal = np.flatnonzero(paired == np.arange(len(X)))
            departure = np.abs(coordinate[nominal])
            worst = int(nominal[np.argmax(departure)])
            coordinates = X[nominal]
            negative, positive = coordinate[coordinate < 0], coordinate[coordinate > 0]
            worst_cell = int(cross[np.argmax(lesser)]) if len(cross) else None
            row['planes'][name] = {
                'plane_coordinate_m': plane, 'straddling_cell_count': len(cross),
                'first_10_straddling_full_element_ids': (cross[:10] + 1).tolist(),
                'maximum_smaller_side_extent_m': float(lesser.max()) if len(lesser) else 0.,
                'worst_straddling_full_element_id': None if worst_cell is None else worst_cell + 1,
                'worst_straddling_min_max_relative_m': None if worst_cell is None else [float(lo[worst_cell]), float(hi[worst_cell])],
                'nearest_negative_coordinate_relative_m': float(negative.max()) if len(negative) else None,
                'nearest_positive_coordinate_relative_m': float(positive.min()) if len(positive) else None,
                'nominal_plane_node_count_from_nearest_id_fixed_points': len(nominal),
                'nominal_plane_nodes_inside_original_tolerance': int((departure <= tolerance).sum()),
                'nominal_plane_coordinate_bounds_m': [coordinates.min(axis=0).tolist(), coordinates.max(axis=0).tolist()],
                'maximum_nominal_plane_departure_m': float(departure.max()),
                'worst_nominal_full_node_id': worst + 1, 'worst_nominal_coordinate_m': X[worst].tolist(),
                'full_node_ids_of_nominal_plane': (nominal + 1).tolist(),
                'scope': 'All original coordinates retained; nominal means nearest-reflection-ID fixed point, not a relaxed plane classification.'}
        row['quarter_eligible'] = all(row['planes'][key]['straddling_cell_count'] == 0 and
            row['reflections'][key]['all_nodes_and_cells_match_original_tolerance'] for key in ('x0', 'y0'))
        row['z_half_geometric_eligibility_only'] = row['planes']['z_midheight']['straddling_cell_count'] == 0 and row['reflections']['z_midheight']['all_nodes_and_cells_match_original_tolerance']
        assert sha(path) == expected
        results[str(N)] = row
    report = {'schema': 'hbe-saved-quarter-eligibility-diagnosis-v1',
              'status': 'quarter_rejected_at_original_geometry_criterion', 'levels': results,
              'source_sha256': sha(__file__), 'elapsed_seconds': time.monotonic() - started,
              'activity': {'saved_meshes_read': 2, 'coordinate_edits': 0, 'new_meshes': 0,
                           'decks_generated': 0, 'native_mesher_calls': 0, 'solver_calls': 0, 'curve_reads': 0},
              'interpretation': 'Cartesian quarter has tiny but disallowed coordinate drift despite matching nearest-ID incidence. Original tolerance is unchanged. Half-height geometric feasibility is separate from equivalence, physics, solver or execution acceptance.'}
    (OUT / 'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'seconds': report['elapsed_seconds'],
                      'quarter_eligible': {N: value['quarter_eligible'] for N, value in results.items()},
                      'z_half_geometric_eligibility_only': {N: value['z_half_geometric_eligibility_only'] for N, value in results.items()}}))


if __name__ == '__main__':
    main()
