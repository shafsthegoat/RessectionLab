"""Independent constructed fields/archives; no patient coordinates or solves."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts import mechanics_patient_interpolation as m
from scripts import mechanics_patient_comparison as c


def tet10(vertices, tetrahedra):
    points = list(np.array(vertices, dtype=float))
    edges = {}
    rows = []
    for raw in tetrahedra:
        cell = list(raw)
        if np.linalg.det(np.stack([points[i]-points[cell[0]] for i in cell[1:]])) < 0:
            cell[1], cell[2] = cell[2], cell[1]
        row = cell.copy()
        for a, b in [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]:
            edge = tuple(sorted((cell[a], cell[b])))
            if edge not in edges:
                edges[edge] = len(points)
                points.append((points[edge[0]] + points[edge[1]]) / 2)
            row.append(edges[edge])
        rows.append(row)
    return np.array(points), np.array(rows, dtype=np.int64)


def simple_mesh():
    return tet10([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01]], [[0, 1, 2, 3]])


def frame():
    return np.diag([1000., 1000., 1000., 1.])


def bind(root, name, payload):
    (root/name).write_bytes(payload)
    return {"path": name, "sha256": hashlib.sha256(payload).hexdigest()}


def bundle(root, x=None, e=None):
    if x is None:
        x, e = simple_mesh()
    mesh = io.BytesIO(); np.savez(mesh, nodes_m=x, tet10_indices=e)
    displacement = io.BytesIO(); np.savez(displacement, displacement_m=np.full_like(x, .0001))
    geometry = {"schema": "native-rest-to-common-ras-v1", "native_reference_frame": "MRI_RAS+",
        "native_length_units": "m", "common_frame": "RAS+", "common_length_units": "mm",
        "native_m_to_common_mm": frame().tolist(), "baseline_alignment_sha256": "1"*64,
        "source_image_sha256": "2"*64, "alignment_status": "provisional_baseline_diagnostic"}
    return {"schema": "frozen-tet10-displacement-field-v1", "frame": "RAS+", "units": "mm",
        "mesh_length_units": "m", "nodal_displacement_units": "m", "tet10_indexing": "zero_based",
        "tet10_order": "vertices0123_edges01_12_20_03_13_23", "direction": "reference_to_displaced",
        "forward_hash": "3"*64, "protocol_sha256": "4"*64,
        "interpolator_entrypoint": "sample_frozen_field", "artifacts": {
            "reference_mesh": bind(root, "mesh.npz", mesh.getvalue()),
            "nodal_displacements": bind(root, "motion.npz", displacement.getvalue()),
            "geometry_frame": bind(root, "frame.json", json.dumps(geometry).encode()),
            "numerical_evidence": bind(root, "evidence.json", b'{"constructed_fixture_only":true}'),
            "interpolator_source": bind(root, "source.py", Path(m.__file__).read_bytes())}}


def test_tiny_overlapping_interiors_are_not_promoted_by_contact_band():
    x, e = tet10([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01],
                 [.007, 0, 0], [0, .007, 0], [0, 0, .007]], [[0, 1, 2, 3], [0, 4, 5, 6]])
    field = m.FrozenTet10Field(x, e, np.full_like(x, .0001), frame())
    assert field.geometry_status == "supported"
    query_native = np.full(3, 1e-13)
    for cell in e:
        local = np.linalg.solve((x[cell[1:4]]-x[cell[0]]).T, query_native-x[cell[0]])
        assert np.all(local > 0) and local.sum() < 1
    result = m.sample_field(field, [query_native*1000])["rows"][0]
    assert result["status"] in ("ambiguous_reference_location", "uncertain_reference_boundary")
    assert result["displacement_ras_mm"] is None


@pytest.mark.parametrize("bad", [None, "not-a-source-hash"])
def test_direct_loader_requires_a_real_forward_hash_identity(tmp_path, bad):
    manifest = bundle(tmp_path)
    manifest["forward_hash"] = bad
    with pytest.raises(ValueError):
        m.load_frozen_field(tmp_path, manifest)


def test_oblique_frame_maps_points_and_vectors_with_distinct_rules():
    x, e = simple_mesh()
    angle = .37
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)],
                         [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    transform = frame()
    transform[:3, :3] = 1000*rotation
    transform[:3, 3] = [-120., 17., 300.]
    def polynomial(p):
        a, b, d = p.T
        return np.column_stack([20*a*b + .2*a - .001,
                                15*d*d - .1*b + .002,
                                -30*b*d + .3*a + .0005])
    field = m.FrozenTet10Field(x, e, polynomial(x), transform)
    q = np.array([[.001, .002, .003], [.003, .001, .002]])
    common = q @ transform[:3, :3].T + transform[:3, 3]
    output = m.sample_field(field, common)
    assert output["counts"] == {"supported": 2}
    np.testing.assert_allclose([r["displacement_ras_mm"] for r in output["rows"]],
        polynomial(q) @ transform[:3, :3].T, rtol=0, atol=2e-13)


def edge_fan():
    return tet10([[0, 0, 0], [0, 0, .01], [.01, 0, 0],
                 [-.005, .008, 0], [-.005, -.008, 0]],
                 [[0, 1, 2, 3], [0, 1, 3, 4], [0, 1, 4, 2]])


def test_three_cell_shared_edge_uses_all_traces_and_is_order_independent():
    x, e = edge_fan()
    u = np.column_stack([x[:, 0]*.3, x[:, 1]*.2, x[:, 2]*.1])
    rows = []
    for cells in [e, e[::-1]]:
        field = m.FrozenTet10Field(x, cells, u, frame())
        row = m.sample_field(field, [[0, 0, 3.]])["rows"][0]
        assert row["status"] == "supported" and len(row["element_ids"]) == 3
        rows.append(row["displacement_ras_mm"])
    np.testing.assert_allclose(rows, [[0, 0, .3], [0, 0, .3]], atol=1e-15)


def test_continuity_checks_pairs_beyond_the_first_trace(monkeypatch):
    x, e = edge_fan()
    u = np.zeros_like(x); u[1, 0] = .001
    field = m.FrozenTet10Field(x, e, u, frame())
    original = m.tet10_values
    def controlled_trace_roundoff(weights):
        # Isolate aggregation with controlled local-trace perturbations. Each
        # differs from first by .75*tolerance; opposite pair differs by1.5*tolerance.
        out = original(weights)
        assert len(out) == 3
        perturb = np.array([0, .75e-9, -.75e-9])
        out[:, 1] += perturb
        out[:, 0] -= perturb
        return out
    monkeypatch.setattr(m, "tet10_values", controlled_trace_roundoff)
    row = m.sample_field(field, [[0, 0, 3.]])["rows"][0]
    assert row["status"] == "nonconforming_reference_mesh"
    assert row["maximum_containing_field_disagreement_m"] > 1e-12
    assert row["displacement_ras_mm"] is None


@pytest.mark.parametrize("status", ["supported", "outside_reference_domain", "uncertain_reference_boundary",
    "unsupported_reference_geometry", "nonconforming_reference_mesh", "ambiguous_reference_location"])
def test_saved_adapter_preserves_every_explicit_status_and_null(tmp_path, status):
    x, e = simple_mesh(); query = (1., 2., 3.)
    if status == "outside_reference_domain":
        query = (30., 20., 10.)
    elif status == "uncertain_reference_boundary":
        query = (1., 2., -.5e-9)
    elif status == "unsupported_reference_geometry":
        x[4, 0] += 1e-6
    elif status == "nonconforming_reference_mesh":
        e = np.vstack([e, e])
    elif status == "ambiguous_reference_location":
        x, e = tet10([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .01],
            [.007, 0, 0], [0, .007, 0], [0, 0, .007]], [[0, 1, 2, 3], [0, 4, 5, 6]])
        query = (1., 1., 1.)
    manifest = bundle(tmp_path, x, e)
    queries = c.FieldQueries((11,), (query,))
    sampled = m.sample_frozen_field(tmp_path, manifest, queries)
    assert sampled.query_sha256 == queries.sha256 and sampled.status == (status,)
    if status == "supported":
        np.testing.assert_allclose(sampled.displacement_ras_mm, [[.1, .1, .1]], atol=1e-14)
    else:
        assert sampled.displacement_ras_mm == (None,)


def test_coherently_rehashed_foreign_source_refuses_before_point_location(tmp_path, monkeypatch):
    manifest = bundle(tmp_path)
    payload = (tmp_path/"source.py").read_bytes() + b"\n# Another source snapshot\n"
    manifest["artifacts"]["interpolator_source"] = bind(tmp_path, "source.py", payload)
    monkeypatch.setattr(m, "sample_field", lambda *args: pytest.fail("Field sources must validate before location"))
    with pytest.raises(ValueError, match="source differs"):
        m.sample_frozen_field(tmp_path, manifest, c.FieldQueries((1,), ((1., 2., 3.),)))


@pytest.mark.parametrize("kind", ["duplicate", "truncated", "object", "fortran"])
def test_malformed_npz_rejects_before_numpy_array_load(monkeypatch, kind):
    out = io.BytesIO()
    values = np.zeros((3, 3), dtype=np.float64)
    if kind == "object":
        values = values.astype(object)
    if kind == "fortran":
        values = np.asfortranarray(values)
    np.save(out, values)
    payload = out.getvalue()
    if kind == "truncated":
        payload = payload[:-8]
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("nodes_m.npy", payload)
        if kind == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                z.writestr("nodes_m.npy", payload)
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Reached allocation before header/member validation"))
    with pytest.raises(ValueError):
        m._npz(archive.getvalue(), {"nodes_m": (3, 10000, "float64")})
