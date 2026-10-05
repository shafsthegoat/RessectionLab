"""Independent asymmetric analytical display controls; no MRI or payload decode."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render_rhuh_fixed_planes as render


def encoded_volume():
    # Every coordinate contributes a distinguishable signed fractional value.
    i, j, k = np.indices((5, 7, 9))
    return (10000 * i - 100 * j + k + .125).astype(np.float32)


def expected_plane(volume, fixed, index):
    other = [axis for axis in range(3) if axis != fixed]
    result = np.empty((volume.shape[other[1]], volume.shape[other[0]]), dtype=volume.dtype)
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            point = [0, 0, 0]
            point[fixed], point[other[0]], point[other[1]] = index, x, y
            result[y, x] = volume[tuple(point)]
    return result


def test_independent_native_coordinates_full_fov_and_exact_float_bits():
    volume = encoded_volume(); before = volume.copy(); indices = (2, 4, 6)
    views = render.plane_views(volume, render.AFFINE, indices)
    labels = [("A", "P", "I", "S"), ("R", "L", "I", "S"), ("R", "L", "A", "P")]
    for fixed, row in enumerate(views):
        expected = expected_plane(volume, fixed, indices[fixed])
        np.testing.assert_array_equal(row["pixels"].view(np.uint32), expected.view(np.uint32))
        assert row["pixels"].dtype == np.dtype("float32")
        assert not np.shares_memory(row["pixels"], volume) and not row["pixels"].flags.writeable
        assert tuple(row[key] for key in ("left", "right", "bottom", "top")) == labels[fixed]
        assert row["extent"] == (-.5, expected.shape[1]-.5, -.5, expected.shape[0]-.5)
    np.testing.assert_array_equal(volume, before)
    assert render.INDICES == (120, 120, 77) and render.SHAPE == (240, 240, 155)


def test_permuted_reflected_anisotropic_axes_have_physical_labels_and_aspect():
    # i points superior by2mm, j points right by3mm, k points posterior by5mm.
    affine = np.array([[0., 3., 0., 17.], [0., 0., -5., -21.], [2., 0., 0., 11.], [0., 0., 0., 1.]])
    figure, views = render.build_figure(encoded_volume(), affine, (2, 4, 6), analytical_fixture=True)
    labels = [("L", "R", "A", "P"), ("I", "S", "A", "P"), ("I", "S", "L", "R")]
    expected_aspects = (5/3, 5/2, 3/2)
    try:
        for col, view in enumerate(views):
            assert tuple(view[key] for key in ("left", "right", "bottom", "top")) == labels[col]
            for row in range(2):
                axis = figure.axes[3*row+col]
                assert axis.get_aspect() == expected_aspects[col]
                assert axis.images[0].origin == "lower"
                assert tuple(axis.get_xlim()) == tuple(view["extent"][:2])
                assert tuple(axis.get_ylim()) == tuple(view["extent"][2:])
                assert {text.get_text() for text in axis.texts} == set(labels[col])
    finally:
        figure.clear()


def test_both_explicit_windows_leave_stored_values_unchanged_and_limit_claims():
    volume = encoded_volume(); indices = (2, 4, 6)
    figure, views = render.build_figure(volume, render.AFFINE, indices, analytical_fixture=True)
    try:
        assert render.WINDOWS == ((-8.976038932800293, 10.492466926574707), (-3., 3.))
        assert len([axis for axis in figure.axes if axis.images]) == 6
        for row, limits in enumerate(render.WINDOWS):
            for col in range(3):
                image = figure.axes[3*row+col].images[0]
                np.testing.assert_array_equal(np.asarray(image.get_array()), expected_plane(volume, col, indices[col]))
                assert image.get_clim() == limits
                assert image.get_cmap().name == "gray"
                assert image.get_interpolation() == "nearest"
        figure_text = "\n".join(text.get_text() for text in figure.texts)
        assert "sampled planes" in figure_text and "no anatomical acceptance" in figure_text
        assert "unsampled anatomy remain unverified" in figure_text
        assert all("physical units unverified" in axis.get_ylabel() for axis in figure.axes[6:])
    finally:
        figure.clear()


@pytest.mark.parametrize("change", ["oblique", "duplicate_axis", "zero_spacing"])
def test_unsupported_frame_never_receives_misleading_cardinal_labels(change):
    affine = np.eye(4)
    if change == "oblique": affine[0, 1] = .01
    elif change == "duplicate_axis": affine[:3, 1] = affine[:3, 0]
    else: affine[:3, 1] = 0
    with pytest.raises(render.Rejected): render.plane_views(encoded_volume(), affine, (2, 4, 6))


def test_changed_original_bytes_fail_before_snapshot_decode_or_figure(monkeypatch, tmp_path):
    """Synthetic file bytes only; all patient-specific metadata functions mocked."""
    expected = b"asymmetric analytical source bytes"
    changed = b"Asymmetric analytical source bytes"
    original = tmp_path / "analytical.fixture"; original.write_bytes(changed)
    output_rel = "outputs/rhuh-fixed-plane-qc-v1/run-analytical-review"
    digest = hashlib.sha256(expected).hexdigest()
    monkeypatch.setattr(render, "ORIGINAL_SHA", digest)
    release = {"schema": "resectionlab.rhuh-fixed-plane-release.v1", "released": True,
        "action": "fixed_three_planes_two_windows_once", "source": render.SOURCE,
        "renderer_sha256": hashlib.sha256(Path(render.__file__).read_bytes()).hexdigest(),
        "decoder_sha256": render.DECODER_SHA, "inspection_sha256": render.REPORT["sha256"],
        "inspection_audit_sha256": render.AUDIT["sha256"], "original_compressed_sha256": digest,
        "output_directory": output_rel}
    decoder = SimpleNamespace(_json=lambda *args: release, _path=lambda root, path: root / path,
        MAX_COMPRESSED=1024, PUBLISHED_TOKEN=hashlib.md5(expected, usedforsecurity=False).hexdigest(),
        Rejected=ValueError)
    request = {"payload": {"path": original.name, "measured_compressed_bytes": len(expected)}}
    monkeypatch.setattr(render, "metadata_preflight", lambda root: (decoder, {}, request))
    monkeypatch.setattr(render, "_decode_snapshot", lambda *args: pytest.fail("Changed bytes must not decode"))
    monkeypatch.setattr(render, "build_figure", lambda *args: pytest.fail("Changed bytes must not render"))
    result = render.execute(tmp_path, {"path": "analytical-release", "sha256": "analytical-only"})
    assert result["accepted"] is False and result["failure"] == "original_compressed_digest"
    assert original.read_bytes() == changed
    assert sorted(path.name for path in (tmp_path / output_rel).iterdir()) == ["receipt.json"]
    assert json.loads((tmp_path / output_rel / "receipt.json").read_text())["accepted"] is False
