"""Small independent numerical controls; no patient/SELECT/UPENN payloads."""
from dataclasses import replace
from itertools import product
import json
import math
from pathlib import Path
import shutil

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef, array_digest, semantic_digest
from resectionlab.structural_evidence import structural_frame_hash
import resectionlab.virtual_glioma as vg


@pytest.fixture(scope="module")
def kernel_cache():
    if not all((vg.DEFAULT_CACHE / name).exists() for name in vg.SOURCE_FILES):
        pytest.skip("Pinned small upstream sources absent; explicitly run ensure_source_assets(allow_download=True)")
    vg.ensure_source_assets(vg.DEFAULT_CACHE)
    return vg.DEFAULT_CACHE


def host():
    shape = (13, 11, 9)
    brain = np.ones(shape, bool)
    wm = np.zeros(shape, bool)
    wm[:7] = True
    gm = ~wm
    # Brain extraction includes this modeled ventricle; classified matter does not.
    wm[6, 5, 4] = gm[6, 5, 4] = False
    affine = np.diag([3., 2., 4., 1.])
    case = CaseData("generated-fk-control", np.ones(shape, np.float32), {"wm": wm, "gm": gm}, affine,
        (SourceRef("generated", "generated://fk-numerical-control", provenance="simulated", native_frame="RAS+"),),
        brain_mask=brain, metadata={"is_synthetic": True, "role": "TRAIN", "patient_group": "generated:fk-control"})
    layers = {name: {"kind": "brain_tissue", "tissue_class": kind, "source_role": "generated_numerical_control",
        "source_sha256": "1" * 64, "source_image_hash": array_digest(case.mri), "source_frame_hash": structural_frame_hash(case)}
        for name, kind in (("wm", "white_matter"), ("gm", "gray_matter"))}
    return replace(case, metadata={**case.metadata, "anatomy_layers": layers})


def oblique_reflected_host():
    """Boundary-filled, anisotropic source with a physical reflection/rotation."""
    case = host()
    shape = (11, 10, 9)
    brain = np.ones(shape, bool)
    brain[0] = False
    wm = np.indices(shape)[0] < 5
    gm = ~wm
    wm[5:7, 4:6, 3:5] = gm[5:7, 4:6, 3:5] = False
    angle = .61
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.],
                         [math.sin(angle), math.cos(angle), 0.], [0., 0., 1.]])
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([-2., 1.5, 2.5])
    affine[:3, 3] = [19., -7., 5.]
    case = replace(case, mri=np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
                   brain_mask=brain, compartments={"wm": wm, "gm": gm},
                   source_compartments={"wm": wm, "gm": gm}, affine=affine)
    layers = {name: {**record, "source_image_hash": array_digest(case.mri),
                    "source_frame_hash": structural_frame_hash(case)}
              for name, record in case.metadata["anatomy_layers"].items()}
    return replace(case, metadata={**case.metadata, "anatomy_layers": layers})


def qualify(case, *, mode=vg.GM_WM_FRACTIONS, shape=None, affine=None):
    return vg.qualify_host(case, patient_group="generated:fk-control", role="TRAIN",
        healthy_source_evidence={"status": "generated_numerical_control", "patient_group": "generated:fk-control",
            "source_url": "generated://fk-control-definition", "source_sha256": "2" * 64,
            "statement": "Analytic software control; not a person or measured physiology"},
        qualified_by="numerical-test", grid_shape=list(case.mri.shape) if shape is None else shape,
        grid_affine=case.affine if affine is None else affine, tissue_mode=mode,
        source_wm_layer="wm" if mode == vg.GM_WM_FRACTIONS else None,
        source_gm_layer="gm" if mode == vg.GM_WM_FRACTIONS else None)


def params(**changes):
    return vg.FKParameters(seed_center_ras_mm=(12., 8., 16.), duration_days=2., **changes)


def independent_finite_volume_step(field, domain, wm, gm, diffusion, ratio, spacing, rho, dt):
    """Pairwise equal/opposite fluxes; no upstream helpers or periodic rolls."""
    change = np.zeros_like(field)
    conductivity = diffusion * (wm + gm / ratio)
    for axis, step in enumerate(spacing):
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        lower, upper = tuple(lower), tuple(upper)
        admitted = domain[lower] & domain[upper]
        face_rate = .5 * (conductivity[lower] + conductivity[upper]) / step ** 2
        flux = np.where(admitted, face_rate * (field[upper] - field[lower]), 0.)
        change[lower] += flux
        change[upper] -= flux
    return field + dt * (change + rho * field * (1 - field))


def test_exact_upstream_sources_license_and_corrupt_cache_refusal(kernel_cache, tmp_path):
    receipt = vg.ensure_source_assets(kernel_cache)
    assert receipt["commit"] == "162c2cbbab5882fff7c3c05d9fedd87453298a0f"
    assert receipt["license"] == "MIT"
    for name in vg.SOURCE_FILES:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(kernel_cache / name, destination)
    (tmp_path / "TumorGrowthToolkit/FK/FK.py").write_bytes(b"raise RuntimeError('must not run')")
    with pytest.raises(ValueError, match="integrity"):
        vg.load_pinned_kernel(tmp_path)


def test_upstream_step_matches_independent_stencil_and_no_flux(kernel_cache):
    kernel, _ = vg.load_pinned_kernel(kernel_cache)
    rng = np.random.default_rng(134)
    domain = np.ones((8, 7, 6), bool)
    domain[2:5, 3, 1:5] = False
    wm = rng.uniform(.1, .7, domain.shape)
    gm = 1 - wm
    initial = rng.uniform(0, 1, domain.shape) * domain
    spacing = np.array([3., 2., 4.])
    coefficients = vg.prepare_fk_coefficients(kernel, domain, wm, gm, tissue_mode=vg.GM_WM_FRACTIONS,
                                               threshold=.1, diffusion=.8, ratio=7.)
    dt = .25
    actual = kernel.FK_update(np.pad(initial, 1), coefficients, .03, dt, *spacing)[1:-1, 1:-1, 1:-1]
    expected = independent_finite_volume_step(initial, domain, wm, gm, .8, 7., spacing, .03, dt)
    assert np.allclose(actual, expected, rtol=0, atol=3e-16)
    diffusion_only = kernel.FK_update(np.pad(initial, 1), coefficients, 0., dt, *spacing)
    assert abs(float(diffusion_only.sum()) - float(initial.sum())) < 5e-14
    assert np.count_nonzero(diffusion_only[~np.pad(domain, 1)]) == 0


def test_zero_padding_prevents_periodic_wraparound_at_source_boundary(kernel_cache):
    kernel, _ = vg.load_pinned_kernel(kernel_cache)
    domain = np.ones((7, 6, 5), bool)
    initial = np.zeros(domain.shape)
    initial[0, 2, 2] = 1.
    padded_d = vg.prepare_fk_coefficients(kernel, domain, None, None, tissue_mode=vg.UNIFORM_DIFFUSION_DOMAIN,
                                           threshold=.1, diffusion=1., ratio=10.)
    actual = kernel.FK_update(np.pad(initial, 1), padded_d, 0., .1, 1., 1., 1.)[1:-1, 1:-1, 1:-1]
    assert actual[-1, 2, 2] == 0.
    assert actual[1, 2, 2] == pytest.approx(.1)
    assert actual[0, 2, 2] == pytest.approx(.5)
    assert actual.sum() == pytest.approx(1., abs=2e-15)
    # Diagnostic witness of why the unguarded original periodic kernel needs the wrapper.
    raw_d = kernel.get_D(domain.astype(float), np.zeros(domain.shape), .1, 1., 1.)
    periodic = kernel.FK_update(initial.copy(), raw_d, 0., .1, 1., 1., 1.)
    assert periodic[-1, 2, 2] == pytest.approx(.1)


def test_positivity_maximum_principle_and_long_diffusion_conservation(kernel_cache):
    kernel, _ = vg.load_pinned_kernel(kernel_cache)
    domain = np.ones((9, 8, 7), bool)
    domain[4, 2:6, 2:5] = False
    d = vg.prepare_fk_coefficients(kernel, domain, None, None, tissue_mode=vg.UNIFORM_DIFFUSION_DOMAIN,
                                    threshold=.1, diffusion=10., ratio=1.)
    limit, _ = vg.stable_dt_limit(d, (.5, 1., 2.), 1.)
    rng = np.random.default_rng(1)
    field = np.pad(rng.uniform(0, 1, domain.shape) * domain, 1)
    field[1, 1, 1], field[2, 1, 1] = 1., 0.
    initial_mass = field.sum()
    for _ in range(100):
        field = kernel.FK_update(field, d, 0., .9 * limit, .5, 1., 2.)
        assert field.min() >= 0 and field.max() <= 1
        assert np.count_nonzero(field[~np.pad(domain, 1)]) == 0
    assert field.sum() == pytest.approx(initial_mass, rel=0, abs=1e-12)
    for _ in range(100):
        field = kernel.FK_update(field, d, 1., .9 * limit, .5, 1., 2.)
        assert 0 <= field.min() <= field.max() <= 1


def test_physical_gaussian_seed_inside_matter_and_masked_ventricle():
    case = host()
    qualification = qualify(case)
    _, affine, _, domain, _, _, _ = vg._validate_qualification(case, qualification)
    field = vg.masked_gaussian_seed(domain, affine, params(seed_sigma_mm=3.))
    assert field[4, 4, 4] == pytest.approx(.2)
    assert field[5, 4, 4] / field[4, 4, 4] == pytest.approx(np.exp(-.5))
    assert domain[6, 5, 4] == 0 and field[6, 5, 4] == 0
    assert np.count_nonzero(field[~domain]) == 0
    assert np.count_nonzero(field[:1]) == 0
    with pytest.raises(ValueError, match="outside the admitted"):
        vg.masked_gaussian_seed(domain, affine, vg.FKParameters(seed_center_ras_mm=(18., 10., 16.)))
    with pytest.raises(ValueError, match="outside the simulation"):
        vg.masked_gaussian_seed(domain, affine, vg.FKParameters(seed_center_ras_mm=(-100., 0., 0.)))


def test_logistic_limit_has_first_order_time_convergence(kernel_cache):
    case = host()
    q = qualify(case)
    errors = []
    for dt in (.4, .2, .1):
        result = vg.simulate_virtual_glioma(case, q, params(diffusion_mm2_per_day=0., growth_per_day=.3, max_dt_days=dt), source_cache=kernel_cache)
        u0 = result.initial_concentration
        exact = u0 * np.exp(.3 * 2.) / (1 - u0 + u0 * np.exp(.3 * 2.))
        errors.append(float(np.max(np.abs(result.concentration - exact))))
        assert result.manifest["timeSeriesMetrics"][0]["timeDays"] == 0.
        assert result.manifest["timeSeriesMetrics"][0]["integralConcentrationMm3"] == pytest.approx(u0.sum() * 24.)
        assert result.manifest["finalTimeDays"] == 2.
    assert 1.7 < errors[0] / errors[1] < 2.2
    assert 1.7 < errors[1] / errors[2] < 2.2


def test_integral_stop_records_actual_step_time_and_initial_stop(kernel_cache):
    case = host()
    q = qualify(case)
    initial = vg.masked_gaussian_seed(vg._validate_qualification(case, q)[3], case.affine, params())
    integral = initial.sum() * 24.
    stopped = vg.simulate_virtual_glioma(case, q, params(diffusion_mm2_per_day=0., growth_per_day=.5,
        max_dt_days=.1, stopping_integral_mm3=integral * 1.001), source_cache=kernel_cache)
    assert stopped.manifest["numericalContract"]["executedSteps"] == 1
    assert stopped.manifest["finalTimeDays"] == pytest.approx(.1)
    assert stopped.manifest["timeSeriesMetrics"][-1]["timeDays"] == pytest.approx(.1)
    stopped0 = vg.simulate_virtual_glioma(case, q, params(stopping_integral_mm3=integral / 2), source_cache=kernel_cache)
    assert stopped0.manifest["finalTimeDays"] == 0.
    assert stopped0.manifest["numericalContract"]["executedSteps"] == 0
    assert np.array_equal(stopped0.concentration, stopped0.initial_concentration)


def test_source_identity_domain_mapping_and_foreign_fraction_refusals(kernel_cache):
    case = host()
    q = qualify(case)
    before = case.semantic_hash
    _, _, _, domain, wm, gm, _ = vg._validate_qualification(case, q)
    wrong = domain.copy()
    wrong[1, 1, 1] = ~wrong[1, 1, 1]
    with pytest.raises(ValueError, match="Supplied domain"):
        vg.simulate_virtual_glioma(case, q, params(), source_cache=kernel_cache, domain_mask=wrong)
    with pytest.raises(ValueError, match="Supplied WM"):
        vg.simulate_virtual_glioma(case, q, params(), source_cache=kernel_cache, wm_fraction=gm)
    with pytest.raises(ValueError, match="identity"):
        vg.simulate_virtual_glioma(replace(case, case_id="another-host"), q, params(), source_cache=kernel_cache)
    changed = json.loads(json.dumps(q))
    changed["mapping"]["admittedDomainHash"] = array_digest(wrong)
    changed["qualificationHash"] = semantic_digest({k: v for k, v in changed.items() if k != "qualificationHash"})
    with pytest.raises(ValueError, match="identity"):
        vg.simulate_virtual_glioma(case, changed, params(), source_cache=kernel_cache)
    result = vg.simulate_virtual_glioma(case, q, params(), source_cache=kernel_cache, domain_mask=domain, wm_fraction=wm, gm_fraction=gm)
    assert case.semantic_hash == before
    assert result.manifest["originalAcquiredStructuresChanged"] is False
    assert result.manifest["newPatientCount"] == 0
    assert result.manifest["kernel"]["upstreamSolveExecuted"] is False
    assert result.manifest["numericalContract"]["badResultClipping"] is False


def test_tissue_mapping_is_linear_without_background_renormalization():
    case = host()
    affine = np.array(case.affine)
    affine[0, 3] += 1.5
    q = qualify(case, shape=[10, 9, 7], affine=affine)
    _, _, _, domain, wm, gm, _ = vg._validate_qualification(case, q)
    assert wm[6, 4, 4] == pytest.approx(.5) and gm[6, 4, 4] == pytest.approx(.5)
    assert wm[6, 5, 4] + gm[6, 5, 4] == pytest.approx(.5)
    assert domain[6, 5, 4]  # half matter admitted at the explicit .1 threshold
    wrong_provenance = {k: dict(v) for k, v in case.metadata["anatomy_layers"].items()}
    wrong_provenance["wm"]["source_image_hash"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="provenance"):
        qualify(replace(case, metadata={**case.metadata, "anatomy_layers": wrong_provenance}))
    with pytest.raises(ValueError, match="overlap"):
        qualify(replace(case, compartments={**case.compartments, "gm": case.brain_mask}))


def test_uniform_mode_is_explicit_fiction_and_does_not_label_brain_white_matter(kernel_cache):
    case = host()
    q = qualify(case, mode=vg.UNIFORM_DIFFUSION_DOMAIN)
    result = vg.simulate_virtual_glioma(case, q, params(), source_cache=kernel_cache)
    assert q["mapping"]["domainRule"] == "brain_envelope_only_explicit_parameter_fiction"
    assert q["mapping"]["wmFractionHash"] is None
    assert "uniform_parameter_fiction_no_GM_WM_classification_or_CSF_barriers" in result.manifest["limitations"]
    assert result.manifest["numericalContract"]["uniformEncoding"].endswith("not white-matter anatomy")
    assert all(row["semantics"] == "declared_scalar_concentration_threshold_only" for row in result.manifest["thresholdMasks"])
    with pytest.raises(ValueError, match="WM/GM"):
        vg.qualify_host(case, patient_group=q["patientGroup"], role="TRAIN", healthy_source_evidence=q["healthySourceEvidence"],
            qualified_by="test", grid_shape=list(case.mri.shape), grid_affine=case.affine,
            tissue_mode=vg.UNIFORM_DIFFUSION_DOMAIN, source_wm_layer="wm")


@pytest.mark.parametrize("declaration", [{"role": "SELECT"}, {"role": "MEASUREMENT_EVAL"},
    {"split": {"role": "development", "development_role": "checkpoint_selection_development"}},
    {"roles": {"SELECT": "true"}}, {"roles": {"SELECT": 1}},
    {"source_collection": {"name": "UPENN-GBM"}},
    {"parent": {"source_collection": {"name": "UPENN-GBM"}}},
    {"source_collection": {"name": "UTSW"}}])
def test_protected_host_rejected_before_arrays_or_files(monkeypatch, declaration):
    import builtins
    case = host()
    case = replace(case, metadata={**case.metadata, **declaration})
    def forbidden(*args, **kwargs):
        pytest.fail("Protected-host metadata must be rejected before array hashing or source access")
    monkeypatch.setattr(vg, "array_digest", forbidden)
    monkeypatch.setattr(builtins, "open", forbidden)
    with pytest.raises(ValueError, match="Protected"):
        qualify(case)
    with pytest.raises(ValueError, match="Protected"):
        vg.simulate_virtual_glioma(case, {"role": "TRAIN"}, params())


def test_no_bad_result_clipping_and_runtime_budget(monkeypatch, kernel_cache):
    case = host()
    q = qualify(case)
    real, receipt = vg.load_pinned_kernel(kernel_cache)
    original = real.FK_update
    def bad_update(*args):
        result = original(*args)
        result[1, 1, 1] = -1e-8
        return result
    real.FK_update = bad_update
    monkeypatch.setattr(vg, "load_pinned_kernel", lambda cache: (real, receipt))
    with pytest.raises(ValueError, match="not silently clipped"):
        vg.simulate_virtual_glioma(case, q, params(), source_cache=kernel_cache)
    with pytest.raises(ValueError, match="budget exceeded"):
        vg.simulate_virtual_glioma(case, q, params(max_dt_days=.00001), source_cache=kernel_cache)


def test_determinism_source_annotation_preservation_and_saved_arrays(kernel_cache, tmp_path):
    case = host()
    q = qualify(case)
    first = vg.simulate_virtual_glioma(case, q, params(), source_cache=kernel_cache)
    second = vg.simulate_virtual_glioma(case, q, params(), source_cache=kernel_cache)
    assert np.array_equal(first.concentration, second.concentration)
    assert first.manifest["simulationId"] == second.manifest["simulationId"]
    private = replace(case, source_compartments={"withheld_reference": np.ones(case.mri.shape, bool)},
                      metadata={**case.metadata, "future_outcome": "not-used-for-growth"})
    alternate = vg.simulate_virtual_glioma(private, qualify(private), params(), source_cache=kernel_cache)
    assert np.array_equal(first.concentration, alternate.concentration)
    assert private.source_compartments["withheld_reference"].all()
    saved = first.save(tmp_path / "result")
    with np.load(saved / "virtual-glioma.npz", allow_pickle=False) as arrays:
        assert np.array_equal(arrays["concentration"], first.concentration)
        for key, mask in first.threshold_masks.items():
            assert np.array_equal(arrays[key], mask)
    with pytest.raises(ValueError):
        first.concentration[0, 0, 0] = 1.


def test_health_statement_identity_and_explicit_role_are_required():
    case = host()
    receipt = qualify(case)
    wrong = dict(receipt["healthySourceEvidence"])
    wrong["patient_group"] = "foreign:person"
    with pytest.raises(ValueError, match="patient group"):
        vg.qualify_host(case, patient_group="generated:fk-control", role="TRAIN", healthy_source_evidence=wrong,
            qualified_by="test", grid_shape=list(case.mri.shape), grid_affine=case.affine,
            source_wm_layer="wm", source_gm_layer="gm")
    with pytest.raises(ValueError, match="host identity"):
        vg.qualify_host(case, patient_group="foreign:person", role="TRAIN", healthy_source_evidence=wrong,
            qualified_by="test", grid_shape=list(case.mri.shape), grid_affine=case.affine,
            source_wm_layer="wm", source_gm_layer="gm")
    with pytest.raises(ValueError, match="declared source role"):
        vg.qualify_host(case, patient_group="generated:fk-control", role="DEVELOPMENT", healthy_source_evidence=receipt["healthySourceEvidence"],
            qualified_by="test", grid_shape=list(case.mri.shape), grid_affine=case.affine,
            source_wm_layer="wm", source_gm_layer="gm")
    real_claim = replace(case, metadata={**case.metadata, "is_synthetic": False})
    with pytest.raises(ValueError, match="source-reported health"):
        qualify(real_claim)


def test_low_fraction_matter_is_retained_without_upstream_half_threshold_crop():
    case = host()
    affine = np.array(case.affine)
    affine[0, 3] += 2.4
    q = qualify(case, shape=[10, 9, 7], affine=affine)
    _, _, _, domain, wm, gm, _ = vg._validate_qualification(case, q)
    assert wm[5, 5, 4] + gm[5, 5, 4] == pytest.approx(.2)
    assert domain[5, 5, 4]


def test_same_oblique_reflected_grid_is_exact_independent_copy():
    case = oblique_reflected_host()
    for array in (case.mri, case.brain_mask, *case.compartments.values()):
        for order in (0, 1):
            mapped = vg._resample(array, case.affine, case.affine.copy(), array.shape, order)
            assert mapped.dtype == np.float64
            assert np.array_equal(mapped, array)
            assert not np.shares_memory(mapped, array)
    q = qualify(case)
    assert q["mapping"]["exactGridIdentityCopy"] is True
    assert np.array_equal(q["mapping"]["targetVoxelToSourceVoxel"], np.eye(4))
    # A real subvoxel translation must still be resampled, even on equal shapes.
    translated = case.affine.copy()
    translated[:3, 3] += case.affine[:3, 0] * .125
    mapped = vg._resample(case.mri, case.affine, translated, case.mri.shape, 1)
    assert mapped[3, 3, 3] == pytest.approx(.875 * case.mri[3, 3, 3] + .125 * case.mri[4, 3, 3])
    assert not np.array_equal(mapped, case.mri)


def test_oblique_linear_fraction_roundoff_is_preserved_with_independent_interpolation():
    case = oblique_reflected_host()
    mapping = np.diag([1.5, 1.25, 1.125, 1.])
    mapping[:3, 3] = [.2, .3, .4]
    affine = case.affine @ mapping
    q = qualify(case, shape=[6, 5, 4], affine=affine)
    _, _, _, domain, wm, gm, _ = vg._validate_qualification(case, q)
    assert max(wm.max(), gm.max()) > 1.  # actual one-ULP regression witness
    for original, mapped in ((case.compartments["wm"], wm), (case.compartments["gm"], gm)):
        expected = np.zeros(mapped.shape)
        for index in np.ndindex(mapped.shape):
            xyz = (mapping @ np.array([*index, 1.]))[:3]
            lower = np.floor(xyz).astype(int)
            weight = xyz - lower
            for corner in product((0, 1), repeat=3):
                offset = np.array(corner)
                expected[index] += original[tuple(lower + offset)] * np.prod(np.where(offset, weight, 1 - weight))
        np.testing.assert_allclose(mapped, expected, atol=5e-15, rtol=0)
    assert q["mapping"]["fractionRoundoffTolerance"] == 8 * np.finfo(np.float64).eps
    assert q["mapping"]["fractionValuesClippedOrRenormalized"] is False
    assert q["mapping"]["mappedFractionRange"]["wm"][1] == float(wm.max())
    assert domain.any()


@pytest.mark.parametrize("bad_fraction", [-1e-8, 1. + 1e-8, float("nan")])
def test_materially_invalid_mapped_fraction_is_rejected_without_clipping(monkeypatch, bad_fraction):
    case = host()
    original = vg._resample
    def corrupt_mapping(array, *args):
        mapped = original(array, *args)
        if array is case.compartments["wm"]:
            mapped[2, 2, 2] = bad_fraction
        return mapped
    monkeypatch.setattr(vg, "_resample", corrupt_mapping)
    with pytest.raises(ValueError, match="bounded disjoint tissue fractions"):
        qualify(case)


def test_close_declared_thresholds_have_distinct_output_masks(kernel_cache):
    case = host()
    result = vg.simulate_virtual_glioma(case, qualify(case), params(concentration_thresholds=(.1, .10000000000001)), source_cache=kernel_cache)
    assert len(result.threshold_masks) == 2
    assert len({row["name"] for row in result.manifest["thresholdMasks"]}) == 2
    result.assert_intact()


def test_runtime_rejects_integral_loss_even_when_field_is_finite_and_bounded(monkeypatch, kernel_cache):
    case = host()
    q = qualify(case)
    real, receipt = vg.load_pinned_kernel(kernel_cache)
    original = real.FK_update
    real.FK_update = lambda *args: original(*args) * .9
    monkeypatch.setattr(vg, "load_pinned_kernel", lambda cache: (real, receipt))
    with pytest.raises(ValueError, match="integral/reaction balance"):
        vg.simulate_virtual_glioma(case, q, params(), source_cache=kernel_cache)
