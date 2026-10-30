"""Frozen-protocol/qualification-checker controls, separate from kernel tests."""
from pathlib import Path
import sys

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import planar_blade_contact_fixtures as fixtures
import qualify_planar_blade_contact as qualification


def valid_receipt():
    return {'maxPenetrationMm': 0., 'maxTensileForceN': 0., 'maxEquilibriumResidualN': 0.,
        'maxComplementarityNmm': 0., 'energyBalanceResidualNmm': 0., 'maxTangentialForceN': 0.,
        'maxElementGradientNorm': .01, 'actuatorWorkNmm': 3., 'storedEnergyNmm': 1.,
        'previousStoredEnergyNmm': 0., 'viscousDissipationNmm': 1., 'numericalDissipationNmm': .5,
        'contactActivationDissipationNmm': .5, 'normalReactionN': 1., 'contactAreaMm2': 4.,
        'footprintAreaMm2': 4., 'minimumGapMm': 0.}


def test_frozen_manifest_identity_and_declared_coverage():
    protocol = fixtures.frozen_protocol()
    assert protocol['frozenBeforeScoring'] is True
    specs = qualification.scenario_specs()
    assert len(specs) == 24
    assert {s['n'] for s in specs} == {2, 4, 8}
    assert {s['order'] for s in specs} == {1, 2, 3, 4}
    assert {s['dt_s'] for s in specs} == {.1, .05, .025}
    assert sum(s['rotated'] for s in specs) == 2


def test_manifest_mutation_is_refused(tmp_path, monkeypatch):
    changed = tmp_path / 'manifest.json'
    changed.write_bytes(fixtures.MANIFEST.read_bytes() + b' ')
    monkeypatch.setattr(fixtures, 'MANIFEST', changed)
    with pytest.raises(ValueError, match='FROZEN_PLANAR_PROTOCOL_HASH_MISMATCH'):
        fixtures.frozen_protocol()


@pytest.mark.parametrize('dt', [.1, .05, .025])
def test_schedule_preserves_ramp_hold_withdrawal_and_bounds(dt):
    steps = fixtures.prescribed_schedule(dt, partial=True)
    assert len(steps) == round(3 / dt)
    assert len(steps) <= fixtures.frozen_protocol()['resources']['maxScheduleSteps']
    assert steps[-1]['timeS'] == 3.
    assert steps[-1]['offsetMm'] == pytest.approx(-.02)
    assert max(s['offsetMm'] for s in steps) == .02
    assert [s['timeS'] for s in steps] == sorted(set(s['timeS'] for s in steps))


def test_checker_retains_all_strict_failures_and_independent_ledger():
    receipt = valid_receipt()
    assert qualification.receipt_failures(receipt, fixtures.frozen_protocol()['tolerances']) == []
    receipt.update(maxPenetrationMm=1e-6, maxTensileForceN=1e-6, maxEquilibriumResidualN=1e-6,
        maxComplementarityNmm=1e-6, maxTangentialForceN=1e-6, maxElementGradientNorm=.21,
        storedEnergyNmm=2., contactAreaMm2=5., minimumGapMm=-1e-5)
    failures = qualification.receipt_failures(receipt, fixtures.frozen_protocol()['tolerances'])
    checks = {f['check'] for f in failures}
    assert {'maxPenetrationMm', 'maxTensileForceN', 'maxEquilibriumResidualN', 'maxComplementarityNmm',
        'maxTangentialForceN', 'maxElementGradientNorm', 'independentReceiptLedgerNmm',
        'boundedContactArea', 'minimumGapMm'} <= checks


def test_checker_refuses_nonfinite_and_missing_measurements():
    receipt = valid_receipt()
    receipt['maxEquilibriumResidualN'] = float('nan')
    receipt.pop('actuatorWorkNmm')
    failures = qualification.receipt_failures(receipt, fixtures.frozen_protocol()['tolerances'])
    assert {'maxEquilibriumResidualN', 'completeFiniteEnergyLedger'} <= {f['check'] for f in failures}


def test_independent_scalar_be_oracle_has_elastic_limit_relaxation_and_release():
    schedule = fixtures.prescribed_schedule(.1)
    elastic = qualification.uniform_analytical(schedule, False)
    maxwell = qualification.uniform_analytical(schedule, True)
    assert elastic[9]['normalReactionN'] == pytest.approx(.00032)
    assert elastic[19]['normalReactionN'] == elastic[9]['normalReactionN']
    assert maxwell[9]['normalReactionN'] > maxwell[19]['normalReactionN'] > elastic[19]['normalReactionN']
    assert elastic[-1]['normalReactionN'] == maxwell[-1]['normalReactionN'] == 0.
    assert maxwell[-1]['indentationMm'] > 0.


def test_rigid_rotation_preserves_orientation():
    r = fixtures.rigid_rotation()
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-15)
    assert np.linalg.det(r) == pytest.approx(1.)


def test_generated_fixture_is_padded_explicit_and_transforms_every_geometry():
    a = fixtures.generated_fixture(2, partial=True, maxwell=True)
    b = fixtures.generated_fixture(2, partial=True, maxwell=True, rotated=True)
    assert a.scene.shape == (4, 4, 4)
    assert len(a.cells) == 8 and len(a.contact_faces) == len(a.support_faces) == 4
    assert np.all(a.scene.labels[[0, -1]] == 0)
    np.testing.assert_allclose(b.scene.affine[:3, :3], b.rotation @ a.scene.affine[:3, :3], atol=1e-15)
    np.testing.assert_allclose(b.blade.origin_mm, b.rotation @ a.blade.origin_mm + b.translation_mm)
    np.testing.assert_allclose(b.blade.inward_normal, b.rotation @ a.blade.inward_normal)


def test_streamed_evidence_preserves_nonfinite_failure_observations(tmp_path):
    import json
    output = tmp_path / 'retained-failure.json'
    qualification.write_report(output, {'failures': [{'observed': float('nan')}],
        'array': np.array([1., np.inf, -np.inf])})
    loaded = json.loads(output.read_text())
    assert loaded == {'failures': [{'observed': 'nan'}], 'array': [1., 'inf', '-inf']}
