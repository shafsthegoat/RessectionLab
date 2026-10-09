"""Exact saved N12 numerical exception; no native execution or data fitting."""
from __future__ import annotations

import pytest

from scripts import mechanics_hbe_v5_n12_admission as admission
from scripts import mechanics_hbe_v5_remaining_one_shot as runner


def _chain(second_sha: str = runner.N12_FAILED_RECEIPT_SHA) -> list[dict]:
    return [
        {'run_id': runner.ORDER[0], 'path': runner.N8_PATH, 'sha256': runner.N8_SHA},
        {'run_id': runner.ORDER[1], 'path': runner.receipt_path(1),
         'sha256': second_sha},
    ]


@pytest.fixture(scope='module')
def reviewed_n12() -> dict:
    return admission.verify_exact()


def test_exact_saved_replay_is_a_numerical_row_with_separate_costs(reviewed_n12):
    assert reviewed_n12['original_receipt_sha256'] == runner.N12_FAILED_RECEIPT_SHA
    assert reviewed_n12['supplement_receipt_sha256'] == admission.SUPPLEMENT_SHA
    assert reviewed_n12['historical_guard_gap'] == admission.GAP
    assert reviewed_n12['readout']['run_id'] == runner.ORDER[1]
    assert reviewed_n12['readout']['frame_count'] == 61
    assert reviewed_n12['physical_validation_pass'] is None
    ledger = runner.validate_prior_chain(
        _chain(), 2,
        expected_profile=reviewed_n12['original_backend_profile'],
        expected_runtime=reviewed_n12['original_runtime_identity'])
    assert ledger['native_calls'] == 2  # N8 and original N12, never a replayed native call.
    assert ledger['supplement_replay_calls'] == 1
    assert ledger['native_seconds'] == 15.259143584175035
    assert ledger['readout_seconds'] == 2.8276635000947863
    assert ledger['prep_seconds'] == 1.348830624949187
    assert ledger['output_bytes'] == 66576514
    assert ledger['supplement_readout_seconds'] == 2.8474308329168707
    assert ledger['supplement_prep_seconds'] == 3.4362154591362923
    assert ledger['supplement_output_bytes'] == 557392
    assert ledger['combined_wall_seconds'] == 25.71928400127217
    assert ledger['combined_output_bytes'] == 67133906


def test_exact_exception_still_requires_same_frozen_profile_and_runtime(
        reviewed_n12, monkeypatch):
    # The original release is authenticated by verify_exact; this tests the
    # separate caller-to-predecessor equality check without repeated disk reads.
    monkeypatch.setattr(admission, 'verify_exact', lambda **_kw: reviewed_n12)
    profile = reviewed_n12['original_backend_profile']
    runtime = reviewed_n12['original_runtime_identity']
    with pytest.raises(ValueError, match='Specific failed N12'):
        runner.validate_prior_chain(_chain(), 2,
            expected_profile={**profile, 'sha256': '0'*64},
            expected_runtime=runtime)
    with pytest.raises(ValueError, match='Specific failed N12'):
        runner.validate_prior_chain(_chain(), 2,
            expected_profile=profile,
            expected_runtime={**runtime, 'sha256': '0'*64})
    with pytest.raises(ValueError, match='frozen runtime ancestry'):
        runner.validate_prior_chain(_chain(), 2)


def test_failed_receipt_hash_cannot_be_substituted(reviewed_n12, monkeypatch):
    monkeypatch.setattr(admission, 'verify_exact',
                        lambda **_kw: pytest.fail('Specific exception must not run'))
    with pytest.raises(ValueError):
        runner.validate_prior_chain(_chain('0'*64), 2,
            expected_profile=reviewed_n12['original_backend_profile'],
            expected_runtime=reviewed_n12['original_runtime_identity'])
