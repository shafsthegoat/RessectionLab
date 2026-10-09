"""Metadata, framing and process controls. Never decode a patient source."""
import copy
import importlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def qc(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    q = importlib.import_module('resect_deferred_qc')
    monkeypatch.setattr(socket, 'create_connection', lambda *a, **k: pytest.fail('network forbidden'))
    monkeypatch.setattr(q, 'verify_pair_files', lambda *a, **k: pytest.fail('patient fixity operation forbidden'))
    monkeypatch.setattr(q, 'inspect_pair', lambda *a, **k: pytest.fail('patient decoding forbidden'))
    return q


@pytest.fixture
def frozen(qc):
    m = qc.preflight()
    records = {k: qc.proof_json(p) for k, p in m['metadata'].items() if k not in ('rights', 'mirror_source')}
    return m, qc.authority.require_manifest(), records


@pytest.fixture
def cache(qc, monkeypatch):
    with tempfile.TemporaryDirectory(prefix='resect-qc-controls-', dir=ROOT / 'build') as folder:
        monkeypatch.setattr(qc, 'CACHE', Path(folder))
        yield Path(folder)


def test_actual_metadata_only_preflight_and_denominator(qc, frozen):
    m, _, _ = frozen
    outcomes = qc.initial_outcomes(m)
    assert len(outcomes) == 25 and len(m['pairs']) == 24
    assert outcomes[0]['pair_id'] == 'Case3-during'
    assert outcomes[0]['status'] == 'inherited_structural_pass'
    assert len(m['missing_annotations']) == 3
    assert all(r['pair']['role'] == 'TRAIN' for r in m['pairs'])
    assert all(r['pair']['patient_group'] != 'RESECT:Case4' for r in m['pairs'])
    assert m['rights']['annotations_license'] == 'CC-BY-NC-SA-4.0'


@pytest.mark.parametrize('mutation', ['remove', 'duplicate', 'role', 'person', 'phase', 'source',
    'image_hash', 'mask_hash', 'revision', 'missing', 'excluded', 'rights', 'denominator', 'claims', 'history'])
def test_worklist_mutations_refused(qc, frozen, mutation):
    m, a, r = copy.deepcopy(frozen)
    first = m['pairs'][0]
    if mutation == 'remove': m['pairs'].pop()
    elif mutation == 'duplicate': m['pairs'][-1] = copy.deepcopy(first)
    elif mutation in ('role', 'person', 'phase'):
        first['pair'][{'role': 'role', 'person': 'patient_group', 'phase': 'phase'}[mutation]] = 'wrong'
    elif mutation == 'source': first['image']['source_url'] = 'https://example.com'
    elif mutation in ('image_hash', 'mask_hash'): first[mutation.replace('_hash', '_sha256')] = '0' * 64
    elif mutation == 'revision': first['mask']['file_revision'] = 1
    elif mutation == 'missing': m['missing_annotations'].pop()
    elif mutation == 'excluded': m['excluded_members'].pop()
    elif mutation == 'rights': m['rights']['commercial_clearance'] = True
    elif mutation == 'denominator': m['denominator']['qualified_pairs'] = 24
    elif mutation == 'claims': m['claims']['training_admitted'] = True
    elif mutation == 'history': m['historical_status_receipts'].pop()
    with pytest.raises(qc.Refusal): qc.validate_manifest(m, a, r)


@pytest.mark.parametrize('field,value', [('acquisition_transport', 'original_osf'),
    ('original_osf_attempts_modified', True), ('import_binding_sha256', '0'*64),
    ('mirror_receipt_sha256', '0'*64), ('training_admitted', True),
    ('status', 'failed'), ('verified_bytes', 1), ('sha256', '0'*64)])
def test_import_receipt_mutations_refused(qc, frozen, monkeypatch, field, value):
    m, _, records = frozen
    row = m['pairs'][0]
    original = qc.proof_json
    def changed(proof, *args):
        result = original(proof, *args)
        if proof == row['import_receipt']: result[field] = value
        return result
    monkeypatch.setattr(qc, 'proof_json', changed)
    with pytest.raises(qc.Refusal): qc.validate_mirror(row, m, records)


@pytest.mark.parametrize('parent', ['import_summary', 'mirror_summary', 'pilot_parent', 'pilot_worker'])
def test_parent_failure_never_becomes_success(qc, frozen, parent):
    m, a, records = copy.deepcopy(frozen)
    records[parent]['status'] = 'failed'
    with pytest.raises(qc.Refusal): qc.validate_manifest(m, a, records)


def test_changed_proof_refused(qc, frozen):
    proof = dict(frozen[0]['metadata']['rights'], sha256='0'*64)
    with pytest.raises(qc.Refusal, match='metadata_proof_changed'): qc.proof_bytes(proof)


@pytest.mark.parametrize('args', [['batch'], ['batch', '--execute', '--manifest-sha', 'wrong', '--run-id', 'test'],
    ['worker', '--execute', '--manifest-sha', 'wrong', '--run-id', 'test']])
def test_cli_requires_explicit_exact_execution(qc, monkeypatch, args):
    monkeypatch.setattr(qc, 'batch', lambda *a: pytest.fail('must not execute'))
    monkeypatch.setattr(qc, 'worker', lambda *a: pytest.fail('must not execute'))
    assert qc.main(args) == 2


@pytest.mark.parametrize('seconds', [0, 4, 601, float('inf'), float('nan'), True])
def test_batch_budget_refusal_before_work(qc, seconds):
    with pytest.raises(qc.Refusal): qc.batch('control', seconds)


def test_decode_workspace_math_only(qc, frozen):
    bounds = frozen[0]['bounds']
    image = qc.streaming.scalar_budget([338, 303, 245], 4, bounds)
    mask = qc.streaming.annotations.decoding_budget([338, 303, 245], 8, bounds)
    assert image['voxels'] == 25091430
    assert mask['chunk_working_bytes_bound'] == 1245184
    with pytest.raises(qc.streaming.Refusal): qc.streaming.scalar_budget([2**30, 2, 2], 8, bounds)


@pytest.mark.parametrize('mutation', ['bitpix', 'magic', 'offset', 'short'])
def test_prefix_framing_refusals_without_image_file(qc, frozen, monkeypatch, mutation):
    import nibabel as nib
    header = nib.Nifti1Header()
    header.set_data_shape((2, 2, 2))
    header['vox_offset'] = 352
    if mutation == 'bitpix': header['bitpix'] = 8
    if mutation == 'magic': header['magic'] = b'ni1\0'
    if mutation == 'offset': header['vox_offset'] = 2**22
    raw = header.binaryblock[:100] if mutation == 'short' else header.binaryblock
    monkeypatch.setattr(qc.gzip, 'open', lambda *a, **k: io.BytesIO(raw))
    with pytest.raises(qc.Refusal):
        qc.check_prefix(ROOT / 'build/control-prefix-no-image', frozen[0]['bounds'], time.monotonic()+10)


def failed_control_receipt(qc, command):
    """Contract-test metadata only; no scientific QC result is manufactured."""
    run_id = command[command.index('--run-id')+1]
    pair_id = command[command.index('--pair')+1]
    run = qc.CACHE/'runs'/run_id
    trial = run/'pairs'/pair_id
    intent = json.loads((trial/'intent.json').read_bytes())
    declaration = json.loads((run/'declaration.json').read_bytes())
    row = next(r for r in qc.preflight()['pairs'] if r['pair']['id'] == pair_id)
    receipt = qc.base_receipt(row, intent, declaration)
    receipt.update(status='review_failed', fixity_before='passed', fixity_after='passed',
                   test_control_only=True)
    qc.save(trial/'receipt.json', receipt)
    return trial, receipt


def test_all_scientific_refusals_keep_25_denominator(qc, cache, monkeypatch):
    def supervise(command, log, *, deadline, on_start):
        on_start(99999999)
        failed_control_receipt(qc, command)
        return 'completed', 0
    monkeypatch.setattr(qc, 'supervise', supervise)
    report = qc.batch('metadata-refusals')
    assert report['status'] == 'bounded_reviews_finished'
    assert report['outcome_counts'] == {'inherited_structural_pass': 1, 'review_failed': 24}
    assert len(report['missing_annotations']) == 3
    assert not report['training_admitted'] and not report['geometric_admission']


@pytest.mark.parametrize('mode', ['prestart', 'poststart', 'failed_receipt', 'bad_receipt', 'late'])
def test_interruption_receipt_and_remaining_denominator(qc, cache, monkeypatch, mode):
    real_check = qc.check_deadline
    expired = [False]
    def check(deadline):
        if expired[0]: raise TimeoutError('control deadline')
        return real_check(deadline)
    monkeypatch.setattr(qc, 'check_deadline', check)
    def supervise(command, log, *, deadline, on_start):
        if mode == 'prestart': raise OSError('start control')
        on_start(99999999)
        if mode in ('failed_receipt', 'bad_receipt', 'late'):
            trial, receipt = failed_control_receipt(qc, command)
            if mode == 'bad_receipt':
                (trial/'receipt.json').write_text('{"status":"review_passed"}')
            if mode == 'late':
                expired[0] = True
                return 'completed', 0
        raise KeyboardInterrupt()
    monkeypatch.setattr(qc, 'supervise', supervise)
    report = qc.batch('interrupted-'+mode)
    assert report['status'] == 'failed_or_incomplete'
    assert len(report['outcomes']) == 25
    assert sum(o['status'] == 'not_attempted' for o in report['outcomes']) == 23
    outcome = report['outcomes'][1]
    assert outcome['attempt'] and outcome['intent_sha256']
    if mode == 'prestart':
        assert outcome['status'] == 'worker_start_failed' and 'worker_pid' not in outcome
    else: assert outcome['worker_pid'] == 99999999
    if mode in ('failed_receipt', 'bad_receipt', 'late'):
        assert outcome['receipt_sha256']
    if mode == 'bad_receipt': assert outcome['receipt_error_type'] == 'Refusal'
    if mode == 'late': assert outcome['completion_error_type'] == 'TimeoutError'


def test_invalid_receipt_retains_raw_hash(qc, frozen, cache):
    row = frozen[0]['pairs'][0]
    trial = cache/'invalid';trial.mkdir()
    raw = b'not-json'
    (trial/'receipt.json').write_bytes(raw)
    outcome = {}
    with pytest.raises(json.JSONDecodeError): qc.retain_receipt(outcome, trial, row, {}, {})
    assert outcome['receipt_sha256'] == qc.digest(raw)


def test_real_process_callback_failure_reaps_child(qc, cache):
    seen = []
    def callback(pid):
        seen.append(pid)
        raise RuntimeError('callback control')
    with pytest.raises(RuntimeError):
        qc.supervise([sys.executable, '-c', 'import time;time.sleep(20)'], cache/'control.log',
                     deadline=time.monotonic()+2, on_start=callback)
    with pytest.raises(ProcessLookupError): os.kill(seen[0], 0)


def test_review_pass_requires_actual_stage_results(qc, frozen):
    row = frozen[0]['pairs'][0]
    declaration = {'run_id': 'contract'}
    receipt = qc.base_receipt(row, {}, declaration)
    receipt.update(status='review_passed', fixity_before='passed', fixity_after='passed')
    with pytest.raises(qc.Refusal, match='review_pass_claim_inconsistent'):
        qc.receipt_contract(receipt, row, {}, declaration)


@pytest.mark.parametrize('difference,expected', [(0., 'passed'), (.009, 'passed'), (.011, 'failed'),
                                              (float('nan'), 'failed')])
def test_geometry_control_never_grants_admission(qc, difference, expected):
    # Coordinate algebra only, no voxel array or image fixture.
    import numpy as np
    a = {'shape': [2, 2, 2], 'affine_ras_mm': np.eye(4).tolist()}
    b = copy.deepcopy(a)
    b['affine_ras_mm'][0][3] = difference
    result = qc.pair_geometry(a, b)
    assert result['status'] == expected and result['geometric_admission'] is False
    json.dumps(result, allow_nan=False)


def test_success_words_without_raw_source_grid_cannot_pass(qc, frozen):
    row = frozen[0]['pairs'][0]
    declaration = {'run_id': 'contract'}
    receipt = qc.base_receipt(row, {}, declaration)
    receipt.update(status='review_passed', fixity_before='passed', fixity_after='passed',
                   image_qc={k: {'status': 'passed'} for k in ('header_qc', 'scalar_qc', 'geometry_qc')},
                   mask_qc={'status': 'passed'}, mask_geometry={'status': 'passed'}, pair_geometry={'status': 'passed'})
    with pytest.raises(qc.Refusal, match='review_pass_metadata_inconsistent'):
        qc.receipt_contract(receipt, row, {}, declaration)


@pytest.mark.parametrize('kind', ['image', 'mask'])
@pytest.mark.parametrize('mutation', ['raw_summary', 'affine', 'shape', 'budget', 'filename'])
def test_reported_header_fields_recomputed_from_bytes(qc, frozen, kind, mutation):
    # Serialized header control only; no voxel buffer or image file exists.
    import nibabel as nib
    from resectionlab.critical_evidence import nifti1_header_record
    h = nib.Nifti1Header()
    h.set_data_shape((2, 2, 2))
    h.set_xyzt_units('mm')
    h.set_sform([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], code=1)
    h['vox_offset'] = 352
    raw = h.binaryblock
    source = {kind + '_sha256': 'a'*64, kind: {'path': 'originals/control.nii.gz'}}
    bounds = frozen[0]['bounds']
    budget = qc.streaming.scalar_budget if kind == 'image' else qc.streaming.annotations.decoding_budget
    value = {'raw_grid': nifti1_header_record(raw, 'a'*64), 'decoding_budget': budget([2, 2, 2], 4, bounds)}
    geometry = qc.geometry_from_header(raw, 'control.nii.gz')
    qc.reported_header_contract(raw, value, geometry, source, bounds, kind)
    if mutation == 'raw_summary': value['raw_grid']['raw_grid']['shape'] = [9, 9, 9]
    elif mutation == 'affine': geometry['affine_ras_mm'][0][3] = 1234.
    elif mutation == 'shape': geometry['shape'] = [2, 2, 3]
    elif mutation == 'budget': value['decoding_budget']['voxels'] = 999
    elif mutation == 'filename': geometry['file'] = 'different.nii.gz'
    with pytest.raises(qc.Refusal):
        qc.reported_header_contract(raw, value, geometry, source, bounds, kind)
