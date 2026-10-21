"""Source-array reuse must preserve fresh artifacts and every storage revision."""
from dataclasses import replace
from io import BytesIO
from zipfile import ZipFile, ZIP_STORED
import gc

import numpy as np
import pytest

from resectionlab import case_array_payload as payloads
from resectionlab.core import CaseData, SourceRef, array_digest, immutable_array
from resectionlab.imaging import load_case, read_case_artifacts, save_case


@pytest.fixture(autouse=True)
def isolated_cache():
    payloads._entries.clear()
    yield
    payloads._entries.clear()


def _case():
    image = np.arange(120, dtype=np.float32).reshape(4, 5, 6)
    mask = image > 50
    return CaseData(case_id='explicit-save-cache-fixture', mri=image,
        affine=np.diag([-1., 1.5, 2., 1.]), compartments={'target': mask},
        brain_mask=np.ones(image.shape, bool),
        source_refs=(SourceRef('fixture', 'synthetic://array-save-cache', provenance='simulated'),))


def _read(payload, name='values'):
    with np.load(BytesIO(payload), allow_pickle=False) as archive:
        return archive[name].copy()


def test_repeated_real_saves_reuse_only_arrays_and_roundtrip_fresh_artifacts(monkeypatch, tmp_path):
    case = _case()
    source_digest = array_digest(case.mri)
    encode = np.savez_compressed
    calls = []
    def counted(*args, **kwargs):
        calls.append(tuple(kwargs))
        return encode(*args, **kwargs)
    monkeypatch.setattr(np, 'savez_compressed', counted)
    first = save_case(case, tmp_path/'first.ressectionlab', artifacts={'cursor': 1})
    second = save_case(case, tmp_path/'second.ressectionlab', artifacts={'cursor': 2})
    assert len(calls) == 1
    assert read_case_artifacts(first) == {'cursor': 1}
    assert read_case_artifacts(second) == {'cursor': 2}
    with ZipFile(first) as old, ZipFile(second) as new:
        assert old.read('arrays.npz') == new.read('arrays.npz')
        assert new.getinfo('arrays.npz').compress_type == ZIP_STORED
    reopened = load_case(second)
    assert reopened.semantic_hash == case.semantic_hash
    np.testing.assert_array_equal(reopened.mri, case.mri)
    np.testing.assert_array_equal(reopened.affine, case.affine)
    np.testing.assert_array_equal(reopened.brain_mask, case.brain_mask)
    assert array_digest(case.mri) == source_digest
    # A real correction must encode its new immutable array set while retaining
    # original source memberships and the first files' exact source contents.
    corrected = replace(case, compartments={'target': np.zeros(case.mri.shape, bool)}, revision=2)
    third = save_case(corrected, tmp_path/'corrected.ressectionlab')
    assert len(calls) == 2
    revised = load_case(third)
    assert revised.semantic_hash == corrected.semantic_hash != case.semantic_hash
    assert not revised.compartments['target'].any()
    np.testing.assert_array_equal(revised.source_compartments['target'], case.source_compartments['target'])
    assert load_case(first).semantic_hash == case.semantic_hash


@pytest.mark.parametrize('storage', ['writable', 'readonly-owner', 'readonly-view'])
def test_mutable_storage_is_never_cached_even_when_view_is_readonly(storage):
    owner = np.arange(12, dtype=np.float32)
    values = owner.view() if storage == 'readonly-view' else owner
    if storage != 'writable':
        values.setflags(write=False)
    first = payloads.encode_array_payload({'values': values})
    if storage == 'readonly-owner':
        owner.setflags(write=True)
    owner[0] = 99.
    if storage == 'readonly-owner':
        owner.setflags(write=False)
    second = payloads.encode_array_payload({'values': values})
    assert _read(first)[0] == 0. and _read(second)[0] == 99.
    assert not payloads._entries


def test_immutable_view_shape_dtype_and_member_names_invalidate_payload():
    values = immutable_array(np.arange(12, dtype=np.int32).reshape(3, 4))
    first = payloads.encode_array_payload({'values': values})
    values.shape = (4, 3)
    second = payloads.encode_array_payload({'values': values})
    assert _read(first).shape == (3, 4) and _read(second).shape == (4, 3)
    values.dtype = np.float32
    third = payloads.encode_array_payload({'values': values})
    assert _read(second).dtype == np.int32 and _read(third).dtype == np.float32
    renamed = payloads.encode_array_payload({'new_name': values})
    np.testing.assert_array_equal(_read(renamed, 'new_name'), values)
    with np.load(BytesIO(renamed), allow_pickle=False) as archive:
        assert archive.files == ['new_name']


def test_structured_dtype_views_bypass_reuse_despite_equal_dtype_string():
    values = immutable_array(np.array([(1, 2)], dtype=[('first', '<i4'), ('second', '<i4')]))
    first = payloads.encode_array_payload({'values': values})
    values.dtype = [('other', '<i4'), ('field', '<i4')]
    second = payloads.encode_array_payload({'values': values})
    assert _read(first).dtype.names == ('first', 'second')
    assert _read(second).dtype.names == ('other', 'field')
    assert not payloads._entries


def test_cache_bounds_and_weak_references_do_not_retain_raw_arrays(monkeypatch):
    monkeypatch.setattr(payloads, 'MAX_CACHE_ENTRIES', 2)
    monkeypatch.setattr(payloads, 'MAX_CACHE_BYTES', 1000)
    values = [immutable_array(np.arange(12, dtype=np.float32) + i) for i in range(3)]
    for array in values:
        payloads.encode_array_payload({'values': array})
    assert len(payloads._entries) <= 2
    assert sum(len(entry[1]) for entry in payloads._entries.values()) <= 1000
    references = [reference for entry in payloads._entries.values() for reference in entry[0]]
    del array, values
    gc.collect()
    assert all(reference() is None for reference in references)
