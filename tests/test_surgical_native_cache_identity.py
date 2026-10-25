"""Optional native modes never share cache or observed/prefix authority."""
import copy

import numpy as np
import pytest

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import SurgicalScene
from resectionlab.surgical_verified_cache import VerifiedSurgicalCache, implementation_fingerprint
from resectionlab.surgical_observed_cache import (
    FrameAdmission, ImplementationChanged, light_implementation_seal)
from resectionlab.surgical_prefix_replay import SurgicalPrefixReplay, PrefixInvalidated
from resectionlab import surgical_idle_native as native


def empty_engine():
    labels = np.ones((2, 2, 2), dtype=np.uint16)
    scene = SurgicalScene(labels, np.eye(4),
        [{'id': 1, 'name': 'generated fixture', 'material': 'tumor', 'target': True}],
        {'kind': 'generated-native-identity-fixture'}, {})
    return SurgicalEngine(scene)


def selected_identity(monkeypatch):
    state = {'mode': 'python', 'binarySha256': None}

    def identity():
        if state.get('unavailable'):
            raise ValueError('selected binary no longer matches its load receipt')
        return copy.deepcopy(state)

    # Emulate one stable adapter function returning changed native metadata.
    # Its loaded code/defaults stay fixed throughout each regression; an identity
    # change must be caught independently of Python bytecode/source changes.
    identity.__module__ = native.__name__
    monkeypatch.setattr(native, 'strict_live_identity', identity)
    return state


@pytest.mark.parametrize('change', [
    {'mode': 'native', 'binarySha256': 'a' * 64},
    {'mode': 'python', 'binarySha256': 'b' * 64},
])
def test_selected_native_metadata_changes_both_full_and_light_seals(monkeypatch, change):
    selected = selected_identity(monkeypatch)
    full, light = implementation_fingerprint(), light_implementation_seal()
    assert full is not None
    assert (implementation_fingerprint(), light_implementation_seal()) == (full, light)
    selected.update(change)
    assert implementation_fingerprint() not in (None, full)
    assert light_implementation_seal() != light


def test_same_numeric_checkpoint_cannot_reuse_cache_across_selected_modes(monkeypatch, tmp_path):
    selected = selected_identity(monkeypatch)
    engine = empty_engine()
    snapshot = engine.snapshot()
    cache = VerifiedSurgicalCache(tmp_path / 'cache')
    restored = SurgicalEngine.restore(engine.scene, snapshot, use_cache=True, verified_cache=cache)
    assert restored.snapshot() == snapshot and cache.publications == 1
    assert cache.restore(SurgicalEngine, engine.scene, snapshot) is not None
    hits = cache.hits
    selected.update(mode='native', binarySha256='c' * 64)
    assert cache.restore(SurgicalEngine, engine.scene, snapshot) is None
    assert cache.hits == hits


def test_unavailable_selected_native_refuses_old_observed_and_prefix_owners(monkeypatch):
    selected = selected_identity(monkeypatch)
    engine = empty_engine()
    snapshot = engine.snapshot()
    admission = FrameAdmission.canonical(engine, snapshot, [])
    prefix = SurgicalPrefixReplay(engine.scene, snapshot)
    selected['unavailable'] = True
    assert implementation_fingerprint() is None
    with pytest.raises(ImplementationChanged):
        light_implementation_seal()
    with pytest.raises(ImplementationChanged):
        admission.check()
    with pytest.raises(PrefixInvalidated):
        prefix.seek(0)
    # The invalidated prefix does not recover by silently changing selection.
    selected.pop('unavailable')
    with pytest.raises(PrefixInvalidated):
        prefix.seek(0)
