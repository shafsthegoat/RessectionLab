"""Frozen replay caches belong to session storage, never to the signed bundle."""
import sys
from types import SimpleNamespace

from resectionlab.desktop_bridge import BridgeSession
from resectionlab.surgical_bridge import _cache_directory, _verified_cache, cache_directories
from resectionlab.surgical_frame_replay import SurgicalFrameCache
from resectionlab.surgical_replay_cache import SurgicalReplayCache


def test_frozen_cache_owners_write_keys_under_validated_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    session = BridgeSession(tmp_path / "transfers", run_dir=tmp_path / "user-data/runs")
    try:
        session.surgical_replay_cache = SurgicalReplayCache(verified_cache=_verified_cache(session))
        session.surgical_frame_cache = SurgicalFrameCache(directory=_cache_directory(session, "observed"))
        verified = session.surgical_replay_cache.verified_cache
        observed = session.surgical_frame_cache.store.owner
        verified._key()
        observed._key()
        assert session.surgical_cache_root == session.run_dir / "surgical-cache"
        assert cache_directories(session) == {
            "verified": str(session.run_dir / "surgical-cache/verified"),
            "observed": str(session.run_dir / "surgical-cache/observed")}
        assert all(path.is_relative_to(session.run_dir) for path in
                   (verified.directory, observed.directory))
    finally:
        session.close()


def test_source_defaults_and_frozen_transfer_fallback(tmp_path, monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)
    source = BridgeSession(tmp_path / "source", run_dir=tmp_path / "runs")
    try:
        assert source.surgical_cache_root is None
        assert _cache_directory(source, "observed") is None
        assert _cache_directory(SimpleNamespace(), "observed") is None
    finally:
        source.close()
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    frozen = BridgeSession(tmp_path / "frozen")
    try:
        assert frozen.surgical_cache_root == frozen.transfers.root / "surgical-cache"
    finally:
        frozen.close()
