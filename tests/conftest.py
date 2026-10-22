"""Keep generated regression caches away from actual application replays."""
import pytest


@pytest.fixture(autouse=True)
def isolated_surgical_caches(tmp_path, monkeypatch):
    # Several public bridge/frame tests intentionally use default constructors.
    # Their tiny fixture publications must not evict a user's patient cache or
    # inherit an authenticated entry created by an earlier test run.
    import resectionlab.surgical_verified_cache as verified
    import resectionlab.surgical_observed_cache as observed
    monkeypatch.setattr(verified, 'DEFAULT_DIRECTORY', tmp_path / 'verified-cache')
    monkeypatch.setattr(observed, 'DEFAULT_DIRECTORY', tmp_path / 'observed-cache')
