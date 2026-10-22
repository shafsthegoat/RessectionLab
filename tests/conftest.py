"""Isolate regression caches and keep learning tests explicitly opt-in."""
import pytest


def pytest_addoption(parser):
    parser.addoption('--run-rl-pilot-tests', action='store_true', default=False,
        help='Run pilot policy-rollout tests, including an actual optimizer update on a generated fixture')


def pytest_configure(config):
    config.addinivalue_line('markers', 'rl_pilot: executes pilot policies or optimizer updates; explicit opt-in required')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--run-rl-pilot-tests'):
        return
    skip = pytest.mark.skip(reason='Pilot policy/optimizer tests require explicit --run-rl-pilot-tests')
    for item in items:
        if item.get_closest_marker('rl_pilot'):
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def isolated_surgical_caches(tmp_path, monkeypatch):
    # Several public bridge/frame tests intentionally use default constructors.
    # Their tiny fixture publications must not evict a user's patient cache or
    # inherit an authenticated entry created by an earlier test run.
    import resectionlab.surgical_verified_cache as verified
    import resectionlab.surgical_observed_cache as observed
    monkeypatch.setattr(verified, 'DEFAULT_DIRECTORY', tmp_path / 'verified-cache')
    monkeypatch.setattr(observed, 'DEFAULT_DIRECTORY', tmp_path / 'observed-cache')
