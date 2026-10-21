"""Generated asset transport controls; no real wheel download/install here."""
import hashlib
import importlib.util
import io
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("setup_atropos", Path(__file__).resolve().parents[1]/"scripts/setup_ixi265_atropos.py")
setup = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(setup)


def wheel():
    data = b"generated-not-a-real-wheel"
    return data, {"url": "https://files.pythonhosted.org/packages/generated.whl", "filename": "generated.whl",
                  "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def test_exact_small_asset_is_verified_and_cannot_retry(tmp_path, monkeypatch):
    data, row = wheel()
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(data))
    assert setup.download_one(row, tmp_path) == len(data)
    assert (tmp_path/row["filename"]).read_bytes() == data
    with pytest.raises(ValueError, match="reuse or retry"): setup.download_one(row, tmp_path)


@pytest.mark.parametrize("payload", [b"wrong-bytes", b"generated-not-a-real-wheel-and-too-many-bytes"])
def test_bad_asset_is_retained_as_failure_not_promoted(tmp_path, monkeypatch, payload):
    _, row = wheel()
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(ValueError): setup.download_one(row, tmp_path)
    assert not (tmp_path/row["filename"]).exists()
    assert (tmp_path/(row["filename"]+".partial")).exists()


@pytest.mark.parametrize("update", [{"url": "https://foreign.test/payload.whl"}, {"filename": "../payload.whl"}])
def test_foreign_source_refused_before_network_or_output(tmp_path, monkeypatch, update):
    _, row = wheel()
    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Unexpected network access"))
    with pytest.raises(ValueError, match="official PyPI"): setup.download_one({**row, **update}, tmp_path)
    assert not list(tmp_path.iterdir())


def test_unreleased_setup_does_not_create_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(setup, "ASSETS", tmp_path/"not-created")
    with pytest.raises(ValueError, match="release"): setup.setup()
    assert not (tmp_path/"not-created").exists()
