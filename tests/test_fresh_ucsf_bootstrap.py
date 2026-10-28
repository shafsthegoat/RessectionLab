"""Metadata-only bootstrap works before any patient pixels are requested."""

import importlib.util
import json
from pathlib import Path

import pytest


def test_development_declaration_is_metadata_only_and_immutable(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_fresh_ucsf_demo.py"
    spec = importlib.util.spec_from_file_location("fresh_ucsf_script", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "SOURCE_ROOT", tmp_path / "source")
    cases = ["UCSF-PDGM-0005", "UCSF-PDGM-0007"]
    path = module.declare_development(cases, "ascending unused mirror IDs before pixel access")
    original = path.read_bytes()
    record = json.loads(original)
    assert record["cases"] == cases and record["role"] == "DEVELOPMENT"
    assert record["image_fetch_cap_bytes"] == 60000000
    assert list(tmp_path.rglob("*.nii.gz")) == []
    assert module.declare_development(cases, None) == path
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="differs"):
        module.declare_development(["UCSF-PDGM-0005"], "different rule")
    with pytest.raises(ValueError, match="four-digit"):
        module.declare_development(["bad-name"], "rule")
