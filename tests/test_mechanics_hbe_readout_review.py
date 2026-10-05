"""Independent constructed primitive controls; never runs FEM or reads measured curves."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import mechanics_hbe_readout as r


SPEC = importlib.util.spec_from_file_location("hbe_readout_owner_fixture",
    Path(__file__).with_name("test_mechanics_hbe_readout.py"))
f = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(f)


def rewrite(root, bindings, key, text):
    path = root / bindings[key]["path"]
    path.write_text(text)
    bindings[key] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("corruption", ["missing_rest", "missing_last_element", "duplicate_node"])
def test_rehashed_incomplete_or_duplicate_primitives_still_reject(tmp_path, corruption):
    bindings = f.fixture(tmp_path)
    if corruption == "missing_rest":
        key = "nodes"
        text = (tmp_path / bindings[key]["path"]).read_text()
        text = text[text.index("*Step = 1\n"):]
    elif corruption == "missing_last_element":
        key = "elements"
        text = (tmp_path / bindings[key]["path"]).read_text()
        text = text[:text.index("*Step = 60\n")]
    else:
        key = "nodes"
        text = (tmp_path / bindings[key]["path"]).read_text()
        text = text.replace("\n2,", "\n1,", 1)
    rewrite(tmp_path, bindings, key, text)
    with pytest.raises(ValueError):
        f.run(tmp_path, bindings)


def test_negative_logged_element_j_cannot_pass_even_with_positive_node_reconstruction(tmp_path):
    bindings = f.fixture(tmp_path)
    lines = (tmp_path / bindings["elements"]["path"]).read_text().splitlines()
    row = lines[-1].split(",")
    row[7] = "-1"  # ID + six stresses + J + sed
    lines[-1] = ",".join(row)
    rewrite(tmp_path, bindings, "elements", "\n".join(lines))
    try:
        receipt, _ = f.run(tmp_path, bindings)
    except ValueError:
        return
    assert receipt["passed"] is False


def test_loading_units_checked_before_any_primitive_iterator(tmp_path, monkeypatch):
    bindings = f.fixture(tmp_path)
    loading = json.loads((tmp_path / bindings["loading"]["path"]).read_text())
    loading["load_coordinate_units"] = "mm"
    rewrite(tmp_path, bindings, "loading", json.dumps(loading))
    def forbidden(*args, **kwargs):
        raise AssertionError("Wrong physical metadata must reject before reading frames")
    monkeypatch.setattr(r, "iter_data_records", forbidden)
    with pytest.raises(ValueError, match="units"):
        f.run(tmp_path, bindings)


def test_source_change_during_readout_detected_by_after_hash(tmp_path, monkeypatch):
    bindings = f.fixture(tmp_path)
    original = r.HexMesh.read_frame
    mutated = []
    def changed(self, *args, **kwargs):
        if not mutated:
            with (tmp_path / bindings["deck"]["path"]).open("a") as stream:
                stream.write("changed during readout")
            mutated.append(True)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(r.HexMesh, "read_frame", changed)
    with pytest.raises(ValueError, match="hash"):
        f.run(tmp_path, bindings)


def test_refinement_connection_uses_common_load_states_and_strict_trend():
    def receipt(value, steps):
        response = np.full(steps + 1, value)
        return {"branch": "tension", "steps": steps, "applied_force_N": response.tolist(),
                "probe_displacements_m": np.zeros((steps + 1, 75, 3)).tolist()}
    # Differences0.002 then0.002 are within2% absolute response tolerance but are not decreasing.
    metrics = r._refinement_group(receipt(1, 60), receipt(1.002, 60), receipt(1.004, 60), kind="mesh")
    assert metrics["reaction"]["actual"] < metrics["reaction"]["limit"]
    assert metrics["reaction_trend"]["comparison"] == "lt"
    assert metrics["reaction_trend"]["actual"] >= metrics["reaction_trend"]["limit"]
    fine = receipt(1.001, 120)
    fine["applied_force_N"][1::2] = [9] * 60
    step = r._refinement_group(receipt(1, 60), receipt(1, 60), fine, kind="step")
    assert step["reaction"]["actual"] == pytest.approx(.001)
    assert step["reaction"]["units"] == "N"
