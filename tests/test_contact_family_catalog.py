"""Metadata-only generated family admission; no task, actor, or native step."""
from __future__ import annotations

from collections import Counter
import pytest

from resectionlab import contact_family_desktop_bridge as candidate


@pytest.fixture(autouse=True)
def unpublished_catalog(monkeypatch):
    """Keep these absent-publication controls independent of local checkpoints."""
    monkeypatch.setattr(candidate, "RELEASE_MANIFEST_SHA256", None)


def test_catalog_has_exact_roles_and_unpublished_learned_methods():
    catalog = candidate.public_contact_family_availability()
    assert set(catalog) == {"version", "fixture", "familyHash", "experimentHash",
                            "releaseHash", "layouts", "methods"}
    assert catalog["version"] == "generated-public-contact-learning-availability-v1"
    assert catalog["fixture"] == candidate.FAMILY_VERSION
    assert catalog["experimentHash"] is None and catalog["releaseHash"] is None
    assert len(catalog["layouts"]) == len({row["layoutId"] for row in catalog["layouts"]}) == 24
    assert Counter(row["role"] for row in catalog["layouts"]) == {
        "TRAIN": 12, "SELECT": 4, "MEASUREMENT_EVAL": 8}
    assert all(row["goals"] == ["surface", "deep"] for row in catalog["layouts"])
    assert all(row["interactive"] == (row["role"] in ("TRAIN", "SELECT"))
               for row in catalog["layouts"])
    controller_ready = candidate._controller_ready()
    for method in ("STOP", "SEARCH"):
        assert catalog["methods"][method] == {"available": controller_ready,
            "reason": None if controller_ready else "backend_controller_not_promoted"}
    for method in ("IL", "RL"):
        assert catalog["methods"][method]["available"] is False
        assert catalog["methods"][method]["reason"]


def test_heldout_and_unpublished_learned_refuse_before_controller(tmp_path, monkeypatch):
    catalog = candidate.public_contact_family_availability()
    heldout = next(row["layoutId"] for row in catalog["layouts"]
                   if row["role"] == "MEASUREMENT_EVAL")
    train = next(row["layoutId"] for row in catalog["layouts"] if row["role"] == "TRAIN")
    monkeypatch.setattr(candidate, "_read_exact_supervisor",
                        lambda: (_ for _ in ()).throw(AssertionError("child controller opened")))
    with pytest.raises(ValueError, match="HELD_OUT_EXECUTION_CLOSED"):
        candidate.execute_public_contact_family_episode(attempts_root=tmp_path,
            layout_id=heldout, goal_id="surface", selector="STOP")
    with pytest.raises(candidate.ContactReleaseUnavailable, match="No reviewed final"):
        candidate.execute_public_contact_family_episode(attempts_root=tmp_path,
            layout_id=train, goal_id="surface", selector="IL")
    assert not list(tmp_path.iterdir())


def test_old_family_source_refuses_before_catalog_or_controller(monkeypatch):
    monkeypatch.setattr(candidate, "FAMILY_VERSION", "generated-public-contact-family-v1")
    with pytest.raises(RuntimeError, match="family v2 source"):
        candidate.public_contact_family_availability()


def test_unbound_controller_marks_nonlearned_methods_unavailable(monkeypatch):
    monkeypatch.setattr(candidate, "SUPERVISOR_SHA256", None)
    catalog = candidate.public_contact_family_availability()
    for method in ("STOP", "SEARCH"):
        assert catalog["methods"][method] == {"available": False,
            "reason": "backend_controller_not_promoted"}
