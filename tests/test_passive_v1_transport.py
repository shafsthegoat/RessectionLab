"""Generated-only transport checks against the unchanged canonical v1 task."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
import hashlib

import numpy as np
import pytest

from resectionlab import persistent_passive as canonical_passive
from resectionlab import passive_desktop_transport as transport
from resectionlab.core import semantic_digest
from resectionlab.desktop_bridge import BridgeSession, BridgeError, _Request


def test_fixed_v1_episode_and_full_source_binding():
    result = transport.execute_persistent_passive_episode()
    episode, binding = result["episode"], result["sourceBinding"]
    assert episode["episodeId"] == canonical_passive.run_passive_development_episode(paired=True)["episodeId"]
    assert episode["episodeId"] == "sha256:33cbb8bdea5873acece2461fbf1c992791ff7cae18ba2099c4bc1724cd76717a"
    assert json.loads(result["episodeCanonicalJson"]) == {k: v for k, v in episode.items() if k != "episodeId"}
    assert binding["caseHash"] == result["case"].semantic_hash
    assert binding["sourceHash"] == episode["sourceHash"]
    assert binding["decisionModelHash"] == episode["decisionModelHash"]
    assert len(binding["tools"]) == 2
    assert binding["bindingHash"] == semantic_digest({k: v for k, v in binding.items() if k != "bindingHash"})


def test_wrong_display_and_renderer_choice_refuse():
    result = transport.execute_persistent_passive_episode()
    task = canonical_passive.make_passive_development_task(paired=True)
    altered = replace(result["case"], mri=np.zeros_like(result["case"].mri))
    with pytest.raises(ValueError, match="Display differs"):
        transport.passive_source_binding(task, altered)
    with pytest.raises(ValueError, match="fixed paired"):
        transport.execute_persistent_passive_episode(paired=False)


@pytest.fixture
def session(tmp_path):
    instance = BridgeSession(tmp_path)
    yield instance
    instance.transfers.close()


def test_live_bridge_publishes_bound_source_and_retains_two_tool_history(session):
    result = session.execute("executePersistentPassiveEpisode",
        {"fixture": "generated-persistent-passive-v1", "paired": True},
        _Request("passive-live"), lambda *args: None)
    source, episode, binding = result["case"], result["episode"], result["sourceBinding"]
    assert source["caseHash"] == binding["caseHash"]
    assert episode["sourceHash"] == binding["sourceHash"]
    descriptor = source["mri"]
    payload = Path(descriptor["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == descriptor["sha256"]
    assert len(payload) == descriptor["byteLength"]
    assert len(episode["frames"]) == 6
    assert all(frame["operative"]["other_tool"]["pose"] is not None for frame in episode["frames"])
    retained = session.cases[source["caseHash"]].episode
    assert retained == {key: result[key] for key in ("episode", "episodeCanonicalJson", "sourceBinding")}


@pytest.mark.parametrize("args", [{}, {"fixture": "patient", "paired": True},
    {"fixture": "generated-persistent-passive-v1", "paired": False},
    {"fixture": "generated-persistent-passive-v1", "paired": True, "path": "untrusted"}])
def test_invalid_live_request_never_installs_case(session, args):
    with pytest.raises(BridgeError):
        session.execute("executePersistentPassiveEpisode", args, _Request("invalid"), lambda *values: None)
    assert not session.cases


def test_cancelled_live_request_never_installs_case(session):
    request = _Request("cancelled")
    request.cancelled.set()
    with pytest.raises(BridgeError):
        session.execute("executePersistentPassiveEpisode",
            {"fixture": "generated-persistent-passive-v1", "paired": True}, request, lambda *args: None)
    assert not session.cases
