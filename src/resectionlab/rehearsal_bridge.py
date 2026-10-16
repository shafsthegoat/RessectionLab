"""Desktop authority for source-bound rehearsal states and saved branches.

The renderer can request a bounded action or select a view. Source masks,
episode history, branch outcomes and replay certification are owned here.
"""
from collections import OrderedDict

from .core import freeze_json, semantic_digest, thaw_json

ARTIFACT_KEY = "patient_rehearsal"
SCHEMA = "patient-rehearsal-workspace-v1"
OPERATIONS = frozenset({"createRehearsal", "inspectRehearsal", "stepRehearsal", "compareRehearsal"})


def _payload(case_hash, episode, alternatives, snapshot_id, saved_view=None):
    exported = episode.export()
    result = {
        "caseHash": case_hash,
        "episode": exported,
        "stateId": exported["finalStateId"],
        "suggestedActions": episode.suggested_actions(),
        "comparison": None,
    }
    if alternatives:
        result["comparison"] = {
            "snapshotId": snapshot_id["snapshotId"],
            "alternatives": [{"id": row["id"], "label": row["label"], "episode": row["episode"].export()} for row in alternatives],
        }
    if saved_view is not None:
        result["savedView"] = saved_view
    return result


def _restore(case, record):
    from .rehearsal_episode import restore_episode
    if not isinstance(record, dict) or set(record) != {"schema", "live", "comparison"} or record["schema"] != SCHEMA:
        raise ValueError("Unsupported saved rehearsal")
    episode = restore_episode(case, record["live"])
    alternatives = []
    comparison = record["comparison"]
    snapshot = None
    if comparison is not None:
        if (not isinstance(comparison, dict) or set(comparison) != {"snapshotId", "snapshot", "alternatives"}
                or not isinstance(comparison["alternatives"], list) or len(comparison["alternatives"]) != 2):
            raise ValueError("Saved comparison requires two source-bound alternatives")
        base = restore_episode(case, comparison["snapshot"])
        if base.export()["finalStateId"] != comparison["snapshotId"]:
            raise ValueError("Saved comparison snapshot identity differs")
        snapshot = {"snapshotId": comparison["snapshotId"], "record": base.serialize()}
        prefix = base.serialize()["actions"]
        for row in comparison["alternatives"]:
            if not isinstance(row, dict) or set(row) != {"id", "label", "record"}:
                raise ValueError("Invalid saved rehearsal branch")
            branch = restore_episode(case, row["record"])
            actions = branch.serialize()["actions"]
            if (actions[:len(prefix)] != prefix or branch.export()["initialStateId"] != base.export()["initialStateId"]
                    or len(branch.frames) < len(base.frames)
                    or branch.frames[:len(base.frames)] != base.frames):
                raise ValueError("Saved branches do not share the exact comparison snapshot")
            alternatives.append({"id": row["id"], "label": row["label"], "episode": branch})
        if len({row["id"] for row in alternatives}) != 2:
            raise ValueError("Saved branch identities must be distinct")
    return episode, alternatives, snapshot


def _record(episode, alternatives, snapshot):
    return {"schema": SCHEMA, "live": episode.serialize(), "comparison": None if not alternatives else {
        "snapshotId": snapshot["snapshotId"], "snapshot": snapshot["record"],
        "alternatives": [{"id": row["id"], "label": row["label"], "record": row["episode"].serialize()} for row in alternatives],
    }}


def checked_view(entry, state, payload):
    """Admit only display preferences for a backend-reconstructed episode."""
    if state is None:
        return None
    if not isinstance(state, dict) or set(state) != {"branchId", "frameIndex", "selectedTool", "visible", "episodeInitialStateId"}:
        raise ValueError("Invalid rehearsal view preferences")
    branches = {"live": payload["episode"]}
    if payload["comparison"]:
        branches.update({row["id"]: row["episode"] for row in payload["comparison"]["alternatives"]})
    episode = branches.get(state["branchId"])
    if episode is None or state["episodeInitialStateId"] != payload["episode"]["initialStateId"]:
        raise ValueError("Rehearsal view belongs to another episode or branch")
    if (type(state["frameIndex"]) is not int or not 0 <= state["frameIndex"] < len(episode["replayFrames"])
            or type(state["visible"]) is not bool
            or state["selectedTool"] not in {tool["toolId"] for tool in episode["tools"]}):
        raise ValueError("Rehearsal view references an unavailable frame or tool")
    return dict(state)



def checked_saved_view(session, entry, state):
    if state is None:
        return None
    raw = thaw_json(entry.artifacts).get(ARTIFACT_KEY)
    if raw is None:
        raise ValueError("Create a source-bound rehearsal before saving its view")
    cached = getattr(session, "rehearsal_sessions", {}).get(entry.case.semantic_hash)
    if cached and cached[0] == semantic_digest(raw):
        episode, alternatives, snapshot = cached[1:]
    else:
        episode, alternatives, snapshot = _restore(entry.case, raw)
    return checked_view(entry, state, _payload(entry.case.semantic_hash, episode, alternatives, snapshot))

def dispatch(session, operation, args, request, progress):
    from .desktop_bridge import BridgeError, _keys, _require_json_budget
    from .rehearsal_episode import create_episode, execute_alternatives, restore_episode
    fields = {"createRehearsal": {"caseHash"}, "inspectRehearsal": {"caseHash"},
              "stepRehearsal": {"caseHash", "expectedStateId", "action"},
              "compareRehearsal": {"caseHash", "expectedStateId"}}
    _keys(args, fields[operation])
    if set(args) != fields[operation]:
        raise BridgeError("INVALID_ARGUMENT", "Use the exact current case, state and action fields")
    _require_json_budget(args, 16 * 1024, "REHEARSAL_REQUEST_LIMIT", "Rehearsal action exceeds the size limit")
    request.check()
    entry = session._get_case(args["caseHash"])
    case = entry.case
    cache = getattr(session, "rehearsal_sessions", None)
    if cache is None:
        session.rehearsal_sessions = cache = OrderedDict()
    raw = thaw_json(entry.artifacts).get(ARTIFACT_KEY)
    artifact_hash = semantic_digest(raw)
    cached = cache.get(case.semantic_hash)
    if operation == "createRehearsal":
        progress(.1, "Preparing source-bound post-exposure research state")
        episode, alternatives, snapshot = create_episode(case), [], None
    elif cached and cached[0] == artifact_hash:
        episode, alternatives, snapshot = cached[1:]
    elif raw is not None:
        progress(.1, "Reconstructing saved actions against original source anatomy")
        episode, alternatives, snapshot = _restore(case, raw)
    elif operation == "inspectRehearsal":
        return None
    else:
        raise BridgeError("REHEARSAL_UNAVAILABLE", "Create a research episode for this case first")
    request.check()
    rejected = None
    if operation in {"stepRehearsal", "compareRehearsal"}:
        if args["expectedStateId"] != episode.export()["finalStateId"]:
            raise BridgeError("REHEARSAL_STATE_STALE", "The rehearsal changed; inspect its current state before acting")
    if operation == "stepRehearsal":
        candidate = episode.fork()
        result = candidate.step(args["action"])
        if result["accepted"]:
            episode = candidate
        else:
            rejected = result
    elif operation == "compareRehearsal":
        progress(.2, "Executing two strategies from the same saved state")
        compared = execute_alternatives(episode)
        snapshot = {"snapshotId": compared["branchPointStateId"], "record": compared["snapshot"]}
        alternatives = [{"id": row["id"], "label": row["label"],
                         "episode": restore_episode(case, row["episode"]["savedEpisode"])} for row in compared["branches"]]
    request.check()
    result = _payload(case.semantic_hash, episode, alternatives, snapshot)
    view = thaw_json(entry.artifacts).get("workspace", {}).get("rehearsalView")
    if view is not None:
        try:
            result["savedView"] = checked_view(entry, view, result)
        except (ValueError, KeyError, TypeError):
            # A prior view can become stale after an explicit new episode or branch.
            pass
    if rejected is not None:
        result["rejectedAction"] = rejected
        result["accepted"] = False
        result["reason"] = rejected["reason"]
    _require_json_budget(result, 16 * 1024 * 1024, "REHEARSAL_RESULT_LIMIT", "Rehearsal replay exceeds the display budget")
    saved = _record(episode, alternatives, snapshot)
    _require_json_budget(saved, 2 * 1024 * 1024, "REHEARSAL_ARCHIVE_LIMIT", "Saved rehearsal exceeds the archive budget")
    request.begin_commit()
    artifacts = thaw_json(entry.artifacts)
    artifacts[ARTIFACT_KEY] = saved
    entry.artifacts = freeze_json(artifacts)
    cache[case.semantic_hash] = (semantic_digest(saved), episode, alternatives, snapshot)
    cache.move_to_end(case.semantic_hash)
    while len(cache) > session.max_cases:
        cache.popitem(last=False)
    return result
