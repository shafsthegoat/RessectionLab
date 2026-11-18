"""Incremental native-study preparation shared by desktop and dataset adapters."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time

from . import anatomy_models
from .anatomy_alignment import alignment_pairs, ensure_alignment, input_key, matches_alignment_key
from .patient_anatomy import PatientImagingCollection, digest

RECEIPTS = "anatomy-preparation.json"
SCHEMA = "native-anatomy-preparation-v1"


def _receipts(collection):
    path = collection.path(RECEIPTS)
    if not path.exists():
        return {}
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("Preparation checkpoint exceeds its size budget")
    record = json.loads(path.read_text())
    if record.get("schema") != SCHEMA or record.get("patient") != collection.document["patient_id"]:
        raise ValueError("Preparation checkpoint belongs to a different study")
    return record["stages"]


def _checkpoint(collection, receipts):
    record = {"schema": SCHEMA, "patient": collection.document["patient_id"], "stages": receipts}
    with tempfile.NamedTemporaryFile(dir=collection.root, mode="w", delete=False) as out:
        try:
            json.dump(record, out, allow_nan=False)
            out.flush()
            os.replace(out.name, collection.path(RECEIPTS))
        finally:
            Path(out.name).unlink(missing_ok=True)


@contextmanager
def _exclusive(collection):
    with collection.path(".preparation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Preparation is already running for this study") from exc
        yield


def _model_input(collection, job):
    sid = job.get("series_id")
    source = collection.series(sid) if sid else {}
    inputs = {k: source.get(k) for k in ("id", "modality", "original", "data", "timepoint", "affine")}
    # Native models are independent of display registrations.
    dependencies = {}
    for key, value in job.get("options", {}).items():
        if key.endswith("_layer_id") and value:
            row = collection.layer(value)
            dependencies[key] = {k: row.get(k) for k in ("id", "revision", "data", "affine", "label_value")}
    return digest({"service": SCHEMA, "source": inputs, "model": job.get("model"),
                   "environment": job.get("environment"), "options": job.get("options", {}),
                   "layers": dependencies, "transforms": job.get("transform_dependencies", {})})


def _stage(collection, job, receipts):
    stage = dict(job)
    stage.setdefault("label", stage.get("capability", stage["id"]).replace("_", " ").capitalize())
    stage["seriesId"] = stage.get("series_id")
    if stage.get("model") == "rigid-registration":
        row = collection.series(stage["series_id"])
        parent = collection.series(stage["parent"])
        stage["inputKey"] = input_key(collection, row, parent)
        transform = collection.document["transforms"].get(row["id"], {})
        if (transform.get("status") in {"source-provided", "reviewed"}
                or matches_alignment_key(collection, row, parent, stage["inputKey"])):
            try:
                collection.registration_matrix(row["id"])
            except ValueError:
                pass
            else:
                stage.update(status="cached", reason="Source-bound " + transform["status"] + " alignment")
                return stage
    else:
        stage["inputKey"] = _model_input(collection, stage)
    receipt = receipts.get(stage["id"], {})
    if receipt.get("inputKey") == stage["inputKey"]:
        if receipt["status"] in {"failed", "empty_output"} and stage["status"] in {"ready", "available"}:
            stage.update(status=receipt["status"], reason=receipt["reason"])
        elif receipt["status"] == "cached" and receipt.get("outputs"):
            rows = {r["id"]: r for r in collection.document["layers"]}
            valid = True
            for output in receipt["outputs"]:
                row = rows.get(output["id"])
                if row is None or output["sha256"] != row.get("original_data", row["data"])["sha256"]:
                    valid = False
                    break
                collection.array(row.get("original_data", row["data"]))
            if valid:
                stage.update(status="cached", reason="Prepared layers retained", outputs=receipt["outputs"])
    return stage


def _jobs(collection, options):
    from .anatomy_model_registry import preparation_model_jobs
    jobs = list(preparation_model_jobs(collection, **options))
    if collection.document.get("reference_series"):
        for row, parent in alignment_pairs(collection):
            jobs.append({"id": "align-" + row["id"], "capability": "registration",
                         "model": "rigid-registration", "label": "Align " + row.get("label", row["modality"]),
                         "series_id": row["id"], "parent": parent["id"], "status": "ready",
                         "reason": "Estimate rigid display alignment; quantitative use requires review",
                         "depends_on": [] if parent["id"] == collection.document["reference_series"]
                         else ["align-" + parent["id"]]})
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate preparation stage identity")
    return jobs


def _plan(collection, options, receipts):
    stages = [_stage(collection, job, receipts) for job in _jobs(collection, options)]
    return {"id": digest({"snapshot": collection.identity(), "options": options,
                          "stages": [(s["id"], s["inputKey"], s["status"]) for s in stages]}),
            "stages": stages}


def preparation_plan(collection, *, indication=None, include_dural_proxy=False, full_head_confirmed=False):
    options = dict(indication=indication, include_dural_proxy=include_dural_proxy,
                   full_head_confirmed=full_head_confirmed)
    return _plan(collection, options, _receipts(collection))


def _run_stage(candidate, stage, cancelled, progress):
    if stage["model"] == "rigid-registration":
        ensure_alignment(candidate, series_id=stage["series_id"], cancelled=cancelled, progress=progress)
        return "cached", "Source-bound display alignment prepared", []
    record = anatomy_models.reconstruct(candidate, stage["series_id"], stage["model"],
                                        cancelled=cancelled, **stage.get("options", {}))
    before = {row["id"] for row in candidate.document["layers"]}
    result = anatomy_models.import_proposal(candidate, stage["series_id"], record)
    outputs = []
    for row in candidate.document["layers"]:
        if row["id"] not in before or row["id"] == result or (
                row.get("provenance", {}).get("output_sha256") == record.get("output_sha256")
                and row.get("source_series") == stage["series_id"] and record.get("output_sha256")):
            outputs.append({"id": row["id"], "sha256": row.get("original_data", row["data"])["sha256"]})
            if row["id"] not in before:
                row["preparation_input"] = stage["inputKey"]
    if not outputs or record.get("foreground_voxels") == 0 or record.get("availability") == "empty_output":
        return "empty_output", "Model returned no labeled anatomy in its output field", outputs
    if len(outputs) == 1 and not candidate.mask(candidate.layer(outputs[0]["id"])).any():
        return "empty_output", "Layer is empty; anatomy remains unassessed", outputs
    return "cached", "Prepared " + str(len(outputs)) + " layer(s) for review", outputs


def prepare_anatomy(collection, *, on_stage=None, cancelled=None, retry_failed=False,
                    indication=None, include_dural_proxy=False, full_head_confirmed=False,
                    progress=None, plan_id=None):
    """Commit completed stages independently, keeping prior revisions on interruption."""
    cancelled = cancelled or (lambda: False)
    progress = progress or (lambda *args: None)
    options = dict(indication=indication, include_dural_proxy=include_dural_proxy,
                   full_head_confirmed=full_head_confirmed)
    outcomes = []
    with _exclusive(collection):
        current = PatientImagingCollection(collection.root)
        if current.identity() != collection.identity():
            raise ValueError("Patient changed; refresh preparation before running")
        receipts = _receipts(collection)
        plan = _plan(collection, options, receipts)
        if plan_id is not None and plan["id"] != plan_id:
            raise ValueError("Preparation inputs changed; refresh the proposed stages")
        completed = {}
        for index, initial in enumerate(plan["stages"]):
            if cancelled():
                raise InterruptedError("Preparation cancelled; completed stages are saved")
            if initial.get("depends_on") or initial.get("model") == "rigid-registration":
                jobs = {j["id"]: j for j in _jobs(collection, options)}
                stage = _stage(collection, jobs.get(initial["id"], initial), receipts)
            else:
                stage = initial
            state = stage["status"]
            reason = stage.get("reason", "")
            if state == "ready" or (retry_failed and state in {"failed", "empty_output"}):
                if any(completed.get(dep) not in {"cached", "available"} for dep in stage.get("depends_on", [])):
                    state, reason = "blocked", "A required preparation stage did not produce usable anatomy"
                else:
                    started = time.monotonic()
                    progress(index / len(plan["stages"]), stage["label"])
                    candidate = PatientImagingCollection(collection.root, collection.document)
                    expected = collection.identity()
                    try:
                        state, reason, outputs = _run_stage(candidate, stage, cancelled,
                            lambda f, m: progress((index + f) / len(plan["stages"]), m))
                    except (OSError, ValueError, RuntimeError, MemoryError) as exc:
                        if cancelled() or isinstance(exc, InterruptedError):
                            raise InterruptedError("Preparation cancelled; completed stages are saved") from exc
                        state, reason, outputs = "failed", str(exc)[:1500], []
                    if cancelled():
                        raise InterruptedError("Preparation cancelled before stage admission")
                    if state != "failed" and candidate.identity() != expected:
                        from .anatomy_workspace import read_view, save_view
                        view = read_view(collection)
                        candidate.save("Prepare anatomy · " + stage["label"], expected=expected)
                        collection.document = candidate.document
                        collection._base_snapshot = candidate.document["snapshot_id"]
                        if view:
                            save_view(collection, {**view, "snapshotId": collection.identity()})
                    elif PatientImagingCollection(collection.root).identity() != expected:
                        raise ValueError("Patient changed during preparation; refresh the study")
                    receipts[stage["id"]] = {"status": state, "reason": reason, "inputKey": stage["inputKey"],
                                               "outputs": outputs, "elapsedSeconds": time.monotonic() - started}
                    _checkpoint(collection, receipts)
            completed[stage["id"]] = state
            outcome = {"id": stage["id"], "status": state, "reason": reason}
            outcomes.append(outcome)
            if on_stage:
                on_stage(outcome)
        return {"document": collection.document, "plan": _plan(collection, options, receipts), "outcomes": outcomes}
