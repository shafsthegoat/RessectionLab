"""Incremental native-study preparation shared by desktop and dataset adapters."""
from contextlib import contextmanager
import fcntl
import json
import math
import os
import re
from pathlib import Path
import tempfile
import time

from . import anatomy_models
from .anatomy_alignment import alignment_pairs, ensure_alignment, input_key, matches_alignment_key
from .patient_anatomy import PatientImagingCollection, digest
from .anatomy_synthstrip_fallback import CPU_MODEL, CPU_RETRY_STATUS, offer_cpu_retry

RECEIPTS = "anatomy-preparation.json"
SCHEMA = "native-anatomy-preparation-v1"


MAX_CHECKPOINT_BYTES = 4 * 1024**2
MAX_STAGE_REASON = 1500
CHECKPOINT_STATUSES = {"cached", "failed", "empty_output", "interrupted"}
RETRY_STATUSES = {"failed", "empty_output"}
RUNNABLE_STATUSES = {"ready", "interrupted"}
SATISFIED_STATUSES = {"cached", "available"}


def _receipt_key(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _unique_object(pairs):
    values = dict(pairs)
    if len(values) != len(pairs):
        raise ValueError("Duplicate preparation checkpoint field")
    return values


def _validate_receipts(record, patient):
    if not isinstance(record, dict) or set(record) != {"schema", "patient", "stages"}:
        raise ValueError("Invalid preparation checkpoint")
    if record["schema"] != SCHEMA or record["patient"] != patient:
        raise ValueError("Preparation checkpoint belongs to a different study")
    receipts = record["stages"]
    if not isinstance(receipts, dict):
        raise ValueError("Invalid preparation checkpoint stage list")
    for stage_id, receipt in receipts.items():
        if not isinstance(stage_id, str) or not stage_id.strip() or not isinstance(receipt, dict) or set(receipt) != {
            "status", "reason", "inputKey", "outputs", "elapsedSeconds",
        }:
            raise ValueError("Invalid preparation checkpoint stage")
        status, reason, elapsed = receipt["status"], receipt["reason"], receipt["elapsedSeconds"]
        if (not isinstance(status, str) or status not in CHECKPOINT_STATUSES
                or not isinstance(reason, str) or len(reason) > MAX_STAGE_REASON
                or not _receipt_key(receipt["inputKey"]) or type(elapsed) not in {int, float}
                or elapsed < 0 or (isinstance(elapsed, float) and not math.isfinite(elapsed))):
            raise ValueError("Invalid preparation checkpoint stage status or identity")
        outputs = receipt["outputs"]
        if not isinstance(outputs, list) or (status in {"failed", "interrupted"} and outputs):
            raise ValueError("Invalid preparation checkpoint outputs")
        seen = set()
        for output in outputs:
            if not isinstance(output, dict) or set(output) != {"id", "sha256"}:
                raise ValueError("Invalid preparation checkpoint output")
            layer_id = output["id"]
            if (not isinstance(layer_id, str) or not layer_id.strip() or layer_id in seen
                    or not _receipt_key(output["sha256"])):
                raise ValueError("Invalid or duplicate preparation checkpoint output")
            seen.add(layer_id)
    return receipts


def _receipts(collection):
    try:
        with collection.path(RECEIPTS).open("rb") as checkpoint:
            payload = checkpoint.read(MAX_CHECKPOINT_BYTES + 1)
    except FileNotFoundError:
        return {}
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Preparation checkpoint exceeds its size budget")
    try:
        record = json.loads(payload, object_pairs_hook=_unique_object)
    except (ValueError, RecursionError) as error:
        raise ValueError("Invalid preparation checkpoint JSON") from error
    return _validate_receipts(record, collection.document["patient_id"])


def _checkpoint(collection, receipts):
    patient = collection.document["patient_id"]
    record = {"schema": SCHEMA, "patient": patient, "stages": receipts}
    _validate_receipts(record, patient)
    payload = json.dumps(record, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_CHECKPOINT_BYTES:
        raise ValueError("Preparation checkpoint exceeds its size budget")
    with tempfile.NamedTemporaryFile(dir=collection.root, mode="wb", delete=False) as out:
        temporary = Path(out.name)
        try:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
            os.replace(temporary, collection.path(RECEIPTS))
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive(collection):
    with collection.path(".preparation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Preparation is already running for this study") from exc
        yield


def _environment_identity(environment):
    if not isinstance(environment, dict):
        return environment
    configuration = {key: value for key, value in environment.items() if key != "file_paths"}
    if isinstance(configuration.get("files"), dict):
        from .anatomy_model_identity import preparation_asset_identities

        configuration["files"] = preparation_asset_identities(configuration["files"])
    return configuration


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
                   "environment": _environment_identity(job.get("environment")), "options": job.get("options", {}),
                   "layers": dependencies, "transforms": job.get("transform_dependencies", {})})


def _stage(collection, job, receipts):
    stage = dict(job)
    stage.setdefault("label", stage.get("capability", stage["id"]).replace("_", " ").capitalize())
    stage["seriesId"] = stage.get("series_id")
    environment = stage.get("environment")
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
        if "environment" in stage:
            stage["environment"] = _environment_identity(environment)
    receipt = receipts.get(stage["id"], {})
    matches = receipt.get("inputKey") == stage["inputKey"]
    if not matches and receipt and stage.get("model") != "rigid-registration":
        from .anatomy_model_cache import legacy_model_environments

        output_ids = {output["id"] for output in receipt.get("outputs", [])}
        lineage = (row for row in collection.document["layers"] if row["id"] in output_ids
                   and row.get("source_series") == stage["seriesId"]
                   and row.get("preparation_input") == receipt["inputKey"])
        matches = any(receipt["inputKey"] == _model_input(collection, {**stage, "environment": legacy})
                      for legacy in legacy_model_environments(stage.get("model"), environment, lineage))
    if matches:
        if receipt["status"] in RETRY_STATUSES and stage["status"] in {"ready", "available"}:
            stage.update(status=receipt["status"], reason=receipt["reason"])
        elif receipt["status"] == "interrupted" and stage["status"] == "ready":
            stage.update(status="interrupted", reason=receipt["reason"])
        elif receipt["status"] == "cached" and receipt["outputs"]:
            rows = {r["id"]: r for r in collection.document["layers"]}
            try:
                for output in receipt["outputs"]:
                    row = rows[output["id"]]
                    stored = row.get("original_data", row["data"])
                    if output["sha256"] != stored["sha256"]:
                        raise ValueError("Output identity changed")
                    collection.array(stored)
            except (KeyError, OSError, ValueError):
                if stage["status"] == "ready":
                    stage["reason"] = "Previously prepared output is missing or changed; recompute from the source"
            else:
                stage.update(status="cached", reason="Prepared layers retained", outputs=receipt["outputs"])
    return stage


def _jobs(collection, options):
    from .anatomy_model_registry import preparation_model_jobs
    from .anatomy_tumor_coverage import retained_crop_jobs

    jobs = [*retained_crop_jobs(collection), *preparation_model_jobs(collection, **options)]
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


def _source_jobs(collection, options, receipts):
    """Keep the original single-source stage ID bound across incremental imports."""
    jobs = _jobs(collection, options)
    renamed = {}
    for base in ("ct-skull", "vascular", "siam-tissues"):
        receipt = receipts.get(base)
        if not receipt:
            continue
        family = [job for job in jobs if job["id"] == base or job["id"].startswith(base + ":")]
        if len(family) < 2:
            continue
        output_ids = {item["id"] for item in receipt["outputs"]}
        owners = {row["source_series"] for row in collection.document["layers"] if row["id"] in output_ids}
        if not owners:
            # Failed and interrupted attempts have no output layer lineage.
            owners = {job["series_id"] for job in family if _model_input(collection, job) == receipt["inputKey"]}
        previous = next((job for job in family if owners == {job["series_id"]}), None)
        if previous is None or previous["id"] == base:
            continue
        preferred = next(job for job in family if job["id"] == base)
        renamed[base] = base + ":" + preferred["series_id"]
        renamed[previous["id"]] = base
    if renamed:
        for job in jobs:
            job["id"] = renamed.get(job["id"], job["id"])
            job["depends_on"] = [renamed.get(dep, dep) for dep in job.get("depends_on", [])]
    return jobs


def _plan(collection, options, receipts):
    stages = []
    for job in _source_jobs(collection, options, receipts):
        stage = _stage(collection, job, receipts)
        stages.append(stage)
        if fallback := offer_cpu_retry(collection, stage, receipts, _stage):
            stages.append(fallback)
    by_id = {stage["id"]: stage for stage in stages}
    for stage in stages:
        if stage["status"] not in RUNNABLE_STATUSES:
            continue
        for dependency in stage.get("depends_on", []):
            parent = by_id[dependency]
            if parent["status"] not in RUNNABLE_STATUSES | SATISFIED_STATUSES:
                stage.update(status="blocked", reason="Requires usable output from " + parent["label"])
                break
    return {"id": digest({"snapshot": collection.identity(), "options": options,
                          "stages": [(s["id"], s["inputKey"], s["status"]) for s in stages]}),
            "stages": stages}


def preparation_plan(collection, *, indication=None, include_dural_proxy=False, full_head_confirmed=False):
    options = dict(indication=indication, include_dural_proxy=include_dural_proxy,
                   full_head_confirmed=full_head_confirmed)
    return _plan(collection, options, _receipts(collection))


def _run_stage(candidate, stage, cancelled, progress):
    if stage["model"] == "restore-tumor-crop-coverage":
        from .anatomy_tumor_coverage import restore_retained_crop

        output = restore_retained_crop(candidate, stage["options"]["proposal_layer_id"])
        return "cached", "Recorded the native model crop from retained evidence", [output]
    if stage["model"] == "rigid-registration":
        ensure_alignment(candidate, series_id=stage["series_id"], cancelled=cancelled, progress=progress)
        return "cached", "Source-bound display alignment prepared", []
    if stage["model"] == CPU_MODEL:
        from .anatomy_synthstrip_fallback import reconstruct_cpu

        record = reconstruct_cpu(candidate, stage, cancelled=cancelled)
    else:
        record = anatomy_models.reconstruct(candidate, stage["series_id"], stage["model"],
                                            cancelled=cancelled, **stage.get("options", {}))
    before = {row["id"] for row in candidate.document["layers"]}
    result = anatomy_models.import_proposal(candidate, stage["series_id"], record)
    outputs = []
    extras = {(item["kind"], item["output_sha256"]) for item in record.get("extra_outputs", [])}
    for row in candidate.document["layers"]:
        provenance = row.get("provenance", {})
        checksum = provenance.get("output_sha256")
        same_model = row["source_series"] == stage["series_id"] and provenance.get("model") == record["model"]
        # Categorical adapters return a shared array. Binary adapters return
        # the layer ID and may also admit separately hashed companion masks.
        categorical = isinstance(result, dict) and checksum == record["output_sha256"]
        if (row["id"] not in before or row["id"] == result
                or (same_model and (categorical or (row["kind"], checksum) in extras))):
            outputs.append({"id": row["id"], "sha256": row.get("original_data", row["data"])["sha256"]})
            if row["id"] not in before:
                row["preparation_input"] = stage["inputKey"]
                if stage["model"] == CPU_MODEL:
                    row["label"] = "Brain envelope · SynthStrip CPU"
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
                jobs = {j["id"]: j for j in _source_jobs(collection, options, receipts)}
                stage = _stage(collection, jobs.get(initial["id"], initial), receipts)
            else:
                stage = initial
            state = stage["status"]
            reason = stage.get("reason") or ""
            published = False
            can_run = stage.get("retryable") is not False and (
                state in RUNNABLE_STATUSES or (retry_failed and state in RETRY_STATUSES | {CPU_RETRY_STATUS})
            )
            if can_run:
                if any(completed.get(dep) not in SATISFIED_STATUSES for dep in stage.get("depends_on", [])):
                    state, reason = "blocked", "A required preparation stage did not produce usable anatomy"
                else:
                    started = time.monotonic()
                    progress(index / len(plan["stages"]), stage["label"])
                    candidate = PatientImagingCollection(collection.root, collection.document)
                    expected = collection.identity()
                    interrupted = None
                    if stage["model"] == CPU_MODEL:
                        # An abandoned or interrupted supervisor is also an
                        # attempt. Persist the one-use budget before launch.
                        receipts[stage["id"]] = {
                            "status": "failed", "reason": "The one CPU retry was started",
                            "inputKey": stage["inputKey"], "outputs": [], "elapsedSeconds": 0,
                        }
                        _checkpoint(collection, receipts)
                    try:
                        state, reason, outputs = _run_stage(candidate, stage, cancelled,
                            lambda f, m: progress((index + f) / len(plan["stages"]), m))
                    except InterruptedError as error:
                        interrupted = error
                    except Exception as error:
                        # The adapter is isolated from revision and checkpoint commits.
                        # A failed optional worker cannot suppress independent stages.
                        state, reason, outputs = "failed", f"{type(error).__name__}: {error}"[:MAX_STAGE_REASON], []
                    if interrupted is not None or cancelled():
                        receipts[stage["id"]] = {
                            "status": "interrupted", "reason": "Previous attempt interrupted; resume to finish this stage",
                            "inputKey": stage["inputKey"], "outputs": [], "elapsedSeconds": time.monotonic() - started,
                        }
                        _checkpoint(collection, receipts)
                        raise InterruptedError("Preparation cancelled; completed stages are saved") from interrupted
                    if state != "failed" and candidate.identity() != expected:
                        from .anatomy_workspace import read_view, save_view
                        view = read_view(collection)
                        candidate.save("Prepare anatomy · " + stage["label"], expected=expected)
                        collection.document = candidate.document
                        collection._base_snapshot = candidate.document["snapshot_id"]
                        published = True
                        if view:
                            save_view(collection, {**view, "snapshotId": collection.identity()})
                    elif PatientImagingCollection(collection.root).identity() != expected:
                        raise ValueError("Patient changed during preparation; refresh the study")
                    receipts[stage["id"]] = {"status": state, "reason": reason, "inputKey": stage["inputKey"],
                                               "outputs": outputs, "elapsedSeconds": time.monotonic() - started}
                    _checkpoint(collection, receipts)
            completed[stage["id"]] = state
            outcome_status = "complete" if state == "cached" and published else state
            outcome = {"id": stage["id"], "label": stage["label"], "status": outcome_status, "reason": reason}
            if "device" in stage:
                outcome["device"] = stage["device"]
            outcomes.append(outcome)
            if on_stage:
                on_stage(outcome)
        return {"document": collection.document, "plan": _plan(collection, options, receipts), "outcomes": outcomes}
