#!/usr/bin/env python3
"""PAT25-only initial access comparison; execution requires a separate release."""
from __future__ import annotations

import time
_BOOT_STARTED = time.perf_counter()

import argparse
import os
import signal
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys

# The executable run path enters this stdlib-only watchdog before importing any
# project/scientific modules. One group owns the orchestrator and every child.
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}


def _small_sha(path):
    if Path(path).stat().st_size>16*1024**2:
        raise ValueError("Release/declaration metadata exceeds its input limit")
    raw=Path(path).read_bytes()
    if len(raw)>16*1024**2: raise ValueError("Release/declaration metadata exceeds its input limit")
    return hashlib.sha256(raw).hexdigest()


def _group_alive(pgid):
    try: os.killpg(pgid,0); return True
    except ProcessLookupError: return False


def _signal_group(pgid, signum):
    try: os.killpg(pgid,signum)
    except ProcessLookupError: pass


def _group_rss(pgid,timeout):
    measurement=subprocess.run(["ps","-axo","pgid=,rss="],capture_output=True,text=True,
                               check=True,timeout=timeout)
    return sum(int(rss)*1024 for group,rss in (line.split() for line in measurement.stdout.splitlines())
               if int(group)==pgid)


def _outer_main(argv):
    parser=argparse.ArgumentParser()
    for name in ("manifest","release","output"): parser.add_argument("--"+name,type=Path,required=True)
    args=parser.parse_args(argv)
    output=args.output.resolve(); receipt_path=output.with_name(output.name+".outer.json")
    log_path=output.with_name(output.name+".outer.log")
    if output.exists() or receipt_path.exists() or log_path.exists():
        raise ValueError("Preserve the existing one-attempt output/outer receipt")
    output.parent.mkdir(parents=True,exist_ok=True)
    receipt={"status":"failed","whole_attempt_seconds_limit":180.,"soft_stop_seconds":174.,
        "hard_cleanup_seconds":178.,"max_group_rss_bytes":6*1024**3,"automatic_retry":False,
        "clock_scope":"first stdlib timestamp through final receipt; interpreter startup before that timestamp and OS scheduling are not bounded"}
    with receipt_path.open("x") as stream: json.dump(receipt,stream)
    process=None; reason=None; maximum=0; samples=0
    try:
        manifest_sha,release_sha=_small_sha(args.manifest),_small_sha(args.release)
        command=[sys.executable,str(Path(__file__).resolve()),"--orchestrator",*argv,
            "--expected-manifest-sha256",manifest_sha,"--expected-release-sha256",release_sha,
            "--outer-parent-pid",str(os.getpid())]
        receipt.update(manifest_sha256=manifest_sha,release_sha256=release_sha,command=command)
        with log_path.open("x") as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,**THREAD_ENV,"PYTHONDONTWRITEBYTECODE":"1"},start_new_session=True)
            receipt["owned_process_group"]=process.pid
            while process.poll() is None:
                elapsed=time.perf_counter()-_BOOT_STARTED
                if elapsed>=174.: reason="outer_whole_attempt_wall_budget"; break
                rss=_group_rss(process.pid,min(.5,174.-elapsed)); samples+=1; maximum=max(maximum,rss)
                if rss>6*1024**3: reason="outer_combined_group_rss_budget"; break
                time.sleep(min(.1,max(0.,174.-(time.perf_counter()-_BOOT_STARTED))))
            if reason:
                _signal_group(process.pid,signal.SIGTERM)
                while _group_alive(process.pid) and time.perf_counter()-_BOOT_STARTED<178.:
                    process.poll(); time.sleep(.02)
            elif _group_alive(process.pid):
                reason="orchestrator_exited_with_surviving_descendants"
    except BaseException as error:
        reason=f"outer_failure:{type(error).__name__}:{error}"
    finally:
        if process is not None:
            _signal_group(process.pid,signal.SIGKILL)  # unconditional known-group cleanup
            try: process.wait(timeout=max(.001,179.-(time.perf_counter()-_BOOT_STARTED)))
            except subprocess.TimeoutExpired: reason=reason or "orchestrator_cleanup_timeout"
            while _group_alive(process.pid) and time.perf_counter()-_BOOT_STARTED<179.:
                time.sleep(.01)
            receipt.update(returncode=process.poll(),group_gone=not _group_alive(process.pid))
            if not receipt["group_gone"]: reason=reason or "process_group_not_reaped"
        else: receipt.update(returncode=None,group_gone=True)
    try:
        acceptance=json.loads((output/"acceptance.json").read_text())
        receipt["acceptance_sha256"]=_small_sha(output/"acceptance.json")
        if (reason is None and receipt["returncode"]==0 and receipt["group_gone"]
                and acceptance.get("status")=="complete" and time.perf_counter()-_BOOT_STARTED<180.):
            receipt["status"]="complete"
        else:
            reason=reason or "orchestrator_or_acceptance_did_not_complete_in_budget"
    except (OSError,ValueError) as error:
        reason=reason or f"missing_or_invalid_acceptance:{error}"
    receipt.update(reason=reason,sampled_peak_group_rss_bytes=maximum,rss_samples=samples,
                   elapsed_seconds=time.perf_counter()-_BOOT_STARTED)
    receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    if time.perf_counter()-_BOOT_STARTED>=180.:
        receipt.update(status="failed",reason="outer_receipt_publication_overrun",elapsed_seconds=time.perf_counter()-_BOOT_STARTED)
        receipt_path.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n")
    return 0 if receipt["status"]=="complete" else 1


if __name__=="__main__" and not any(flag in sys.argv for flag in
        ("--declare","--preflight","--worker","--orchestrator","--help","-h")):
    raise SystemExit(_outer_main(sys.argv[1:]))


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import diagnose_pat25_access as legacy
import diagnose_training_observation_coverage as visibility
from preflight_real_spatial_policy import peak_rss_bytes, sha256, write_json

prep = legacy.prep
VERSION = "pat25-screened-ingress-initial-comparison-v1"
SUBJECT = "sub-PAT25"
SETTINGS = {"whole_worker_envelope_seconds": 180., "cooperative_seconds": 170.,
    "max_wall_seconds": 174., "max_rss_bytes": 6 * 1024**3, "cpu_threads": 1,
    "max_static_screens": 468, "max_initial_previews_per_inventory": 78, "max_total_previews": 156,
    "max_initial_inventories": 2, "automatic_retry": False}
SCRIPT_PATHS = ("scripts/diagnose_pat25_ingress_access.py", "scripts/diagnose_pat25_access.py",
    "scripts/diagnose_training_observation_coverage.py", "scripts/prepare_real_training_cases.py",
    "scripts/preflight_real_spatial_policy.py")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def source_inventory():
    paths = [ROOT / name for name in SCRIPT_PATHS] + list((ROOT / "src/resectionlab").rglob("*.py"))
    if any(not path.resolve().is_relative_to(ROOT) for path in paths):
        raise ValueError("Source closure cannot resolve outside its archive")
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(paths)}


def metadata_inventory():
    direct = f"{legacy.SAVED}/{SUBJECT}/preparation.json"
    saved = direct if (ROOT / direct).is_file() else f"{legacy.SAVED}/completed-run.tar.gz"
    names = (*prep.RECEIPT_PATHS, prep.COMMON_PATH, prep.COHORT_PATH, saved)
    return {name: sha256(ROOT / name) for name in names}


def declaration():
    """Read/hash code and saved metadata only. Never decode or hash a bundle."""
    prior = legacy.declaration()
    return {"version": VERSION, "subject": SUBJECT, "role": "TRAIN",
        "declared_at": prior["declared_at"], "settings": SETTINGS,
        "source_sha256": source_inventory(), "metadata_sha256": metadata_inventory(),
        "legacy_input": prior, "authorized_execution": False,
        "selection": "six existing exits; any initial ingress pose; distance/axis/outward_sign",
        "comparison_order": ["original", "screened_selected"],
        "prohibited": ["committed_cut", "search", "policy_forward", "checkpoint", "optimizer",
                       "fallback_after_selected_motion_failure", "other_patient_decode", "retry"],
        "scope": "initial geometry preparation only; no resection or clinical validation"}


def validate(record):
    # Gate roles before reading even registry/saved metadata.
    if (record.get("version") != VERSION or record.get("subject") != SUBJECT
            or record.get("role") != "TRAIN" or record.get("settings") != SETTINGS):
        raise ValueError("Only the fixed PAT25 TRAIN ingress diagnostic is permitted")
    expected = declaration()
    expected["declared_at"] = record.get("declared_at")
    expected["legacy_input"]["declared_at"] = record.get("declared_at")
    if canonical(expected) != canonical(record):
        raise ValueError("Prospective source, registry, historical inputs or diagnostic scope changed")
    return legacy.validate(record["legacy_input"])[0]


def _local(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("A bound relative path cannot escape its declared root")
    return path


def preflight(manifest_path, release_path, output, *, check=lambda: None):
    """Verify metadata, immutable Git closure and encoded bundle bytes; no decode."""
    manifest_path, release_path, output = map(Path, (manifest_path, release_path, output))
    record = json.loads(manifest_path.read_text())
    if record.get("subject") != SUBJECT or record.get("role") != "TRAIN":
        raise ValueError("Only PAT25 TRAIN is permitted before source access")
    release = json.loads(release_path.read_text())
    if (release.get("version") != VERSION + "-release" or release.get("authorized") is not True
            or release.get("subject") != SUBJECT or release.get("role") != "TRAIN"
            or release.get("phase") != "initial_inventory_diagnostic"
            or release.get("declaration_sha256") != sha256(manifest_path)
            or Path(release.get("manifest_path", "")).resolve() != manifest_path.resolve()
            or Path(release.get("archive_root", "")).resolve() != ROOT
            or Path(release.get("output", "")).resolve() != output.resolve()):
        raise ValueError("An exact, separately authorized archive/output release is required")
    repository = Path(release["repository_root"]).resolve()
    commit = release.get("source_commit", "")
    if repository == ROOT or len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("Execution requires a separate immutable full-commit archive")
    check()
    saved = validate(record)
    check()
    closure = {**record["source_sha256"], **record["metadata_sha256"]}
    for name, digest in closure.items():
        check()
        path = _local(ROOT, name)
        raw = subprocess.run(["git", "-C", str(repository), "show", f"{commit}:{name}"],
                             check=True, capture_output=True, timeout=10).stdout
        if hashlib.sha256(raw).hexdigest() != digest or sha256(path) != digest:
            raise ValueError("Archive/source metadata differs from the exact committed Git closure")
    for module in (legacy, visibility, prep):
        if Path(module.__file__).resolve() != ROOT / "scripts" / Path(module.__file__).name:
            raise ValueError("A runtime helper was imported from another checkout")
    for name, module in tuple(sys.modules.items()):
        if name == "resectionlab" or name.startswith("resectionlab."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(ROOT / "src/resectionlab"):
                raise ValueError("Scientific imports must resolve inside the frozen source archive")
    check()
    member = record["legacy_input"]["member"]
    bundle = _local(ROOT, member["case_bundle"])
    if sha256(bundle) != member["case_bundle_sha256"]:
        raise ValueError("PAT25 encoded bundle differs before decoding")
    bound = {str(_local(ROOT, name)): digest for name, digest in closure.items()}
    bound.update({str(bundle): member["case_bundle_sha256"], str(manifest_path.resolve()): sha256(manifest_path),
                  str(release_path.resolve()): sha256(release_path)})
    check()
    return record, saved, bound, release


def unchanged(bound):
    try:
        return all(sha256(Path(path)) == digest for path, digest in bound.items())
    except OSError:
        return False


class InitialOnlyTrace(legacy.PreviewTrace):
    """Reuse exact preview-return diagnostics with a two-phase whole-worker gate."""
    def __init__(self, check=lambda: None):
        super().__init__(check)
        self.phase = None
        self.total = 0
        self.phases = []
        self.violation = None

    def begin(self, phase):
        if self.violation is not None: raise RuntimeError("Sticky observer violation prevents another phase")
        if phase not in ("original", "screened_selected") or phase in self.phases or self.phase is not None:
            raise ValueError("Each of the two initial inventories can run only once")
        self.phase = phase; self.phases.append(phase)
        self.rows = []; self.preview_calls = 0

    def end(self):
        self.phase = None
        if self.violation is not None: raise RuntimeError("Sticky observer violation prevents phase completion")

    def assert_healthy(self):
        if sys.getprofile()!=self._observe and self.violation is None:
            self.violation="Profile observer is no longer installed"
        if self.violation is not None:
            raise RuntimeError("Initial-only observer is disabled or has a sticky violation: "+str(self.violation))

    def __exit__(self, *args):
        healthy=sys.getprofile()==self._observe
        if not healthy and self.violation is None:
            self.violation="Profile observer was disabled before context exit"
        super().__exit__(*args)
        if args[0] is None and self.violation is not None:
            raise RuntimeError("Initial-only observer lost authority: "+str(self.violation))

    def _observe(self, frame, event, returned):
        try: self._observe_checked(frame,event,returned)
        except BaseException as error:
            self.violation=f"{type(error).__name__}: {error}"
            raise

    def _observe_checked(self, frame, event, returned):
        if event == "call":
            module, name = frame.f_globals.get("__name__", ""), frame.f_code.co_name
            if ((module == "resectionlab.native_spatial_task" and name == "step")
                    or (module == "resectionlab.native_resection" and name in {"commit_preview", "execute_stroke"})
                    or (module == "resectionlab.spatial_policy" and name in
                        {"__init__", "forward", "gradient_step", "reinforce_loss", "behavior_cloning_loss"})
                    or (module == "resectionlab.observed_search" and name != "<module>")
                    or (module.startswith("torch.optim.") and name in {"__init__", "step"})):
                raise RuntimeError("Committed motion, search, policy and optimization are prohibited")
            if frame.f_code is self.preview_code:
                self.check()
                if self.phase is None or self.total >= SETTINGS["max_total_previews"]:
                    raise RuntimeError("A native preview is outside its declared phase/count budget")
                self.total += 1
        super()._observe(frame, event, returned)


def actor_visibility(task, observation, inventory):
    """Legacy actor view only: exact target/cavity counts plus shaft centerlines."""
    import numpy as np
    tool_map = {tool.tool_id: tool for tool in task.case.tools}
    rows = []
    for row in inventory["emitted"]:
        entry, tip = np.asarray(row["entry_mm"]), np.asarray(row["tip_mm"])
        axis = (tip-entry)/np.linalg.norm(tip-entry)
        tool = tool_map[row["tool_id"]]
        spans = {"entry_to_tip": (entry, tip),
            "approach_shaft_centerline": (entry-tool.working_length_mm*axis, entry-tool.tip_length_mm*axis),
            "deepest_shaft_centerline": (tip-tool.working_length_mm*axis, tip-tool.tip_length_mm*axis)}
        rows.append({"action_id": row["action_id"], "accepted": row["feasible"],
            "segments": {name: visibility.segment_visibility(a,b,observation.affine_ras_mm,
                observation.image_channels.shape[1:],observation.coverage) for name,(a,b) in spans.items()}})
    return {"view": visibility.view_description(observation), "actions": rows,
        "cavity_source_cells": int(task._engine.removed_mask.sum()),
        "cavity_visible_cells": int(np.count_nonzero(observation.image_channels[3])),
        "cavity_channel_available": bool(observation.channel_available[3]),
        "scope": "unchanged actor crop; shaft centerline visibility is not complete-tool clearance"}


def inspect_initial(factory, exit_record, phase, trace, check):
    from resectionlab.spatial_policy_diagnostics import spatial_coverage, runtime_proposal_coverage
    started = time.perf_counter(); task = None
    trace.assert_healthy()
    trace.begin(phase)
    try:
        task = factory()
        check()
        invariants = visibility.task_invariants(task)
        inventory, observation = task.candidate_inventory(), task.observation()
        if (not inventory["complete"] or not inventory["ledger_complete"]
                or inventory["declared_slots"] != 78 or inventory["omitted_count"]
                or trace.preview_calls != len(trace.rows) or len(trace.rows) != inventory["emitted_count"]):
            raise RuntimeError("A complete unchanged 78-slot initial inventory is required")
        for ray, returned in zip(inventory["emitted"], trace.rows, strict=True):
            if any(ray[key] != returned[key] for key in ("tool_id", "entry_mm", "tip_mm", "feasible", "reason")):
                raise RuntimeError("Preview return trace differs from the initial inventory")
        native = task.case._native_affine_ras_mm
        result = {"status": "complete", "exit": exit_record, "initial_inventory": inventory,
            "trace": list(trace.rows), "preview_calls": trace.preview_calls,
            "invariants": invariants, "decision_model_hash": task.decision_model_hash,
            "reward": asdict(task.reward_spec), "horizon": task.max_steps,
            "actor_coverage": spatial_coverage(observation, source_shape=task.case.observed_support.shape,
                source_affine=task.case.affine_ras_mm, nominal_target=task.case.nominal_target, native_affine=native),
            "proposal_coverage": runtime_proposal_coverage(task.case,inventory,observation,native_affine=native),
            "actor_visibility": actor_visibility(task,observation,inventory),
            "elapsed_seconds": time.perf_counter()-started}
        result["invariants_after"] = visibility.task_invariants(task)
        if invariants != result["invariants_after"]:
            raise RuntimeError("Diagnostic reporting changed initial geometry or observation")
        check()
        trace.assert_healthy()
        return task, result
    except BaseException as error:
        error.diagnostic = {"status": "unassessed", "exit": exit_record,
            "error": f"{type(error).__name__}: {error}", "partial_preview_trace": list(trace.rows),
            "started_preview_calls": trace.preview_calls, "complete": False,
            "elapsed_seconds": time.perf_counter()-started}
        raise
    finally:
        # Keep the original interruption and its partial trace. A callback
        # exception already poisons the observer; end must not replace it.
        if sys.exc_info()[0] is not None:
            trace.phase=None
        else:
            trace.end()


def prepare_candidates(source, exits, check):
    """Replace only access on the existing source; never instantiate six tasks."""
    from resectionlab.geometry import AccessWindow
    from resectionlab.native_ingress import IngressCandidate
    from resectionlab.native_resection import NativeResectionEngine
    candidates, sources = [], {}
    for exit_record in exits:
        check()
        key = exit_record["axis"], exit_record["outward_sign"]
        derived = source if exit_record["selected_original"] else replace(source,access=AccessWindow(**exit_record["access"]))
        derived.assert_intact()
        sources[key] = derived
        candidates.append(IngressCandidate(*key,exit_record["distance_mm"],
            NativeResectionEngine(derived._native_config),derived._nominal_proposer))
    return tuple(candidates), sources


def worker(manifest, release, output, *, expected_manifest_sha256=None, expected_release_sha256=None, started_at=None):
    output = Path(output)
    if (output / "receipt.json").exists():
        raise ValueError("An existing attempt cannot be restarted")
    started = time.perf_counter() if started_at is None else started_at
    bound = {}; trace = InitialOnlyTrace()
    report = {"version": VERSION, "subject": SUBJECT, "role": "TRAIN", "status": "preparing",
        "settings": SETTINGS, "original": {"status": "not_executed"},
        "screening": {"status": "not_executed"}, "screened_selected": {"status": "not_executed"},
        "executed_transitions": 0, "commits": 0, "searches": 0, "policy_forwards": 0,
        "checkpoints": 0, "optimizer_updates": 0, "automatic_retry": False}
    def check():
        if time.perf_counter()-started >= SETTINGS["cooperative_seconds"] or peak_rss_bytes()>SETTINGS["max_rss_bytes"]:
            raise TimeoutError("Whole-worker preparation wall/memory envelope exhausted")
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_bytes=peak_rss_bytes(),
                      total_preview_calls=trace.total)
        write_json(output / "receipt.json",report)
    save()
    try:
        if (expected_manifest_sha256 is None or expected_release_sha256 is None
                or sha256(Path(manifest))!=expected_manifest_sha256 or sha256(Path(release))!=expected_release_sha256):
            raise ValueError("Worker requires the parent's exact manifest/release bytes before preflight or decode")
        record, saved, bound, released = preflight(manifest,release,output,check=check)
        report["execution_binding"] = {"source_commit":released["source_commit"],
            "archive_root":str(ROOT),"inputs_before":bound,"declaration_sha256":sha256(Path(manifest)),
            "release_sha256":sha256(Path(release))}
        check()
        case, reproduced_saved, exits = legacy.load_bound_case(record["legacy_input"],check)
        if canonical(saved)!=canonical(reproduced_saved):
            raise ValueError("Saved source binding changed during load")
        report["derived_exits"] = exits
        original = next(row for row in exits if row["selected_original"])
        common = record["legacy_input"]["common_task"]
        trace.check = check
        with trace:
            try:
                task, result = inspect_initial(lambda: prep._construct_task(case,saved["binding"]["member"],common,
                    lambda: time.perf_counter()-started>=SETTINGS["cooperative_seconds"]),original,"original",trace,check)
                if (canonical(result["initial_inventory"])!=canonical(saved["initial_inventory"])
                        or result["decision_model_hash"]!=saved["binding"]["decision_model_hash"]
                        or result["reward"]!=common["objective"] or result["horizon"]!=common["max_steps"]):
                    raise ValueError("Original PAT25 initial inventory/model/objective no longer reproduces")
                result["historical_inventory_exactly_reproduced"] = True
                report["original"] = result; save()
            except BaseException as error:
                report["original"] = getattr(error,"diagnostic",{"status":"unassessed","error":str(error)})
                raise
            setup_start=time.perf_counter()
            candidates, sources = prepare_candidates(task.case,exits,check)
            report["candidate_setup_seconds"]=time.perf_counter()-setup_start
            from resectionlab.native_ingress import screen_axis_accesses
            screen_start=time.perf_counter()
            screening=screen_axis_accesses(candidates,cancelled=lambda: time.perf_counter()-started>=SETTINGS["cooperative_seconds"])
            report["screening"]=screening.to_dict(); report["screening_seconds"]=time.perf_counter()-screen_start; save()
            if not screening.complete or screening.screen_count>SETTINGS["max_static_screens"]:
                raise RuntimeError("Ingress selection is incomplete; no selected inventory may run")
            check()
            if screening.selected_exit is None:
                report["screened_selected"]={"status":"not_executed_no_ingress_admissible_access"}
            else:
                selected=next(row for row in exits if (row["axis"],row["outward_sign"])==screening.selected_exit)
                from resectionlab.native_spatial_task import NativeSpatialTask
                try:
                    _,result=inspect_initial(lambda: NativeSpatialTask(sources[screening.selected_exit],max_steps=common["max_steps"],
                        reward=task.reward_spec,cancelled=lambda: time.perf_counter()-started>=SETTINGS["cooperative_seconds"]),
                        selected,"screened_selected",trace,check)
                    if result["reward"]!=report["original"]["reward"] or result["horizon"]!=report["original"]["horizon"]:
                        raise ValueError("Selected access changed the frozen reward/horizon")
                    report["screened_selected"]=result; save()
                except BaseException as error:
                    report["screened_selected"]=getattr(error,"diagnostic",{"status":"unassessed","error":str(error)})
                    raise
        check()
        if trace.violation is not None: raise RuntimeError("Sticky observer violation: "+trace.violation)
        if not unchanged(bound):
            raise ValueError("Bound source/archive/metadata/bundle/release bytes changed")
        report.update(status="complete",inputs_unchanged=True)
    except BaseException as error:
        report.update(status="incomplete",error=f"{type(error).__name__}: {error}",inputs_unchanged=unchanged(bound) if bound else None)
    finally:
        report["observer_violation"]=trace.violation
        report["observer_health_verified"]=trace.violation is None and bool(trace.phases)
        if trace.violation is not None:
            report.update(status="incomplete",executed_transitions=None,commits=None,searches=None,
                          policy_forwards=None,checkpoints=None,optimizer_updates=None)
        save()
    return report



def supervise_worker(command,output,settings,declaration_sha256):
    """Inner native-process observation; outer owns the single process group."""
    started=time.perf_counter(); process=None; reason=None; maximum=0; samples=0
    try:
        with (Path(output)/"worker.log").open("x") as log:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,**THREAD_ENV,"PYTHONDONTWRITEBYTECODE":"1"},start_new_session=False)
            while process.poll() is None:
                remaining=settings["max_wall_seconds"]-(time.perf_counter()-started)
                if remaining<=0: reason="parent_wall_budget_exceeded"; break
                measurement=subprocess.run(["ps","-o","rss=","-p",str(process.pid)],
                    capture_output=True,text=True,check=False,timeout=min(.5,remaining))
                if measurement.stdout.strip():
                    maximum=max(maximum,int(measurement.stdout.strip())*1024);samples+=1
                elif process.poll() is None: raise RuntimeError("Running native-worker RSS unavailable")
                if maximum>settings["max_rss_bytes"]: reason="parent_sampled_rss_budget_exceeded";break
                time.sleep(min(.1,max(0.,remaining)))
    except BaseException as error: reason=f"inner_supervision_error:{type(error).__name__}:{error}"
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()
                try: process.wait(timeout=1)
                except subprocess.TimeoutExpired: reason=reason or "inner_worker_cleanup_timeout"
    code=None if process is None else process.poll()
    result={"status":"complete" if code==0 and reason is None else "failed","returncode":code,
        "termination_reason":reason,"timed_out":reason=="parent_wall_budget_exceeded",
        "seconds":time.perf_counter()-started,"sampled_peak_rss_bytes":maximum,"rss_samples":samples,
        "automatic_retry":False,"declaration_sha256":declaration_sha256,"inherits_outer_process_group":True}
    write_json(Path(output)/"supervisor.json",result)
    return result

def run(manifest,release,output, *, started_at=None, expected_manifest_sha256=None, expected_release_sha256=None):
    output=Path(output)
    started=time.perf_counter() if started_at is None else started_at
    def check():
        if time.perf_counter()-started >= SETTINGS["cooperative_seconds"]:
            raise TimeoutError("Parent input validation exhausted the whole-attempt budget")
    if (expected_manifest_sha256 is not None and sha256(Path(manifest))!=expected_manifest_sha256
            or expected_release_sha256 is not None and sha256(Path(release))!=expected_release_sha256):
        raise ValueError("Orchestrator inputs differ from the outer launcher's exact byte pins")
    record,saved,bound,released=preflight(manifest,release,output,check=check)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"execution-baseline.json",{"inputs":bound,"source_commit":released["source_commit"],"settings":SETTINGS})
    receipt={"status":"failed","automatic_retry":False,"parent_preflight_seconds":time.perf_counter()-started}
    try:
        command=[sys.executable,str(Path(__file__).resolve()),"--worker","--manifest",str(Path(manifest).resolve()),
                 "--release",str(Path(release).resolve()),"--output",str(output.resolve()),
                 "--orchestrator-pid",str(os.getpid()),
                 "--expected-manifest-sha256",bound[str(Path(manifest).resolve())],
                 "--expected-release-sha256",bound[str(Path(release).resolve())]]
        remaining=SETTINGS["max_wall_seconds"]-(time.perf_counter()-started)
        if remaining<=0: raise TimeoutError("No supervisor budget remains after preflight")
        supervisor_settings={**SETTINGS,"max_wall_seconds":remaining}
        receipt["supervisor_settings"]=supervisor_settings
        supervisor=supervise_worker(command,output,supervisor_settings,sha256(Path(manifest)))
        worker_record=json.loads((output/"receipt.json").read_text())
        receipt.update(supervisor=supervisor,worker_receipt_sha256=sha256(output/"receipt.json"))
        if (supervisor["status"]!="complete" or worker_record.get("status")!="complete"
                or worker_record.get("version")!=VERSION or worker_record.get("subject")!=SUBJECT
                or worker_record.get("inputs_unchanged") is not True or not unchanged(bound)
                or time.perf_counter()-started>SETTINGS["whole_worker_envelope_seconds"]):
            raise RuntimeError("Supervision, worker completion or source preservation failed")
        receipt["status"]="complete"
    except BaseException as error:
        receipt["error"]=f"{type(error).__name__}: {error}"
    receipt.update(inputs_unchanged=unchanged(bound),elapsed_seconds=time.perf_counter()-started)
    if not receipt["inputs_unchanged"]:
        receipt.update(status="failed",error="Final bound inputs changed before acceptance publication")
    write_json(output/"acceptance.json",receipt)
    if time.perf_counter()-started>=SETTINGS["whole_worker_envelope_seconds"]:
        receipt.update(status="failed",error="Acceptance publication exceeded the whole-attempt budget",
                       elapsed_seconds=time.perf_counter()-started)
        write_json(output/"acceptance.json",receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--declare",action="store_true")
    parser.add_argument("--preflight",action="store_true")
    parser.add_argument("--worker",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--orchestrator",action="store_true",help=argparse.SUPPRESS)
    parser.add_argument("--outer-parent-pid",type=int,help=argparse.SUPPRESS)
    parser.add_argument("--orchestrator-pid",type=int,help=argparse.SUPPRESS)
    parser.add_argument("--expected-manifest-sha256",help=argparse.SUPPRESS)
    parser.add_argument("--expected-release-sha256",help=argparse.SUPPRESS)
    parser.add_argument("--manifest",type=Path); parser.add_argument("--release",type=Path); parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    if args.declare:
        print(json.dumps(declaration(),sort_keys=True,indent=2,allow_nan=False)); return
    if not all((args.manifest,args.release,args.output)):
        parser.error("Metadata preflight and execution require --manifest, --release and --output")
    if args.preflight:
        _,_,bound,_=preflight(args.manifest,args.release,args.output)
        print(json.dumps({"status":"preflight_only","bindings":len(bound),"patient_decoded":False})); return
    pins={"expected_manifest_sha256":args.expected_manifest_sha256,"expected_release_sha256":args.expected_release_sha256}
    if not all(pins.values()): parser.error("Actual children require exact parent byte pins")
    if args.worker:
        if args.orchestrator_pid!=os.getppid() or os.getpgrp()!=args.orchestrator_pid:
            parser.error("Native worker must inherit its live orchestrator's process group")
        result=worker(args.manifest,args.release,args.output,started_at=_BOOT_STARTED,**pins)
    else:
        if not args.orchestrator or args.outer_parent_pid!=os.getppid() or os.getpgrp()!=os.getpid():
            parser.error("Execution requires the live outer-owned process group")
        def interrupted(signum,frame): raise TimeoutError("Outer whole-attempt watchdog requested termination")
        signal.signal(signal.SIGTERM,interrupted)
        result=run(args.manifest,args.release,args.output,started_at=_BOOT_STARTED,**pins)
    if result["status"]!="complete": raise SystemExit(1)


if __name__=="__main__":
    main()
