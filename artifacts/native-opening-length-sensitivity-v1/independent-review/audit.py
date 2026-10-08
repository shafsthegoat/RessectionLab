"""Saved-output audit using stdlib JSON/scalars and byte hashes only."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ART = ROOT / "artifacts/native-opening-length-sensitivity-v1"
COMMIT = "dcf894641fa7c9d6389b280df5978565b91fe09f"
EXPECTED = {
    "result.json": "71a44c9fdc91d0254bde907d108d1c0bb9737f21b8ff4c04be8140fbd4ee91e5",
    "initial.json": "c0d3d11cad61ea7bdca27a68c172f9d6813422ef8e6fc5499eadfdd50c6012eb",
    "RL256.json": "e5a15613c0d0d837c561243ad41e01fd15e185c8395ef344715a770b63097af8",
    "supervisor.json": "a6cc1481ccdeffa469fe358231862a14501bd2bbca35438b29adeef8ee20b994",
}
checks = 0


def require(value, label):
    global checks
    checks += 1
    if not value:
        raise AssertionError(label)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def close(a, b, label, tolerance=1e-12):
    require(math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, label)


def probabilities(logits):
    maximum = max(logits)
    values = [math.exp(x-maximum) for x in logits]
    denominator = math.fsum(values)
    return [x/denominator for x in values]


def main():
    began = time.perf_counter()
    require(not (OUT / "verification.json").exists(), "review output is fresh")
    for name, expected in EXPECTED.items():
        require(sha(ART/name) == expected, "parent-bound output: " + name)
    index = read(ART/"output-sha256.json")
    actual = {str(p.relative_to(ART)): sha(p) for p in ART.rglob("*") if p.is_file() and p != ART/"output-sha256.json"}
    require(actual == index, "index covers all original output bytes, excludes only its own root")
    for relative, expected in index.items():
        require(actual[relative] == expected, "indexed file: " + relative)
    nested_index = "input-metadata/artifacts/native-opening-rl-capacity-v1/output-sha256.json"
    require(nested_index in index, "inherited index is included")
    declaration = read(ART/"declaration-input.json")
    declaration_sha = sha(ART/"declaration-input.json")
    preflight_path = ROOT/"build/native-opening-length-sensitivity-independent-review-v1/verification.json"
    require(sha(preflight_path) == "6ca66f6e1583ab19cd194fe96580cfa363b00fcae15ca7e8f1830b4fd0dd332d", "prior independent review unchanged")
    preflight = read(preflight_path)
    require(declaration_sha == preflight["final_files_sha256"]["manifests/experiments/native-opening-length-sensitivity-v1.json"], "executed declaration is independently reviewed")
    require(declaration["runtime"] == preflight["runtime"], "runtime binding unchanged")
    source_ids = declaration["source_sha256"]
    require(len(source_ids) == 29, "exact 29-file source closure")
    for relative, expected in source_ids.items():
        require(sha(ART/"source-snapshot"/relative) == expected, "retained source: " + relative)
        require(sha(ROOT/relative) == expected, "current source: " + relative)
        committed = subprocess.run(["git", "show", COMMIT+":"+relative], cwd=ROOT, check=True, capture_output=True).stdout
        require(hashlib.sha256(committed).hexdigest() == expected, "committed source: " + relative)
    for relative, expected in preflight["final_files_sha256"].items():
        committed = subprocess.run(["git", "show", COMMIT+":"+relative], cwd=ROOT, check=True, capture_output=True).stdout
        require(hashlib.sha256(committed).hexdigest() == expected, "reviewed runner/tests/manifest in source commit")
    require(len(declaration["input_sha256"]) == 7, "exact seven generated metadata inputs")
    for relative, expected in declaration["input_sha256"].items():
        require(sha(ART/"input-metadata"/relative) == expected, "retained generated metadata: " + relative)
        require(sha(ROOT/relative) == expected, "unchanged original generated metadata: " + relative)
    proposal_path = ROOT/"build/native-opening-length-sensitivity-v1/proposal.json"
    require(sha(proposal_path) == declaration["approved_proposal_sha256"], "prospective proposal byte identity")
    proposal = read(proposal_path)
    inputs, result, supervisor = [read(ART/name) for name in ("inputs.json", "result.json", "supervisor.json")]
    prep = read(ART/"input-metadata/artifacts/native-opening-learning-v1/preparation.json")
    old_result = read(ART/"input-metadata/artifacts/native-opening-rl-capacity-v1/result.json")
    old_review = read(ART/"input-metadata/artifacts/native-opening-rl-capacity-v1/independent-verification.json")
    old_index = read(ART/nested_index)
    binding = preflight["input_identity"]
    for name in ("original_observation_hash", "original_source_hash", "original_task_model_hash"):
        required = {"original_observation_hash": declaration["original_observation_hash"],
                    "original_source_hash": declaration["original_source_hash"],
                    "original_task_model_hash": declaration["original_task_model_hash"]}[name]
        require(inputs[name] == required, "input identity: " + name)
    require(inputs["original_observation_hash"] == binding["original_fingerprint"] == prep["initial_observation_hash"], "original whole DTO identity")
    require(inputs["intervention_observation_hash"] == binding["intervention_fingerprint"], "intervention whole DTO identity")
    require(inputs["original_source_hash"] == prep["source_hash"], "generated source ancestry")
    require(inputs["original_task_model_hash"] == prep["decision_model_hash"], "original task model ancestry")
    require(inputs["original_action_geometry"] == binding["original_action_geometry"], "original geometry matches independent pure reconstruction")
    require(inputs["intervention_action_geometry"] == binding["intervention_action_geometry"], "modified geometry matches independent pure reconstruction")
    old_geometry, new_geometry = inputs["original_action_geometry"], inputs["intervention_action_geometry"]
    require(len(old_geometry) == len(new_geometry) == 5, "five choices retained")
    require(all(len(a) == len(b) == 16 for a,b in zip(old_geometry,new_geometry)), "16-column geometry retained")
    differences = [[i,j] for i in range(5) for j in range(16) if old_geometry[i][j] != new_geometry[i][j]]
    require(differences == [[1,12],[2,12],[3,12],[4,12]], "only four intended length descriptors differ")
    require([old_geometry[i][12] for i in range(1,5)] == [2.2,12.,2.2,12.], "original lengths")
    require([new_geometry[i][12] for i in range(1,5)] == [120.]*4, "new lengths")
    require(inputs["action_mask"] == [True]*5, "unchanged five-choice action mask")
    require(inputs["state_features"] == [0.,2.,0.,3.9,4.,.5,0.,0.,1.,2.4], "unchanged generated root state")
    require("not modified physical-tool certification" in inputs["source_id_interpretation"], "modified provenance interpretation retained")
    for field, identities in binding["all_other_arrays"].items():
        require(identities["original"] == identities["intervention"], "previously verified unchanged tensor bytes: " + field)
    require(result["status"] == supervisor["status"] == "complete", "worker and supervisor terminal success")
    require(supervisor["returncode"] == 0 and not supervisor["timed_out"] and supervisor["termination_reason"] is None, "successful bounded exit")
    for field in ("checkpoint_attempts", "forward_attempts", "completed_forwards"):
        require(result["ledger"][field] == ["initial","RL256"], "exactly two ordered calls: " + field)
    for field in ("executed_actions", "native_previews", "new_baseline_forwards", "optimizer_updates", "patient_inputs"):
        require(result[field] == 0, "no new prohibited operation: " + field)
    require(result["declaration_sha256"] == supervisor["declaration_sha256"] == declaration_sha, "run binds exact prospective declaration")
    require(set(result["methods"]) == {"initial", "RL256"}, "all and only two checkpoint slots")
    numerical = {}
    checkpoint_ids = {}
    for name in ("initial", "RL256"):
        row, slot, spec = read(ART/(name+".json")), result["methods"][name], declaration["checkpoints"][name]
        update = str(spec["updates"])
        matching = [s for s in old_result["readouts"][update]["states"] if s["observation_hash"] == inputs["original_observation_hash"]]
        require(len(matching) == 1, "one original saved root baseline")
        baseline = matching[0]
        require(inputs["saved_baselines"][name] == baseline, "baseline is original authenticated readout")
        require(proposal["checkpoints"][name]["original_saved_root"] == baseline, "baseline fixed prospectively")
        require(row["action_ids"] == inputs["action_ids"] == baseline["action_ids"], "same ordered five choices")
        require(row["original_observation_hash"] == inputs["original_observation_hash"], "output original DTO binding")
        require(row["intervention_observation_hash"] == inputs["intervention_observation_hash"], "output modified DTO binding")
        require(row["checkpoint"] == spec, "checkpoint specification exact")
        require(sha(ROOT/spec["path"]) == spec["file_sha256"] == old_index[Path(spec["path"]).name], "checkpoint bytes remain unchanged")
        require(spec["parameter_hash"] == old_review["checkpoints"][update]["tensor_sha256"] == old_result["readouts"][update]["parameter_hash"], "independent original checkpoint tensor identity")
        require(row["architecture_hash"] == declaration["architecture_hash"] == old_result["architecture_hash"], "same architecture")
        require(row["weights_unchanged"] is True, "runner verified unchanged weight and gradient state")
        require(slot["status"] == row["status"] == "complete" and slot["outputs"] == name+".json", "method terminal output")
        logits, probs, old_logits, old_probs = row["logits"], row["probabilities"], baseline["logits"], baseline["probabilities"]
        require(len(logits) == len(probs) == len(old_logits) == len(old_probs) == 5, "exact logits/probability dimensions")
        require(all(math.isfinite(x) for x in logits+probs+old_logits+old_probs+[row["value_uncalibrated"]]), "all scalar outputs finite")
        pcalc, poldcalc = probabilities(logits), probabilities(old_logits)
        for i in range(5):
            close(probs[i], pcalc[i], "independent output softmax", 1e-7)
            close(old_probs[i], poldcalc[i], "independent saved baseline softmax", 1e-7)
            close(row["per_action_logit_change"][i], logits[i]-old_logits[i], "per-action logit delta")
            close(row["per_action_probability_change"][i], probs[i]-old_probs[i], "per-action probability delta")
        close(math.fsum(probs),1.,"output probability mass",1e-7)
        close(math.fsum(old_probs),1.,"saved probability mass",1e-7)
        margin, old_margin = logits[0]-max(logits[1:]), old_logits[0]-max(old_logits[1:])
        close(row["stop_minus_best_nonstop_logit"],margin,"output STOP margin")
        close(row["saved_original_margin"],old_margin,"baseline margin from actual stored logits")
        close(row["margin_change"],margin-old_margin,"primary margin change")
        close(row["saved_original_stop_probability"],old_probs[0],"saved STOP probability")
        close(row["stop_probability"],probs[0],"output STOP probability")
        close(row["stop_probability_change"],probs[0]-old_probs[0],"STOP probability change")
        close(row["stop_logit_change"],logits[0]-old_logits[0],"STOP logit delta")
        require(logits[0] == old_logits[0] and row["stop_head_invariant_within_1e_minus6"] is True, "STOP logit exactly unchanged")
        entropy = -math.fsum(p*math.log(p) for p in probs if p)
        close(row["entropy"],entropy,"independent entropy")
        argmax = row["action_ids"][max(range(5),key=logits.__getitem__)]
        baseline_argmax = baseline["action_ids"][max(range(5),key=old_logits.__getitem__)]
        require(row["highest_ranked_id_not_executed"] == argmax, "output argmax from stored logits")
        require(baseline["selected_action"] == baseline_argmax, "original argmax from stored logits")
        for field in ("margin_change","stop_probability","stop_logit_change","highest_ranked_id_not_executed"):
            require(slot[field] == row[field], "summary matches detailed output: " + field)
        require(0 <= row["forward_seconds"] <= row["load_and_forward_seconds"] < result["elapsed_seconds"], "nested per-method clock ordering")
        numerical[name] = {"original_stop_probability":old_probs[0],"intervention_stop_probability":probs[0],
            "stop_probability_change":probs[0]-old_probs[0],"original_stop_minus_best_nonstop_logit":old_margin,
            "intervention_stop_minus_best_nonstop_logit":margin,"margin_change":margin-old_margin,
            "stop_logit_exactly_unchanged":True,"stop_logit":logits[0],"original_argmax":baseline_argmax,
            "intervention_argmax":argmax,"intervention_entropy":entropy,
            "nonstop_logit_changes":[logits[i]-old_logits[i] for i in range(1,5)],
            "softmax_max_abs_error":max(abs(a-b) for a,b in zip(pcalc,probs)),
            "baseline_softmax_max_abs_error":max(abs(a-b) for a,b in zip(poldcalc,old_probs)),
            "saved_baseline_cached_margin_rounding_difference":baseline["stop_minus_best_nonstop_logit"]-old_margin,
            "forward_seconds":row["forward_seconds"],"load_and_forward_seconds":row["load_and_forward_seconds"]}
        checkpoint_ids[name] = spec
    require(numerical["RL256"]["original_argmax"] != "STOP" and numerical["RL256"]["intervention_argmax"] == "STOP", "trained descriptor-only preference flip")
    require(numerical["initial"]["original_argmax"] != "STOP" and numerical["initial"]["intervention_argmax"] != "STOP", "initial remains movement-preferring")
    launch_path = ROOT/"build/native-opening-length-sensitivity-execution-v1/launch.json"
    terminal_path = ROOT/"build/native-opening-length-sensitivity-execution-v1/terminal.json"
    launch, terminal = read(launch_path), read(terminal_path)
    require(launch["source_commit"] == terminal["source_commit"] == COMMIT, "execution source commit")
    require(all(terminal[k] == v for k,v in launch.items()), "launch preserved in terminal closure")
    require(terminal["exit_code"] == 0 and not terminal["automatic_retry"], "outer process terminal success without retry")
    require(terminal["command"][-1] == "--execute" and "artifacts/native-opening-length-sensitivity-v1" in terminal["command"], "exact declared executable scope")
    require(0 < result["elapsed_seconds"] < supervisor["seconds"] < terminal["elapsed_seconds"] < 20., "nested worker/supervisor/outer wall clocks")
    require(0 < supervisor["sampled_peak_rss_bytes"] < 1024**3 and 0 < result["peak_worker_rss_bytes"] < 1024**3, "worker memory within declared bounds")
    require(declaration["settings"]["cpu_threads"] == 1 and declaration["settings"]["max_wall_seconds"] == 20. and declaration["settings"]["max_rss_bytes"] == 1024**3, "unchanged 20s/1GiB/one-thread declaration")
    require(not declaration["clinical_correctness_measured"] and not declaration["planning_efficacy_measured"], "no clinical/planning endpoint claim")
    timings = {"outer_wall_seconds":terminal["elapsed_seconds"],"supervisor_seconds":supervisor["seconds"],
        "worker_seconds":result["elapsed_seconds"],"worker_reconstruction_seconds":result["reconstruction_seconds"],
        "sum_of_two_disjoint_forward_seconds":math.fsum(v["forward_seconds"] for v in numerical.values()),
        "sum_of_two_disjoint_load_forward_validation_seconds":math.fsum(v["load_and_forward_seconds"] for v in numerical.values()),
        "worker_self_peak_rss_bytes":result["peak_worker_rss_bytes"],"supervisor_sampled_worker_peak_rss_bytes":supervisor["sampled_peak_rss_bytes"],
        "rss_samples":supervisor["rss_samples"],"interpretation":"Outer includes supervision; worker includes reconstruction and both load/forward/validation calls; forward clocks are nested. Do not add overlapping clocks. Prior baseline work is excluded. RSS is sampled with possible transient misses."}
    bindings = {"source_commit":COMMIT,"output_index_sha256":sha(ART/"output-sha256.json"),
        "declaration_sha256":declaration_sha,"inputs_sha256":sha(ART/"inputs.json"),
        **{name:sha(ART/name) for name in EXPECTED},"pre_execution_independent_review_sha256":sha(preflight_path),
        "launch_sha256":sha(launch_path),"terminal_sha256":sha(terminal_path),"proposal_sha256":sha(proposal_path)}
    report = {"schema":"native-opening-length-sensitivity-saved-output-review-v1","status":"passed",
        "created_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),"checks_passed":checks,"issues":[],
        "bindings":bindings,"indexed_file_count":len(index),"indexed_bytes":sum((ART/p).stat().st_size for p in index),
        "source_file_count":len(source_ids),"generated_metadata_file_count":len(declaration["input_sha256"]),
        "indexed_file_sha256":index,"checkpoint_bindings":checkpoint_ids,
        "input_identity":{"source_ancestry":inputs["original_source_hash"],"task_model_ancestry":inputs["original_task_model_hash"],
            "original_dto_fingerprint":inputs["original_observation_hash"],"modified_dto_fingerprint":inputs["intervention_observation_hash"],
            "changed_coordinates":differences,"original_lengths_mm":[2.2,12.,2.2,12.],"modified_lengths_mm":[120.]*4,
            "unchanged_action_ids":inputs["action_ids"],"unchanged_action_mask":inputs["action_mask"],
            "other_arrays_evidence":"Full original/modified DTO fingerprints match the pre-execution independent byte-equality audit, joined to frozen runner assertions before/after each saved forward; no new array reconstruction or model call in this review.",
            "source_id_interpretation":inputs["source_id_interpretation"]},
        "actual_experiment_accounting":{"new_forwards":2,"new_baseline_forwards":0,"native_engine_or_proposer_calls":0,"executed_actions":0,"optimizer_updates":0,"patient_inputs":0,"same_choice_count":5,"attempt_order":result["ledger"]},
        "numerical_results":numerical,"timing_and_resources":timings,
        "conclusion":"Changing only the four nonSTOP working-length descriptors to120mm is sufficient to flip this trained frozen policy from movement to STOP on the familiar generated root. STOP logit is exactly unchanged; lowered movement logits produce the increased STOP probability. Initial weights remain movement-preferring.",
        "claim_limits":["One trained checkpoint and one generated root, one fixed descriptor intervention; no generalization or interaction sweep.","Does not establish the full cause of PAT05 preference or a clinical error.","No executed planning endpoint, reward, removal, safety, clinical-risk probability or real transfer measured.","120mm descriptors were not certified as physical tools or a complete/legal modified action inventory.","Original negative and later positive simulator findings are unchanged."],
        "audit_scope":{"method":"stdlib saved JSON, independently computed scalar arithmetic, source/metadata/checkpoint byte hashes; no repository code imported","checkpoint_deserializations":0,"model_forwards":0,"patient_decodes":0,"engine_or_proposer_calls":0,"rollouts":0,"updates":0,"tracked_writes":0},
        "audit_script_sha256":sha(Path(__file__)),"audit_seconds":time.perf_counter()-began}
    (OUT/"verification.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"status":"passed","checks":checks,"indexed_files":len(index),"verification_sha256":sha(OUT/"verification.json"),"numerical_results":numerical,"timing_and_resources":timings},indent=2))


if __name__ == "__main__":
    main()
