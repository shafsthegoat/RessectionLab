"""Independent arithmetic on saved decision JSON; no model or patient-array imports."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARCHIVE = ROOT / "artifacts/remaining-training-frozen-spatial-float64-v1/completed-run.tar.gz"
ARCHIVE_SHA = "ee11dc1b7ca194a43c51ea7bac0194e62c7c6d00a89329c5dbccc52bfde9bfea"
SOURCE_SHA = "792b248d27df8b380f0939abe46453a66a66133afdbc63d6bb64a6d33afbf332"
SUBJECTS = ["sub-PAT16", "sub-PAT20", "sub-PAT22", "sub-PAT25", "sub-PAT28"]


def need(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024**2):
            value.update(chunk)
    return value.hexdigest()


def close(left, right, tolerance=1e-9):
    need(math.isfinite(left) and math.isfinite(right) and abs(left - right) <= tolerance,
         f"arithmetic mismatch: {left!r} != {right!r}")


def check():
    need(digest(ARCHIVE) == ARCHIVE_SHA, "original archive hash mismatch")
    source = HERE / "analyze.py"
    need(digest(source) == SOURCE_SHA, "diagnostic source changed")
    summary_path = HERE / "summary.json"
    summary_sha = digest(summary_path)
    diagnostic = json.loads(summary_path.read_bytes())
    need(diagnostic["analysis_source_sha256"] == SOURCE_SHA and
         diagnostic["archive_sha256"] == ARCHIVE_SHA, "diagnostic provenance mismatch")
    need(diagnostic["prescribed_patients"] == 5 and
         [row["subject"] for row in diagnostic["patients"]] == SUBJECTS, "five-case denominator changed")
    members_read = {}
    rows = []
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        # Only JSON members are decoded. Images, checkpoints and arrays are never extracted.
        names = [item.name for item in archive.getmembers()]
        need(len(names) == len(set(names)), "ambiguous archive names")

        def read(name):
            member = archive.getmember(name)
            need(member.isfile() and name.endswith(".json") and member.size <= 20 * 1024**2,
                 "unexpected metadata member")
            data = archive.extractfile(member).read()
            members_read[name] = hashlib.sha256(data).hexdigest()
            return json.loads(data)

        original = read("summary.json")
        declaration = read("declaration-input.json")
        need(declaration["subjects"] == SUBJECTS and
             [p["subject"] for p in original["patients"]] == SUBJECTS, "original cohort mismatch")
        need(original["optimizer_updates"] == 0 and original["adaptation"] is False,
             "frozen inference scope changed")
        need(declaration["checkpoint"]["training_patient"] == "sub-PAT05" and
             declaration["checkpoint"]["parameter_hash"] == original["checkpoint_parameter_hash"],
             "checkpoint lineage mismatch")

        for patient, claimed in zip(original["patients"], diagnostic["patients"]):
            subject = patient["subject"]
            need(patient["status"] == claimed["status"] and
                 patient["preparation_status"] == claimed["preparation_status"], "status mismatch")
            if subject in SUBJECTS[:2]:
                need(patient["status"] == "preparation_blocked" and
                     patient["preparation_status"] == "blocked_support_conflict" and
                     claimed["diagnostic"] is None, "blocked case converted to an outcome")
                for method in ("frozen_policy", "greedy_search"):
                    need(patient["methods"][method]["status"] == "not_executed" and
                         patient["methods"][method]["outcomes"] is None, "blocked outcome is not null")
                rows.append({"subject": subject, "status": "blocked_support_conflict", "diagnostic": None})
                continue

            need(patient["status"] == "complete", "incomplete primary pair")
            policy = read(subject + "/frozen_policy.json")
            greedy = read(subject + "/greedy_search.json")
            decisions = (policy["decisions"], greedy["decisions"])
            returns = []
            for episode in (policy, greedy):
                need(episode["status"] == "complete" and
                     episode["independent_evaluation"]["accepted"] is True and
                     episode["independent_evaluation"]["complete_episode"] is True,
                     "primary episode lacks saved independent acceptance")
                need(episode["behavior_parameter_hash"] == original["checkpoint_parameter_hash"],
                     "behavior parameter identity mismatch")
                rewards = []
                for decision in episode["decisions"]:
                    need(decision["status"] == "returned", "unfinished decision")
                    close(decision["reward"], decision["committed_info"]["reward"])
                    rewards.append(decision["reward"])
                total = math.fsum(rewards)
                close(total, episode["simulated_return"])
                close(total, episode["independent_evaluation"]["outcomes"]["total_reward"])
                returns.append(total)

            initial, teacher = decisions[0][0], decisions[1][0]
            for key in ("observation_hash", "action_ids", "action_mask"):
                need(initial[key] == teacher[key], "initial states or legal choices differ")
            ids, mask, logits, probabilities = (initial[key] for key in
                                                 ("action_ids", "action_mask", "logits", "probabilities"))
            need(len(ids) == len(mask) == len(logits) == len(probabilities), "action-vector length mismatch")
            need(len(set(ids)) == len(ids), "duplicate action identity")
            legal = [index for index, allowed in enumerate(mask) if allowed]
            selected, target = initial["selected_index"], teacher["selected_index"]
            need(selected in legal and target in legal and ids[selected] == initial["action_id"] and
                 ids[target] == teacher["action_id"], "illegal or mislabeled selection")
            ordered = sorted(legal, key=lambda i: (-logits[i], i))
            need(selected == ordered[0] and initial["decision_rule"] == "argmax_first; STOP wins exact ties",
                 "saved frozen behavior is not deterministic first argmax")
            need(all(math.isfinite(logits[i]) for i in legal), "nonfinite legal logit")
            need(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), "invalid probability")
            need(all(probabilities[i] == 0 for i in range(len(mask)) if not mask[i]),
                 "illegal action has nonzero mass")
            close(math.fsum(probabilities), 1.0, 1e-6)
            maximum = logits[ordered[0]]
            exp_values = {i: math.exp(logits[i] - maximum) for i in legal}
            normalizer = math.fsum(exp_values.values())
            for i in legal:
                close(probabilities[i], exp_values[i] / normalizer, 1e-7)
            # Ranking by sorted tie groups is independent of the producer's greater-count formula.
            tied_positions = [position + 1 for position, i in enumerate(ordered) if logits[i] == logits[target]]
            entropy = math.fsum(-p * math.log(p) for i, p in enumerate(probabilities) if mask[i] and p > 0)
            uniform_entropy = math.log(len(legal))
            need(-1e-12 <= entropy <= uniform_entropy + 1e-6, "entropy outside categorical range")
            gap = returns[1] - returns[0]
            first_gap = teacher["reward"] - initial["reward"]
            remainder = math.fsum(d["reward"] for d in decisions[1][1:]) - math.fsum(
                d["reward"] for d in decisions[0][1:])
            close(gap, first_gap + remainder)
            saved = claimed["diagnostic"]
            exact = {"initial_observation_hash": initial["observation_hash"], "legal_actions": len(legal),
                     "search_initial_action_rank_min": min(tied_positions),
                     "policy_step_rewards": [d["reward"] for d in decisions[0]],
                     "search_step_rewards": [d["reward"] for d in decisions[1]]}
            for key, value in exact.items():
                need(saved[key] == value, f"saved {key} mismatch")
            numbers = {"policy_selected_probability": probabilities[selected],
                       "search_initial_action_probability": probabilities[target],
                       "policy_minus_search_action_logit": logits[selected] - logits[target],
                       "entropy_nats": entropy,
                       "first_step_search_minus_policy_reward": first_gap,
                       "total_search_minus_policy_reward": gap,
                       "later_steps_arithmetic_remainder": remainder}
            for key, value in numbers.items():
                close(value, saved[key])
            if len(legal) > 1:
                close(entropy / uniform_entropy, saved["entropy_fraction_of_uniform_max"])
                close(remainder / gap, saved["later_steps_fraction_of_total_gap"])
                need(initial["action_id"] != teacher["action_id"] and
                     saved["later_states_are_matched"] is False and
                     all(a["observation_hash"] != b["observation_hash"]
                         for a, b in zip(decisions[0][1:], decisions[1][1:])), "later-state divergence mismatch")
            else:
                need(subject == "sub-PAT25" and ids[selected] == ids[target] == "STOP" and
                     len(decisions[0]) == len(decisions[1]) == 1 and returns == [0, 0] and
                     saved["entropy_fraction_of_uniform_max"] is None and
                     saved["later_steps_fraction_of_total_gap"] is None and
                     saved["later_states_are_matched"] is None, "STOP-only case changed")
            rows.append({"subject": subject, "status": "complete", "legal_actions": len(legal),
                         "policy_action": initial["action_id"], "greedy_action": teacher["action_id"],
                         "greedy_rank_tie_interval": [min(tied_positions), max(tied_positions)],
                         "entropy_nats": entropy, "maximum_entropy_nats": uniform_entropy,
                         "deterministic_argmax_verified": True, "episode_returns": returns,
                         "first_step_greedy_minus_policy": first_gap,
                         "later_steps_arithmetic_only": remainder, "total_greedy_minus_policy": gap})

    for name, wanted in diagnostic["input_members_sha256"].items():
        need(members_read[name] == wanted, "diagnostic input-member binding mismatch")
    need(set(diagnostic["input_members_sha256"]) == set(members_read) - {"declaration-input.json"},
         "unexpected diagnostic input set")
    need(digest(ARCHIVE) == ARCHIVE_SHA and digest(source) == SOURCE_SHA and
         digest(summary_path) == summary_sha, "input drift during independent check")
    return {"schema": "resectionlab.saved-decision-diagnostic-independent-check.v1",
            "accepted": True, "checked_at": datetime.now(timezone.utc).isoformat(),
            "checker": {"path": Path(__file__).relative_to(ROOT).as_posix(), "sha256": digest(Path(__file__))},
            "analysis_source_sha256": SOURCE_SHA, "analysis_summary_sha256": summary_sha,
            "original_archive_sha256": ARCHIVE_SHA, "read_json_members_sha256": members_read,
            "prescribed_train_cases": 5, "blocked_cases": 2, "complete_primary_pairs": 3,
            "stop_only_primary_pairs": 1, "patients": rows, "source_and_archive_unchanged": True,
            "checker_failures": [], "patient_array_files_opened": 0, "checkpoint_loads": 0,
            "model_forwards": 0, "native_runs": 0, "updates": 0,
            "scope": "Hash-pinned saved TRAIN decision JSON only; no producer import or execution. No patient-array/checkpoint members were opened and no cell or voxel arrays were used for these calculations.",
            "interpretation": ["PAT16 and PAT20 remain blocked/null; PAT25 remains a legitimate STOP-only zero-return result.",
                               "PAT22 and PAT28 share the initial observation across methods, then follow different actions and different later observations.",
                               "Later-step reward subtraction is descriptive, not a causal attribution, controlled state comparison or counterfactual regret.",
                               "Near-maximum categorical entropy describes stored action probabilities; actual selections are deterministic argmax. It is neither clinical uncertainty nor evidence of random argmax behavior.",
                               "This is post-hoc arithmetic on development TRAIN cases, not held-out generalization or surgical validation."]}


if __name__ == "__main__":
    result = check()
    output = HERE / "independent-check.json"
    need(not output.exists(), "preserve existing independent result")
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"accepted": True, "receipt_sha256": digest(output),
                      "checker_sha256": result["checker"]["sha256"]}))
