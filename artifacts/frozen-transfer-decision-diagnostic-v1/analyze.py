"""Post-hoc arithmetic on completed frozen-transfer JSON, with no model execution."""
from pathlib import Path
import hashlib
import json
import math
import tarfile

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "artifacts/remaining-training-frozen-spatial-float64-v1/completed-run.tar.gz"
EXPECTED_SHA = "ee11dc1b7ca194a43c51ea7bac0194e62c7c6d00a89329c5dbccc52bfde9bfea"
SUBJECTS = ["sub-PAT16", "sub-PAT20", "sub-PAT22", "sub-PAT25", "sub-PAT28"]


def analyze():
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == EXPECTED_SHA
    inputs = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        def read(name):
            member = archive.getmember(name)
            assert member.isfile() and member.size <= 20 * 1024**2
            raw = archive.extractfile(member).read()
            inputs[name] = hashlib.sha256(raw).hexdigest()
            return json.loads(raw)

        summary = read("summary.json")
        assert [p["subject"] for p in summary["patients"]] == SUBJECTS
        assert summary["optimizer_updates"] == 0 and summary["adaptation"] is False
        rows = []
        for patient in summary["patients"]:
            subject = patient["subject"]
            row = {"subject": subject, "status": patient["status"],
                   "preparation_status": patient["preparation_status"], "diagnostic": None}
            rows.append(row)
            if patient["status"] == "preparation_blocked":
                continue
            assert patient["status"] == "complete"
            policy = read(subject + "/frozen_policy.json")
            search = read(subject + "/greedy_search.json")
            for episode in (policy, search):
                assert episode["status"] == "complete" and episode["independent_evaluation"]["accepted"] is True
                assert episode["behavior_parameter_hash"] == summary["checkpoint_parameter_hash"]
                assert math.isclose(sum(d["reward"] for d in episode["decisions"]),
                                    episode["simulated_return"], abs_tol=1e-10, rel_tol=0)
            a, b = policy["decisions"][0], search["decisions"][0]
            for key in ("observation_hash", "action_ids", "action_mask"):
                assert a[key] == b[key], key
            legal = [i for i, mask in enumerate(a["action_mask"]) if mask]
            probabilities, logits = a["probabilities"], a["logits"]
            selected, teacher = a["selected_index"], b["selected_index"]
            assert selected in legal and teacher in legal
            assert a["action_ids"][selected] == a["action_id"]
            assert a["action_ids"][teacher] == b["action_id"]
            assert selected == max(legal, key=lambda i: logits[i])
            assert all(math.isfinite(probabilities[i]) and 0 <= probabilities[i] <= 1 for i in legal)
            assert math.isclose(sum(probabilities), 1., abs_tol=1e-6, rel_tol=0)
            peak = max(logits[i] for i in legal)
            exponents = [math.exp(logits[i] - peak) for i in legal]
            total = sum(exponents)
            assert all(math.isclose(probabilities[i], value / total, abs_tol=1e-7, rel_tol=0)
                       for i, value in zip(legal, exponents))
            entropy = -sum(probabilities[i] * math.log(probabilities[i]) for i in legal if probabilities[i] > 0)
            first_gap = b["reward"] - a["reward"]
            total_gap = search["simulated_return"] - policy["simulated_return"]
            row["diagnostic"] = {
                "initial_observation_hash": a["observation_hash"], "legal_actions": len(legal),
                "policy_selected_probability": probabilities[selected],
                "search_initial_action_probability": probabilities[teacher],
                "search_initial_action_rank_min": 1 + sum(logits[i] > logits[teacher] for i in legal),
                "policy_minus_search_action_logit": logits[selected] - logits[teacher],
                "entropy_nats": entropy,
                "entropy_fraction_of_uniform_max": entropy / math.log(len(legal)) if len(legal) > 1 else None,
                "first_step_search_minus_policy_reward": first_gap,
                "total_search_minus_policy_reward": total_gap,
                "later_steps_arithmetic_remainder": total_gap - first_gap,
                "later_steps_fraction_of_total_gap": (total_gap - first_gap) / total_gap if total_gap else None,
                "policy_step_rewards": [d["reward"] for d in policy["decisions"]],
                "search_step_rewards": [d["reward"] for d in search["decisions"]],
                "later_states_are_matched": False if a["action_id"] != b["action_id"] else None,
            }
    assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == EXPECTED_SHA
    return {"analysis": "post_hoc_saved_decision_diagnostic_v1", "archive_sha256": EXPECTED_SHA,
            "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_members_sha256": inputs, "prescribed_patients": len(SUBJECTS), "patients": rows,
            "patient_arrays_opened": 0, "checkpoint_loads": 0, "policy_forwards": 0, "updates": 0,
            "scope": "Saved TRAIN trajectories only. Softmax probabilities are action probabilities, not clinical probabilities. Later-step differences are descriptive arithmetic on diverged states, not counterfactual regret or causal attribution."}


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, allow_nan=False))
