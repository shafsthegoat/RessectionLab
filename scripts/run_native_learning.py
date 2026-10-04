#!/usr/bin/env python3
"""Native source-cell development CLI using the shared frozen experiment runner."""
from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import threading
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_patient_learning import run_experiment
from resectionlab.imaging import load_case
from resectionlab.learning import TrainingConfig, train_patient_policy
from resectionlab.native_simulation import make_native_patient_simulator


def run_native_experiment(case: Any, output: Path, *, config: TrainingConfig,
                          seeds: tuple[int, ...] = (11, 23, 47),
                          candidate_count: int = 4, max_steps: int = 3,
                          max_actions: int = 7, evaluate: bool = False,
                          trainer: Callable[..., Any] = train_patient_policy,
                          cancelled: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """Independent native geometry is checked even when final worlds stay closed."""
    synthetic = case.metadata.get("is_synthetic") or (case.source_refs and
        all(ref.provenance == "simulated" for ref in case.source_refs))
    return run_experiment(lambda: make_native_patient_simulator(case,
        candidate_count=candidate_count, max_steps=max_steps, max_actions=max_actions,
        cancelled=cancelled), output, config=config, seeds=seeds,
        counts=(3, 2, 3, 2), source_case=case,
        independent_patient_count=0 if synthetic else 1,
        evaluate=evaluate, trainer=trainer, cancelled=cancelled)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--wall-seconds", type=float, default=30.)
    parser.add_argument("--gradient-steps", type=int, default=32)
    parser.add_argument("--algorithm", choices=("reinforce", "ppo"), default="reinforce")
    parser.add_argument("--evaluate", action="store_true", help="Open final/stress worlds only after freeze and native geometry audit")
    args = parser.parse_args()
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    config_type, trainer = TrainingConfig, train_patient_policy
    if args.algorithm == "ppo":
        from resectionlab.learning_ppo import PPOConfig, train_patient_ppo
        config_type, trainer = PPOConfig, train_patient_ppo
    settings = config_type(max_environment_steps=256, max_gradient_steps=args.gradient_steps,
        max_wall_seconds=args.wall_seconds, hidden_features=16, max_episode_steps=4,
        episodes_per_update=2, checkpoint_interval=2)
    print(run_native_experiment(load_case(args.case_bundle), args.output, config=settings,
        seeds=tuple(args.seeds), evaluate=args.evaluate, trainer=trainer, cancelled=stop.is_set))


if __name__ == "__main__":
    main()
