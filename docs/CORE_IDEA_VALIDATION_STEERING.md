Reprioritize RessectionLab around validating the core idea as quickly and credibly as possible.

The immediate question is: **Can a model learn useful surgical-planning behavior from real patients and apply it to new patients, with a measurable advantage over a reasonable search baseline?**

We need an end-to-end experiment before investing further in surrounding infrastructure. Preserve the existing code, data, patient splits, negative results and geometric baseline. Read this alongside the existing project specifications; do not create another giant plan.

1. **Identify the shortest credible experiment.**
   Inspect what already works. Choose one clearly defined planning task that the current system can support. State its observations, available actions, success criteria, failure conditions and limitations. Resolve any action-coverage problem that makes the target unreachable before training.

2. **Validate the environment only to the level that this experiment requires.**
   Check tool geometry, coordinates, state transitions, rewards and failure handling. Confirm that the agent cannot earn success by exploiting missing collision checks, incorrect tissue accounting or inaccessible information. Add tests for substantive risks and observed bugs; avoid building a general testing framework.

   Begin small learning-pipeline experiments once these checks pass. Full surgical simulation is not a prerequisite for testing whether the learning pipeline works. However, a geometric-only experiment must be labeled as such and cannot prove realistic cutting, retraction or tissue-damage prediction. Keep the necessary physical-validation work focused on one narrow, measurable interaction.

3. **Use real patient data and realistic inputs.**
   Preserve the existing TRAIN/SELECT/unopened patient assignments. Use only information available at the intended time of deployment. Keep supplied annotations and scan-only inference clearly distinguished. Do not expose hidden simulator properties, future observations, evaluation labels or postoperative information to the policy. Do not use synthetic patients as evidence of patient generalization; analytical fixtures remain acceptable for software checks.

4. **Get a small RL experiment running early.**
   Start with one suitable algorithm and one modest model using existing components. Demonstrate that parameters actually update, behavior changes and learning produces useful improvement. Compare against random legal actions and a reasonable search baseline. Include imitation learning when existing search demonstrations make it inexpensive, and distinguish imitation gains from gains produced by RL.

   First establish that learning works on a development case. Then test population training and transfer to separate patients. Report immediate predictions separately from predictions improved through patient-specific adaptation.

5. **Make the comparison fair and informative.**
   Methods must receive the same permitted information and operate under clearly reported budgets. Measure task success, invalid actions, relevant tissue-interaction costs, route quality and runtime. Include preprocessing, search and adaptation costs where applicable. Use independent replay or evaluation to check proposed plans.

   Training reward alone is insufficient evidence. Before claiming a reliable advantage, check repeatability across seeds and performance on patients not used for training or model selection.

6. **Optimize the measured bottleneck.**
   When an experiment fails, identify whether the problem is the task definition, action space, observations, reward, implementation, optimization or computational cost. Profile before rewriting. Make one justified change and rerun the relevant comparison. Keep the configuration budget small and record negative results.

   Seek the best demonstrated tradeoff under the stated budget. Do not claim global optimality or assume RL must beat search. If search wins, report where it wins and investigate whether RL offers another useful advantage, such as faster initial proposals or reduced adaptation time.

7. **Defer infrastructure that does not unblock this experiment.**
   Defer UI polish, packaging, generalized orchestration systems and broad refactors. Reuse existing simulation frameworks and execution tools. Keep documentation and provenance sufficient to reproduce the work, without producing repeated plans, manifests or reviews that do not address a concrete risk.

   Give agents narrow roles in environment correctness, training/baselines and independent evaluation. Integrate their work promptly into a runnable experiment.

8. **Deliver evidence at each checkpoint.**
   Show the exact task and patient roles, what actually ran, learning curves, a compact baseline comparison, failure examples and the next single experiment. Clearly distinguish completed results from preparation.

The next milestone is a reproducible demonstration of whether the learning approach works on the defined task, followed by honest testing on new patients. Physical fidelity, generalization and surgical usefulness remain separate claims that each require evidence.

Stay local-first and avoid paid infrastructure or billing changes. Commit coherent increments under skamal23 <sayemkamal12@gmail.com>, with Co-authored-by: shafsthegoat <shafrir.p@gmail.com> on every commit.

Start by giving a brief assessment of the smallest remaining blockers, then execute the first meaningful experiment.