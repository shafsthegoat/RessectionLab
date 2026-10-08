# Generated native opening learning

The October 8 instruction permits synthetic data and simulator-generated
experience for research training. This narrow experiment uses the existing
`make_native_opening_task` and native resection engine. It contains six tissue
cells in a 9×9×7 grid, two distal target cells, a shaft obstacle and two
complementary tools. A paid opening makes the distal cut possible. The existing
finite-tree check gives a best geometric return of 1.1; either tool alone and
immediate greedy search give zero. These are mathematical task properties, not
patient or tissue-mechanics measurements.

The task preserves removed tissue, retained contact, connected cavity, current
tool, remaining action budget and complete-tool constraints. Each action is a
certified insertion/cut/withdrawal macro-stroke. It does not simulate arbitrary
in-tissue tool poses, force, deformation, hemostasis, skull opening or neurological
injury. Partial-contact cells remain occupied and receive no removal credit.

The existing spatial CNN receives six channels with coverage and availability,
the full tiny image, committed simulated cavity and tool/ray geometry. The
generated scan uses the declared 0.2/0.8 tissue/target signal; its nominal target
is the fixed 0.5 threshold. That rule is not an MRI tumor estimator. Hidden
reference arrays, per-action removal sums and evaluator rewards do not enter
actor or critic observations. Functional channels remain unavailable. The DTO
name `observed_cavity` means simulator-observed state in this experiment; its
dataset lineage is explicitly `simulator_generated`, never a recorded operation.

The prospective declaration fixes seed 11, the existing candidate-aware critic,
16 BC updates and 16 scratch REINFORCE updates from four complete episodes each.
Both models start with exactly the same fresh weights and separate Adam
optimizers, learning rate 0.001. The corrected episodic REINFORCE objective,
physical returns, entropy 0.01, value weight 0.5 and global clipping norm 5 are
unchanged. Initial and fixed final checkpoints are reported without selection.
There is no population training, held-out anatomy or patient access in this
credit-assignment microtask.

STOP, three fixed-seed random legal episodes, the untrained policy and immediate
greedy search are reference points. Complete depth-two nominal search is the
competent search comparator and BC teacher. It retains negative opening prefixes
and labels every reachable nonterminal observation, including off-optimal cavity
states. Conflicting labels on identical observations cause failure. Its costs
are recorded once and explicitly attributed to both the search result and shared
offline teacher generation. Each labeled state additionally requires an actual
root-prefix plus optimal-completion replay, independent native audit and exact
observation/action match; this validation cost is charged offline. Every actual evaluation and RL collection episode
must terminate and pass the independent native geometry/cell/reward audit.

Online methods have one uninterrupted 10-second/2,048-native-preview guard over
cloning, planning, decisions, replay, successor inventories and durable terminal
export. Search model calls and native preview entries are distinct counters.
Shared initial preparation, offline training and independent audits are reported
separately. All costs still fall inside the whole 300-second, 2-GiB worker
envelope on one CPU thread. Cooperative guards and sampled RSS are not guarantees
against transient resource overshoot. No retry or cap extension is allowed.

Before that experiment, `--profile` performs exactly one declared opening→cut
episode and one BC gradient using its independently audited actions. It has a
60-second envelope; those weights are discarded. This checks execution and
cost only, and cannot choose a winning training configuration.

The three spatial loss/update APIs now require an explicit
`GeneratedDevelopmentContext`. It binds one generated source, horizon two,
declared task identity and declaration hash. DTO source/track/horizon are checked
before loss forwards; the runner separately verifies task/reward identity, which
an observation cannot establish by itself. Losses bind parameter bytes, actual
parameter objects, architecture and context, and can be consumed only once.
Optimizer membership must match exactly before backward. Historical patient,
PPO, model-loading, structural-evidence and clinical permission gates remain
closed unless separately admitted. This context is a research-use contract, not
an authentication proof against arbitrary Python code.

The runner preserves failed attempts, full method denominator, partial committed
history, immutable input/source snapshots, actual gradients, checkpoint hashes,
all episode traces and independent audits. Null partial outcomes cannot become
STOP scores. The first result may be negative; successful same-fixture learning
does not demonstrate surgical utility or transfer. BTC TRAIN05/16/20/22/25/28,
SELECT26/27, transfer29/31 and UPenn external roles remain unopened and unchanged.
Physical measurements and held-out real patients remain separate future
evaluation requirements.
