# Spatial scan-conditioned policy: first bounded prototype

October 4, 2026. This implements the small 3D encoder recommended in `MASTER_PLAN.md` §9.4 while preserving the existing feature policies and historical studies. The immediate question is whether a policy can learn useful sequential decisions from a permitted scan across different synthetic anatomies. A procedural screen does not establish transfer to MRI, patients or clinical outcomes.

## Primary-source rationale

| Primary work | Relevant evidence and resulting choice | Limit |
|---|---|---|
| [Ghesu et al., multi-scale 3D landmark RL](https://pubmed.ncbi.nlm.nih.gov/29990011/) | Image-driven navigation can learn from volumetric neighborhoods; use actual 3D intensities and spatial features. | Landmark localization is not tissue removal or surgical safety. No reported performance is imported into this project. |
| [PerAct](https://peract.github.io/) | The authors learn action-centered features from voxel observations with behavior cloning. Spatially aligning candidate tool rays with learned features is a useful small-model design. | This prototype is a small CNN, not a reproduction of PerAct's transformer or tabletop benchmark. BC is not RL. |
| [SurRoL](https://arxiv.org/abs/2108.13035) | A surgical robot simulator separates tasks, instruments and learning algorithms. | Its robot manipulation tasks and transfer results do not validate glioma resection mechanics or neurological outcomes. |
| [DEX](https://arxiv.org/abs/2302.09772), [official implementation](https://github.com/med-air/DEX) | Demonstration guidance can address exploration in surgical manipulation. The repository generates demonstrations with scripted SurRoL controllers. Compare search-generated imitation with scratch RL and imitation followed by RL. | Search demonstrations here are simulated labels, never surgeon demonstrations. We do not implement DEX's particular algorithm. |
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html) | Actions change the observation distribution, so offline imitation can compound errors. Evaluate full held-out episodes, not just teacher-action accuracy. | The first bounded screen uses fixed BC data; it does not claim DAgger's interactive aggregation or guarantees. |
| [AWAC](https://arxiv.org/abs/2006.09359) | Prior data plus online updates is a substantive research direction, and transition from offline data to online RL is not automatically effective. | The initial algorithm is explicitly BC plus on-policy REINFORCE, not AWAC, PPO, or an offline critic method. |
| [Reverse Curriculum Generation](https://arxiv.org/abs/1707.05300) | Structured task difficulty can help exploration. Start with tiny paid-access branching anatomy and known small-problem competitors. | Do not adapt held-out anatomy difficulty after seeing results. Any later curriculum needs a separate frozen rule. |
| [Domain Randomization](https://arxiv.org/abs/1703.06907) | Vary the declared synthetic image formation and physical geometry, and test disjoint anatomy/tool combinations. | Randomization is an engineering robustness probe, not evidence of clinical transfer or calibrated anatomical uncertainty. |

These are targeted primary-source checks, not an exhaustive novelty review or reproduction. Paper abstracts/author project descriptions were inspected; no new numerical claim relies on an uninspected full method. The Ghesu author-hosted full PDF exceeded the browser fetch limit, so its primary abstract supports only the limited statement above.

## Observation and architecture

The primary track carries synthetic scan intensity, coverage/availability, scan-derived support, observed cavity and declared tool/access/budget state. Target and motor/language channels are unavailable unless an explicitly permitted scan estimator supplies them. Reference targets and hidden hazards belong only to outcome scoring. The observation adapter accepts no simulator, per-action removal volume, hazard integral, predicted return or postoperative field. Provenance strings alone cannot prove upstream behavior; counterfactual hidden-truth tests are required in the task.

The fixed six image channels each have value, coverage and availability planes, giving 18 CNN inputs. Unavailable or uncovered values are zeroed before convolution. Two 3D convolutions use 8 and 16 channels with ReLU. Five equally spaced entry-to-tip locations sample the feature grid using the declared physical affine. The implementation explicitly maps repository C,X,Y,Z to PyTorch's Z,Y,X grid coordinates, uses `align_corners=True`, and sets every sample outside the voxel-center bounds to zero with a separate in-bounds flag. A declared 64×float64-epsilon band in normalized coordinates snaps machine-roundoff corner errors to the exact boundary; this is not a geometric safety tolerance.

A shared 32-unit action head consumes sampled features, complete geometric tool parameters, procedure state and a coarse spatial summary. STOP has its own head. The critic also consumes the 2×2×2 pooled spatial encoding, rather than only the old six scalar state features. Physical tool lengths use a fixed 10 mm reference unit; angles use 90 degrees. Image intensities receive no policy-dependent normalization. Argmax chooses the first tied action, including STOP at index zero; ties are not evidence of learned stopping.

The source affine maps geometry into the existing image grid; the model does not select an anatomical tangent or infer a cross-patient coordinate convention. Physical geometry stays float64 through the observation boundary and affine conversion; proper re-expression of the same frame is tested within normalized model float32 precision. An ordinary CNN is not invariant to voxel reindexing, crop changes or image rotations. Five ray samples and coarse pooling can miss small structures and do not certify tool clearance. The independent geometry engine remains responsible for legal execution.

## First algorithm and accounting

The isolated module provides supervised cross-entropy on search actions and masked Monte Carlo policy gradients with a spatial value baseline. Complete episodes are required. Returns remain in the task's declared units. Actor score-function terms are summed within trajectories, weighted by gamma to the action time, then averaged across episodes. This targets start-state discounted return without upweighting short episodes. Value MSE and entropy use explicit per-action averages within episodes, then a batch mean; their weights, discount and global clipping are frozen by the experiment declaration. A gradient step records actual encoder/actor/STOP/critic norms and before/after parameter hashes.

An initial draft incorrectly averaged actor terms by each trajectory's length. Root/protocol review identified the bias before any screen ran. Two analytic gradient regressions failed as expected on the preserved draft; the repaired formulation is checked against unequal-length and discounted examples. The draft and failures are retained in `artifacts/spatial-policy-prerequisites-v1/`.

Rollout action selection and the loss-building autograd forward are separate calls and must both be counted. Search expansion, teacher generation, preprocessing, BC, RL, checkpoint selection, frozen inference and independent checks have separate costs. The runner must keep weights unchanged while collecting each on-policy batch and use a fresh optimizer for the BC-to-RL arm. Frozen evaluation uses no optimizer or gradient updates. Search uses only an observed-scan model on the same action set; withheld scoring truth must never become its lookahead oracle.

The prospective screen, anatomy splits, budgets and release authority belong to the separate experiment declaration. Single-gradient unit checks establish differentiability and data flow only. Unknown vascular anatomy, continuous tool trajectories, tissue forces, deformation, retraction damage and clinical deficits remain unmodeled. See [the surgical realism evidence note](rl-surgical-realism-evidence.md) for the separate mechanics requirements.

## Opt-in candidate context for the critic

The real-patient development runner can explicitly request
`SpatialPolicyConfig(critic_candidate_context=True)`. This version concatenates
the mean and componentwise maximum of normalized geometry and sampled ray
features across legal non-STOP candidates, followed by their count divided by
the maximum non-STOP inventory size. A STOP-only inventory contributes zeros.
Masked candidates and STOP placeholders contribute neither values nor gradients.
The actor and its initialization are unchanged; value gradients can also train
the shared image encoder through candidate ray features.

The option defaults to false, preserving v1 layer shapes, initialization and
architecture records/hashes exactly. The new architecture identifies itself as
`spatial-scan-ray-conv-candidate-critic-v2`; v1 critic weights are shape-incompatible
and cannot be silently reused. Default size grows from 24,331 to 30,827 parameters
when enabled. No throughput or planning improvement has been measured for it.

Controlled algebra tests demonstrate that changing permitted tool geometry can
change the new critic while leaving the old critic input unchanged. This is an
input-capacity check, not evidence of a learned return difference. Only current
legal candidates are summarized: tools unavailable now but useful after an
opening remain absent, so this does not prove a complete Markov observation.
That limitation requires a separately versioned full-catalog observation if it
affects actual patient decisions. No synthetic training or anatomy benchmark
was run for this change; unit derivative checks make no parameter updates.
