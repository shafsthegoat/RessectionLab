# The guarded planner preserves the original four-case results

The frozen scorer with conditional checking reproduces all original balanced
policy actions and histories on TRAIN002/015/018/045. The independent saved audit
confirms no quality loss against retained exhaustive search. All four plans are
sealed before uncached replay; each separate complete-tool evaluator accepts.

| TRAIN case | Target cells removed | Outside-target removal, mm³ | Online previews | Source through plan, s |
| --- | ---: | ---: | ---: | ---: |
| 002 | 0 | 0 | 117 | 9.221 |
| 015 | 58 | 0 | 197 | 3.242 |
| 018 | 0 | 0 | 75 | 2.473 |
| 045 | 51 | 0 | 187 | 5.166 |

There is a cost: four final STOP checks expand the unqueried catalogue, adding
344 previews. Online checking increases from the original 232 to 576 (+148.3%).
The current-tool branch triggers zero times because every retained positive
action keeps the same tool. This confirms the saved prediction and guards against
the premature STOP failure measured on050, without changing these four routes.
Historical wall times are not a matched speed comparison; the table reports only
this run's costs. The previously measured050 quality/cost improvement remains a
separate development-case result, not a universal speed advantage.

The source, physical actions, complete histories, outcomes and all metrics match
the original balanced run, apart from the explicitly changed outer policy
identity. Quality comparison retains path length, movement and switch counts,
contact bounds, full-target denominators and null clinical probabilities. The
only positive removals remain 58 and 51 cells, each less than 1% of its supplied
whole-tumor annotation. Existing STOP cases are preserved rather than omitted.

The parent completes in 49.228 s at 737,296,384 B sampled peak with 5,065,107 B
output and clean cleanup. Planning plus replay totals 1,152 previews, 58
transitions, 58 ridge scores and eight source visits. One existing model is loaded;
there are zero fits, optimizer updates or neural forward calls. Four generated
runner controls pass before the acquired run, and source/terminal independent
checks agree. Original fixed-four admission is preserved; the outside-fit
development-transfer capability is deliberately not used for fitted subjects.

This is integration regression on previously fitted anatomy, not new-patient
generalization, RL improvement or physical validation. No SELECT/EVAL patient,
new dataset or material property is introduced. The separate complete-route
failure remains unresolved and motivates the single longer-horizon experiment.

The index preserves compact sources, controls, complete plans, metrics and
certificates. Original immutable policy/helper/supervisor are referenced by hash,
and duplicate inventories, raw patient data and model weights remain outside Git.
Canonical runtime is pinned to `e8fe1bb`.
