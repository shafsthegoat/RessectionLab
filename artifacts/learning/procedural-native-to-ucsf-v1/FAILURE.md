# Preserved failed development attempt

This attempt stopped at the first adaptation validator after 158.44 seconds. Shared procedural training, search/frozen inspection and the three scratch runs produced raw records. Adapted seed 11 was rejected before optimizer creation; adapted seeds 23 and 47 were not attempted. No candidate freeze or independent final-candidate geometry audit completed. Final and stress worlds stayed closed. These partial records are not a validated comparative result.

The saved world manifests match the declaration. The failure was a direct Python comparison between three tuple-valued generator fields and their JSON list representations, in both optimization and selection panels. Canonical hashes of the complete panels match. The earlier test-only path constructed both sides in Python and missed this public-declaration serialization boundary.

The failure, checkpoints and exact source remain intact under commit `dfab4c4adaba978a687c3e80cd7fffff4a071b6b` and design hash `sha256:3abf10162ed9d096a7e21ec84a87c1a88ebf12bb074345a78040dfdebb828ac9`. See `experiment-status.json`, `comparison/status.json` and `world-panel-failure-diagnosis.json`.

The proposed second attempt changes the serialization check and adds an actual patient-world preflight before any gradients. It retains the design's geometry, sources, seeds, rewards, policies and budgets. It requires fresh pretraining and all comparison arms, a new output directory, tested committed source and a separate execution release. No model setting was selected from this attempt's scores.
