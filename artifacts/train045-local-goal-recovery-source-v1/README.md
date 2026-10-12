# Reverse-curriculum checkpoint recovery

Frozen and released October 12, 2026; this source record does not claim a completed training run. Source index SHA-256: `54543f2b71a8262c4340dbb3aa71c54ca4d5f7d3df0b63d16392d64443298d28`.

Resume the verified reverse512 checkpoint (353 actual Adam updates) for 512 new PPO decisions and one fresh, independently checked full-start evaluation at model-clock1024. Preserve the completed uniform1024 branch. The original reset schedule continues at offset256; physical episode and RNG restart, with six old live decisions retired. This is checkpoint recovery, not seamless episode continuation. At least320 decisions and40 updates in the failed unsaved tail remain paid outside the restored model lineage. See the [terminal failure](../train045-local-goal-curriculum-interruption-v1/RESULT.md).

The only implementation difference admitted from the interrupted run is the tested [input-memory change](../train045-local-goal-input-memory-v1/RESULT.md). Task, rewards, actions, optimizer settings, batch4, frozen encoder and evaluation rules remain fixed. Its assembly-only RSS result does not guarantee the full worker will stay below4GiB.

Six generated recovery controls pass in1.025 seconds; the supervised batch takes2.245 seconds, peaks at351,535,104 bytes and reaps its child. They check failed-attempt admission, checkpoint/schedule continuation and cost accounting without acquired patient payloads or native simulation calls. The freezer authenticates the failed parent, source postmortem, memory receipt, checkpoint and preserved uniform result. Runtime still must complete and pass independent replay, full-tool geometry and local reward reconstruction.

The single recovery is capped at4GiB RSS, two CPU threads,4,470 worker seconds/4,500 parent seconds,180,000 previews,1,400 transitions and16MiB output within the existing192MiB stage allowance. No new dataset, dependency or patient split is introduced. Weights and bulk traces stay outside Git.

Storage measurement before release: extension high-water growth1,853,288,448 bytes; free space352,230,809,600 bytes. The archive/Git reservation is amended from80 to96MiB within the unchanged10GiB/50GiB user policy, preserving the prior20,480-byte internal allowance overrun. All open reservations total311,427,072 bytes.

This remains one reused TRAIN045 anatomy with annotation-assisted simulator experience. It does not establish physical, neurological, clinical or held-out-patient benefit. The imitation follow-up stays unreleased.
