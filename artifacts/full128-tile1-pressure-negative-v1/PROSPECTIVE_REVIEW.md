# Prospective full128³ generated tile1 feasibility control

At the prospective freeze, `tiled/` did not exist. The original full128³ tile4
failure in `artifacts/full128-tiled-feasibility-negative-v1/` remains separate
and negative:
the pressure guard stopped its forward at 2,394,734,592 sampled RSS bytes,
without an accepted output. This release does not reconstruct or replace it.

The proposed worker uses the **exact same original generated 128³ input**
(NPY SHA `daa9ffd646347a94c89afe8357b5eb321ecc56cf8b17116b12f45ebe43858712`),
checkpoint/model hashes, CPU FP32 architecture, complete feature-map
InstanceNorm, alias checks and safe `weights_only=True` load. The only model
operation change is `output_depth_tile: 4 → 1` in the tracked optional Conv3d
adapter. Its worker source diff from the original changes only the gate/tile
check and result metadata. The full128 output would be `[1,3,128,128,128]`.

The parent’s new identity-bound supervisor is copied from the independently
reviewed and successfully used 64³ v2 controller, with only the full128 gate,
scope and lock name changed. The launch gate prevents worker exec until
PID/start/PGID binding, synchronous bound-PGID inventory, host recheck and a
durable launch receipt. It uses identity-checked fail-closed finalization, no
bare process-group signal. The earlier 64³ generated tile1-vs-tile4 comparison
found byte-identical logits and a 16.8% lower sampled peak for tile1 in one
instrumented run. This motivates testing tile1 at full128; it cannot predict
the full128 outcome because the full decoder concatenation and complete
normalized feature maps still grow substantially with patch volume.

The old and new contracts match on **25 pinned numerical and resource fields**,
including the 3 GiB sampled process-group RSS cap, 120 s wall cap, 30 s
preflight, pressure-mask 2/4 abort, free-memory and swap bounds, input/model
hashes, expected output shape and no TTA/sliding window. The expired original
acquisition exception is replaced by one exact TractoInferno intake identity
from `acquisition-allowlist.json` (PID/PGID 92434, kernel start/command and
source/declaration hashes; RSS ≤512 MiB). The same live identity passed the
reviewed `slow_inventory` matcher at 48,608 KiB. If it ends, changes identity,
or exceeds its bound, preflight must defer; the controller never signals the
downloader. All other project compute remains blocking. Nine tiny generated
launch/failure tests pass; they do not load the model or patient data.

This is a **tiled-only resource feasibility** measurement. There is no
native full128 comparator, patient input, search/RL evaluation or clinical
claim. An independent exact-source review preceded the parent-owned single
model call recorded in `RESULT.md`. Its pressure failure remains negative;
resource caps were not raised and an unchanged call was not retried.
