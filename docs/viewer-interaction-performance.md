# Viewer interaction performance — 15 October 2026

Implemented in the existing `codex/patient-specific-planner` checkout, against baseline `67ce3216`. Source data, segmentation, physical coordinates, original picking surfaces, and final rendering quality are preserved. This is a software responsiveness evaluation.

## Results

The final production Electron build was compared with an archived copy of the baseline source on the same Apple M2 Pro / 16 GB Mac, macOS 26.5.2, 2560 × 1440 external display at 100 Hz. The application window was 1600 × 1000. Both runs used the same sub-06 snapshot (`be803617eed66a7ab9321cda5fb8b8d783b02560f41a88504ed217b682a45c2c`), input paths, layouts, and 38-layer stress configuration.

Recurring gestures have five measured repetitions after warm-up. The first brain-region load is one separately measured **cold residency / warm filesystem** operation per version. It is not a cold-disk benchmark. Source/renderer startup and screenshot capture are outside recurring gesture timings.

| Measure | Baseline | Updated |
|---|---:|---:|
| Zoom: median CPU frame-submission interval | 92.0 ms | 10.1 ms |
| Zoom: p95 CPU frame-submission interval | 104.1 ms | 20.3 ms |
| Slice scroll: median CPU frame-submission interval | 94.9 ms | 19.9 ms |
| Slice scroll: p95 CPU frame-submission interval | 109.7 ms | 21.2 ms |
| First brain-region load, request to ready | 9.14 s | 4.44 s |
| First brain-region load, asset bytes read | 511.4 MB | 208.9 MB |
| First brain-region load, asset reads | 174 | 156 |
| Canvas replacements during that load | 2 | 0 |
| Navigation available during that load | No | Yes |
| MRI slice redraws during measured camera-only rotation / pan / zoom | Previously all visible panes every frame | 0 |

The biggest improvement is removal of the long zoom/slice backlog and of the blank viewer during new-layer loading. Ordinary rotation and panning already used an interactive render path; their measured submission cadence is approximately unchanged. Sidebar scrolling submits no anatomy frames.

**Timing boundaries matter:** CPU submission is not GPU completion or presentation. Asynchronous GPU fences give a completion upper bound; Electron frame subscriptions measure captured frame delivery and have observer overhead. Raw traces retain all gaps and stalls. The summary separately reports active-gesture capture intervals and whole-scenario intervals, including the deliberate quiet period before refinement. It does not silently discard long frames. The capture-free rotation control is retained in each raw report.

## Acceptance status and remaining limits

| Goal | Final evidence | Status |
|---|---|---|
| Roughly 60 displayed FPS; p95 interval ≤33 ms | Active capture p95: rotation 49.4 ms, pan 50.8 ms, zoom 41.2 ms; CPU submission p95 ≤21.2 ms | **Not established by displayed-frame capture** |
| Slice response p95 ≤100 ms | Input timestamp → first applicable GPU completion p95 37.9 ms | GPU completion target met; presentation latency is not directly established |
| Resident toggle response p95 ≤100 ms | Median 48.8 ms, p95 103.7 ms over 10 changes | **Narrow miss**; an already submitted full-quality frame can delay a new toggle |
| Full-detail restoration p95 ≤300 ms | Rotation 257.2 ms; pan 274.6 ms; zoom 260.6 ms; slices 250.5 ms; 38-layer rotation 268.9 ms | Met in these scenarios, including 150 ms quiet time |
| New layers keep navigation usable and avoid blanking | Same canvas, interactive frames during load, inline pending state, successful retry | Met; pending-label latency was functionally checked, not independently timestamped |
| Dense configuration ≥30 FPS | 38 selected layers; median active capture interval 31.5 ms, p95 49.6 ms | Median exceeds 30 FPS; a sustained 30 FPS floor is not established |

The remaining bottleneck is a full-quality volume pass already in GPU execution: it cannot be preempted when new input arrives. Progressive, cancellable tiled refinement is a possible follow-up. The current bounded queue prevents additional stale frames from accumulating. This implementation makes no universal FPS claim for other datasets, screens, or GPUs.

## Implementation

- Wheel, slice, and appearance interactions keep a gesture alive through 150 ms of quiet. Pending refinement is superseded by new movement. Camera state is persisted independently of visual refinement.
- Adaptive offscreen 3D rendering targets a 16.7 ms budget. The shared canvas and source slices keep stable resolution. A nonblocking GPU fence bounds in-flight work; pending state is coalesced.
- Source slices use explicit offscreen caches. Camera restoration does not invalidate them. Pane bounds are invalidated only by layout, resize, or ancestor scrolling.
- MRI slices and volume rendering share one scalar GPU texture. At the 439 × 439 × 361 source grid this avoids a duplicate 278 MB scalar texture, before driver overhead.
- Label membership work is skipped when successive ray samples have the same packed value, preserving entry-boundary compositing and isolation semantics. Final output is unchanged.
- Native layer residency is keyed by source identity and revision/preview generation. Missing layers load serially; latest visibility is retained. Failure keeps existing anatomy and provides Retry. Source changes reject stale responses.
- Label packing runs in a worker. Bounded copies preserve source arrays and yield between large blocks; only preparation copies are transferred. Existing meshes, image buffers, shared categorical arrays, and transfer descriptors are reused. Backend cache publication is transactional and its transfer paths remain retained until invalidation.

## Correctness and regression evidence

- `pnpm test`: **595 passed** — 123 main-process, 351 renderer, 121 viewer tests.
- TypeScript checking and the production Vite build passed. The existing bundle-size advisory remains.
- Focused Python coverage: **57 passed**, including anatomy, imaging, workspace persistence and patient workspace tests.
- Live failure injection: canvas preserved after a failed transfer and retry; rapid selection retained its latest visibility; pending source switch rejected stale results.
- Twenty resident visibility changes caused **zero asset reads**, with stable GPU resource counts.
- The real Electron/Python correction verifier passed preview, undo/redo, original comparison, rejection, accepted revision, Save, and reopen on a disposable APFS copy. The original sub-06 snapshot remains revision 5.
- The baseline and updated settled default 3D image crops were **pixel-identical**. Slice crops differed by at most one 8-bit channel level, with mean absolute channel differences below 0.00002. This is a scoped screenshot comparison, supplemented by coordinate, mask, opacity, overlap, source-picking, registration, and replay tests; it is not an exhaustive visual-equivalence proof.
- CT/MRI switching, scan volume on/off, four-view layout, 38 layers, opacity changes, and the 1050 × 720 window were exercised in Electron. Unit coverage includes clipping, opaque/zero-opacity isolation, categorical overlap, stale identities, cancellation, cache invalidation and resource disposal.

## Reproduce and inspect

Build before running the benchmark; measurements should run alone, without another GPU workload. The benchmark disables Electron background throttling and explicitly focuses its window. It neither saves nor edits the supplied collection.

```sh
cd desktop
pnpm build
INTERACTION_OUTPUT=../outputs/interaction-performance/recheck \
  pnpm exec electron electron/benchmark-interactions.cjs \
  --anatomy /path/to/patient-anatomy.json

# Additional failure/retry/source-switch validation:
INTERACTION_VERIFY_ONLY=1 INTERACTION_OUTPUT=../outputs/interaction-performance/lifecycle-recheck \
  pnpm exec electron electron/benchmark-interactions.cjs \
  --anatomy /path/to/patient-anatomy.json
```

From the repository root:

```sh
.venv/bin/python scripts/summarize_viewer_performance.py \
  outputs/interaction-performance/before-final/report.json \
  outputs/interaction-performance/after-verified/report.json \
  outputs/interaction-performance/comparison.json
```

- [Comparison JSON](../outputs/interaction-performance/comparison.json)
- [Baseline raw trace](../outputs/interaction-performance/before-final/report.json) and [updated raw trace](../outputs/interaction-performance/after-verified/report.json)
- [Failure/retry/lifecycle results](../outputs/interaction-performance/lifecycle/report.json)
- [Correction/save/reopen results](../outputs/interaction-performance/corrections/report.json)
- [Pixel comparison](../outputs/interaction-performance/pixel-comparison.json)
- [Before recording](../outputs/interaction-performance/before.mp4) and [after recording](../outputs/interaction-performance/after.mp4)

Recordings are separate visual demonstrations with capture overhead; they are not the timed benchmark. Earlier calibration attempts are retained locally but are excluded from the comparison. Timing artifacts and patient data are not added to Git.
