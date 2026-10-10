# Saved audit of live generated contact-family desktop attempts

Three root-owned pcf-10/surface **TRAIN** attempts completed under the same public source, initial state, initial observed channels and goal. The result and supervision SHA-256 values equal each attempt's durable completion receipt. Every episode body matches its canonical JSON and episode ID; each native strategy matches its sealed action history. All three workers exited zero with no stop reason, cleanup error or unresolved PID. The sampled RSS limit was 1 GiB and the worker wall limit was 20 s; sampled peaks can miss transients.

| Method | Saved action history | Public goal | Return | New optimizer updates | Actor forwards | Elapsed | Sampled RSS peak |
|---|---|---:|---:|---:|---:|---:|---:|
| SEARCH | aspiration → probe | contacted and retained | +0.704 | 0 | 0 | 2.945 s | 254.7 MB |
| IL final32 | STOP | not contacted | 0 | 0 | 1 | 5.248 s | 288.0 MB |
| RL final32 | STOP | not contacted | 0 | 0 | 1 | 5.256 s | 288.7 MB |

SEARCH used 20 bounded planning transitions without a time or call cap and removed 1 mm³ in the generated source. IL and RL each made one verified final-checkpoint actor decision, then stopped without removal. Their saved authorship matches the published checkpoint file, parameter and training-lineage hashes; both records say completed 32 updates and zero inference updates. The backend release receipt binds the final freeze and completed pilot. This is an interactive **TRAIN** demonstration, not an additional held-out measurement or evidence that the learned methods generalized. The independent pilot's held-out result remains IL/RL 0/16 each and SEARCH 12/16.

A fourth, separate SEARCH attempt was performed to check the desktop camera fit after the UI patch. Its saved receipt also completed cleanly (2.952 s, 254.8 MB sampled peak). It has the same public initial state, sealed strategy, aspiration/probe actions and outcome as the first SEARCH attempt. Its episode ID differs because its planning record contains run-specific timing. The screenshot is at `build/episode-camera-fit-v1/live-probe-final.png`; this saved-result audit did not inspect its pixels or rerun the UI. It is a camera regression, not a timing replicate or another method comparison.

`audit.json` contains compact hashes, sizes, identities, outcomes and cleanup data for the original three attempts. `camera-audit.json` separately covers the fourth. No full result, checkpoint, medical image or screenshot was copied into this package. The scripts read saved metadata/result JSON, the published manifest and freeze; they did not deserialize weights, run a model or native transition, or operate the UI. The source closure was enforced by the completed backend controller, but the attempt directories do not themselves contain per-run source snapshots, so this audit does not claim independent byte-level reconstruction of executed source.

All four episodes are generated software evidence with patient admission and clinical validation false. Public goal contact and removal counts are not clinical outcomes or vascular injury measurements.
