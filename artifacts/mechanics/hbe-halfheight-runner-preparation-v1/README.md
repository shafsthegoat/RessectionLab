# Half-height runner source preparation

This is source preparation for the fixed lower-half-height/full-cross-section comparison. No native tool, actual mesh extraction, solver output replay, or measured response member was opened by this lane. The four axial N8/N12 S60 reference cases and original full-model records stay fixed by declaration `0df5587a…`. The abandoned Cartesian-quarter scaffold is preserved separately and is not an executable module.

The runner uses the established supervisor, process-group output watcher, terminal solver wrapper and exact solver-only backend transform. It has no fitting, calibration or response-reader call path. Every full-reference readout, execution receipt and six primitive files is bound before either phase. Source checking requires the exact 14 members listed in `source-files.txt`, the Git archive commit comment, actual imported module paths and the current interpreter bytes.

Two releases are required. `prepare` performs only two pure saved-mesh extractions and writes four decks in the new study's `preparation/` tree. It has a separate 60-second/3-GiB cap and zero native calls. `solve` requires the accepted preparation result, same source/runtime/declaration, successful bounded supervision and every frozen mesh, reconstruction and case file. Its supervised worker regenerates these pure inputs in memory and compares exact JSON/XML/loading before its first solver invocation. It never writes to the accepted preparation tree.

The native phase allows four calls at most, 90 seconds each, 420 seconds aggregate including readout and revalidation, 3 GiB sampled process-group RSS, one numerical thread, 128 MiB active output and 512 MiB total new-study output. First native or equivalence failure terminates the phase. Partial primitive logs and actual invocation records survive; later cases remain unexecuted. Each completed case must pass the independently implemented half/full readout before the next case begins. There is no automatic finer-mesh promotion or physical-validation success label.

`owner-checks.json` records 14 mocked orchestration controls. The initial incomplete XML test fixture was rejected by the real backend parser; `initial-check.json` and the original test snapshot preserve that setup failure. Only the fixture hierarchy changed. Helper and independent review receipts are separate.

## Concrete release handoff

After source review and root's commit, create a prefix-free Git archive of exactly `source-files.txt` and extract it to a fresh ignored source directory. Bind all twelve script files at their actual extracted paths. `source_commit` must equal the archive's Git PAX comment. Bind the archive, interpreter and unchanged accepted backend profile shown in `prepare-release-draft.json`. Its unsupported draft schema and `authorized:false` deliberately refuse execution.

Root's separately authorized final preparation release must use schema `hbe-halfheight-release-v1`, `authorized:true`, `phase:"prepare"`, exact declaration binding, actual committed source/archive/module bindings, interpreter and backend profile. Invoke the frozen extracted `scripts/mechanics_hbe_halfheight_experiment.py` with:

```
<bound-python> <frozen-runner> --root <repository-root> --phase prepare --release <root-relative-final-release> --release-sha256 <exact-sha256> --execute
```

Omitting `--execute` performs metadata/hash checks only. The native solve release uses the same schema and source closure with `phase:"solve"` and adds `preparation:{path,sha256}` pointing to the accepted `outputs/mechanics/hbe-01-03-halfheight-equivalence-v1/preparation/result.json`. Its CLI differs only in phase and release binding. Each phase creates an exclusive marker and fresh directory; an attempted phase cannot be silently restarted.
