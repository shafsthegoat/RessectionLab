# Completed native axis preflight V2

Read `RESULT.md` for the cost finding and its limits. `cost-summary.json` contains exact source-derived values; `report-source.json` binds the original inputs. The execution baseline is retained separately at `artifacts/validation/native-axis-public-v2/execution-baseline.json`.

The five `profile/*/episode.json.gz` files preserve all original episode bytes, including native cell histories. Their raw JSON counterparts remain unchanged locally and are narrowly ignored by Git. `compression-roundtrip.json` records byte and JSON equivalence, both hashes, sizes, and byte-identical reproduction of the report from a private copy with only compressed episode files. No episode or gradient was run while reporting or compressing.

`report.py` accepts either raw JSON or gzip and refuses disagreement when both exist. Run it with the existing Python environment; it reads completed evidence and writes only `RESULT.md`, `cost-summary.json` and `report-source.json`. `pack-evidence.py` is a separate preservation/reproduction utility; it requires the retained raw files and never deletes them.

The independent completed-artifact audit is at `artifacts/native-axis-v2-result-audit/audit-02.json`; its 25 adversarial tests and lossless-compression verification are recorded in that directory's `verification.json`. Full local auditing also uses the preserved initial checkpoint and verified source bundle, which are excluded from Git. A Git checkout can reproduce the cost report using the compressed episode evidence without those large/local model inputs.

Original launcher/worker/profile/configuration records and the immutable executed source were not edited. The failure from V1 remains a separate retained attempt. No training budget or learning-method conclusion is declared by this preflight.
