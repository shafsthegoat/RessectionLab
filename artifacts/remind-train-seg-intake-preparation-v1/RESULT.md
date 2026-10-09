PREPARED, NOT RELEASED: ReMIND TRAIN SEG byte-intake extension.

Exact scope: 242 SEG series / 242 DICOM objects / 424,167,640 bytes, spanning all 79 existing frozen TRAIN people. No new patient, patient-level filtering, role reassignment, protected participant payload, or ReMIND-001 redownload. The completed raw MR/US receipt and its original no-SEG scope remain unchanged.

Source authority: official IDC v24 metadata projection plus 242 fresh verified-TLS official idc-open-data S3 series-prefix listings. Every object has an exact identity, URL, size and single-part ETag MD5. There are no multipart objects in this scope. Exact object-manifest SHA256: 87eb3b3fc1b78276f1d58ae873665d89b62e21f65244084cb77a7ee34b736525. All 242 source-index licenses are CC BY 4.0 with DOI 10.7937/3rag-d070, consistent with the same-day TCIA/DataCite rights snapshot. Retain the original source attribution and release bindings.

Metadata-only exposure: source series names/study labels and public object listings, with no DICOM body, DICOM header, segmentation values or annotation decoding. Study-description counts are 200 Preop and 42 Intraop; these describe source metadata and are not proofs of annotation availability at a surgical decision. The scope includes residual, prior-cavity and target-named annotations; all labels are unavailable to the preoperative actor, planner inputs or rewards, and remain unreviewed and unadmitted. Later intended-use review must establish timing, provenance and referenced frames. Existing cohort diagnosis and overlap restrictions remain intact. Unknown annotation ancestry is preserved without reviving a prohibition on generated/simulator training labels.

Prepared scope: proposed-scope.json, SHA256 d04c2581a37a089b2f74d26053be5ce50904bcad56df32072506b1ff8df059eb.
Prepared declaration: prepared-declaration.json, SHA256 2c1eed8fbf576d9b961b64bf8c2c68f7a11b1c316cb58cc13797e179ba808da8.
Runner: run-intake.py, SHA256 8afdb0ebeea18c3e1a487c0fb9f4d02d141a9165460d5fe5f35c20725bd56830.
Source-authority audit: source-authority-audit.json, SHA256 0fcbe8540ef5004e9e1d572e4f8586495eb0ed780ad753b9d1321aac039a340b.

Transport: narrow adapter of the completed ReMIND raw runner, using unchanged source snapshots of acquire_public_case, acquire_btc_case and real_intake_io. Two workers; verified TLS and hostnames; no redirects; exact URL/If-Match/length contract; strict Range resume; exact source size+MD5 before atomic publication; additional MD5/SHA256 receipt; immutable intent/HTTP/result history; at most three attempts per object across restarts; persistent host Retry-After cooldown; 100 GiB free reserve. Predecessor ReMIND and NFBS completion receipts are bound, and their queue locks must be free. No model-overlap exemption is included.

Zero-network controls PASS: valid check-only returns 242 files / 424,167,640 bytes / 79 people. Unreleased ordinary execution refuses before network access or transport/queue writes. A failing network sentinel received no call, and staged source file hashes remained unchanged. Receipt: preflight-controls.json, SHA256 6ca123bf09f1dbc05cdcbf0b6f2f1dedabdc8442825beef454247d40d547ee37.

Pending only bounded independent/root review and frozen scope/declaration binding before authorized payload launch. No payload worker has been launched, and no anatomy-QC pause is part of acquisition.
