# Independent BTC spatial acquisition review

The final independent review passed **29 offline tests in 0.12 seconds**. No
image or metadata request, socket connection, patient reconstruction, training,
or app execution occurred. Test bodies are small constructed byte strings;
they are not substituted for any source imaging.

The cohort manifest remains byte-identical to commit `9881c17`, SHA-256
`962d964e1d71427f3625cdbebc0f7e4759e5d2345d8f95cb211ed45810ed2985`.
PAT22/PAT25 remain population-training development cases; PAT26/PAT27 remain
checkpoint-selection development cases. PAT29/PAT31 retain their unopened
frozen-development-transfer role. They cannot enter through the command's
allowlist or a forged supplied verified manifest. The original PAT05/PAT16/
PAT20/PAT28 acquisition manifests still match their predeclared byte hashes.

The initial review found and retained three failing tests: once a SHA-256 was
known, image reacquisition accepted a fake response with an incorrect or absent
S3 version, or a final URL naming PAT29. The original pending-checksum path
already rejected these before reading the body. A separate offline construction
also demonstrated that Python's default redirect handler would create a
follow-up GET before the previous post-response URL check could reject it.
No such GET was sent.

The owner's repair routes both image states through the same exact URL/version/
length checks before body reads, and verifies SHA-256 when available plus the
annex MD5 before publishing the file. Its default BTC transport rejects redirects
before following them. Independent tests cover all five redirect status codes
301/302/303/307/308 to both sealed subjects, installation of the refusal handler
for pending images, known images and metadata, corrupted bodies, bounded size,
and preservation of existing local files.

`receipt.json` records source hashes, the retained initial failures, final test
results and review limits. Initial and final source copies, logs and JUnit files
are preserved. The final source remained unchanged during the independent run.
The owner separately reported 135 passing combined acquisition/preparation and
review checks; that execution is outside this receipt's independent commands.

Run the independent checks with:

```sh
.venv/bin/python -m pytest -q tests/test_btc_spatial_acquisition_review.py
```

This clears the bounded acquisition implementation, not anatomical or planning
eligibility. Source metadata remains archival cohort-selection evidence, with
unknown preoperative availability; no downloaded image acquires reviewed brain
support or cortical access merely by passing transport checks.
