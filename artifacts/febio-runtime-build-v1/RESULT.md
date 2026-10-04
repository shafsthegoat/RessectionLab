# FEBio 4.13 local runtime result

One acquisition, one configure and one build/install completed on local arm64
macOS under the committed runtime proposal. No source patch, additional runtime
dependency, global installation or failed-run retry was used.

| Stage | Elapsed | Peak sampled process-group RSS | Outcome |
| --- | ---: | ---: | --- |
| Acquisition | 11.842 s | 62,832,640 bytes | Exact dependency hashes matched |
| Configure | 4.435 s | 136,151,040 bytes | Declared cache/private OpenMP verified |
| Build + install | 386.739 s | 664,715,264 bytes | Completed under 900 s / two jobs / 3 GiB |
| Version information | 3.502 s | 360,448 bytes | Reported 4.13.0; expected raw exit 1 |

The version command deliberately used `-info -norun -noconfig`. Tagged
`FEBioApp::ParseCmdLine` returns false for `-norun`, causing `main` to return 1.
The preserved generic supervisor therefore records a nonzero process outcome;
[its separate interpretation](version-01/interpretation.json) verifies the exact
version text without treating it as a successful mechanics solve. Sampling can
miss brief RSS excursions. The probe did not initialize the full solver library.

[Runtime identity](runtime-identity.json) binds the source tag/commit, actual
executable, twelve FEBio libraries, private OpenMP and verification receipts.
The executable SHA256 is
`00a3ad9ff8d28b2d2b8ba41fd63890f555ec9ff3e102c1c3eeef649cfec4fc9c`.
[Linkage inspection](linkage-review.json) found arm64 binaries and only private
FEBio/OpenMP or macOS system dependencies. All 6,164 original acquired source
and tool entries were rechecked unchanged after the build. Unresolved lazy
symbols and numerical behavior still require the analytical controls.

The source archive is 53,891,813 bytes, SHA256
`a1c4f65597c9b3f89a02daeeee6c3ae00d9f4bf64a85824b33e817043fd1ac66`;
2,410 entries expand to 74,394,570 bytes. Total downloads were 102,042,925 bytes.
Original notices remain in the isolated runtime. Their presence and upstream
terms were inspected for this local build; this is not an app redistribution
or legal-compliance conclusion.

The driver passed 19 focused controls, including actual wall/memory stops and
descendant cleanup, and independent review repeated them successfully. Review
found and repaired stage-order/parent-acceptance and child-loader-environment
gaps **before** the first acquisition. See
[the independent receipt](../febio-runtime-driver-v1/independent-review.json).

This receipt ends before mechanics execution: zero material/specimen/patient
models have run here, and no HBE response curves were opened. Geometry owns the
single separately released five-deck analytical run and its independent checks.
A successful runtime build establishes neither physical tissue validation nor
patient-specific material properties.

For publication use [the acquisition summary](acquisition-public-summary.json),
which omits transient public CDN signed-query parameters while retaining payload
hashes and the original receipt hash. Keep the original `acquire-01/acquisition.json`
and `worker-progress.json` local. Full downloaded upstream source/build trees
remain under ignored `data/optional-runtimes/febio-4.13/`; the earlier individual
source-inspection copies are also local research material, not files to vendor
into Git.
