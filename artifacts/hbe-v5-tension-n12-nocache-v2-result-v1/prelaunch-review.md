# Independent exact-byte review: HBE v5 ordinal-8 no-cache v2

**GO for the exact final candidate release bytes at HEAD `0ece80d179e28f183b379ba2298961d0a2fd2c18`.** This is a source, release and numerical-specimen input review, not a native execution result. Root must take a fresh direct host sample at launch and allow the launcher to complete its full F_NOCACHE predecessor validation and unchanged 45% gates. A failed gate remains terminal with no retry or cap change.

| File | SHA-256 |
| --- | --- |
| `build/hbe-v5-tension-n12-s60-nocache-v2-release-prep/INNER_TEMPLATE_NOT_RELEASED.json` | `8623b67ced7fef2e7cde5ae6d821225d0927b446083987d8ed7610ab73a38254` |
| `build/hbe-v5-tension-n12-s60-nocache-v2-release-prep/OUTER_TEMPLATE_NOT_RELEASED.json` | `da7eb15e65ae7ea7555c60c4ab3ad8c95b5fd25ca3fa36e69bfd67963d04f223` |
| `build/hbe-v5-tension-n12-s60-nocache-v2-final-candidate/inner-release.json` | `7f41865a1bfef4a785b4be262832af6d92d79472abd808a905773dc7ddc181d8` |
| `build/hbe-v5-tension-n12-s60-nocache-v2-final-candidate/outer-envelope.json` | `2fc24ff3c33923ba33e16ae0842ceb97c1d89b6ede74b2b0e733b0127f74feb7` |

I reconstructed the fresh inner template exactly from the previously reviewed original ordinal-8 template with only its source commit changed. The final inner JSON differs from that template only by `status=root_released_one_native_call`. The final outer JSON differs from the fresh outer template only by `status=root_released_one_native_call_with_declared_io_policy_v2` and its exact inner path/SHA binding to the final candidate above. Canonical JSON bytes matched these constructions. The actual v2 launcher's read-only `_read_exact_release` parser accepted the exact outer/inner pair.

All 20 frozen HBE source bindings and three v2 extension bindings matched both working bytes and current-HEAD Git blobs. All eight small predecessor receipt files matched the exact release hashes; ordinal 1's original compression-N12 receipt remains `failed_or_incomplete` and is admitted only through its separate reviewed saved-output supplement. No coarse row was retried. The current check did not reread the multi-gigabyte predecessor raw outputs; that full F_NOCACHE verification is deliberately retained in the root launcher and must pass before native reservation.

The seven bound row metadata/source-map/profile/runtime/source-deck/mesh files matched their SHA-256s. Independent pure-adapter replay reproduced the exact 61-frame, 60-step adapted deck bytes, SHA-256 `6dbf980c8d60309f62d4d6e8e395d881b4bacfd1be73796eedfad6112f9505a9`, and the complete adapter receipt. The source/mesh topology replay found 3,199 nodes, 2,592 Hex8 cells, 1,374 boundary conditions, and the declared 457 top/457 bottom nodes. The accepted backend verifier replayed the repaired `accelerate_csc_v1` runtime/profile and its 115 bound runtime/control inputs; the runtime executable and its source/installed inventory hashes matched. No solver was launched for this review.

The row remains exactly `tension:N12:S60:reference`, ordinal 8, in the original native output path. Caps remain one native call/attempt, no retry, 90 s native, 600 s separate readout, 150 s preparation, 3 GiB sampled process-group RSS per supervised stage, 64 MiB active output, one numerical thread, and the frozen all-12 aggregate limits. The v2 envelope retains 54%/normal initial host admission, unchanged 45%/normal checks before reservation and supervision, exact predecessor hint counts/bytes, one MiB sidecar reservation, and bounded wrapper work. Its native output and v1/v2 sidecar attempt paths were absent at final review. These numerical software checks do not validate physical force, brain tissue, a patient or a clinical decision.

From the repository root, the **exact root-only launch command** for these reviewed bytes is:

```sh
.venv/bin/python -B -m launchers.hbe_v5_nocache_v2 --execute --envelope build/hbe-v5-tension-n12-s60-nocache-v2-final-candidate/outer-envelope.json
```

Before running it, root should verify HEAD and the two final candidate SHA-256s above and take a fresh direct 54%/normal host observation. The launcher itself must pass full predecessor, deck, runtime, source, storage, sidecar and post-validation/immediate-pre-native host gates. A change in HEAD or any reviewed byte voids this exact GO and requires refreshed bindings/review. No measured response, patient array or native run was accessed here.
