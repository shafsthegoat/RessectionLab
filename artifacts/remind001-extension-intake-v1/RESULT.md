# ReMIND001 imaging extension

Acquisition completed **400 objects across eight series, 412,786,204 bytes**, for the existing development patient. These comprise ventricles segmentation, four MRI series and three staged ultrasound objects. All bytes remain unreviewed and undecoded; anatomical and registration quality have not been accepted.

The 397 MRI/segmentation objects use verified TLS and source single-part MD5 checksums. The three ultrasound objects use the official IDC GCS mirror, pinned object generations and publisher whole-object CRC32C. Multipart S3 ETags are preserved only as identity evidence, never interpreted as whole-file MD5. Local SHA-256 identifies every payload. Transfers support resumption; these 400 completed on their first request from offset zero.

The acquisition agent rechecked all files after download. Root separately streamed every file to recheck SHA-256 and MD5 against the completion records, and confirmed that existing patient-role files still match Git. Root did not independently implement or rerun CRC32C. The [byte proofs](byte-proofs.json) bind public source objects to local sizes/hashes; the [summary](summary.json) binds the original completion and acquisition-agent fixity records.

ReMIND001 retains its development annotation-assisted geometry role. No new patient, split reassignment, training admission, optimization update or independent held-out result is claimed. Intraoperative ultrasound remains excluded from preoperative policy inputs. The scans do not provide measured tool forces or an observed action-to-force mapping. Patient data remain local and out of Git.

Source: TCIA ReMIND version 1 / IDC v24, CC BY 4.0, [dataset citation](https://doi.org/10.7937/3rag-d070). Existing guarded transport was reused through ignored adapters; this result does not claim a newly shipped general-purpose downloader.
