# Lausanne original TRAIN acquisition complete

October 8, 2026: all 840 frozen original-image/sidecar files are byte-verified,
covering 199 TRAIN people and 210 T1/TOF sessions, 10,020,802,851 bytes. The
continuous download-only queue acquired the final 52 files (739,875,061 bytes)
and reused 788 verified files. No SELECT or evaluation assignment changed.

`completion.json` joins every file to the frozen source index, successful immutable
download outcomes and exact receipt hashes. This is a metadata-only completion
checkpoint, not an additional image-content review. Full local transfer receipts
and originals remain outside Git; shared declarations are retained once.

The previous independently reviewed image-QC checkpoint covers 197 sessions:
191 full pairs from 180 people passed header/scalar checks, while T1 conflicts
for sub253, sub410, sub411, sub418, sub424 and sub473 remain unresolved. All 197
TOFs passed that image-level gate, which does not resolve their paired T1 failures
or narrower annotation-grid questions. The last 13 newly acquired sessions
(26 images) await separate header/scalar QC. Anatomy, source-frame reconciliation,
interscan alignment and clinical coverage remain separate evidence requirements.

The completed 144-mask annotation intake has its own record. Its 27 historical
missing-reference outcomes and sub454 grid failure are not rewritten by these
downloads. A separate targeted QC phase will review new originals and reference
joins. No component fitting or policy update used these Lausanne records.

RESECT acquisition continues independently, including provider cooldowns. There
are no further original Lausanne TRAIN files to acquire in this frozen release.
