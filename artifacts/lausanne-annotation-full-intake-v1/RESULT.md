# Full eligible TRAIN annotation intake

October 8, 2026. All 144 metadata-qualified Lausanne masks were acquired or reused
from verified cache: 13,977,015 compressed bytes, 106 people and 117 sessions.
The batch completed in 95.1011 seconds with 142 GETs; its two previously acquired
masks were reused. Four metadata exclusions remain outside this eligible set.
No SELECT or measurement-evaluation records were acquired or inspected here.

Independent streaming checks verified every mask's compressed fixity, gzip
integrity, extension framing and binary content: 6,685,598,720 source voxels,
including 1,134,965 positive voxels. The review used 21.3515 seconds and an observed
peak of 51,986,432 bytes RSS. Batch RSS was not recorded. All worker processes exited.

At this historical checkpoint, 116 masks passed their reference-grid checks,
27 had no original reference receipt when processed, and sub454 retained an
unresolved reference/annotation grid. Twelve of those 27 references appeared by
the independent review; no historical status was rewritten. Subsequent reference
review must create a new result. Downloads continue independently of these checks.

Source semantics remain limited: 111 masks have unresolved manual-region subtype,
31 have an author-crosswalk date mismatch, and only two have exact voxelwise
crosswalks. Background remains unknown. Content/grid QC is not anatomical review,
complete vascular coverage, scanner-frame acceptance or spatial-planning admission.
No fitting or RL updates used these records. The original sub022 format failure
and later successful cached repair remain preserved in their separate records.

`batch.json`, `declaration.json`, `source.json` and `worker-receipts.json` retain
execution and source bindings. `independent-verification.json` has SHA-256
`756a12b35e2634efa5d53629903d25798fb434b2c82d397a6d20cf958aacfcd9`;
the accompanying independent audit and completion controls preserve its method
and terminal process checks. Original images and masks remain outside Git.
