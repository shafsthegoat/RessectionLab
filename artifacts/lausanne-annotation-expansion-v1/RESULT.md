# TRAIN annotation metadata qualification

The pinned ds003949 Git release `896b8846d899acee68c0246cc987ca96e77267d4`
and immutable S3 metadata qualify144mask files from106TRAIN people/117sessions:
13,977,015prospective source bytes. No new mask payload was acquired.

[Independent review](independent-review.json) recomputed the source-pointer sizes
and MD5s, same-session original links, S3 versions, TRAIN roles and categories:
111manual regions with unresolved subtype;31voxelwise-crosswalk date mismatches;
two exact contour crosswalks (sub476/sub478). A region is not a voxelwise contour;
exact crosswalk identity does not prove mask/grid quality.

[Four failed records](metadata-qualification.json) remain excluded: sub148 lesion1
and sub163 lesion1 timed out; sub163 lesion2 and sub220 lesion1 identify different
patients/sessions in RawSources. No inferred repair or automatic retry occurred.
297metadata requests transferred494,583response bytes in267.979seconds.

The full local index is `build/lausanne-annotation-expansion-qualification-v1/qualification-index.json`,
SHA256 `11bc772164ad3308583c00cc813f1f7e93d22aa96bbfd79d3c721794aa04017b`.
Raw response bodies and source provenance remain beside that index. Independent
review verifies the supplied candidate set; missing raw Git-tree response prevents
a new claim of tree completeness. No clinical, training or planning admission
follows. Next: bounded payload acquisition and per-file original-grid checks.
