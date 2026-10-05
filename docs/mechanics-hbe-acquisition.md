# HBE specimen mechanics acquisition

The creator-supplied [Hyperelastic Human Brain 1–7](https://zenodo.org/records/8095559)
record is version 1.0, DOI `10.5281/zenodo.8095559`, published September 7,
2023, under CC BY 4.0. Attribution: Jan Hinrichsen, Nina Reiter, Friedrich
Paulsen, Lars Bräuer and Silvia Budday. The exact API response and prospective
file pins are retained in `artifacts/mechanics/hbe-8095559-acquisition-v1/`
and `manifests/hbe_8095559_acquisition.json`.

All three declared files downloaded once in 18.566 seconds, totaling
12,985,030 bytes. Every length and provider MD5 matched; local SHA256 hashes
are in the acquisition receipt. The unchanged source files remain ignored in
`data/mechanics/zenodo-8095559/`. No downloaded code was executed.

This is complementary ex-vivo material evidence. The README describes
moving-average filtering, interpolation and averaging of loading/unloading
curves to approximate quasi-static hyperelastic response. Compression/tension
columns are displacement in metres and force in newtons; torsion columns are
angle in radians and torque in newton-metres. These data alone do not establish
patient material properties, dynamic response or cutting mechanics.

The metadata-only inventory reconciles 182 lookup rows with 182 geometry files
and 2,184 expected curve filenames. Donors `HBE_01` through `HBE_07` contain
21, 20, 40, 25, 27, 29 and 20 specimens respectively. Every specimen has
compression/tension and two torsion amplitudes, each for cycles 1 and 3, with
separate torsion signs. The ZIP also contains two extra `HBE_06_07` files named
`_raw.csv` and `_proc.csv`; their contents remain unopened and excluded.

Geometry provides positive `height`, `radius` and cylindrical shape in metres.
The source uses both `cylinder` and `Cylinder`; exact spelling is retained.
An initial case-sensitive inspection assertion stopped on this difference;
the corrected metadata parser accepts the two spellings without changing data.
The creator says height was determined from test data, so it is not an
independent geometry measurement or a supplied uncertainty estimate.

Sorting actual numeric donor and specimen identifiers gives `HBE_01_03` as
the first metadata-eligible specimen: source radius 0.004 m, height 0.00489159 m,
governing region cortex, region motor cortex. Selection eligibility uses only
geometry completeness and required filenames. The separate validation
declaration owns its concrete roles. No curve CSV was opened during this
acquisition: headers, delimiters, row counts, force/torque values, plots and
fits remain uninspected. Metadata for all seven donors has been consulted;
their measured curves remain sealed until explicitly assigned and released.

The retained acquisition driver enforces the prospective manifest hash,
refuses redirects and overwrites, limits each transfer to its declared size,
and keeps failures. `inspect-lookup.mjs` reads the unchanged workbook with the
bundled spreadsheet runtime; `inspect_metadata.py` reads only lookup metadata,
ZIP directory entries and geometry members. The full directory inventory is
compressed metadata, not measured curves. No specimen curve SHA is claimed
before reading that member; the verified archive SHA binds all stored bytes.
