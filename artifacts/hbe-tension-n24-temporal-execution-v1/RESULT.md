# Tension increment check passed

The actual retained N24 specimen model completed one native solve with 120 loading
increments. Independent reconstruction reproduced all 121 states and both original
spatial-triplet comparisons exactly. No measured tissue response was accessed.

| Common-state difference | Measured numerical change | Fixed allowance |
|---|---:|---:|
| Maximum force change | 6.515066e−12 N | 5.316157e−5 N |
| Maximum probe displacement change | 1.414195e−14 m | 8e−7 m |

All 61 common states and 75 probes per state were included. Replacing only N24/S60
with N24/S120 changed no pass/fail classification in either spatial triplet.
These mixed-increment comparisons are sensitivity checks, not a new homogeneous
spatial convergence study or rigorous continuum error bound. The earlier global
spatial failure remains preserved.

The native solve took 245.951626 seconds; preparation took 4.752022 seconds and
was included in the 301.955898-second phase through durable result publication.
Sampled peak process-family RSS was 557,350,912 bytes; retained output including
closeout was 394,515,352 bytes. All declared limits held, with zero remeshing and
no retries. The independent replay took 49.709868 seconds and ran no solver.

Together with the compression increment check, this supports a prospective
branch-specific calibration release. Calibration is still unreleased. Two actual
fitted axial confirmations and frozen predictions must precede withheld torsion
measurements. Any later fit applies to this human specimen and its declared loading
conditions; it cannot establish patient-specific properties, retraction, cutting
forces or injury prediction.

Source commit: `2911d31`; exact 32-file archive SHA256
`860527629a93f34f4a331c7b29dce0a845e7b16b997b47bdd269c37659e7c5b6`.
See the independently reproduced result and original bindings below this directory.
