# Fixed-scene native discretization result

All 16 numerical rows completed. Eleven native strokes were accepted; five were
rejected by the retained shaft-clearance rule without changing the cavity or
contact state. **Ten accepted histories passed independent checking. The final
0.125-mm oblique/shifted history reached the 170-second cooperative deadline and
remains unverified.** There was no retry or cap increase. Launcher exit zero did
not override `independent_audits_complete: false`.

This is one analytical numerical control, with zero patients, policy updates or
training examples. Physical anatomy remained `z >= 0`; the entry, endpoint,
4-mm aperture and native fine instrument stayed fixed. Its tip radius was
1.25 mm, shaft radius 0.45 mm, active length 2 mm and working length 120 mm.
Every grid was newly sampled from that same anatomy. Grid phase/orientation
changed relative to the tool; this was not a joint rigid coordinate transform.
The 0.0625-mm microstep was unchanged across resolutions.

The ideal geometric active-sweep volume inside the half-space is
**23.725570 mm³**. It is a geometrical reference, not a clinical removal target.
Native whole-cell credits below refer to the conservative voxelized mask.
“Physical” volume clips those cells against the unchanged analytical plane;
partially contacted cells remain occupied and receive no removal credit.

| Spacing mm | Grid | Native whole-cell removal mm³ | Physical removal mm³ | Static contact upper bound mm³ | Native / independent outcome |
|---:|---|---:|---:|---:|---|
| 1 | Aligned | 5.000 | 4.500 | 49.500 | Accepted / passed |
| 1 | Half-voxel shift | 0 | 0 | 64.000 | Shaft rejected / no committed history |
| 1 | 20° tilt | 0 | 0 | 60.500 | Shaft rejected / no committed history |
| 1 | Oblique + fractional shift | 0 | 0 | 61.720 | Shaft rejected / no committed history |
| 0.5 | Aligned | 10.750 | 10.188 | 36.688 | Accepted / passed |
| 0.5 | Half-voxel shift | 14.000 | 14.000 | 42.000 | Accepted / passed |
| 0.5 | 20° tilt | 0 | 0 | 40.563 | Shaft rejected / no committed history |
| 0.5 | Oblique + fractional shift | 0 | 0 | 41.336 | Shaft rejected / no committed history |
| 0.25 | Aligned | 18.391 | 17.914 | 31.680 | Accepted / passed |
| 0.25 | Half-voxel shift | 15.375 | 15.375 | 32.188 | Accepted / passed |
| 0.25 | 20° tilt | 17.641 | 17.055 | 32.133 | Accepted / passed |
| 0.25 | Oblique + fractional shift | 17.391 | 16.757 | 31.915 | Accepted / passed |
| 0.125 | Aligned | 20.840 | 20.569 | 27.423 | Accepted / passed |
| 0.125 | Half-voxel shift | 19.953 | 19.953 | 27.563 | Accepted / passed |
| 0.125 | 20° tilt | 20.539 | 20.198 | 27.751 | Accepted / passed |
| 0.125 | Oblique + fractional shift | 20.457 | 20.080 | 27.692 | Accepted / audit cancelled at cap |

Grid sampling materially changes the decision problem. At 1 mm, shifting the
grid alone changes an accepted stroke to rejection. At 0.5 mm, both rotated grids
still reject despite static contained physical masses of 10.0625 and 11.1216 mm³:
static containment cannot excuse a shaft collision or become executed removal.
The relative physical-removal ordering of aligned and half-shifted grids also
changes between 0.5 mm and 0.25 mm. No instrument dimension was changed to obtain
these results.

The full-cell rule leaves partially covered cells occupied. Changing their phase
or orientation changes which of those retained cells intersects the trailing
shaft, even though the continuous tool and tissue plane are unchanged. Static
contact or eventual containment cannot clear a cell before the causal shaft
check. Finer cells reduce this obstruction in the tested scene without weakening
that ordering rule.

Every static physical bracket contained the analytical reference. Across the
four samplings, bracket widths were 45–64 mm³ at 1 mm, 26.5–30.5 mm³ at 0.5 mm,
13.766–16.813 mm³ at 0.25 mm, and 6.854–7.612 mm³ at 0.125 mm. Brackets narrowed
in these tested sequences; this does not establish monotonic convergence for
arbitrary grids. Even at 0.125 mm, the contained physical volume is
**13.30%–15.90% below** the reference. The three independently accepted finest
histories also retain 6.854–7.609 mm³ of contacted-but-unremoved cell volume as
an upper bound. Partial cells were never erased to admit the shaft.

Native full-cell volume is not exact continuum tissue mass. For example, the
aligned 1-mm stroke receives 5 mm³ of source-cell credit, while only 4.5 mm³ of
those cells lies inside the analytical tissue. This boundary overcoverage is
reported separately. The fractional calculation is an audit of this known plane;
it does not infer unknown partial tissue volumes in patient images.

All numerical rows took 18.204 s, including 17.162 s in native strokes, 0.203 s in
static bounds and 0.064 s in grid/configuration preparation. The separate native
audits took 151.954 s: 124.997 s for ten completed checks and 26.957 s for the
cancelled final check. Independent checking dominated this run. The finest tilted
accepted audit alone took 43.436 s, versus 3.511 s for that native stroke. These
are one-run local costs, not isolated-machine benchmarks or learning-speed
claims. Worker time was 170.229 s; parent time was 170.690 s. The 170-s cooperative
deadline was observed between audit microsteps, producing 0.229 s overshoot; the
180-s parent kill was not triggered. Peak worker RSS was 685,096,960 bytes
(0.638 GiB), below the unchanged 2-GiB cap.

The evidence supports treating coarse source-grid acceptance, removal and contact
as resolution-dependent model outputs before learning comparisons. It does not
authorize relaxing the guards, silently enlarging tools, or claiming the finest
grid is converged. A future change to subcell or surface geometry needs its own
fixed-scene comparison and independent history checks; no production change was
made here. The source-axis proposal generator was not exercised, and outside-
image tool uncertainty remains recorded by the native engine.

Execution used committed `7abecf3426eb88fc9233be3d72332fa282a2c25a` and Git archive
SHA256 `304b788f019fe00cff20273580384eb36579fe53ccf948c2ba4f3a8860000667`.
All 52 original archived file hashes and archive bytes remain unchanged.
[Baseline](execution-baseline.json), [raw rows](run-01/report.json),
[launcher outcome](run-01/launcher.json), [summary](summary.json) and
[read-only verification](verification.json) preserve the exact inputs, timing,
certificate status and source identities. No source tree is duplicated in Git.
