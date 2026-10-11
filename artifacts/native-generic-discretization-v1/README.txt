Source-only preparation: actual-tool fixed-scene resolution control

No numerical execution or acquired input is authorized by these files. Root owns
the next serial generated-test slot and any subsequent benchmark release/output
reservation. No canonical code is modified. Source/tests fit the acknowledged
1 MiB reservation. The existing scripts/probe_native_discretization.py is reused
by an adapter, including its bounded one-child launcher and numerical/audit loop.

Question: with the actual generic instruments, how much do source-grid spacing,
phase and orientation change acceptance and credited removal for one prescribed
physical side-clearing sequence? This is a necessary resolution-consistency
check, not patient-resolution validation, optimized route search or RL scoring.

The continuous anatomy is z>=0. The aperture is radius6 mm in z=-2. Each tool
has nine shallow cuts ending at z=1 mm, followed by the same central target cut
ending at z=2.2 mm. The nine entries form a3x3 physical lattice with spacing1 mm
(suction) or1.8 mm (aspirator). Order is fixed: center, four cardinals, four
corners. Every cut is axial with full withdrawal; inter-insertion transfer stays
unassessed, as in the canonical primitive. Nothing is optimized per grid.

Continuum rationale before outcomes
1. The actual suction tip/shaft radii are1.0/1.4 mm, active length3 mm; aspirator
   radii1.7/2.7 mm, active length4 mm. Every shallow shaft ends strictly below
   tissue: highest z=-0.6 mm or-0.3 mm. Thus ideal shallow cutting does not need
   to borrow clearance from its own unexecuted cut.
2. At z in[0,1], the9 completed active cylinders cover the square[-R,R]^2, hence
   the central shaft disk. A conservative nearest-lattice-center distance bound
   is sqrt(2)*max(a/2,R-a). This leaves lateral margins>0.29 mm and>0.42 mm.
3. During the central target stroke, highest shaft tissue-contact z is0.6 mm or
   0.9 mm, strictly below the completed shallow depth1 mm. Its entire swept shaft
   in tissue therefore lies within the already cut ideal continuous union.
4. Even the far corner insertion fits the unchanged aperture with>0.75 mm margin.
5. The target is the analytic half-space z>=2.8 mm. Preparation active caps end
   at z<=2.0 or2.7 mm, before that plane. The final cap extends to3.2 or3.9 mm,
   so it has strictly positive continuous target intersection.
These are continuous geometric enclosures, not native certificates. At coarse
resolution a box touching the shaft may extend outside an individual cutting
capsule; contacts do not erase boxes. A native rejection is the measured outcome.

The18-row denominator is2 actual tools x3 samplings x3 spacings. Samplings are
aligned, half-voxel phase, and20-degree tilt relative to the fixed physical
scene/tool. Spacings are1,0.5,0.25 mm. The microstep stays0.0625 mm in every row;
it does not increase with coarse spacing. There is no joint scene/tool rotation.

The borrowed analytic outer voxelizer and exact half-space cell fractions are
used directly. Report native whole-cell credit separately from plane-clipped
physical support/target volume. Static unions of contained/contacted cells from
all10 ideal active sweeps are explicitly non-executable bounds. Complete native
temporal, shaft, surface/frontier and aperture rules remain intact.

Reward weights remain the canonical DEFAULT_NATIVE_SPATIAL_REWARD. The adapter
calls the existing _score_record on exact analytic target fractions stored as
float32, with the actual complete tool travel. This ancillary readout quantifies
grid-dependent source-cell accounting; it is not a selected policy return and
does not define benchmark success. No new reward or optimizer is introduced.

Each sequence stops at its first refusal. Accepted prefixes still receive the
existing independent full-tool/history checker, using its declared batch256
backend. Later prescribed actions remain unexecuted rather than skipped. Missing
rows, unfinished audits and rejected target strokes remain in the18-row/180-slot
denominator. Numerical completion is not audit completion. The source guard and
170-second worker/180-second parent/2-GiB/one-thread bounds are fixed before runs;
no automatic retry, tool retuning, grid-specific action changes or cap increase.

Generated test file is prepared but unrun. It checks real tool/reward identity,
physical action invariance, relative-grid changes, analytic target fractions,
changed-tool refusal, two coarse numerical rows with prefix geometry checks, and
missing-row/batch-hook/overwrite handling. No success count is assumed in advance.

Once root explicitly releases a numerical slot, the existing style command is:
  .venv/bin/python -B build/native-generic-discretization-v1/probe.py --output <fresh-reserved-directory>
Until then the files are source-only. Existing source/output trees are untouched.
