# Eight actual repaired-backend numerical controls

The separately released attempt ran all eight fixed cases once and passed:
zero motion, rigid translation, finite stretch, shear, doubled stiffness,
tet10 affine deformation, rigid multipoint constraints and nonrigid multipoint
constraints. Every actual solver log selected Accelerate; each input differs
from the original only in the declared linear-solver subtree.

The supervised worker took 5.09111 s, with 45,416,448 bytes peak sampled process
group RSS, within 60 s / 3 GiB / one numerical thread. Total launcher time was
5.78290 s. There were exactly eight solver calls, no retry, no resource kill,
unchanged bound inputs and no summary-publication error.

All five states of the stiffness-scaling comparison retained the same motion
and Jacobian while doubling reaction, stress and energy under the unchanged
original checker. Both complete group summaries and all raw analytical records
are preserved byte-for-byte. Independent saved-output replay is separate.

These are numerical software controls, not measured tissue agreement or patient
validation. They permit evaluation of the fixed specimen experiment with the
repaired backend; they do not establish cutting, retraction forces, anatomical
fidelity or policy superiority. No measured specimen curve or patient motion
was read in this attempt.
