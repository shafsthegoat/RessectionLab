# Lower-half-height mechanical contract review

The declared reconstruction is consistent for the **midheight-symmetric axial equilibrium branch**, with homogeneous isotropic alpha-2 Ogden material, the unchanged three-field formulation, identical bonded physical plates, free lateral boundary and no body force. This is a pre-execution analytical review, not implementation or numerical-equivalence acceptance. No measured responses, solver fields, patient arrays or solver execution were accessed.

Let `S=diag(1,1,-1)`, `X'=SX+H e_z`, and `x=X+u`. Reflection of the current configuration is `x'=Sx+(H+d)e_z`, hence `u'=(u_x,u_y,d-u_z)` and `F'=SFS`. Therefore `det(F')=det(F)`, the isochoric invariant is unchanged, and `W(F')=W(F)` for the declared law. The same identity holds for reference-volume-weighted cell-average J and its volumetric energy. Stress transforms as `S σ S`; a reflected hex connectivity must receive the proper orientation permutation, without changing original coordinates.

On the cut `Z=H/2`, continuity requires `u_z=d/2`. Tangential displacement remains free: imposing zero x/y there would introduce a bonded artificial plate. Mirroring the fixed physical bottom produces the original fixed-x/y, prescribed-z physical top. The current symmetry plane lies at `(H+d)/2`, not at the original rest midplane after loading.

Raw FEBio reactions use the existing **body-on-constraint** convention. At distinct mirrored nodes use `R'=S R`. At a shared midplane node **sum** both contributions: its normal components cancel; its tangential components add and must remain available to residual/equivalence checks. Displacements are single-valued and are never summed. With equilibrium and no external body/side loads,

`F_full = −Σ(top full R_z) = +Σ(bottom half R_z) = −Σ(cut half R_z)`.

Force scale is **one**, since the full cross-section is retained. The last equality is an equilibrium check, not permission to replace one independently calculated plate force by the other. Both signs and load coordinates remain signed under compression. `E_full=2 E_half`; writing `q=d/2`, `W_full=∫F_full dd=2∫F_cut dq`. Do not double force or use full `d` when integrating half work.

Implementation checks needed before execution:

- Existing `HexMesh.read_frame` and `read_run` treat both physical plates as prescribed x/y/z and assume the original full H. They cannot be applied directly to the raw half mesh with its free cut x/y. Use explicit half-boundary checks, then the unchanged full readout on the lifted original full mesh.
- Preserve original H in full loading, comparison scales and 75 probe coordinates; retain H/2 separately as modeled domain height. Compare independently doubled half energy/work and reconstructed full totals without double-counting either.
- Recompute moments about the common origin using actual reconstructed current positions. Moments are axial vectors: under an improper reflection their transformation includes `det(S)=-1`, and changing origins also contributes a force-dependent term. Applying the force-vector reflection to a moment would be incorrect.
- Reconstruct the three-field energy with cell-average J. Logged pointwise-average `sed` remains diagnostic, as in the existing full readout.

Symmetry removes asymmetric perturbations and can exclude buckling or other branches. A lifted symmetric stationary solution need not be unique or stable in the unrestricted full domain. Matching all four saved N8/N12 cases and all 61 states supports only those numerical equivalences; it does not prove stability, locking immunity, finer-mesh convergence or material fidelity. Positive sampled J is not a global injectivity/stability proof. Torsion, heterogeneous/anisotropic properties without the required reflection symmetry, unequal bonding, gravity or contact require a separate argument. No contract amendment is required for the declared axial study; implementation and actual saved-output review remain outstanding.
