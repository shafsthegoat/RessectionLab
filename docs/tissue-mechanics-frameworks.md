# Tissue-mechanics framework decision

Primary sources checked October 4, 2026. This is a scoped framework review,
not an installation, completed validation or change to the geometric engine.
Suitability judgments below are engineering recommendations; no candidate was
installed or tested for this review.

**Recommend FEBio for the first measured, finite-strain specimen deformation
experiment.** The measurement review found the creators' [Hyperelastic Human
Brain 1–7 dataset](https://zenodo.org/records/8095559): specimen geometry,
compression/tension displacement–force and torsion angle–torque observations.
Its curves are filtered and averaged to approximate quasi-static hyperelastic
response; they are not raw viscoelastic histories, glioma-patient measurements
or retractor/cutting outcomes. This supports a narrower physical check than
immediately predicting deformation around a patient's cavity. The associated
[identification study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10511383/)
also shows that inferred parameters depend on assumed compressibility and
conditioning. Do not transfer its fitted constants directly between differently
defined strain-energy functions or into an uploaded patient.

| Framework | Suitable narrow role and important limits | Local Mac and license evidence |
|---|---|---|
| **FEBio** | Existing finite-strain constitutive models, prescribed rigid motion, reaction forces, and sliding/tied contact fit a specimen-loading benchmark and later measured retraction. Supported tetrahedra include tet4/tet10/tet15. Nearly incompressible behavior requires the appropriate element/formulation; the documented three-field recommendation concerns hex/penta, while `ut4-solid` uses nodal integration. [Elements](https://febiosoftware.github.io/febio-docs/user/chapter3/3.6-mesh-section/), [three-field](https://febiosoftware.github.io/febio-docs/features/features/solid_soliddomain_three-field-solid/), [UT4](https://febiosoftware.github.io/febio-docs/features/features/solid_soliddomain_ut4-solid/). | Source/SDK are MIT; website binaries use a separate license prohibiting redistribution, with commercial-use restriction removed. The published Apple Silicon note describes x86/Rosetta and an optional licensed ARM Pardiso route. A source build without MKL instead defaults to Skyline: paid Pardiso is unnecessary for a small prospective bench, but its local build, speed and memory remain untested. [Licensing](https://febio.org/about/licensing/), [Mac note](https://febio.org/knowledgebase/tutorials/febio-on-apple-silicon/), [build guide](https://github.com/febiosoftware/FEBio/blob/develop/BUILD.md). |
| **SfePy** | A smaller Python-facing choice for displacement-constrained, small-strain anatomy experiments, with existing elastic/hyperelastic terms. Its shipped mixed elasticity example uses **P1 displacement/P0 pressure at ν=0.4**; that example alone establishes neither locking resistance nor stability near incompressibility. Quadratic/bubble simplex fields are supported, permitting a separately verified element choice. [Example](https://sfepy.org/doc/examples/linear_elasticity-linear_elastic_up.html), [field definitions](https://sfepy.org/doc/users_guide.html#fields). | BSD three-clause terms; pip/conda installation and SciPy-based serial solvers are documented. Current installation documentation explicitly names **Intel MacOS**: Apple Silicon compatibility and exact compiled dependencies require a bounded test. Prefer an isolated runtime. [License](https://sfepy.org/doc/license.html), [installation](https://sfepy.org/doc/installation.html). |
| **FEniCSx/DOLFINx** | Strong alternative for an explicit mixed displacement–pressure formulation, tagged boundaries and transparent convergence studies. The official hyperelastic demo is **compressible** neo-Hookean; it is not a verified nearly incompressible brain model. Contact or evolving cuts would require additional formulation/integration work, beyond the proposed first experiment. [Official demo](https://docs.fenicsproject.org/dolfinx/v0.11.0.post0/cpp/demos/demo_hyperelasticity.html). | LGPL-3.0-or-later. Official Mac recommendation is conda; the stack includes compiled FEM/MPI components and commonly PETSc. An isolated pinned conda-forge environment is practical to investigate, but no local installation or footprint has been measured. [Upstream installation/license](https://github.com/FEniCS/dolfinx/blob/main/README.md). |
| **SOFA** | Best aligned with later interactive tool contact, multiple mechanical/collision representations and topology changes. Existing corotational/hyperelastic tetrahedral components and contact solvers avoid writing an engine. A visually plausible example does not validate locking, force prediction or brain cutting. [Hyperelastic/contact example](https://sofa-framework.github.io/doc/components/solidmechanics/fem/hyperelastic/tetrahedronhyperelasticityfemforcefield/). | Libraries LGPL-2.1-or-later; applications GPL-2.0-or-later; plugins/dependencies need their own inventory. Official Mac build instructions support ARM64 and describe Python/native dependency requirements. Local binary architecture and plugin compatibility must be inspected, not assumed. [Distribution/license](https://www.sofa-framework.org/download/), [Mac build](https://sofa-framework.github.io/doc/getting-started/build/macos/). |

These are upstream terms, not a completed redistribution audit. Keep the first
solver external to the app; pin its source/build and dependencies separately.

**Meshing and model definition.** For the first specimen, generate its measured
height and documented diameter with an established mesher; preserve dimensional
uncertainty. For later anatomy, segmentation must become a closed, oriented,
quality-checked physical mesh with tissue labels and separate skull/opening,
support and cavity facet tags. Preserve coordinates/units and quantify surface
approximation error; do not silently fix poor geometry or translate an access
annotation into a mechanical clamp. FEBioStudio documents Netgen/Tetgen paths
and higher-order tetrahedra. Mesher licensing is separate: Gmsh is GPL-2.0-or-later
with its stated exception, not BSD merely because a Python wrapper imports it.
[Meshing documentation](https://febiosoftware.github.io/febio-docs/studio/chapter5/5.4-creating-and-editing-a-mesh/),
[Gmsh terms](https://gmsh.info/).

**Three prospective checks before mechanics feeds optimization:**

1. **Numerical verification:** pin one existing constitutive law and its exact
   parameter convention; verify an analytic homogeneous loading case, rigid
   translation, force balance and residuals. On at least three refinements,
   check displacement/reaction-force convergence, positive element Jacobians,
   volume change and sensitivity to near-incompressibility. Tighten load steps
   and solver tolerances independently. For a tetrahedral specimen, compare the
   chosen formulation against a verified higher-order or structured reference;
   a solver's success flag is insufficient.
2. **Measured response:** reproduce the study's attachment to sandpaper on
   **both plates (no slip)** and recorded displacement/rotation. A homogeneous
   free-slip formula does not reproduce this fixture. The paper uses a modified
   one-term Ogden law; establish equivalence of its volumetric/deviatoric energy
   and parameter definitions before comparing a FEBio implementation. Freeze calibration versus
   withheld loading mode/specimen/donor before fitting. Compare withheld
   forces/torques in their actual units, retaining systematic residuals and
   simpler constitutive baselines. Published curve fit is calibration; a
   matched displacement boundary is an input, not a successful force prediction.
3. **Transfer boundary:** only after those checks, evaluate a separately
   declared real-anatomy displacement experiment against withheld observations
   and rigid/interpolation baselines. A homogeneous linear elastic model driven
   solely by prescribed displacement and traction-free surfaces cannot identify
   absolute Young's modulus from the resulting displacement field.
   [Primary identifiability discussion](https://pmc.ncbi.nlm.nih.gov/articles/PMC3600363/).

FEBio has published analytical/cross-code verification and a separate
independent comparison with PolyFEM; the latter also reports difficult contact
failures, so neither establishes universal reliability or our model's physical
validity. [Original verification](https://pmc.ncbi.nlm.nih.gov/articles/PMC3705975/),
[independent comparison](https://doi.org/10.1016/j.cmpb.2023.107938).

Actual retraction needs tool-surface motion/contact constraints and independent
displacement or force observations. A published phantom study used tracked
retractor surfaces and CT-visible beads, illustrating the missing measurements;
its paper alone is not an executable public validation dataset.
[Retraction experiment](https://pmc.ncbi.nlm.nih.gov/articles/PMC4082653/).
Cavity evolution additionally changes topology, free surfaces and stress history.
Deleting occupied cells is not a verified cutting or stress-release law. None of
these frameworks supplies patient-specific stiffness, tissue injury thresholds
or clinical safety from structural MRI alone.
