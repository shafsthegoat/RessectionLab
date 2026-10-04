# Derived-frame diagnostics: independent review

Nine focused analytic checks passed in 1.45 seconds with unchanged source hashes. The reporting mismatch identified during the core roundoff review is resolved: original geometry determines discrete proposal indices; the declared derived geometry determines physical depth and crop-coordinate inversion. Both matrices remain visible in reports. Default behavior agrees with explicitly supplying the original affine.

The added test uses an actually accepted tiny roundoff transform and checks the original crop indices to 1e-14 voxels. Fake-worker default, derived and mismatched declarations verify exact propagation and rejection before diagnostics on mismatch. Optimizer, gradient and checkpoint calls remain forbidden in these profile branch tests. The prior profiler/resource checks also remain passing.

Eight static checks of draft v2 confirm unchanged patient, source bundle, access, support, tools, objective, crop, policy and resource limits. The only adapter option added is explicit reconciliation. Its exact expected matrix and displacement agree with the pure 4×4 helper calculation from the already saved header JSON. The draft's empty numerical source map still prevents execution.

Final source freeze, declaration hashing and patient release remain root responsibilities. No patient bundle, header or image was opened here; no patient task, preview, policy or learning ran. No final source tree is duplicated in this evidence directory.
