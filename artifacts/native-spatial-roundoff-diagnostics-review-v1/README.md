# Derived-frame diagnostics: independent review

Nine focused analytic checks passed in 1.45 seconds with unchanged source hashes. The reporting mismatch identified during the core roundoff review is resolved: original geometry determines discrete proposal indices; the declared derived geometry determines physical depth and crop-coordinate inversion. Both matrices remain visible in reports. Default behavior agrees with explicitly supplying the original affine.

The added test uses an actually accepted tiny roundoff transform and checks the original crop indices to 1e-14 voxels. Fake-worker default, derived and mismatched declarations verify exact propagation and rejection before diagnostics on mismatch. Optimizer, gradient and checkpoint calls remain forbidden in these profile branch tests. The prior profiler/resource checks also remain passing.

Eight static checks of draft v2 confirm unchanged patient, source bundle, access, support, tools, objective, crop, policy and resource limits. The only adapter option added is explicit reconciliation. Its exact expected matrix and displacement agree with the pure 4×4 helper calculation from the already saved header JSON. The draft's empty numerical source map still prevents execution.

Final source freeze, declaration hashing and patient release remain root responsibilities. No patient bundle, header or image was opened here; no patient task, preview, policy or learning ran. No final source tree is duplicated in this evidence directory.

After production commit `89f01e4c3a1a8564d0ec00d926eb2a5e5174c332`, `final-declaration-review.json` independently verifies the sealed v2 declaration `30213ab0174a77f5df50afc1bd93bafa0a38e442419d22d5933af9c52f25ad94`: all 53 source files match that commit and the current closed inventory. The reviewed expected-grid fields are unchanged. Root release remains separate; this verification opened no patient files.
