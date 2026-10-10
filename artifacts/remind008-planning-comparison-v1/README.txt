ReMIND-008: completed SEARCH / one-update IL / one-update RL comparison

REPORT.txt is the saved-only scientific interpretation. audit.json contains
verified scalar results, plan identities and hashes of original external saved
histories. Failed v2 preflight, replay-only completion, failed private evaluation
v1 and completed private evaluation v2 are all retained as distinct events.
No patient arrays, images, DICOM objects, weights, bulk histories or profiler
payloads are packaged. Their original local files remain in place.

The one-case result is negative: zero supplied-target progress for every method;
SEARCH STOP return 0 versus IL -1.06485 and RL -1.14221 after one update each.
The bounded beam is not a global optimum certificate. Moving methods have UNKNOWN
ventricular encounter status because their sweeps have no known reference
coverage. Automatic source BrainLab annotation is not manual truth or clinical
injury evidence. No population training or cross-patient transfer is established.

files/ mirrors original repository-relative paths for exact archived sources and
small receipts. copy-list.txt lists every copied original. SHA256SUMS binds every
package member except itself. All original bytes are unchanged. Exact runtime
dependencies are in each source-index and supervision declaration. The original
preflight ran at HEAD 0a6ba3c251bc2f8918890e6b84400f2e7437a5df; replay completion
ran at 26d6a66d077687d24d47f778bae96cad04a22796. Canonical library code remains in
repository history and is not needlessly duplicated here.

Reproduction sources are archived at their original paths under files/build:
actual-preflight-v2/{patient_worker.py,run_owned.py,release.json,source-index.json};
replay-completion-v1/{public_patient_factory.py,replay_worker.py,run_owned.py,
release.json,input-index.json,source-index.json}; both postseal-evaluation versions;
and the shared owned supervisor/sampler. The evaluator source is included too.
Use original repository-relative locations, qualified local inputs and the exact
pinned canonical dependencies. Workers retain existing release/output/parent
checks. This archive is not permission for another run; changing output or source
bindings requires a new explicit experiment release, not reuse of an old seal.

The archived saved-only audit script belongs at its original path to resolve
existing local outputs. It uses standard-library JSON/hash/scalar arithmetic;
it does not reconstruct anatomy or execute methods. Its result was produced
before publication packaging. Packaging made no scientific computation or rerun.

Three remaining fixed TRAIN conversion results are summarized separately in
REMAINING-TRAIN-QC.txt; they do not change the 008 result or grant admission.
