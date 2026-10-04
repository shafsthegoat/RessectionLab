# Independent nominal/cavity review

The provider passes seven analytic checks and an exact comparison against the committed default fixed-lattice module at `3d9278de4735640150e4d9b83ae0ec1d88c3d407`. The comparison covers three states and two transitions, including source/model hashes, observations, historical inventory fields, rewards and native history. These tiny arrays are code tests, not training or benchmark cases. No patient images or policies were loaded.

`provider-receipt.json` records the frozen source and limitations. Reproduce the default comparison with `.venv/bin/python artifacts/nominal-cavity-independent-review-v1/audit_legacy_default.py` from the repository root; it requires that Git object. Final committed source is identified by hashes and Git, without copying another complete source tree.

The initial run had six passes and four failures. Three exposed a real diagnostic issue: accepted IDs could retain their names while entry, tip or tool changed. Their initial uncommitted source and failing tests are preserved as gzip files, with `initial-pytest.log`. The fourth was an independent test assumption: retained contact was checked after the final cut had removed it; the corrected check measures the intermediate cavity. No provider change resulted.

The provider is reviewed separately from the dynamic diagnostic reporting fix, which remains pending. The new generator emits only a bounded family of proposals and every emitted candidate receives the unchanged full native geometry preview. It supplies no clinical approval or claim of whole-patient reachability.
