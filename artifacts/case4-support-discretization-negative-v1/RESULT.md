# Case4 source-coverage discretization negative

The first source-support map attempt stopped because independent SciPy and ANTs reconstructions differed at 49,293 T1 and 24,923 FLAIR voxels. No support map or success result was saved. Its supervisor recorded clean resource and process cleanup; peak sampled memory was 1,527,513,088 bytes. This is a preserved preparation failure, not an anatomical or model result.

Generated boundary tests reproduce a difference between SciPy constant padding and ANTs nearest-neighbor sampling near half-voxel boundaries. Grid-constant sampling matches those generated controls, including the separately tested oblique and nonidentity transforms. This does not yet establish that the replacement explains every patient-scan mismatch. A separately reviewed, bounded read-only comparison is pending; the original parity tolerance remains unchanged.

MANIFEST.json binds the twelve exact evidence/source snapshots; this root-authored summary is additional. The included read-only replay draft is unexecuted and is not an execution release. No patient arrays, model predictions, training, planning or physical validation are included. Case4 remains DEVELOPMENT with unresolved inferior mask omissions.
