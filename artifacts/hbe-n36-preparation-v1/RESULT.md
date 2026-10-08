# Actual N36 preparation and single solve release

One Gmsh generation produced the declared N36 half-height specimen mesh: 39,610 nodes, 34,992 hexahedral cells and nine axial layers. Full reflection has 75,259 nodes and 69,984 cells. Independent saved-output checks verified the mesh, loading, reconstruction, source archive and 266 unchanged baseline inputs. Maximum reflection coordinate error was 8.67e-19 m.

Preparation took 17.9955 seconds through result publication against a 60-second cap. Sampled process-family RSS peaked at 687,112,192 bytes; retained preparation output was 66,029,975 bytes. Its closing receipt explicitly excludes its own final fsync from that observed interval. No solver or measured response data was used in preparation.

The separately issued solve release binds the preparation result, publication check and independent review. One FEBio solve has now launched with the unchanged 1,500-second native / 1,800-second aggregate, 3-GiB sampled RSS and one-thread caps. It is a terminal uniform-resolution diagnostic; no automatic retry or further uniform refinement is authorized by this declaration.

The prior N32 negative result remains. This milestone establishes preparation integrity only: spatial convergence, temporal convergence, physical fidelity and calibration are not accepted. The actual solve result will be recorded separately, including a cap failure if one occurs.
