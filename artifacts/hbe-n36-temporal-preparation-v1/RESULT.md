# Prepare the finest-mesh loading-increment check

The new adapter reuses the accepted N36 mesh, reconstruction, material model, loading endpoints and FEBio runtime. Only the fixed compression schedule changes from 60 to 120 increments. It will read all 121 native states and compare the 61 common states and 75 probes against the independently checked S60 result.

Original force and displacement criteria remain unchanged. A separate signed sensitivity calculation tests whether the small change reverses the earlier conditional spatial classification. Passing this diagnostic does not establish continuum convergence, tissue calibration, patient-specific force or surgical validity. No measured biological curve is accessed by this run.

The prospective limits are one native call, zero mesher calls, 2,100 seconds for the native solver, 2,400 seconds in total, 3 GiB sampled process-family RSS, one runtime thread and 2 GiB new retained output. Pure deck preparation has a 60-second sublimit within the total allowance. Case-specific streaming ceilings accommodate all S120 records without widening legacy parser defaults. There is no automatic retry or calibration release.

Five new files preserve all 20 inherited source modules. Twenty-two owner controls passed in 16.60 seconds, including reconstruction of the actual saved geometry and confirmation that the generated deck changes only the permitted schedule. No native solve or mesher ran. A targeted independent implementation review precedes the single execution.

The archived proposal, design review and owner evidence preserve their original hashes. Their original local build paths remain named in the source-bound declaration; restoring those exact metadata files and reproducing the prior saved N36 baseline is required when preparing another checkout. New runtime output and scientific logs stay local.
