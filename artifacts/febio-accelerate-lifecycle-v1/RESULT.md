# Corrected-only adapter controls

The separately released attempt from preparation commit `0eeda40` passed all
four actual-SDK structure initialization profiles and fourteen literal-method
mock resource-lifetime scenarios. Both binaries were compiled with ASan/UBSan.
The supervised group completed once in 5.45914 s, with 312,262,656 bytes peak
sampled RSS, no limit violation, and all eighteen bound inputs unchanged.

The lifecycle compilation retained two warnings about upstream integer-valued
iterative tolerances. That iterative branch is excluded and aborts if entered
in the fixture. Four factorization-error messages are expected outputs from
defined mock failure scenarios. No sanitizer diagnostic was reported.

The fixture uses SDK data types but mocks factorization, solve and cleanup
allocation behavior. It verifies the corrected adapter's ownership transitions;
it does not validate private SDK allocation behavior, actual numerical solves,
physical mechanics, or the original faulty implementation. No original faulty
code, FEBio model, patient array or measured curve was executed or accessed.

The result, compiler/runtime logs, supervision, immutable attempt marker and
parent acceptance are retained here. The isolated runtime build and its eight
actual numerical controls remain separate prerequisites for the specimen test.
