# Combined repair identity review

The narrow backend update passed **54 focused controls in 1.15 seconds**; all
nine bound files were unchanged during the run. The previous 127-check generic
integration review and its four preserved failures remain untouched.

The identity record, combined patch and patched C++ source match their exact
bytes committed at `0eeda40b29a501b9b270299eac7600c492c2c6e4`:

- Identity: `54198366eccb90a9fcdf4fcbf2a5055919a5658a1db0fdd7db8cfeede3b49eee`
- Patch: `67a1858f2d55046796d7ccb46758ca06e5bfb673e75a539d158b1a17652f3340`
- Patched source: `476ac8471ea681a99c298352a50aba2a7a5baa71635e1678155a7f58b72ba04a`

The backend now authenticates that exact nested identity and both referenced
files, adding them to its input inventory. Accepted runtime records must retain
all five bindings for the separate adapter controls. Their contents are checked
by the runtime producer before publication; the consumer rechecks their bytes
and binds them through the accepted candidate/final identity.

An AST comparison against the earlier reviewed backend confirms that deck
transformation, runtime byte checks, eight-control replay and prepared-deck
verification are unchanged. The profile ID, private prefix, upstream commit,
solver XML, OpenMP exception, case inventory and checker pins are unchanged.
The experiment and access source hashes also remain unchanged.

The independent placeholder-summary regression now traverses the authentic
committed nested repair record before expecting the missing-control-evidence
error. It cannot pass merely because its old flat patch fixture became obsolete.
Owner controls reject old identities, changed patch/source bytes and missing
adapter evidence. Other fixtures remain explicitly constructed files and checker
doubles, not actual solver-validation evidence.

This supersedes only the pending identity-update caveat in the prior generic
review. It does not independently revalidate the C++ repair, establish a working
sparse runtime, or authorize execution. No solver, compiler, patient data or
measured curves were accessed. The actual sparse executable remains absent;
future execution still requires the separately reviewed runtime and actual
eight-control profile.
