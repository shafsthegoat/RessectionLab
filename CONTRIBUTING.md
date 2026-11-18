# Keep the repository focused

This policy governs code, documentation and artifacts across the repository.
It supersedes older instructions to keep obsolete material in the tracked tree.

- Keep tracked material that supports a current workflow, maintained shared
  component, or named active branch/milestone. Remove dead code, abandoned
  experiments, superseded documentation and unused artifacts when that need ends.
  Speculative future value is a reason to archive useful material outside Git.
- Retire related code, dedicated tests, obsolete fixtures and documentation
  together. Check dynamic/path-based consumers and planned active integrations,
  update references, and run the relevant checks. Preserve supported behavior,
  scientific constraints and shared fixtures that still have consumers.
- Generated runs, raw logs, captures, checkpoints, bulk data and repeated source
  snapshots default to ignored locations. Promote only the small evidence needed
  to review current work. Reuse an existing source bundle instead of copying it
  for every seed or attempt.
- Before removing valuable or irreproducible research, preserve exact bytes in
  a verified existing external archive and retain a concise index/restore pointer.
  Verify restoration and independent durable preservation before removal. Keep
  scientific failures, exclusions and provenance inspectable there. Ordinary
  dead code needs a normal reversible Git deletion, not a permanent legacy tree.
- Update the existing current guide or result summary. Replace stale instructions;
  avoid duplicate status reports, handoffs, giant inventories and per-step diaries.
  Keep essential commands, contracts, conclusions, limitations and evidence links.
- At branch/milestone closeout, remove or archive material whose active purpose
  ended. Review newly tracked files and `git diff --stat` before committing; clean
  up alongside the work rather than postponing another large repository purge.

Historical documents describe their recorded work; they do not require indefinite
tracking of every supporting file. Permanent user datasets, active untracked work
and installed runtimes are separate from repository cleanup.

An optional user-only HDD holds additional datasets, models, projects and archives.
See the [local storage ledger](docs/local-storage-ledger.md) for its location,
contents and collaborator access limits; maintain that ledger when assets change.

## Desktop interface changes

Use the [desktop UI principles and review checklist](desktop/src/README.md#desktop-ui-principles)
when changing a visible workflow. In the change description, identify the task
that gets easier and where the action fits in the existing workspace. For a
layout or rendering change, describe what you saw in the actual app at normal
and compact window sizes.
