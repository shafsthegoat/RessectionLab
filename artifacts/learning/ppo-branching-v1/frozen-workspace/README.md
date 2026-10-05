# RessectionLab

A local macOS research workspace for inspecting patient-specific glioma access
routes and testing simulated resection strategies.

Development is in progress. See [PROJECT_STATUS.md](PROJECT_STATUS.md) for what
has actually been run. The October 4, 2026 [master plan](MASTER_PLAN.md),
[annotated references](ANNOTATED_REFERENCES.md), and
[implementation handoff](IMPLEMENTING_AGENT_HANDOFF.md) define the project.

## Local development

The initial tested dependency environment is Apple Silicon, macOS 26.6 and
Python 3.12. Use an isolated environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e . --no-deps
.venv/bin/python -m pytest
```

`requirements-lock.txt` records exact versions installed on the development
machine. It is not evidence that every dependency combination or platform is
validated. Mac application packaging is a separate acceptance gate.

## Evidence and interpretation

- Source imaging, supplied annotations, estimated anatomy, population priors and
  simulator-generated experience have separate provenance.
- Route accessibility is distinct from tissue removed in a legal simulated
  sequence. All geometry uses a declared physical frame and millimeters.
- Motor/language proximity and modeled structural encounters are research
  surrogates. Clinical neurological-deficit probabilities are unavailable.
- Missing diffusion, functional mapping, vessels or skull anatomy remain
  unassessed. Public MRI plus a plausible visualization does not establish
  surgical safety or clinical readiness.
- Patient-specific learning updates a policy under frozen simulator assumptions.
  Optimization, checkpoint selection and independent evaluation use separate
  worlds. Search remains a required comparator and may be the better planner.

## Data and privacy

Raw public imaging, local case bundles and checkpoints are excluded from Git.
Download only named cases under verified terms and retain source checksums.
Mirrored data is identified as such; a mirror checksum does not prove byte
equivalence to an authoritative release. The application is local-first; no
cloud processing or paid infrastructure is required.

The interface is original. Medivis supplies workflow inspiration only; no
affiliation, compatibility, endorsement or device clearance is implied.
