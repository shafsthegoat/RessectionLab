# Historical Qt migration evidence

The user requested an Electron/React interface on October 4, 2026. The tested
Qt prototype remains in `src/resectionlab/app`; active interface work is in
`desktop/`.

`qt-unfinished.patch` preserves unfinished Qt changes made before that decision.
`qt-native-workflow-unfinished.py` is their unfinished GUI exercise script. It
references those draft controls and has not passed against the restored Qt
prototype. It is archived outside pytest collection so the current numerical
test suite does not import optional Qt solely to collect a historical script.

`../app-refinement-smoke/` preserves an earlier coarse synthetic training smoke
run. It shows actual numerical updates, not a verified native Qt workflow or
source-cell resection performance. Current native patient evidence is recorded
in `../desktop-native-bridge/` and `../learning/native-ucsf0004-v1/`.
