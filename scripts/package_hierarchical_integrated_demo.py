#!/usr/bin/env python3
"""Freeze and verify the local hierarchical sector-policy demonstration.

This is a separate package profile. It never changes the historical or v6
archives. Source is copied from one named Git commit; sealed cases, checkpoints,
and receipts are copied byte for byte. Outputs belong under ignored testcases/.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

import package_integrated_demo as historical

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = Path('testcases/outputs/integrated-goal-20261026')
FRESH = EVIDENCE/'fresh-patients'
FIT = EVIDENCE/'learning/patient-sector-fit-v4'
RAW = Path('testcases/downloaded-datasets/UCSF-fresh-development-20261026')
SCHEMA = 'resectionlab.integrated-hierarchical-local-package.v1'
SEEDS = (17, 29, 43)
METHODS = ('bc', 'ppo')
CASES = {'train': '0007', 'transfer': '0005'}

# The legacy v1 manifest has no profile field. Keep its builder and verifier
# below unchanged; a v2 archive must opt into these exact input identities.
EXIT_PROFILE_ID = 'exit-preflight-v1'
EXIT_SCHEMA = 'resectionlab.integrated-hierarchical-local-package.v2'
EXIT_FIT_NAME = 'patient-sector-fit-v5'
EXIT_SOURCE_VERSION = 'v4'
EXIT_SOURCE_SHA256 = {
    '0007': '2441fc3f9c39ceaa0be495806fe9b28882d973527970685b373b645f8e084143',
    '0005': '4047e1dcd8197b2319add61ba24c0c0b36782b15725c8cb156ecd864d288a8c7',
}


def _exit_case_path(patient):
    return f'cases/{patient}-sector-{EXIT_SOURCE_VERSION}.ressectionlab'


def _exit_source_dir(patient):
    return ROOT/FRESH/f'UCSF-PDGM-{patient}'/f'sector-comparison-{EXIT_SOURCE_VERSION}'


def _exit_profile(fit_sha256, desktop_report_shas):
    """Semantic contract stored in v2 manifests, independent of host paths."""
    return {
        'id': EXIT_PROFILE_ID,
        'sourceVersion': EXIT_SOURCE_VERSION,
        'sources': {patient: {'path': _exit_case_path(patient),
                              'sha256': EXIT_SOURCE_SHA256[patient],
                              'receipt': f'evidence/sector-comparison-{patient}-{EXIT_SOURCE_VERSION}/receipt.json'}
                    for patient in ('0007', '0005')},
        'fitting': {'receipt': f'evidence/{EXIT_FIT_NAME}/receipt.json',
                    'sha256': fit_sha256},
        'desktopInference': {patient: {'report': f'evidence/desktop-{patient}/report.json',
                                       'sha256': desktop_report_shas[patient]}
                             for patient in ('0007', '0005')},
        'selectedMethod': 'ppo',
        'selectedSeed': 17,
        'seeds': list(SEEDS),
        'taskGates': {'0007': True, '0005': False},
        'cleanupComplete': {'0007': True, '0005': True},
    }


def _json(path):
    return json.loads(Path(path).read_text())


def _under_testcases(path):
    path = Path(path).resolve()
    if not path.is_relative_to((ROOT/'testcases').resolve()):
        raise ValueError('Output must be under ignored testcases/')
    return path


def _data_members(desktop_reports):
    fixed = {}
    for case, patient in CASES.items():
        source = FRESH/f'UCSF-PDGM-{patient}/sector-comparison-v3'
        fixed[f'cases/{patient}-sector-v3.ressectionlab'] = ROOT/source/'sector-comparison.ressectionlab'
        fixed[f'evidence/sector-comparison-{patient}-v3/receipt.json'] = ROOT/source/'receipt.json'
        for item in sorted((ROOT/source).rglob('*')):
            if item.is_file() and item.name not in ('receipt.json', 'sector-comparison.ressectionlab'):
                fixed[f'evidence/sector-comparison-{patient}-v3/{item.relative_to(ROOT/source)}'] = item
    for item in sorted((ROOT/FIT).rglob('*')):
        if item.is_file():
            fixed[f'evidence/patient-sector-fit-v4/{item.relative_to(ROOT/FIT)}'] = item
    for patient in ('0007', '0005'):
        companion = ROOT/EVIDENCE/f'learning/sector-cold-companion-{patient}-v1'
        for item in sorted(companion.rglob('*')):
            if item.is_file():
                fixed[f'evidence/cold-companion-{patient}/{item.relative_to(companion)}'] = item
        for failure in sorted((ROOT/EVIDENCE).glob(f'electron-sector-policy-execute-{patient}-v*')):
            if not failure.is_dir() or not (failure/'report.json').is_file():
                continue
            if _json(failure/'report.json').get('passed') is not False:
                continue
            label = failure.name.removeprefix('electron-sector-policy-execute-')
            for item in sorted(failure.rglob('*')):
                if item.is_file():
                    fixed[f'evidence/desktop-failed-{label}/{item.relative_to(failure)}'] = item
            log = failure.with_suffix('.log')
            if log.is_file():
                fixed[f'evidence/desktop-failed-{label}.log'] = log
    for label in ('electron-sector-generate-0007-v2',
                  'electron-sector-inspect-0005-v1'):
        proof = ROOT/EVIDENCE/label
        report = _json(proof/'report.json')
        if report.get('passed') is not True:
            raise ValueError('Sector generation/inspection desktop proof is not passed')
        for item in sorted(proof.rglob('*')):
            if item.is_file():
                fixed[f'evidence/{label}/{item.relative_to(proof)}'] = item
    for patient, report_path in desktop_reports.items():
        report_path = Path(report_path).resolve()
        report = _json(report_path)
        expected_source = fixed[f'cases/{patient}-sector-v3.ressectionlab']
        if (report.get('passed') is not True or report.get('mode') != 'sector-execute'
                or Path(report.get('input', '')).resolve() != expected_source.resolve()
                or report_path.name != 'report.json'
                or not report_path.is_relative_to((ROOT/EVIDENCE).resolve())):
            raise ValueError(f'Desktop {patient} report is not successful evidence for the source case')
        report_root = report_path.parent
        run_receipt = Path(report['executionReceipt']).resolve()
        run_root = run_receipt.parent.parent
        allowed = (ROOT/'testcases/outputs/sector-desktop').resolve()
        if not run_root.is_relative_to(allowed) or run_receipt != run_root/'execution/receipt.json':
            raise ValueError('Desktop native execution lies outside its ignored run')
        saved = Path(report.get('savedCase', '')).resolve()
        if saved != report_root/'desktop-saved.ressectionlab' or not saved.is_file():
            raise ValueError('Desktop saved native evidence differs')
        execution = _json(run_receipt)
        if (execution.get('status') != 'evaluated' or execution.get('optimizerUpdates') != 0
                or execution['source']['unchanged'] is not True
                or execution['source']['sha256'] !=
                    historical.sha(run_root/'input.ressectionlab')):
            raise ValueError('Desktop frozen inference source/freeze differs')
        for item in sorted(report_root.rglob('*')):
            if item.is_file():
                fixed[f'evidence/desktop-{patient}/{item.relative_to(report_root)}'] = item
        for item in sorted(run_root.rglob('*')):
            if item.is_file():
                fixed[f'evidence/desktop-{patient}-run/{item.relative_to(run_root)}'] = item
    for folder in ('data/models/synthstrip-v1', 'data/models/raidionics/MRI_GBM'):
        for item in sorted((ROOT/folder).rglob('*')):
            if item.is_file():
                fixed[str(item.relative_to(ROOT))] = item
    for item in sorted((ROOT/RAW).rglob('*')):
        if item.is_file():
            fixed[str(item.relative_to(ROOT))] = item
    for target, source in sorted(fixed.items()):
        role = ('case' if target.startswith('cases/') else
                'source-input' if target.startswith('testcases/') else
                'model' if target.startswith('data/models/') else 'evidence')
        yield target, source, role


def _instructions(commit):
    return f'''# Hierarchical sector-policy local reproduction

Source commit: `{commit}`. This package contains source-anatomy research
trajectories, not clinical guidance. Both source MRIs are skull-stripped; the
brain layer is an unreviewed SynthStrip estimate. Whole-head access, vessels,
functional protection and calibrated clinical risk remain unknown. The 0005
transfer outcome is incomplete and must remain a failed task gate.

All six seed 17/29/43 BC and PPO checkpoints, their twelve train/transfer
native trial workspaces, both v3 sector source cases, training receipt,
independent audit, producer snapshot, selected seed 17 cold companion runs,
and real Electron positive/negative evidence are included. Earlier failed
Electron attempts remain labeled failed. Seed 17 was declared for the demo;
it was not selected by maximizing transfer performance. The ten original
pinned NIfTIs and DEVELOPMENT declaration, source receipts, SynthStrip and
MRI_GBM installed bytes, acquisition receipts and license notices are included.
Historical absolute paths inside sealed receipts are provenance only. Runtime
commands below explicitly pass bundled source and checkpoint paths.

The archive does not provision Python, torch, numpy, scipy, the imaging model
runtimes, Node modules or Electron. Frozen verification requires a matching
same-host Python environment and does not prove fresh-install inference or a
new Electron installation. The checkpoint binds exact producer source and
runtime versions. From the extracted bundle, use a matching Python executable:

```sh
PY=/absolute/path/to/matching/.venv/bin/python
"$PY" scripts/package_hierarchical_integrated_demo.py verify --bundle .
"$PY" scripts/package_hierarchical_integrated_demo.py verify-models --bundle .
"$PY" scripts/package_hierarchical_integrated_demo.py verify-native --bundle .
"$PY" scripts/package_hierarchical_integrated_demo.py cold-run --bundle . \\
  --output ../testcases/hierarchical-cold-all-six-v1
```

`cold-run` executes each checkpoint on both 0007 and 0005 from the bundled
intact v3 source, using the packaged executor and producer source. It performs
zero optimizer updates, compares each actual native endpoint with its fitting
trial, verifies source hashes unchanged, and writes a cold verification receipt.
It has a 600-second aggregate and 120-second per-run bound. Use a fresh output
directory for another run. A single direct seed 17 execution is:

```sh
"$PY" scripts/execute_patient_sector_policy.py \\
  --source cases/0007-sector-v3.ressectionlab \\
  --checkpoint evidence/patient-sector-fit-v4/ppo-seed17.pt \\
  --source-role DEVELOPMENT --output ../testcases/my-cold-0007-seed17
```
'''


def _exit_instructions(commit):
    return f'''# Exit-preflight hierarchical policy reproduction

Profile: `{EXIT_PROFILE_ID}`. Source commit: `{commit}`. The two sealed source
cases are sector-comparison-v4; training is patient-sector-fit-v5. Seed 17 was
chosen before evaluating these cases. UCSF-PDGM-0007 passes the modeled task
gate; 0005 completes instrument cleanup but still fails the task gate. Neither
outcome is clinical validation. Missing skull, vascular, and functional anatomy
and uncalibrated simulation effects remain explicit limitations.

The manifest pins each source case, fitting receipt, and real Electron inference
report by relative path and SHA256. The bundle includes all six BC/PPO frozen
checkpoints, twelve fitting native trials, raw DEVELOPMENT input and model bytes.
Absolute paths inside historical receipts record provenance; runtime commands
use bundled files. The archive does not install Python, torch, imaging runtimes,
Node modules, or Electron. Model checks need the matching same-host runtime.
Electron's native input snapshots can have different archive bytes after workspace
serialization. Each snapshot is pinned to its frozen execution receipt, and
`verify-native` compares its source case, comparison seals and every original
branch state/action identity with the sealed v4 source.

```sh
PY=/absolute/path/to/matching/.venv/bin/python
"$PY" scripts/package_hierarchical_integrated_demo.py verify --bundle .
"$PY" scripts/package_hierarchical_integrated_demo.py verify-models --bundle .
"$PY" scripts/package_hierarchical_integrated_demo.py verify-native --bundle .
"$PY" scripts/package_hierarchical_integrated_demo.py cold-run --bundle . \\
  --output ../testcases/exit-preflight-cold-all-six-v1
```

The cold run executes each checkpoint on both unchanged v4 sources with zero
optimizer updates and compares the actual native endpoint with its fitting
trial. It has a 600-second aggregate and 120-second per-run bound.
'''


def _exit_data_members(desktop_reports):
    fixed = {}
    for patient in ('0007', '0005'):
        source_dir = _exit_source_dir(patient)
        source = source_dir/'sector-comparison.ressectionlab'
        if historical.sha(source) != EXIT_SOURCE_SHA256[patient]:
            raise ValueError(f'Exit profile {patient} source hash differs')
        fixed[_exit_case_path(patient)] = source
        for item in sorted(source_dir.rglob('*')):
            if item.is_file() and item != source:
                fixed[f'evidence/sector-comparison-{patient}-{EXIT_SOURCE_VERSION}/{item.relative_to(source_dir)}'] = item
    fit_dir = ROOT/EVIDENCE/'learning'/EXIT_FIT_NAME
    if not (fit_dir/'receipt.json').is_file() or not (fit_dir/'independent-audit.json').is_file():
        raise ValueError('Exit profile requires fit-v5 receipt and independent audit')
    for item in sorted(fit_dir.rglob('*')):
        if item.is_file():
            fixed[f'evidence/{EXIT_FIT_NAME}/{item.relative_to(fit_dir)}'] = item
    for patient in ('0007', '0005'):
        comparison = ROOT/EVIDENCE/f'electron-sector-escape-{patient}-v1'
        report = _json(comparison/'report.json')
        if (report.get('passed') is not True or report.get('mode') != 'sector' or
                Path(report.get('input', '')).resolve() !=
                    _exit_source_dir(patient)/'sector-comparison.ressectionlab'):
            raise ValueError('Exit profile comparison Electron report differs')
        for item in sorted(comparison.rglob('*')):
            if item.is_file():
                fixed[f'evidence/desktop-comparison-{patient}/{item.relative_to(comparison)}'] = item
        report_path = Path(desktop_reports[patient]).resolve()
        inference = _json(report_path)
        expected_source = _exit_source_dir(patient)/'sector-comparison.ressectionlab'
        if (report_path.name != 'report.json' or
                not report_path.is_relative_to((ROOT/EVIDENCE).resolve()) or
                inference.get('passed') is not True or
                inference.get('mode') != 'sector-execute' or
                inference.get('windowMode') != 'hidden' or
                inference.get('windowEverVisible') is not False or
                Path(inference.get('input', '')).resolve() != expected_source):
            raise ValueError(f'Exit profile {patient} Electron inference report differs')
        report_root = report_path.parent
        run_receipt = Path(inference['executionReceipt']).resolve()
        run_root = run_receipt.parent.parent
        if (not run_root.is_relative_to((ROOT/'testcases/outputs/sector-desktop').resolve()) or
                run_receipt != run_root/'execution/receipt.json' or
                Path(inference.get('savedCase', '')).resolve() !=
                    report_root/'desktop-saved.ressectionlab'):
            raise ValueError('Exit profile Electron native run path differs')
        for item in sorted(report_root.rglob('*')):
            if item.is_file():
                fixed[f'evidence/desktop-{patient}/{item.relative_to(report_root)}'] = item
        for item in sorted(run_root.rglob('*')):
            if item.is_file():
                fixed[f'evidence/desktop-{patient}-run/{item.relative_to(run_root)}'] = item
    for folder in ('data/models/synthstrip-v1', 'data/models/raidionics/MRI_GBM'):
        for item in sorted((ROOT/folder).rglob('*')):
            if item.is_file():
                fixed[str(item.relative_to(ROOT))] = item
    for item in sorted((ROOT/RAW).rglob('*')):
        if item.is_file():
            fixed[str(item.relative_to(ROOT))] = item
    for target, source in sorted(fixed.items()):
        role = ('case' if target.startswith('cases/') else
                'source-input' if target.startswith('testcases/') else
                'model' if target.startswith('data/models/') else 'evidence')
        yield target, source, role


def _exit_build(output, commit, desktop_reports):
    output = _under_testcases(output)
    if output.exists():
        raise ValueError('Choose a new package output directory')
    if set(desktop_reports) != {'0007', '0005'}:
        raise ValueError('Supply successful Electron reports for both development cases')
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', commit+'^{commit}'],
                                     cwd=ROOT, text=True).strip()
    if len(commit) != 40:
        raise ValueError('Freeze a complete committed source revision')
    helper = 'scripts/package_hierarchical_integrated_demo.py'
    committed_helper = subprocess.run(['git', 'show', f'{commit}:{helper}'],
                                      cwd=ROOT, capture_output=True)
    if committed_helper.returncode == 0 and committed_helper.stdout != Path(__file__).read_bytes():
        raise ValueError('Committed package helper differs from active builder; commit it before build')
    members = list(_exit_data_members(desktop_reports))
    output.mkdir(parents=True)
    bundle = output/'bundle'
    bundle.mkdir()
    roles, origins = {}, {}
    for path, mode, oid in historical.committed_files(commit):
        data = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=ROOT)
        historical.write_bytes(bundle/path, data, 0o755 if mode == '100755' else 0o644)
        roles[path], origins[path] = 'committed-source', f'git:{commit}:{path}'
    if helper in roles:
        if historical.sha(bundle/helper) != historical.sha(Path(__file__)):
            raise ValueError('Committed package helper differs from active builder; commit it before build')
    else:
        historical.copy_exact(Path(__file__), bundle/helper)
        roles[helper], origins[helper] = 'package-helper', 'local package builder outside actor fingerprint'
    for target, source, role in members:
        historical.copy_exact(source, bundle/target)
        roles[target], origins[target] = role, str(source.relative_to(ROOT))
    historical.write_bytes(bundle/'VERIFY.md', _exit_instructions(commit).encode())
    roles['VERIFY.md'], origins['VERIFY.md'] = 'instructions', 'generated package instructions'
    files = [{'path': p, 'role': roles[p], 'origin': origins[p],
              'bytes': (bundle/p).stat().st_size, 'sha256': historical.sha(bundle/p)}
             for p in sorted(roles)]
    fit_path = bundle/f'evidence/{EXIT_FIT_NAME}/receipt.json'
    fitting = _json(fit_path)
    selected = next(row for row in fitting['seedRuns'] if row['seed'] == 17)
    profile = _exit_profile(historical.sha(fit_path),
                            {patient: historical.sha(bundle/f'evidence/desktop-{patient}/report.json')
                             for patient in ('0007', '0005')})
    tree = subprocess.check_output(['git', 'rev-parse', commit+'^{tree}'], cwd=ROOT,
                                   text=True).strip()
    manifest = {'schema': EXIT_SCHEMA, 'profile': profile,
                'sourceCommit': commit, 'sourceTree': tree,
                'files': files, 'fileCount': len(files),
                'contentBytes': sum(row['bytes'] for row in files),
                'selectedDemoSeed': 17,
                'selectedDemo': {
                    'checkpoint': f'evidence/{EXIT_FIT_NAME}/ppo-seed17.pt',
                    'positiveSource': _exit_case_path('0007'),
                    'negativeSource': _exit_case_path('0005'),
                    'positiveTaskGateMet': selected['ppo']['evaluations']['train']['report']['taskGateMet'],
                    'negativeTaskGateMet': selected['ppo']['evaluations']['transfer']['report']['taskGateMet']},
                'limitations': ['Same-host producer/runtime fingerprint required',
                                '0005 cleanup complete but modeled task gate failed',
                                'Historical absolute provenance is informational after relocation',
                                'Fresh-install imaging inference and Electron installation not verified']}
    historical.write_bytes(bundle/'manifest.json',
        (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode())
    verify(bundle)
    archive = output/'integrated-hierarchical-local.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb',
                                                 mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for path in sorted(p for p in bundle.rglob('*') if p.is_file()):
                info = tarfile.TarInfo(str(path.relative_to(bundle)))
                info.size, info.mode = path.stat().st_size, path.stat().st_mode & 0o777
                info.uid = info.gid = info.mtime = 0
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    result = {'schema': 'resectionlab.integrated-hierarchical-package-build.v2',
              'profile': EXIT_PROFILE_ID, 'archive': str(archive),
              'archiveSha256': historical.sha(archive),
              'archiveBytes': archive.stat().st_size, 'bundle': str(bundle),
              'sourceCommit': commit, 'fileCount': len(files),
              'contentBytes': manifest['contentBytes']}
    (output/'build-receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def build(output, commit, desktop_reports, *, profile='legacy-v1'):
    if profile == EXIT_PROFILE_ID:
        return _exit_build(output, commit, desktop_reports)
    if profile != 'legacy-v1':
        raise ValueError('Unknown hierarchical package profile')
    output = _under_testcases(output)
    if output.exists():
        raise ValueError('Choose a new package output directory')
    if set(desktop_reports) != {'0007', '0005'}:
        raise ValueError('Supply successful Electron reports for both development cases')
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', commit+'^{commit}'],
                                     cwd=ROOT, text=True).strip()
    if len(commit) != 40:
        raise ValueError('Freeze a complete committed source revision')
    output.mkdir(parents=True)
    bundle = output/'bundle'
    bundle.mkdir()
    roles, origins = {}, {}
    for path, mode, oid in historical.committed_files(commit):
        data = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=ROOT)
        historical.write_bytes(bundle/path, data, 0o755 if mode == '100755' else 0o644)
        roles[path], origins[path] = 'committed-source', f'git:{commit}:{path}'
    required = {'src/resectionlab/surgical_engine.py',
                'src/resectionlab/surgical_prefix_json_native.c',
                'src/resectionlab/surgical_sector_planning.py',
                'src/resectionlab/patient_sector_learning.py',
                'scripts/run_patient_sector_learning.py',
                'scripts/execute_patient_sector_policy.py',
                'scripts/compare_anatomy_sectors.py',
                'desktop/pnpm-lock.yaml'}
    if not required <= roles.keys():
        raise ValueError('Committed source snapshot lacks required hierarchical source')
    helper = 'scripts/package_hierarchical_integrated_demo.py'
    if helper not in roles:
        historical.copy_exact(Path(__file__), bundle/helper)
        roles[helper], origins[helper] = 'package-helper', 'local package builder outside actor fingerprint'
    for target, source, role in _data_members(desktop_reports):
        historical.copy_exact(source, bundle/target)
        roles[target], origins[target] = role, str(source.relative_to(ROOT))
    historical.write_bytes(bundle/'VERIFY.md', _instructions(commit).encode())
    roles['VERIFY.md'], origins['VERIFY.md'] = 'instructions', 'generated package instructions'
    files = [{'path': p, 'role': roles[p], 'origin': origins[p],
              'bytes': (bundle/p).stat().st_size, 'sha256': historical.sha(bundle/p)}
             for p in sorted(roles)]
    fitting = _json(bundle/'evidence/patient-sector-fit-v4/receipt.json')
    selected = next(row for row in fitting['seedRuns'] if row['seed'] == 17)
    tree = subprocess.check_output(['git', 'rev-parse', commit+'^{tree}'], cwd=ROOT,
                                   text=True).strip()
    manifest = {'schema': SCHEMA, 'sourceCommit': commit, 'sourceTree': tree,
                'files': files, 'fileCount': len(files),
                'contentBytes': sum(row['bytes'] for row in files),
                'selectedDemoSeed': 17,
                'selectedDemo': {
                    'checkpoint': 'evidence/patient-sector-fit-v4/ppo-seed17.pt',
                    'positiveSource': 'cases/0007-sector-v3.ressectionlab',
                    'negativeSource': 'cases/0005-sector-v3.ressectionlab',
                    'positiveTaskGateMet': selected['ppo']['evaluations']['train']['report']['taskGateMet'],
                    'negativeTaskGateMet': selected['ppo']['evaluations']['transfer']['report']['taskGateMet']},
                'limitations': ['Same-host producer/runtime fingerprint required',
                                '0005 remains incomplete with failed task gate',
                                'Historical absolute provenance is informational after relocation',
                                'Fresh-install imaging inference and Electron installation not verified']}
    historical.write_bytes(bundle/'manifest.json',
        (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode())
    verify(bundle)
    archive = output/'integrated-hierarchical-local.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb',
                                                 mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for path in sorted(p for p in bundle.rglob('*') if p.is_file()):
                info = tarfile.TarInfo(str(path.relative_to(bundle)))
                info.size, info.mode = path.stat().st_size, path.stat().st_mode & 0o777
                info.uid = info.gid = info.mtime = 0
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    result = {'schema': 'resectionlab.integrated-hierarchical-package-build.v1',
              'archive': str(archive), 'archiveSha256': historical.sha(archive),
              'archiveBytes': archive.stat().st_size, 'bundle': str(bundle),
              'sourceCommit': commit, 'fileCount': len(files),
              'contentBytes': manifest['contentBytes']}
    (output/'build-receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def _exit_verify(bundle, manifest):
    commit = manifest.get('sourceCommit')
    if (manifest.get('schema') != EXIT_SCHEMA or
            not isinstance(commit, str) or len(commit) != 40 or
            any(ch not in '0123456789abcdef' for ch in commit)):
        raise ValueError('Exit package schema or source commit differs')
    declared = set()
    for row in manifest['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or row['path'] in declared:
            raise ValueError('Unsafe or duplicate package path')
        path = bundle/rel
        if (path.is_symlink() or not path.is_file() or
                not path.resolve().is_relative_to(bundle) or
                path.stat().st_size != row['bytes'] or historical.sha(path) != row['sha256']):
            raise ValueError('Exit package member changed: '+row['path'])
        declared.add(row['path'])
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*')
              if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}
    if (actual != declared | {'manifest.json'} or len(declared) != manifest['fileCount'] or
            sum(row['bytes'] for row in manifest['files']) != manifest['contentBytes']):
        raise ValueError('Exit package inventory differs')
    required = {'src/resectionlab/surgical_engine.py',
                'src/resectionlab/surgical_prefix_json_native.c',
                'src/resectionlab/surgical_sector_planning.py',
                'src/resectionlab/patient_sector_learning.py',
                'scripts/run_patient_sector_learning.py',
                'scripts/execute_patient_sector_policy.py',
                'scripts/compare_anatomy_sectors.py',
                'scripts/package_hierarchical_integrated_demo.py',
                'desktop/pnpm-lock.yaml'}
    if not required <= declared:
        raise ValueError('Exit package source snapshot is incomplete')
    profile = manifest.get('profile')
    if not isinstance(profile, dict):
        raise ValueError('Exit package profile is missing')
    fit_path = bundle/f'evidence/{EXIT_FIT_NAME}/receipt.json'
    desktop_shas = {patient: historical.sha(bundle/f'evidence/desktop-{patient}/report.json')
                    for patient in ('0007', '0005')}
    expected_profile = _exit_profile(historical.sha(fit_path), desktop_shas)
    if profile != expected_profile:
        raise ValueError('Exit package profile paths or hashes differ')
    fit = _json(fit_path)
    audit_root = bundle/f'evidence/{EXIT_FIT_NAME}'
    audit = _json(audit_root/'independent-audit.json')
    if (fit.get('status') != 'saved' or fit.get('seeds') != list(SEEDS) or
            len(fit.get('seedRuns', [])) != len(SEEDS) or
            audit.get('allPassed') is not True or
            audit.get('runReceiptSha256') != historical.sha(fit_path)):
        raise ValueError('Exit fitting receipt or independent audit differs')
    selected = next((row for row in fit['seedRuns'] if row['seed'] == 17), None)
    if selected is None or {row['seed'] for row in fit['seedRuns']} != set(SEEDS):
        raise ValueError('Exit fitting seeds differ')
    demo = manifest.get('selectedDemo')
    reference = selected['ppo']['evaluations']
    if (manifest.get('selectedDemoSeed') != 17 or
            demo != {'checkpoint': f'evidence/{EXIT_FIT_NAME}/ppo-seed17.pt',
                     'positiveSource': _exit_case_path('0007'),
                     'negativeSource': _exit_case_path('0005'),
                     'positiveTaskGateMet': True, 'negativeTaskGateMet': False} or
            reference['train']['report'].get('taskGateMet') is not True or
            reference['transfer']['report'].get('taskGateMet') is not False or
            any(reference[case]['report'].get('cleanupComplete') is not True
                for case in CASES)):
        raise ValueError('Exit selected demo gate or cleanup differs')
    producer_paths = {
        'run_patient_sector_learning.py': 'scripts/run_patient_sector_learning.py',
        'patient_sector_learning.py': 'src/resectionlab/patient_sector_learning.py',
        'surgical_sector_planning.py': 'src/resectionlab/surgical_sector_planning.py',
        'surgical_rl_env.py': 'src/resectionlab/surgical_rl_env.py',
    }
    for name, source in producer_paths.items():
        expected = audit['producerSnapshot'][name]
        if (historical.sha(audit_root/'producer-snapshot'/name) != expected or
                historical.sha(bundle/source) != expected):
            raise ValueError('Exit committed producer differs from checkpoint seal')
    source_shas = {}
    for case, patient in CASES.items():
        source = bundle/_exit_case_path(patient)
        source_shas[case] = historical.sha(source)
        receipt = _json(bundle/f'evidence/sector-comparison-{patient}-{EXIT_SOURCE_VERSION}/receipt.json')
        if (source_shas[case] != EXIT_SOURCE_SHA256[patient] or
                source_shas[case] != fit['sources'][case]['fileSha256'] or
                receipt.get('savedSha256') != source_shas[case] or
                receipt.get('sourcePreserved') is not True or
                receipt.get('sectorPlanning', {}).get('declaredGate', {}).get('checkedCleanup') is not True):
            raise ValueError('Exit source case or comparison receipt differs')
        comparison = _json(bundle/f'evidence/desktop-comparison-{patient}/report.json')
        if (comparison.get('passed') is not True or comparison.get('mode') != 'sector' or
                comparison.get('sourcePreserved') is not True):
            raise ValueError('Exit real Electron comparison proof differs')
    declaration = bundle/RAW/'development-declaration.json'
    for case in CASES:
        if historical.sha(declaration) != fit['sources'][case]['roleDeclaration']['declarationSha256']:
            raise ValueError('Exit development role declaration differs')
    for patient in ('UCSF-PDGM-0005', 'UCSF-PDGM-0007'):
        original = _json(bundle/RAW/patient/'source-receipt.json')
        for item in original['files']:
            path = bundle/item['path']
            if path.stat().st_size != item['size_bytes'] or historical.sha(path) != item['sha256']:
                raise ValueError('Exit pinned original NIfTI differs')
    synth = _json(bundle/'data/models/synthstrip-v1/poc-acquisition.json')
    for item in synth['files']:
        path = bundle/item['path']
        if path.stat().st_size != item['bytes'] or historical.sha(path) != item['sha256']:
            raise ValueError('Exit SynthStrip installation differs')
    gbm = _json(bundle/'data/models/raidionics/MRI_GBM/installation.json')
    for filename, expected in gbm['files'].items():
        if historical.sha(bundle/'data/models/raidionics/MRI_GBM'/filename) != expected:
            raise ValueError('Exit MRI_GBM installation differs')
    for run in fit['seedRuns']:
        for method in METHODS:
            model = run[method]['model']
            checkpoint = audit_root/Path(model['path']).name
            if historical.sha(checkpoint) != model['fileSha256']:
                raise ValueError('Exit fitting checkpoint differs')
            for case in CASES:
                evaluation = run[method]['evaluations'][case]
                native = audit_root/Path(evaluation['derived']['workspacePath']).name
                if historical.sha(native) != evaluation['derived']['fileSha256']:
                    raise ValueError('Exit fitting native trial differs')
    for case, patient in CASES.items():
        report_root = bundle/f'evidence/desktop-{patient}'
        report = _json(report_root/'report.json')
        run_root = bundle/f'evidence/desktop-{patient}-run'
        execution_receipt = _json(run_root/'execution/receipt.json')
        expected = reference[case]['report']
        execution = execution_receipt['execution']
        derived = run_root/'execution'/Path(execution_receipt['derived']['workspacePath']).name
        if (report.get('passed') is not True or report.get('mode') != 'sector-execute' or
                report.get('windowMode') != 'hidden' or
                report.get('windowEverVisible') is not False or
                report.get('sourcePreserved') is not True or
                execution_receipt.get('status') != 'evaluated' or
                execution_receipt.get('optimizerUpdates') != 0 or
                execution_receipt['source']['unchanged'] is not True or
                execution_receipt['source']['sha256'] !=
                    historical.sha(run_root/'input.ressectionlab') or
                execution_receipt['model']['fileSha256'] !=
                    selected['ppo']['model']['fileSha256'] or
                execution_receipt['derived']['reopenedExact'] is not True or
                historical.sha(derived) != execution_receipt['derived']['fileSha256'] or
                not (report_root/'desktop-saved.ressectionlab').is_file() or
                any(execution.get(key) != expected.get(key) for key in (
                    'status', 'taskGateMet', 'cleanupComplete', 'finalStateId',
                    'deltaTargetMm3', 'deltaNonTargetMm3', 'perSectorRemovedMm3'))):
            raise ValueError('Exit real Electron frozen inference differs')
    return {'verified': True, 'profile': EXIT_PROFILE_ID, 'sourceCommit': commit,
            'files': len(declared), 'contentBytes': manifest['contentBytes'],
            'seeds': list(SEEDS), 'checkpoints': 6, 'fittingNativeTrials': 12}


def verify(bundle):
    bundle = Path(bundle).resolve()
    manifest = _json(bundle/'manifest.json')
    if manifest.get('schema') == EXIT_SCHEMA:
        return _exit_verify(bundle, manifest)
    commit = manifest.get('sourceCommit')
    if (manifest.get('schema') != SCHEMA or manifest.get('selectedDemoSeed') != 17
            or not isinstance(commit, str) or len(commit) != 40
            or any(ch not in '0123456789abcdef' for ch in commit)):
        raise ValueError('Hierarchical package schema or source commit differs')
    declared = set()
    for row in manifest['files']:
        rel = Path(row['path'])
        if rel.is_absolute() or '..' in rel.parts or row['path'] in declared:
            raise ValueError('Unsafe or duplicate package path')
        path = bundle/rel
        if (path.is_symlink() or not path.is_file() or
                not path.resolve().is_relative_to(bundle) or
                path.stat().st_size != row['bytes'] or historical.sha(path) != row['sha256']):
            raise ValueError('Hierarchical package member changed: '+row['path'])
        declared.add(row['path'])
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*')
              if (p.is_file() or p.is_symlink()) and '__pycache__' not in p.parts}
    if (actual != declared | {'manifest.json'} or len(declared) != manifest['fileCount']
            or sum(row['bytes'] for row in manifest['files']) != manifest['contentBytes']):
        raise ValueError('Hierarchical package inventory differs')
    essential = {'src/resectionlab/surgical_engine.py',
                 'src/resectionlab/surgical_prefix_json_native.c',
                 'src/resectionlab/surgical_sector_planning.py',
                 'src/resectionlab/patient_sector_learning.py',
                 'scripts/run_patient_sector_learning.py',
                 'scripts/execute_patient_sector_policy.py',
                 'scripts/compare_anatomy_sectors.py',
                 'desktop/pnpm-lock.yaml'}
    if not essential <= declared:
        raise ValueError('Hierarchical package source snapshot is incomplete')
    fit = _json(bundle/'evidence/patient-sector-fit-v4/receipt.json')
    audit = _json(bundle/'evidence/patient-sector-fit-v4/independent-audit.json')
    if (fit.get('status') != 'saved' or fit.get('seeds') != list(SEEDS)
            or len(fit.get('seedRuns', [])) != 3 or audit.get('allPassed') is not True
            or historical.sha(bundle/'evidence/patient-sector-fit-v4/receipt.json') !=
                audit['runReceiptSha256']):
        raise ValueError('V4 fitting receipt or independent audit differs')
    demo = manifest['selectedDemo']
    if (demo['checkpoint'] != 'evidence/patient-sector-fit-v4/ppo-seed17.pt'
            or demo['positiveSource'] != 'cases/0007-sector-v3.ressectionlab'
            or demo['negativeSource'] != 'cases/0005-sector-v3.ressectionlab'
            or demo['positiveTaskGateMet'] is not True
            or demo['negativeTaskGateMet'] is not False):
        raise ValueError('Predeclared positive/negative seed 17 demo differs')
    producer_paths = {
        'run_patient_sector_learning.py': 'scripts/run_patient_sector_learning.py',
        'patient_sector_learning.py': 'src/resectionlab/patient_sector_learning.py',
        'surgical_sector_planning.py': 'src/resectionlab/surgical_sector_planning.py',
        'surgical_rl_env.py': 'src/resectionlab/surgical_rl_env.py',
    }
    for name, source in producer_paths.items():
        snapshot = bundle/'evidence/patient-sector-fit-v4/producer-snapshot'/name
        if (historical.sha(snapshot) != audit['producerSnapshot'][name] or
                historical.sha(bundle/source) != audit['producerSnapshot'][name]):
            raise ValueError('Packaged committed producer bytes differ from V4 actor seal')
    source_shas = {}
    for case, patient in CASES.items():
        name = f'cases/{patient}-sector-v3.ressectionlab'
        source_shas[case] = historical.sha(bundle/name)
        if source_shas[case] != fit['sources'][case]['fileSha256']:
            raise ValueError('V3 source archive differs from fitting proof')
        source_receipt = _json(bundle/f'evidence/sector-comparison-{patient}-v3/receipt.json')
        if source_receipt['savedSha256'] != source_shas[case] or not source_receipt['sourcePreserved']:
            raise ValueError('V3 sector comparison source receipt differs')
    declaration = bundle/RAW/'development-declaration.json'
    for case in CASES:
        if historical.sha(declaration) != fit['sources'][case]['roleDeclaration']['declarationSha256']:
            raise ValueError('Development declaration differs from source role seal')
    for patient in ('UCSF-PDGM-0005', 'UCSF-PDGM-0007'):
        source_receipt = _json(bundle/RAW/patient/'source-receipt.json')
        for item in source_receipt['files']:
            path = bundle/item['path']
            if path.stat().st_size != item['size_bytes'] or historical.sha(path) != item['sha256']:
                raise ValueError('Pinned original NIfTI differs')
    synth = _json(bundle/'data/models/synthstrip-v1/poc-acquisition.json')
    for item in synth['files']:
        path = bundle/item['path']
        if path.stat().st_size != item['bytes'] or historical.sha(path) != item['sha256']:
            raise ValueError('SynthStrip installation differs')
    gbm = _json(bundle/'data/models/raidionics/MRI_GBM/installation.json')
    for filename, expected in gbm['files'].items():
        if historical.sha(bundle/'data/models/raidionics/MRI_GBM'/filename) != expected:
            raise ValueError('MRI_GBM installation differs')
    for run in fit['seedRuns']:
        seed = run['seed']
        if seed not in SEEDS:
            raise ValueError('Unexpected fitting seed')
        for method in METHODS:
            model = run[method]['model']
            checkpoint = bundle/'evidence/patient-sector-fit-v4'/Path(model['path']).name
            if historical.sha(checkpoint) != model['fileSha256']:
                raise ValueError('Fitting checkpoint differs')
            for case in CASES:
                derived = run[method]['evaluations'][case]['derived']
                native = bundle/'evidence/patient-sector-fit-v4'/Path(derived['workspacePath']).name
                if historical.sha(native) != derived['fileSha256']:
                    raise ValueError('Fitting native trial differs')
    for patient in ('0007', '0005'):
        companion_root = bundle/f'evidence/cold-companion-{patient}'
        companion = _json(companion_root/'receipt.json')
        native = companion_root/Path(companion['derived']['workspacePath']).name
        case = 'train' if patient == '0007' else 'transfer'
        selected_model = fit['seedRuns'][0]['ppo']['model']
        reference = fit['seedRuns'][0]['ppo']['evaluations'][case]['report']
        if (companion['status'] != 'evaluated' or companion['optimizerUpdates'] != 0
                or companion['source']['unchanged'] is not True
                or companion['source']['sha256'] != source_shas[case]
                or companion['model']['fileSha256'] != selected_model['fileSha256']
                or companion['model']['actorSha256'] != selected_model['actorSha256']
                or any(companion['execution'].get(key) != reference.get(key) for key in (
                    'status', 'taskGateMet', 'finalStateId', 'deltaTargetMm3',
                    'deltaNonTargetMm3', 'perSectorRemovedMm3'))
                or companion['exactReplay']['exactNativeReplay'] is not True
                or historical.sha(native) != companion['derived']['fileSha256']):
            raise ValueError('Selected seed 17 cold companion differs')
        desktop_root = bundle/f'evidence/desktop-{patient}'
        report = _json(desktop_root/'report.json')
        if report.get('passed') is not True or report.get('sourcePreserved') is not True:
            raise ValueError('Real Electron hierarchical execution proof differs')
        desktop_receipt = _json(bundle/f'evidence/desktop-{patient}-run/execution/receipt.json')
        execution = desktop_receipt['execution']
        if (desktop_receipt.get('optimizerUpdates') != 0 or
                desktop_receipt.get('status') != 'evaluated' or
                desktop_receipt['source']['unchanged'] is not True or
                desktop_receipt['source']['sha256'] !=
                    historical.sha(bundle/f'evidence/desktop-{patient}-run/input.ressectionlab') or
                desktop_receipt['model']['fileSha256'] !=
                    fit['seedRuns'][0]['ppo']['model']['fileSha256'] or
                desktop_receipt['derived']['reopenedExact'] is not True or
                historical.sha(bundle/f'evidence/desktop-{patient}-run/execution'/
                               Path(desktop_receipt['derived']['workspacePath']).name) !=
                    desktop_receipt['derived']['fileSha256'] or
                any(execution.get(key) != reference.get(key) for key in (
                    'status', 'taskGateMet', 'finalStateId', 'deltaTargetMm3',
                    'deltaNonTargetMm3', 'perSectorRemovedMm3'))):
            raise ValueError('Real Electron frozen inference differs')
    return {'verified': True, 'sourceCommit': commit, 'files': len(declared),
            'contentBytes': manifest['contentBytes'], 'seeds': list(SEEDS),
            'checkpoints': 6, 'fittingNativeTrials': 12}


def extract(archive, output):
    output = _under_testcases(output)
    if output.exists():
        raise ValueError('Choose a new extraction directory')
    output.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe hierarchical archive member')
            path = output/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target, 1024*1024)
            path.chmod(member.mode & 0o777)
    return verify(output)


def _package_modules(bundle):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(bundle/'src'))
    sys.path.insert(0, str(bundle/'scripts'))
    import resectionlab
    import run_patient_sector_learning as runner
    if (not Path(resectionlab.__file__).resolve().is_relative_to(bundle/'src') or
            Path(runner.__file__).resolve() != bundle/'scripts/run_patient_sector_learning.py'):
        raise ValueError('Verification imported producer code outside bundle')
    return runner


def _bundle_paths(bundle):
    schema = _json(bundle/'manifest.json').get('schema')
    if schema == EXIT_SCHEMA:
        return bundle/f'evidence/{EXIT_FIT_NAME}', {
            case: bundle/_exit_case_path(patient) for case, patient in CASES.items()}, True
    if schema == SCHEMA:
        return bundle/'evidence/patient-sector-fit-v4', {
            case: bundle/f'cases/{patient}-sector-v3.ressectionlab'
            for case, patient in CASES.items()}, False
    raise ValueError('Unknown hierarchical package schema')


def model_check(bundle):
    bundle = Path(bundle).resolve()
    verify(bundle)
    runner = _package_modules(bundle)
    from resectionlab.patient_sector_learning import frozen_actor, parameter_hash
    fit_root, _, _ = _bundle_paths(bundle)
    fit = _json(fit_root/'receipt.json')
    producer = runner.producer_identity()
    if producer != fit['producerIdentity']:
        raise ValueError('Packaged producer fingerprint differs from six checkpoint seals')
    checked = []
    for run in fit['seedRuns']:
        for method in METHODS:
            model = run[method]['model']
            path = fit_root/Path(model['path']).name
            actor = frozen_actor(path, expected_producer=producer)
            if parameter_hash(actor) != model['actorSha256']:
                raise ValueError('Packaged frozen actor parameters differ')
            checked.append({'seed': run['seed'], 'stage': method,
                            'fileSha256': model['fileSha256'],
                            'actorSha256': model['actorSha256']})
    return {'verified': True, 'producerIdentity': producer, 'models': checked}


def _native_endpoint(bundle, path, expected_state=None):
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_planning import validate_saved_comparison
    from resectionlab.surgical_sector_planning import validate_saved_sector_comparison
    case, artifacts, _ = load_workspace(path)
    scene = load_environment(path)
    record = artifacts['surgical_toolbox']
    validate_saved_comparison(scene, record, record['planning'])
    validate_saved_sector_comparison(scene, record, record['sectorPlanning'])
    learned = record.get('learnedSectorExecutions', [])
    learned_ids = {row['branchId'] for row in learned}
    original_branches = [
        {'branchId': branch_id, 'stateId': branch['stateId'],
         'actionsHash': semantic_digest(branch['actions'])}
        for branch_id, branch in sorted(record['branches'].items())
        if branch_id not in learned_ids
    ]
    if expected_state is not None:
        if len(learned) != 1:
            raise ValueError('Expected one learned native branch')
        branch = record['branches'][learned[0]['branchId']]
        replay = SurgicalEngine.restore(scene, branch)
        if replay.state_id != expected_state:
            raise ValueError('Learned native endpoint state differs')
    return {'sourceCaseHash': case.semantic_hash,
            'sceneId': scene.identity,
            'accessComparisonSeal': record['planning']['comparisonSeal'],
            'sectorComparisonSeal': record['sectorPlanning']['comparisonSeal'],
            'originalBranchInventory': original_branches,
            'learnedBranches': len(learned)}


def _assert_same_native_source(source, snapshot):
    if source != snapshot:
        raise ValueError('Desktop input snapshot differs from original source case/comparison/branch state')


def native_check(bundle):
    bundle = Path(bundle).resolve()
    verify(bundle)
    _package_modules(bundle)
    fit_root, sources, is_exit = _bundle_paths(bundle)
    fit = _json(fit_root/'receipt.json')
    checked = []
    source_signatures = {}
    for case, patient in CASES.items():
        source = sources[case]
        source_signatures[patient] = _native_endpoint(bundle, source)
        checked.append({'name': f'{case}-source', **source_signatures[patient]})
    for run in fit['seedRuns']:
        for method in METHODS:
            for case in CASES:
                eval_row = run[method]['evaluations'][case]
                path = fit_root/Path(eval_row['derived']['workspacePath']).name
                checked.append({'name': f'{case}-{method}-seed{run["seed"]}',
                                **_native_endpoint(bundle, path, eval_row['report']['finalStateId'])})
    for patient in ('0007', '0005'):
        if not is_exit:
            root = bundle/f'evidence/cold-companion-{patient}'
            receipt = _json(root/'receipt.json')
            native = root/Path(receipt['derived']['workspacePath']).name
            checked.append({'name': f'cold-companion-{patient}',
                            **_native_endpoint(bundle, native, receipt['execution']['finalStateId'])})
        desktop = bundle/f'evidence/desktop-{patient}'
        report = _json(desktop/'report.json')
        desktop_receipt = _json(bundle/f'evidence/desktop-{patient}-run/execution/receipt.json')
        input_signature = _native_endpoint(bundle,
            bundle/f'evidence/desktop-{patient}-run/input.ressectionlab')
        _assert_same_native_source(source_signatures[patient], input_signature)
        checked.append({'name': f'desktop-input-{patient}', **input_signature})
        checked.append({'name': f'desktop-saved-{patient}',
                        **_native_endpoint(bundle, desktop/Path(report['savedCase']).name,
                                           desktop_receipt['execution']['finalStateId'])})
    return {'verified': True, 'codeRoot': str(bundle/'src'), 'native': checked}


def cold_check(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    model_check(bundle)
    fit_root, sources, is_exit = _bundle_paths(bundle)
    fit = _json(fit_root/'receipt.json')
    runs = []
    for run in fit['seedRuns']:
        seed = run['seed']
        for method in METHODS:
            model = run[method]['model']
            for case, patient in CASES.items():
                folder = output/f'{case}-{method}-seed{seed}'
                receipt = _json(folder/'receipt.json')
                source = sources[case]
                reference = run[method]['evaluations'][case]['report']
                actual = receipt['execution']
                checkpoint = fit_root/Path(model['path']).name
                if (receipt['status'] != 'evaluated' or receipt['optimizerUpdates'] != 0
                        or receipt['source']['unchanged'] is not True
                        or receipt['source']['sha256'] != historical.sha(source)
                        or receipt['model']['fileSha256'] != historical.sha(checkpoint)
                        or receipt['model']['actorSha256'] != model['actorSha256']
                        or receipt['sourceRoleDeclaration']['declarationSha256'] !=
                            fit['sources'][case]['roleDeclaration']['declarationSha256']
                        or receipt['sourceRoleDeclaration']['patientId'] != f'UCSF-PDGM-{patient}'
                        or receipt['exactReplay']['exactNativeReplay'] is not True):
                    raise ValueError(f'Cold {case} {method} seed {seed} source/model/freeze differs')
                fields = ('status', 'taskGateMet', 'finalStateId', 'deltaTargetMm3',
                          'deltaNonTargetMm3', 'totalNonTargetMm3', 'perSectorRemovedMm3',
                          'admittedSectorOptions', 'cleanupComplete')
                if any(actual.get(field) != reference.get(field) for field in fields):
                    raise ValueError(f'Cold {case} {method} seed {seed} endpoint differs')
                native = folder/Path(receipt['derived']['workspacePath']).name
                if (historical.sha(native) != receipt['derived']['fileSha256'] or
                        receipt['derived']['reopenedExact'] is not True):
                    raise ValueError('Cold native export bytes or reopen verification differ')
                _native_endpoint(bundle, native, actual['finalStateId'])
                runs.append({'case': patient, 'seed': seed, 'method': method,
                             'status': actual['status'], 'taskGateMet': actual['taskGateMet'],
                             'finalStateId': actual['finalStateId'],
                             'targetMm3': actual['deltaTargetMm3'],
                             'nonTargetMm3': actual['deltaNonTargetMm3'],
                             'sourceUnchanged': True, 'optimizerUpdates': 0,
                             'nativeSha256': historical.sha(native)})
    return {'schema': 'resectionlab.hierarchical-cold-verification.v2' if is_exit
                      else 'resectionlab.hierarchical-cold-verification.v1',
            'verified': True, 'runCount': len(runs), 'runs': runs}


def cold_run(bundle, output, *, max_total_seconds=600):
    bundle = Path(bundle).resolve()
    output = Path(output).resolve()
    if 'testcases' not in output.parts or output.exists():
        raise ValueError('Choose a new ignored testcases/ cold output directory')
    model_check(bundle)
    fit_root, sources, _ = _bundle_paths(bundle)
    fit = _json(fit_root/'receipt.json')
    output.mkdir(parents=True)
    started = time.monotonic()
    for run in fit['seedRuns']:
        for method in METHODS:
            checkpoint = fit_root/Path(run[method]['model']['path']).name
            for case, patient in CASES.items():
                remaining = max_total_seconds-(time.monotonic()-started)
                if remaining <= 0:
                    raise TimeoutError('Aggregate cold execution budget exhausted')
                folder = output/f'{case}-{method}-seed{run["seed"]}'
                command = [sys.executable, str(bundle/'scripts/execute_patient_sector_policy.py'),
                           '--source', str(sources[case]),
                           '--checkpoint', str(checkpoint), '--source-role', 'DEVELOPMENT',
                           '--output', str(folder)]
                proc = subprocess.run(command, cwd=bundle, text=True, capture_output=True,
                                      timeout=min(120, remaining))
                (output/f'{folder.name}.log').write_text(proc.stdout+'\n'+proc.stderr)
                if proc.returncode != 0:
                    raise RuntimeError(f'Cold execution failed: {folder.name}; see saved log')
    result = cold_check(bundle, output)
    result['wallSeconds'] = time.monotonic()-started
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'verify', 'extract', 'verify-models',
                                            'verify-native', 'verify-cold', 'cold-run'))
    parser.add_argument('--commit')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--desktop-0007-report', type=Path)
    parser.add_argument('--desktop-0005-report', type=Path)
    parser.add_argument('--profile', choices=('legacy-v1', EXIT_PROFILE_ID),
                        default='legacy-v1', help='Build profile; verification reads the manifest')
    parser.add_argument('--max-total-seconds', type=float, default=600)
    args = parser.parse_args()
    if args.command == 'build':
        if not (args.commit and args.output and args.desktop_0007_report and args.desktop_0005_report):
            parser.error('Build requires --commit, --output and both desktop report paths')
        result = build(args.output, args.commit,
                       {'0007': args.desktop_0007_report, '0005': args.desktop_0005_report},
                       profile=args.profile)
    elif args.command == 'verify':
        result = verify(args.bundle)
    elif args.command == 'extract':
        result = extract(args.archive, args.output)
    elif args.command == 'verify-models':
        result = model_check(args.bundle)
    elif args.command == 'verify-native':
        result = native_check(args.bundle)
    elif args.command == 'verify-cold':
        result = cold_check(args.bundle, args.output)
    else:
        result = cold_run(args.bundle, args.output, max_total_seconds=args.max_total_seconds)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
