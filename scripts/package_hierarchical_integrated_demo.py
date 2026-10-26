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


def build(output, commit, desktop_reports):
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


def verify(bundle):
    bundle = Path(bundle).resolve()
    manifest = _json(bundle/'manifest.json')
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


def model_check(bundle):
    bundle = Path(bundle).resolve()
    verify(bundle)
    runner = _package_modules(bundle)
    from resectionlab.patient_sector_learning import frozen_actor, parameter_hash
    fit = _json(bundle/'evidence/patient-sector-fit-v4/receipt.json')
    producer = runner.producer_identity()
    if producer != fit['producerIdentity']:
        raise ValueError('Packaged producer fingerprint differs from six checkpoint seals')
    checked = []
    for run in fit['seedRuns']:
        for method in METHODS:
            model = run[method]['model']
            path = bundle/'evidence/patient-sector-fit-v4'/Path(model['path']).name
            actor = frozen_actor(path, expected_producer=producer)
            if parameter_hash(actor) != model['actorSha256']:
                raise ValueError('Packaged frozen actor parameters differ')
            checked.append({'seed': run['seed'], 'stage': method,
                            'fileSha256': model['fileSha256'],
                            'actorSha256': model['actorSha256']})
    return {'verified': True, 'producerIdentity': producer, 'models': checked}


def _native_endpoint(bundle, path, expected_state=None):
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_planning import validate_saved_comparison
    from resectionlab.surgical_sector_planning import validate_saved_sector_comparison
    _, artifacts, _ = load_workspace(path)
    scene = load_environment(path)
    record = artifacts['surgical_toolbox']
    validate_saved_comparison(scene, record, record['planning'])
    validate_saved_sector_comparison(scene, record, record['sectorPlanning'])
    learned = record.get('learnedSectorExecutions', [])
    if expected_state is not None:
        if len(learned) != 1:
            raise ValueError('Expected one learned native branch')
        branch = record['branches'][learned[0]['branchId']]
        replay = SurgicalEngine.restore(scene, branch)
        if replay.state_id != expected_state:
            raise ValueError('Learned native endpoint state differs')
    return {'sceneId': scene.identity,
            'accessComparisonSeal': record['planning']['comparisonSeal'],
            'sectorComparisonSeal': record['sectorPlanning']['comparisonSeal'],
            'learnedBranches': len(learned)}


def native_check(bundle):
    bundle = Path(bundle).resolve()
    verify(bundle)
    _package_modules(bundle)
    fit = _json(bundle/'evidence/patient-sector-fit-v4/receipt.json')
    checked = []
    source_signatures = {}
    for case, patient in CASES.items():
        source = bundle/f'cases/{patient}-sector-v3.ressectionlab'
        source_signatures[patient] = _native_endpoint(bundle, source)
        checked.append({'name': f'{case}-source', **source_signatures[patient]})
    for run in fit['seedRuns']:
        for method in METHODS:
            for case in CASES:
                eval_row = run[method]['evaluations'][case]
                path = bundle/'evidence/patient-sector-fit-v4'/Path(eval_row['derived']['workspacePath']).name
                checked.append({'name': f'{case}-{method}-seed{run["seed"]}',
                                **_native_endpoint(bundle, path, eval_row['report']['finalStateId'])})
    for patient in ('0007', '0005'):
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
        if input_signature != source_signatures[patient]:
            raise ValueError('Desktop input snapshot differs from original scene/comparison/branch state')
        checked.append({'name': f'desktop-input-{patient}', **input_signature})
        checked.append({'name': f'desktop-saved-{patient}',
                        **_native_endpoint(bundle, desktop/Path(report['savedCase']).name,
                                           desktop_receipt['execution']['finalStateId'])})
    return {'verified': True, 'codeRoot': str(bundle/'src'), 'native': checked}


def cold_check(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    model_check(bundle)
    fit = _json(bundle/'evidence/patient-sector-fit-v4/receipt.json')
    runs = []
    for run in fit['seedRuns']:
        seed = run['seed']
        for method in METHODS:
            model = run[method]['model']
            for case, patient in CASES.items():
                folder = output/f'{case}-{method}-seed{seed}'
                receipt = _json(folder/'receipt.json')
                source = bundle/f'cases/{patient}-sector-v3.ressectionlab'
                reference = run[method]['evaluations'][case]['report']
                actual = receipt['execution']
                checkpoint = bundle/'evidence/patient-sector-fit-v4'/Path(model['path']).name
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
    return {'schema': 'resectionlab.hierarchical-cold-verification.v1',
            'verified': True, 'runCount': len(runs), 'runs': runs}


def cold_run(bundle, output, *, max_total_seconds=600):
    bundle = Path(bundle).resolve()
    output = Path(output).resolve()
    if 'testcases' not in output.parts or output.exists():
        raise ValueError('Choose a new ignored testcases/ cold output directory')
    model_check(bundle)
    fit = _json(bundle/'evidence/patient-sector-fit-v4/receipt.json')
    output.mkdir(parents=True)
    started = time.monotonic()
    for run in fit['seedRuns']:
        for method in METHODS:
            checkpoint = bundle/'evidence/patient-sector-fit-v4'/Path(run[method]['model']['path']).name
            for case, patient in CASES.items():
                remaining = max_total_seconds-(time.monotonic()-started)
                if remaining <= 0:
                    raise TimeoutError('Aggregate cold execution budget exhausted')
                folder = output/f'{case}-{method}-seed{run["seed"]}'
                command = [sys.executable, str(bundle/'scripts/execute_patient_sector_policy.py'),
                           '--source', str(bundle/f'cases/{patient}-sector-v3.ressectionlab'),
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
    parser.add_argument('--max-total-seconds', type=float, default=600)
    args = parser.parse_args()
    if args.command == 'build':
        if not (args.commit and args.output and args.desktop_0007_report and args.desktop_0005_report):
            parser.error('Build requires --commit, --output and both desktop report paths')
        result = build(args.output, args.commit,
                       {'0007': args.desktop_0007_report, '0005': args.desktop_0005_report})
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
