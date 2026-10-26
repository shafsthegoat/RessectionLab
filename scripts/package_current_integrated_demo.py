#!/usr/bin/env python3
"""Build and verify the post-novelty local demonstration without altering v4.

The build freezes one explicit Git commit and copies sealed case/checkpoint
bytes unchanged. It requires a same-host compatible runtime for cold execution.
All package and test outputs belong under ignored testcases/.
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

import package_integrated_demo as historical

ROOT = Path(__file__).resolve().parents[1]
FRESH = Path('testcases/outputs/integrated-goal-20261026/fresh-patients')
EVIDENCE = Path('testcases/outputs/integrated-goal-20261026')
RAW = Path('testcases/downloaded-datasets/UCSF-fresh-development-20261026')
SCHEMA = 'resectionlab.integrated-current-local-package.v1'


def _data_members():
    fixed = {
        'cases/0005-access-v6.ressectionlab': FRESH/'UCSF-PDGM-0005/access-comparison-v6/access-comparison.ressectionlab',
        'cases/0007-access-fitting-v4.ressectionlab': FRESH/'UCSF-PDGM-0007/access-comparison-v4/access-comparison.ressectionlab',
        'cases/0007-learned-v6.ressectionlab': FRESH/'UCSF-PDGM-0007/cusa-continuation-trained-v6/learned-continuations.ressectionlab',
        'cases/0005-frozen-bc-v6.ressectionlab': FRESH/'UCSF-PDGM-0005/cusa-continuation-frozen-bc-v6/frozen-transfer.ressectionlab',
        'cases/0005-frozen-ppo-v6.ressectionlab': FRESH/'UCSF-PDGM-0005/cusa-continuation-frozen-ppo-v6/frozen-transfer.ressectionlab',
        'checkpoints/0007-bc-v6.pt': FRESH/'UCSF-PDGM-0007/cusa-continuation-trained-v6/bc-policy.pt',
        'checkpoints/0007-ppo-v6.pt': FRESH/'UCSF-PDGM-0007/cusa-continuation-trained-v6/ppo-policy.pt',
        'evidence/0005-access-v6.json': FRESH/'UCSF-PDGM-0005/access-comparison-v6/receipt.json',
        'evidence/0007-access-v4.json': FRESH/'UCSF-PDGM-0007/access-comparison-v4/receipt.json',
        'evidence/0007-training-v6.json': FRESH/'UCSF-PDGM-0007/cusa-continuation-trained-v6/receipt.json',
        'evidence/0005-frozen-bc-v6.json': FRESH/'UCSF-PDGM-0005/cusa-continuation-frozen-bc-v6/receipt.json',
        'evidence/0005-frozen-ppo-v6.json': FRESH/'UCSF-PDGM-0005/cusa-continuation-frozen-ppo-v6/receipt.json',
    }
    for patient in ('0007', '0005'):
        report_path = EVIDENCE/f'electron-cusa-execute-{patient}-v1/report.json'
        report = json.loads((ROOT/report_path).read_text())
        run_receipt = Path(report['executionReceipt']).resolve()
        run_root = run_receipt.parent.parent
        allowed = (ROOT/'testcases/outputs/cusa-desktop').resolve()
        if not run_root.is_relative_to(allowed) or run_receipt != run_root/'execution/receipt.json':
            raise ValueError('Desktop execution receipt is outside its ignored run')
        expected_input = ROOT/FRESH/f'UCSF-PDGM-{patient}'/(
            'access-comparison-v4' if patient == '0007' else 'access-comparison-v6')/'access-comparison.ressectionlab'
        if Path(report['input']).resolve() != expected_input.resolve() or report.get('passed') is not True:
            raise ValueError('Desktop execution report source or status differs')
        fixed.update({
            f'cases/{patient}-electron-saved.ressectionlab': EVIDENCE/f'electron-cusa-execute-{patient}-v1/desktop-saved.ressectionlab',
            f'cases/{patient}-desktop-input.ressectionlab': run_root.relative_to(ROOT)/'input.ressectionlab',
            f'cases/{patient}-desktop-frozen.ressectionlab': run_root.relative_to(ROOT)/'execution/frozen-transfer.ressectionlab',
            f'evidence/{patient}-electron-report.json': report_path,
            f'evidence/{patient}-electron-provenance.png': EVIDENCE/f'electron-cusa-execute-{patient}-v1/01-fresh-learned-provenance.png',
            f'evidence/{patient}-electron-playback.png': EVIDENCE/f'electron-cusa-execute-{patient}-v1/02-authoritative-playback.png',
            f'evidence/{patient}-desktop-execution.json': run_root.relative_to(ROOT)/'execution/receipt.json',
            f'evidence/{patient}-desktop-execution.log': run_root.relative_to(ROOT)/'execution.log',
        })
    for destination, origin in sorted(fixed.items()):
        yield destination, ROOT/origin, 'case' if destination.startswith('cases/') else \
            'checkpoint' if destination.startswith('checkpoints/') else 'evidence'
    for folder in ('data/models/synthstrip-v1', 'data/models/raidionics/MRI_GBM'):
        for source in sorted((ROOT/folder).iterdir()):
            if source.is_file():
                yield str(source.relative_to(ROOT)), source, 'model'
    for source in sorted((ROOT/RAW).rglob('*')):
        if source.is_file():
            yield str(source.relative_to(ROOT)), source, 'source-input'


def _readme(commit):
    return f'''# Current local integrated demonstration

Source commit: `{commit}`. This package is a source-bound research replay, not
clinical guidance or complete surgical access. The 0005 and 0007 source MRI
is skull-stripped; brain is an unreviewed SynthStrip estimate. Complete skull,
vessel, functional and calibrated harm anatomy remains unavailable.

The ten pinned NIfTIs, original DEVELOPMENT declaration and mirror receipts
are under `testcases/downloaded-datasets/UCSF-fresh-development-20261026/`.
The mirror declares CC BY 4.0; official TCIA byte equivalence and model
pretraining overlap are unknown. SynthStrip weight, runner and FreeSurfer
notice are under `data/models/synthstrip-v1/` (offered weight license MIT;
runner has its separate notice). Installed Raidionics MRI_GBM ONNX/configuration,
hash receipt and BSD-2-Clause notice are under
`data/models/raidionics/MRI_GBM/`. No unrelated models or datasets are here.

Native v6 artifacts include the novel 0007 fitting access, v6 BC/PPO actor
checkpoints and learned branch, 0005 access and both incomplete frozen BC/PPO
continuations, plus both real Electron execution reports, native input/output
snapshots, saved workspaces and screenshots. The 0005 frozen policy **abstains**:
it does not meet the target task gate. Earlier v4 package is separate immutable
historical evidence. Stored absolute paths in source/receipt provenance are
historical and are not rewritten; commands below pass bundled inputs explicitly.

## Same-host verification

The package does not contain the Python venv, model inference runtimes, Node
modules or Electron binary. The checkpoint binds producer source bytes, Python,
platform, numpy, scipy and torch. Use the matching host runtime. Installed
model bytes are provided, but fresh-install inference and Electron dependency
installation are **not** verified by the following replay test.

From this extracted bundle directory:

```sh
PY=/absolute/path/to/matching/.venv/bin/python
"$PY" scripts/package_current_integrated_demo.py verify --bundle .
"$PY" scripts/package_current_integrated_demo.py verify-native --bundle .
mkdir -p ../testcases
"$PY" scripts/run_patient_cusa_learning.py \\
  --source cases/0007-access-fitting-v4.ressectionlab \\
  --branch access-superior-access --source-role DEVELOPMENT \\
  --frozen-model checkpoints/0007-ppo-v6.pt \\
  --output ../testcases/cold-positive-0007
"$PY" scripts/run_patient_cusa_learning.py \\
  --source cases/0005-access-v6.ressectionlab \\
  --branch access-superior-access --source-role DEVELOPMENT \\
  --frozen-model checkpoints/0007-ppo-v6.pt \\
  --output ../testcases/cold-negative-0005
"$PY" scripts/package_current_integrated_demo.py verify-cold --bundle . \\
  --positive-output ../testcases/cold-positive-0007 \\
  --negative-output ../testcases/cold-negative-0005
```

The two frozen runs perform **zero optimizer updates**. The positive 0007
endpoint must independently replay to the saved v6 outcome (+36 mm³ target,
0 mm³ non-target). The 0005 run must independently replay to its preserved
abstained outcome; no failed gate is promoted. Each run must leave its input
archive unchanged. Use new output directories on repeat runs.
'''


def build(output, commit):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'testcases').resolve()) or output.exists():
        raise ValueError('Choose a new output directory under ignored testcases/')
    commit = subprocess.check_output(['git', 'rev-parse', '--verify', commit+'^{commit}'],
                                     cwd=ROOT, text=True).strip()
    if len(commit) != 40:
        raise ValueError('Freeze a complete committed source revision')
    output.mkdir(parents=True)
    bundle = output/'bundle'
    bundle.mkdir()
    roles = {}
    origins = {}
    for path, mode, oid in historical.committed_files(commit):
        data = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=ROOT)
        historical.write_bytes(bundle/path, data, 0o755 if mode == '100755' else 0o644)
        roles[path] = 'committed-source'
        origins[path] = f'git:{commit}:{path}'
    helper = 'scripts/package_current_integrated_demo.py'
    if helper not in roles:
        historical.copy_exact(Path(__file__), bundle/helper)
        roles[helper] = 'package-helper'
        origins[helper] = 'local package builder; outside frozen producer fingerprint'
    for target, source, role in _data_members():
        historical.copy_exact(source, bundle/target)
        roles[target] = role
        origins[target] = str(source.relative_to(ROOT))
    historical.write_bytes(bundle/'VERIFY.md', _readme(commit).encode())
    roles['VERIFY.md'] = 'instructions'
    origins['VERIFY.md'] = 'generated package instructions'
    files = [{'path': path, 'role': roles[path], 'origin': origins[path],
              'bytes': (bundle/path).stat().st_size, 'sha256': historical.sha(bundle/path)}
             for path in sorted(roles)]
    tree = subprocess.check_output(['git', 'rev-parse', commit+'^{tree}'],
                                   cwd=ROOT, text=True).strip()
    manifest = {'schema': SCHEMA, 'sourceCommit': commit, 'sourceTree': tree,
                'files': files, 'fileCount': len(files),
                'contentBytes': sum(row['bytes'] for row in files),
                'limitations': ['Same-host source/runtime fingerprint required for frozen actor',
                                'Historical absolute provenance is informational after relocation',
                                '0005 frozen transfer is abstained, not a successful task gate',
                                'Fresh-install inference and Electron installation not checked']}
    historical.write_bytes(bundle/'manifest.json',
        (json.dumps(manifest, indent=2, sort_keys=True)+'\n').encode())
    verify(bundle)
    archive = output/'integrated-current-local.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb',
                                                 mtime=0, filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for path in sorted(p for p in bundle.rglob('*') if p.is_file()):
                info = tarfile.TarInfo(str(path.relative_to(bundle)))
                info.size = path.stat().st_size
                info.mode = path.stat().st_mode & 0o777
                info.uid = info.gid = info.mtime = 0
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    result = {'schema': 'resectionlab.integrated-current-package-build.v1',
              'archive': str(archive), 'archiveSha256': historical.sha(archive),
              'archiveBytes': archive.stat().st_size, 'bundle': str(bundle),
              'sourceCommit': commit, 'fileCount': len(files),
              'contentBytes': manifest['contentBytes']}
    (output/'build-receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def verify(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle/'manifest.json').read_text())
    commit = manifest.get('sourceCommit')
    if (manifest.get('schema') != SCHEMA or not isinstance(commit, str) or
            len(commit) != 40 or any(ch not in '0123456789abcdef' for ch in commit)):
        raise ValueError('Current package schema or source commit differs')
    declared = set()
    for row in manifest['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or row['path'] in declared:
            raise ValueError('Unsafe or duplicate package path')
        path = bundle/relative
        if (path.is_symlink() or not path.is_file() or
                not path.resolve().is_relative_to(bundle) or
                path.stat().st_size != row['bytes'] or
                historical.sha(path) != row['sha256']):
            raise ValueError('Current package member changed: '+row['path'])
        declared.add(row['path'])
    if len(declared) != manifest['fileCount'] or \
            sum(row['bytes'] for row in manifest['files']) != manifest['contentBytes']:
        raise ValueError('Current package count or byte ledger differs')
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*')
              if p.is_file() or p.is_symlink() if '__pycache__' not in p.parts}
    if actual != declared | {'manifest.json'}:
        raise ValueError('Undeclared current package members')
    if not {'src/resectionlab/surgical_engine.py',
            'src/resectionlab/surgical_prefix_json_native.c',
            'src/resectionlab/surface_policy.json',
            'scripts/run_patient_cusa_learning.py',
            'desktop/pnpm-lock.yaml'} <= declared:
        raise ValueError('Frozen native/runner/desktop source is incomplete')
    for patient in ('UCSF-PDGM-0005', 'UCSF-PDGM-0007'):
        receipt = json.loads((bundle/RAW/patient/'source-receipt.json').read_text())
        for item in receipt['files']:
            source = bundle/item['path']
            if source.stat().st_size != item['size_bytes'] or historical.sha(source) != item['sha256']:
                raise ValueError('Pinned fresh-development NIfTI differs')
    declaration = bundle/RAW/'development-declaration.json'
    training = json.loads((bundle/'evidence/0007-training-v6.json').read_text())
    if historical.sha(declaration) != training['roleDeclaration']['declarationSha256']:
        raise ValueError('Bundled development declaration differs from fitting proof')
    synth = json.loads((bundle/'data/models/synthstrip-v1/poc-acquisition.json').read_text())
    for item in synth['files']:
        path = bundle/item['path']
        if path.stat().st_size != item['bytes'] or historical.sha(path) != item['sha256']:
            raise ValueError('SynthStrip model installation differs')
    gbm = json.loads((bundle/'data/models/raidionics/MRI_GBM/installation.json').read_text())
    for filename, expected in gbm['files'].items():
        if historical.sha(bundle/'data/models/raidionics/MRI_GBM'/filename) != expected:
            raise ValueError('MRI_GBM model installation differs')
    for method in ('bc', 'ppo'):
        negative = json.loads((bundle/f'evidence/0005-frozen-{method}-v6.json').read_text())
        if (negative.get('status') != 'evaluated' or
                negative['execution']['status'] != 'abstained' or
                negative['execution']['taskGateMet'] is not False or
                negative['source']['unchanged'] is not True):
            raise ValueError('Frozen 0005 negative outcome differs')
    return {'verified': True, 'sourceCommit': commit,
            'files': len(declared), 'contentBytes': manifest['contentBytes']}


def extract(archive, output):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'testcases').resolve()) or output.exists():
        raise ValueError('Choose a new extraction directory under ignored testcases/')
    output.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe current package archive member')
            path = output/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target, 1024*1024)
            path.chmod(member.mode & 0o777)
    return verify(output)


def native_check(bundle):
    bundle = Path(bundle).resolve()
    verify(bundle)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(bundle/'src'))
    import resectionlab
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_planning import validate_saved_comparison
    if not Path(resectionlab.__file__).resolve().is_relative_to(bundle/'src'):
        raise ValueError('Native check imported source outside package')
    checked = []
    names = ('0005-access-v6', '0007-access-fitting-v4', '0007-learned-v6',
             '0005-frozen-bc-v6', '0005-frozen-ppo-v6',
             '0007-electron-saved', '0005-electron-saved',
             '0007-desktop-input', '0005-desktop-input',
             '0007-desktop-frozen', '0005-desktop-frozen')
    for name in names:
        path = bundle/'cases'/(name+'.ressectionlab')
        _, artifacts, _ = load_workspace(path)
        scene = load_environment(path)
        record = artifacts['surgical_toolbox']
        validate_saved_comparison(scene, record, record['planning'])
        branches = [p['branchId'] for p in record['planning']['alternatives']]
        branches += [p['branchId'] for p in record.get('learnedExecutions', [])]
        for branch in set(branches):
            snapshot = record['branches'][branch]
            replay = SurgicalEngine.restore(scene, snapshot)
            if replay.state_id != snapshot['stateId']:
                raise ValueError('Current native endpoint differs: '+name+' '+branch)
        checked.append({'archive': name, 'sceneId': scene.identity,
                        'replayedBranches': len(set(branches))})
    training = json.loads((bundle/'evidence/0007-training-v6.json').read_text())
    if (historical.sha(bundle/'cases/0007-access-fitting-v4.ressectionlab') !=
            training['prefix']['sourceSha256']):
        raise ValueError('Current fitting source archive differs')
    return {'verified': True, 'codeRoot': str(bundle/'src'),
            'sourceCommit': json.loads((bundle/'manifest.json').read_text())['sourceCommit'],
            'native': checked}


def cold_check(bundle, positive_output, negative_output):
    bundle = Path(bundle).resolve()
    native_check(bundle)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(bundle/'src'))
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    training = json.loads((bundle/'evidence/0007-training-v6.json').read_text())
    expected_negative = json.loads((bundle/'evidence/0005-frozen-ppo-v6.json').read_text())
    checkpoint = historical.sha(bundle/'checkpoints/0007-ppo-v6.pt')
    results = {}
    for patient, name, output, reference, status, gate in (
            ('0007', '0007-access-fitting-v4', positive_output, training['ppoExecution'], 'success', True),
            ('0005', '0005-access-v6', negative_output, expected_negative['execution'], 'abstained', False)):
        source = bundle/'cases'/(name+'.ressectionlab')
        before = historical.sha(source)
        output = Path(output).resolve()
        receipt = json.loads((output/'receipt.json').read_text())
        execution = receipt['execution']
        if (receipt.get('status') != 'evaluated' or
                receipt.get('optimizerUpdates') != 0 or
                receipt['source']['unchanged'] is not True or
                receipt['source']['sha256'] != before or
                historical.sha(source) != before or
                receipt['model']['fileSha256'] != checkpoint or
                receipt['model']['actorSha256'] != training['ppoModel']['actorSha256'] or
                receipt['roleDeclaration']['declarationSha256'] !=
                    training['roleDeclaration']['declarationSha256'] or
                receipt['roleDeclaration']['patientId'] != 'UCSF-PDGM-'+patient or
                execution['status'] != status or execution['taskGateMet'] is not gate or
                execution['finalStateId'] != reference['finalStateId'] or
                execution['deltaTargetMm3'] != reference['deltaTargetMm3'] or
                execution['deltaNonTargetMm3'] != reference['deltaNonTargetMm3'] or
                receipt['exactReplay']['exactNativeReplay'] is not True):
            raise ValueError('Cold current '+patient+' source, model, or task endpoint differs')
        derived = output/'frozen-transfer.ressectionlab'
        _, artifacts, _ = load_workspace(derived)
        scene = load_environment(derived)
        record = artifacts['surgical_toolbox']
        learned = record.get('learnedExecutions', [])
        if len(learned) != 1 or learned[0]['sourceArchiveSha256'] != before:
            raise ValueError('Current cold derived branch source differs')
        replay = SurgicalEngine.restore(scene, record['branches'][learned[0]['branchId']])
        if replay.state_id != execution['finalStateId']:
            raise ValueError('Current cold native replay endpoint differs')
        results[patient] = {'status': status, 'taskGateMet': gate,
                            'sourceSha256': before, 'sourcePreserved': True,
                            'nativeFinalStateId': replay.state_id,
                            'deltaTargetMm3': execution['deltaTargetMm3'],
                            'deltaNonTargetMm3': execution['deltaNonTargetMm3'],
                            'optimizerUpdates': 0}
    return {'verified': True, 'ppoActorSha256': training['ppoModel']['actorSha256'],
            'coldRuns': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'verify', 'extract', 'verify-native', 'verify-cold'))
    parser.add_argument('--commit')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--positive-output', type=Path)
    parser.add_argument('--negative-output', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        if not args.commit or not args.output:
            parser.error('Build requires --commit and --output')
        result = build(args.output, args.commit)
    elif args.command == 'verify':
        result = verify(args.bundle)
    elif args.command == 'extract':
        result = extract(args.archive, args.output)
    elif args.command == 'verify-native':
        result = native_check(args.bundle)
    else:
        result = cold_check(args.bundle, args.positive_output, args.negative_output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
