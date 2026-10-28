#!/usr/bin/env python3
"""Build and verify a local, source-frozen integrated demonstration bundle.

The archive contains no runtime environment. All bulk output stays under the
ignored testcases tree. Sealed native archives and checkpoints are copied byte
for byte; historical absolute provenance inside them is never rewritten.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
COMMIT = 'd793e6925139bbec835acced6e7b9e25720e11ac'
FRESH = Path('testcases/outputs/integrated-goal-20261026/fresh-patients')
EVIDENCE = Path('testcases/outputs/integrated-goal-20261026')
RAW = Path('testcases/downloaded-datasets/UCSF-fresh-development-20261026')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_bytes(path, data, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.write_bytes(data)
    path.chmod(mode)


def copy_exact(source, destination):
    source = Path(source)
    if not source.resolve().is_file():
        raise ValueError(f'Missing regular package input: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    # Dereference any input symlink so extracted archives never depend on its
    # original machine. The manifest authenticates the included target bytes.
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
    destination.chmod(0o644)


def committed_files(commit):
    tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', commit], cwd=ROOT)
    for item in tree.split(b'\0'):
        if not item:
            continue
        meta, raw_path = item.split(b'\t', 1)
        mode, kind, oid = meta.decode().split(' ')
        path = raw_path.decode()
        if (path.startswith(('src/', 'desktop/', 'packaging/')) or
                path.startswith('scripts/') and path.endswith(('.py', '.sh')) or
                path in ('pyproject.toml', 'requirements-lock.txt',
                         'requirements-legacy-qt-lock.txt',
                         'requirements-scan-target-macos-arm64-lock.txt', 'README.md') or
                path in ('docs/integrated-goal-demonstration.md',
                         'docs/fresh-patient-reproduction.md', 'docs/artifact-reproducibility.md')):
            if kind != 'blob' or mode not in ('100644', '100755'):
                raise ValueError(f'Unsupported committed source member {path}: {mode} {kind}')
            yield path, mode, oid


def data_members():
    """Explicit development inputs and receipts; no older/evaluation datasets."""
    cases = {
        'cases/0005-access-v4.ressectionlab': FRESH / 'UCSF-PDGM-0005/access-comparison-v4/access-comparison.ressectionlab',
        'cases/0007-access-fitting-v1.ressectionlab': FRESH / 'UCSF-PDGM-0007/access-comparison-v1/access-comparison.ressectionlab',
        'cases/0007-learned-v4.ressectionlab': FRESH / 'UCSF-PDGM-0007/cusa-continuation-trained-v4/learned-continuations.ressectionlab',
        'cases/0007-electron-saved.ressectionlab': EVIDENCE / 'electron-cusa-0007-v2/desktop-saved.ressectionlab',
        'checkpoints/0007-bc-v4.pt': FRESH / 'UCSF-PDGM-0007/cusa-continuation-trained-v4/bc-policy.pt',
        'checkpoints/0007-ppo-v4.pt': FRESH / 'UCSF-PDGM-0007/cusa-continuation-trained-v4/ppo-policy.pt',
        'evidence/0005-access-v4.json': FRESH / 'UCSF-PDGM-0005/access-comparison-v4/receipt.json',
        'evidence/0007-access-fitting-v1.json': FRESH / 'UCSF-PDGM-0007/access-comparison-v1/receipt.json',
        'evidence/0007-access-v2.json': FRESH / 'UCSF-PDGM-0007/access-comparison-v2/receipt.json',
        'evidence/0007-cusa-training-v4.json': FRESH / 'UCSF-PDGM-0007/cusa-continuation-trained-v4/receipt.json',
        'evidence/0005-frozen-bc-v4.json': FRESH / 'UCSF-PDGM-0005/cusa-continuation-frozen-bc-v4/receipt.json',
        'evidence/0005-frozen-ppo-v4.json': FRESH / 'UCSF-PDGM-0005/cusa-continuation-frozen-ppo-v4/receipt.json',
        'evidence/0005-frozen-transfer-v2.json': FRESH / 'UCSF-PDGM-0005/cusa-continuation-frozen-transfer-v2/receipt.json',
        'evidence/0007-electron-v2.json': EVIDENCE / 'electron-cusa-0007-v2/report.json',
        'evidence/0007-electron-provenance.png': EVIDENCE / 'electron-cusa-0007-v2/01-fresh-learned-provenance.png',
        'evidence/0007-electron-playback.png': EVIDENCE / 'electron-cusa-0007-v2/02-authoritative-playback.png',
    }
    for target, source in cases.items():
        yield target, ROOT / source, 'case' if target.startswith('cases/') else 'evidence'
    for folder in ('data/models/synthstrip-v1', 'data/models/raidionics/MRI_GBM'):
        for source in sorted((ROOT / folder).iterdir()):
            if source.is_file():
                yield str(source.relative_to(ROOT)), source, 'model'
    for source in sorted((ROOT / RAW).rglob('*')):
        if source.is_file():
            yield str(source.relative_to(ROOT)), source, 'source-input'


def verification_readme():
    return f'''# Local integrated demo reproduction

Frozen source commit: `{COMMIT}`. This bundle is local research software, not
clinical guidance. The saved 0005 and 0007 cases use supplied enhancing target
labels and unreviewed SynthStrip brain estimates. Skull, complete vasculature,
functional anatomy and calibrated harm outcomes are absent. The frozen policy
met its bounded 0007 development task; its 0005 transfer failed and that receipt
is included.

## Contents and licenses

`src/` is the complete tracked source tree from the commit, including native C
and surface policy data. `scripts/` and `desktop/` contain committed source and
lockfiles. `testcases/downloaded-datasets/UCSF-fresh-development-20261026/`
contains the ten pinned NIfTIs, original development declaration, per-case LFS
receipts and source attribution (MedOtter mirror declares CC BY 4.0; official
TCIA byte equivalence unknown). `data/models/synthstrip-v1/` contains the
weight, runner, acquisition hashes and FreeSurfer runner notice; the offered
SynthStrip weight license is MIT. `data/models/raidionics/MRI_GBM/` contains the
official-release installed ONNX, configurations, installation receipt and
BSD-2-Clause notice. No other model or dataset is included.

Original receipt and scene provenance contains absolute source paths from the
creation host. These are historical statements. They are left intact; the
relocated native replay below uses the bundled archives and model files.

## Runtime and fixity

Use the **same host and exact Python 3.12, numpy 2.5.3, scipy 1.18.1, torch
2.14.1 runtime** as the checkpoint producer. The venv, RaidionicsSeg runtime,
ONNX Runtime, pnpm modules, Electron binary, and operating system are not
included. The checkpoint rejects changed producer source or runtime. Installed
SynthStrip and MRI_GBM bytes plus source inputs are included for reconstruction,
but fresh-install inference is not validated by this package check; install
the documented runtimes separately before claiming an inference reproduction.

From the extracted bundle directory, using a compatible host Python:

```sh
PY=/absolute/path/to/compatible/.venv/bin/python
"$PY" scripts/package_integrated_demo.py verify --bundle .
"$PY" scripts/package_integrated_demo.py verify-native --bundle .
mkdir -p ../testcases
"$PY" scripts/run_patient_cusa_learning.py \\
  --source cases/0007-access-fitting-v1.ressectionlab \\
  --branch access-superior-access --source-role DEVELOPMENT \\
  --frozen-model checkpoints/0007-ppo-v4.pt \\
  --output ../testcases/cold-frozen-ppo-0007
"$PY" scripts/package_integrated_demo.py verify-cold --bundle . \\
  --cold-output ../testcases/cold-frozen-ppo-0007
```

The cold command performs no fitting or optimizer update. `verify-cold` checks
source bytes, native replay of the exported branch, the frozen receipt, and its
endpoint against the independently saved v4 fitting receipt. Use a **new**
`--output` directory for each repeat. The two original access cases are
`cases/0005-access-v4.ressectionlab` and `cases/0007-access-fitting-v1.ressectionlab`;
`cases/0007-learned-v4.ressectionlab` opens the learned BC/PPO branches in the
desktop. To run the desktop, install its locked pnpm dependencies and Electron
on a compatible host; these are not provided here.
'''


def build(output, commit=COMMIT):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / 'testcases').resolve()) or output.exists():
        raise ValueError('Choose a new output directory under ignored testcases/')
    if commit != COMMIT:
        raise ValueError('This package definition freezes the declared d793e692 source commit')
    subprocess.check_output(['git', 'cat-file', '-e', commit + '^{commit}'], cwd=ROOT)
    output.mkdir(parents=True)
    bundle = output / 'bundle'
    bundle.mkdir()
    roles = {}
    for path, mode, oid in committed_files(commit):
        data = subprocess.check_output(['git', 'cat-file', 'blob', oid], cwd=ROOT)
        write_bytes(bundle / path, data, 0o755 if mode == '100755' else 0o644)
        roles[path] = 'committed-source'
    # This build/verification helper is new tooling; it does not replace or
    # alter any fingerprint-bound committed producer source.
    helper = 'scripts/package_integrated_demo.py'
    copy_exact(Path(__file__), bundle / helper)
    roles[helper] = 'package-helper'
    for target, source, role in data_members():
        copy_exact(source, bundle / target)
        roles[target] = role
    write_bytes(bundle / 'VERIFY.md', verification_readme().encode())
    roles['VERIFY.md'] = 'instructions'
    files = [{'path': path, 'role': roles[path], 'bytes': (bundle/path).stat().st_size,
              'sha256': sha(bundle/path)} for path in sorted(roles)]
    source_tree = subprocess.check_output(['git', 'rev-parse', commit + '^{tree}'],
                                           cwd=ROOT, text=True).strip()
    manifest = {'schema': 'resectionlab.integrated-local-package.v1',
                'sourceCommit': commit, 'sourceTree': source_tree,
                'files': files, 'fileCount': len(files),
                'contentBytes': sum(row['bytes'] for row in files),
                'limits': ['Same-host pinned Python/runtime required for frozen checkpoint',
                           'Historical sealed absolute provenance is informational after relocation',
                           'Fresh-install model inference and Electron dependency install not checked']}
    write_bytes(bundle / 'manifest.json',
                (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode())
    verify(bundle)
    archive = output / 'integrated-demo-local.tar.gz'
    with archive.open('xb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0,
                                                 filename='') as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for path in sorted(p for p in bundle.rglob('*') if p.is_file()):
                name = str(path.relative_to(bundle))
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = path.stat().st_mode & 0o777
                info.uid = info.gid = info.mtime = 0
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    result = {'schema': 'resectionlab.integrated-package-build.v1',
              'archive': str(archive), 'archiveSha256': sha(archive),
              'archiveBytes': archive.stat().st_size,
              'bundle': str(bundle), 'sourceCommit': commit,
              'fileCount': len(files), 'contentBytes': manifest['contentBytes']}
    (output / 'build-receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify(bundle):
    bundle = Path(bundle).resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text())
    if manifest.get('schema') != 'resectionlab.integrated-local-package.v1' or \
            manifest.get('sourceCommit') != COMMIT:
        raise ValueError('Package manifest or frozen source commit differs')
    declared = set()
    for row in manifest['files']:
        relative = Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts or row['path'] in declared:
            raise ValueError('Unsafe or duplicate package path')
        path = bundle / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(bundle):
            raise ValueError('Missing/unsafe package file: ' + row['path'])
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError('Package member changed: ' + row['path'])
        declared.add(row['path'])
    if len(declared) != manifest['fileCount'] or \
            sum(row['bytes'] for row in manifest['files']) != manifest['contentBytes']:
        raise ValueError('Package count/size ledger differs')
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*') if p.is_file() or p.is_symlink()
              if not (p.is_relative_to(bundle/'testcases/outputs') or '__pycache__' in p.parts)}
    if actual != declared | {'manifest.json'}:
        raise ValueError('Package contains undeclared or missing members: ' + str(sorted(actual ^ (declared | {'manifest.json'}))[:8]))
    sources = {p for p in declared if p.startswith('src/')}
    if not sources or not any(p.endswith('.c') for p in sources) or \
            'src/resectionlab/surface_policy.json' not in sources:
        raise ValueError('Complete native source tree is missing')
    for case in ('UCSF-PDGM-0005', 'UCSF-PDGM-0007'):
        receipt = json.loads((bundle / RAW / case / 'source-receipt.json').read_text())
        for item in receipt['files']:
            source = bundle / item['path']
            if source.stat().st_size != item['size_bytes'] or sha(source) != item['sha256']:
                raise ValueError('Pinned development NIfTI differs')
    synth = json.loads((bundle/'data/models/synthstrip-v1/poc-acquisition.json').read_text())
    for item in synth['files']:
        path = bundle/item['path']
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('SynthStrip installation differs')
    gbm = json.loads((bundle/'data/models/raidionics/MRI_GBM/installation.json').read_text())
    for filename, expected in gbm['files'].items():
        if sha(bundle/'data/models/raidionics/MRI_GBM'/filename) != expected:
            raise ValueError('MRI_GBM installation differs')
    for name in ('0005-frozen-bc-v4', '0005-frozen-ppo-v4',
                 '0005-frozen-transfer-v2'):
        failed = json.loads((bundle/'evidence'/(name+'.json')).read_text())
        if (failed.get('status') != 'evaluated' or
                failed.get('execution', {}).get('status') != 'incomplete' or
                failed.get('source', {}).get('unchanged') is not True):
            raise ValueError('0005 failed transfer evidence differs: '+name)
    return {'verified': True, 'sourceCommit': COMMIT, 'files': len(declared),
            'contentBytes': manifest['contentBytes']}


def extract(archive, output):
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT/'testcases').resolve()) or output.exists():
        raise ValueError('Choose a new extraction directory under ignored testcases/')
    output.mkdir(parents=True)
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            relative = Path(member.name)
            if not member.isfile() or relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe archive member')
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, destination.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024*1024)
            destination.chmod(member.mode & 0o777)
    return verify(output)


def native_check(bundle):
    bundle = Path(bundle).resolve()
    verify(bundle)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(bundle/'src'))
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    from resectionlab.surgical_planning import validate_saved_comparison
    import resectionlab
    if not Path(resectionlab.__file__).resolve().is_relative_to(bundle/'src'):
        raise ValueError('Native verification imported code outside the package')
    checked = []
    for name in ('0005-access-v4', '0007-access-fitting-v1', '0007-learned-v4',
                 '0007-electron-saved'):
        path = bundle/'cases'/(name+'.ressectionlab')
        _, artifacts, _ = load_workspace(path)
        scene = load_environment(path)
        record = artifacts['surgical_toolbox']
        validate_saved_comparison(scene, record, record['planning'])
        branch_ids = [plan['branchId'] for plan in record['planning']['alternatives']]
        if record.get('learnedExecutions'):
            branch_ids += [entry['branchId'] for entry in record['learnedExecutions']]
        for identity in branch_ids:
            snapshot = record['branches'][identity]
            engine = SurgicalEngine.restore(scene, snapshot)
            if engine.state_id != snapshot['stateId']:
                raise ValueError('Native endpoint differs: '+name+' '+identity)
        checked.append({'case': name, 'sceneId': scene.identity,
                        'replayedBranches': len(branch_ids)})
    training = json.loads((bundle/'evidence/0007-cusa-training-v4.json').read_text())
    if sha(bundle/'cases/0007-access-fitting-v1.ressectionlab') != training['prefix']['sourceSha256']:
        raise ValueError('Bundled fitting source differs from checkpoint receipt')
    return {'verified': True, 'codeRoot': str(bundle/'src'), 'native': checked,
            'fittingSourceSha256': training['prefix']['sourceSha256']}


def cold_check(bundle, cold_output):
    bundle = Path(bundle).resolve()
    cold_output = Path(cold_output).resolve()
    original = sha(bundle/'cases/0007-access-fitting-v1.ressectionlab')
    native_check(bundle)
    receipt = json.loads((cold_output/'receipt.json').read_text())
    training = json.loads((bundle/'evidence/0007-cusa-training-v4.json').read_text())
    if (receipt.get('status') != 'evaluated' or receipt['source']['sha256'] != original or
            receipt['source']['unchanged'] is not True or sha(bundle/'cases/0007-access-fitting-v1.ressectionlab') != original or
            receipt['model']['fileSha256'] != sha(bundle/'checkpoints/0007-ppo-v4.pt') or
            receipt['model']['actorSha256'] != training['ppoModel']['actorSha256'] or
            receipt['execution']['finalStateId'] != training['ppoExecution']['finalStateId'] or
            receipt['execution']['deltaTargetMm3'] != training['ppoExecution']['deltaTargetMm3'] or
            receipt['execution']['deltaNonTargetMm3'] != training['ppoExecution']['deltaNonTargetMm3'] or
            receipt['exactReplay']['exactNativeReplay'] is not True or
            receipt['optimizerUpdates'] != 0):
        raise ValueError('Cold frozen PPO source, model, or endpoint differs')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(bundle/'src'))
    from resectionlab.workspace_bundle import load_workspace
    from resectionlab.surgical_environment import load_environment
    from resectionlab.surgical_engine import SurgicalEngine
    derived = cold_output/'frozen-transfer.ressectionlab'
    _, artifacts, _ = load_workspace(derived)
    scene = load_environment(derived)
    record = artifacts['surgical_toolbox']
    learned = record['learnedExecutions']
    if len(learned) != 1 or learned[0]['sourceArchiveSha256'] != original:
        raise ValueError('Cold derived native branch source differs')
    engine = SurgicalEngine.restore(scene, record['branches'][learned[0]['branchId']])
    if engine.state_id != receipt['execution']['finalStateId'] or \
            engine.state_id != training['ppoReplay']['stateId']:
        raise ValueError('Cold derived native replay endpoint differs')
    return {'verified': True, 'sourcePreserved': True,
            'sourceSha256': original, 'modelSha256': receipt['model']['actorSha256'],
            'nativeFinalStateId': engine.state_id,
            'deltaTargetMm3': receipt['execution']['deltaTargetMm3'],
            'deltaNonTargetMm3': receipt['execution']['deltaNonTargetMm3'],
            'optimizationUpdates': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'verify', 'extract', 'verify-native', 'verify-cold'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--cold-output', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.output)
    elif args.command == 'verify':
        result = verify(args.bundle)
    elif args.command == 'extract':
        result = extract(args.archive, args.output)
    elif args.command == 'verify-native':
        result = native_check(args.bundle)
    else:
        result = cold_check(args.bundle, args.cold_output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
