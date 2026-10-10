"""Root-only source/metadata freeze and NON-executable template preparation.

Imports the canonical protocol constructor to derive exact limits, but constructs
no patient, policy or optimizer. No array load, search, test or pilot is started.
Root reviews the resulting template and separately releases its one attempt.
"""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from pilot_contract import (OUTPUT, TRAIN, CLOSED, MODE, PUBLIC_INDEX, PUBLIC_SHA, COHORT,
    COHORT_SHA, PARENT_SECONDS, SUPERVISION_BYTES, canonical_configuration, sha, semantic,
    validate_release, BASELINE_FILES, BASELINE_DIRECTORY, require_matched_baseline)


def relative(path): return str(path.relative_to(ROOT))


def project_module(name):
    if name == 'resectionlab': return ROOT/'src/resectionlab/__init__.py'
    if not name.startswith('resectionlab.'): return None
    base = ROOT/'src'/name.replace('.', '/')
    if base.with_suffix('.py').is_file(): return base.with_suffix('.py')
    if (base/'__init__.py').is_file(): return base/'__init__.py'
    return None


def source_closure():
    """Conservative static import closure, including lazy imports and package init."""
    paths = {HERE/name for name in ('pilot_contract.py', 'cohort_worker.py', 'run_owned.py', 'freeze-runtime.py')}
    paths.update(ROOT/'build/goal-conditioned-policy-v1'/name
        for name in ('run_contact_owned.py', 'darwin_fast_sampler.py'))
    paths.add(ROOT/'src/resectionlab/__init__.py')
    visited = set()
    while paths-visited:
        path = sorted(paths-visited)[0]; visited.add(path)
        if path.is_symlink() or not path.is_file(): raise ValueError('Regular source required: '+str(path))
        tree = ast.parse(path.read_text())
        in_package = path.is_relative_to(ROOT/'src/resectionlab')
        if in_package:
            package = '.'.join(path.parent.relative_to(ROOT/'src').parts)
        for node in ast.walk(tree):
            modules = []
            if isinstance(node, ast.Import): modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level and in_package:
                    base = '.'.join(package.split('.')[:len(package.split('.'))-node.level+1])
                    module = base + ('.'+node.module if node.module else '')
                elif node.level == 0: module = node.module or ''
                else: continue
                modules.append(module)
                # from . import preflight-style module imports, as well as
                # package-exported submodules, must not disappear from closure.
                modules.extend(module+'.'+alias.name for alias in node.names if alias.name != '*')
            for module in modules:
                dependency = project_module(module)
                if dependency is not None: paths.add(dependency)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--head', required=True)
    args = parser.parse_args()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True, timeout=5).strip()
    if head != args.head: raise ValueError('Root supplied canonical HEAD differs')
    output = ROOT/OUTPUT
    if any(path.exists() or path.is_symlink() for path in (output, output.with_name(output.name+'.supervision'))):
        raise FileExistsError('The one attempt or its reservation already exists')
    for name in ('source-index.json', 'release-template.json'):
        if (HERE/name).exists(): raise FileExistsError('Freeze metadata already exists; refuse overwrite: '+name)
    if sha(ROOT/PUBLIC_INDEX) != PUBLIC_SHA or sha(ROOT/COHORT) != COHORT_SHA:
        raise ValueError('Original public index or cohort changed')
    public = json.loads((ROOT/PUBLIC_INDEX).read_text())
    rows = public['cases']
    if (len(rows) != 4 or {r['patient_id'] for r in rows} != set(TRAIN)
            or any(r['role'] != 'TRAIN' for r in rows) or public['cohort_sha256'] != COHORT_SHA):
        raise ValueError('Exactly original four TRAIN metadata entries required')
    metadata = {PUBLIC_INDEX: PUBLIC_SHA, COHORT: COHORT_SHA}
    for row in rows:
        path = Path(row['path'])
        if (not path.is_absolute() or not path.is_relative_to(ROOT) or path.is_symlink()
                or not path.is_file() or path.stat().st_size > 1024**2 or sha(path) != row['sha256']):
            raise ValueError('Pinned public manifest metadata changed: '+row['patient_id'])
        metadata[relative(path)] = row['sha256']
    paths = source_closure()
    bindings = {relative(path): sha(path) for path in paths}
    for path in paths:
        if path.is_relative_to(ROOT/'src'):
            committed = subprocess.check_output(['git', 'show', head+':'+relative(path)], cwd=ROOT, timeout=5)
            if committed != path.read_bytes(): raise ValueError('Canonical bytes differ from HEAD: '+relative(path))
    # Derive the configuration from the already source-bound canonical APIs.
    # A fresh absent cache prefix prevents an existing bytecode cache from being
    # selected; -B semantics keep this constructor-only preparation read-only.
    cache = HERE/'never-written-freeze-pycache'
    if cache.exists(): raise FileExistsError('Fresh protocol-import cache namespace required')
    sys.dont_write_bytecode = True; sys.pycache_prefix = str(cache)
    sys.path.insert(0, str(ROOT/'src'))
    protocol, limits = canonical_configuration()
    baseline=require_matched_baseline(protocol,limits)
    if head==baseline['source_index']['head']:
        raise ValueError('Freeze only after root commits the completed one-update evidence')
    prior_sources=baseline['source_index']['source_files']
    # The evidence commit may advance HEAD, but this follow-on must execute the
    # identical canonical learner/task/source closure. Only these four driver
    # files change; even an unrelated closure drift requires explicit review.
    old_drivers={BASELINE_DIRECTORY+'/'+name for name in
        ('pilot_contract.py','cohort_worker.py','run_owned.py','freeze-runtime.py')}
    new_drivers={relative(HERE/name) for name in
        ('pilot_contract.py','cohort_worker.py','run_owned.py','freeze-runtime.py')}
    if ({k:v for k,v in prior_sources.items() if k not in old_drivers}
            !={k:v for k,v in bindings.items() if k not in new_drivers}):
        raise ValueError('Executing source closure differs from completed one-update pilot')
    if metadata!=baseline['source_index']['metadata_files']:
        raise ValueError('Four public input metadata bindings differ from pilot')
    metadata.update({ref['path']:ref['sha256'] for ref in BASELINE_FILES.values()})
    if any(sha(ROOT/path) != digest for path, digest in bindings.items()):
        raise ValueError('Source changed during canonical configuration construction')
    if any(sha(ROOT/path) != digest for path, digest in metadata.items()):
        raise ValueError('Public metadata changed during freeze')
    if subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True, timeout=5).strip() != head:
        raise ValueError('Canonical HEAD changed during preparation')
    index = {'version': 'fixed-four-TRAIN-eight-updates-runtime-v1', 'head': head,
        'source_files': bindings, 'metadata_files': metadata,
        'closure_scope': 'conservative repository Python import closure, including lazy branches; external Python/NumPy/Torch environment is separately recorded by the owned worker',
        'scope': 'one fixed four-TRAIN cohort; no SELECT/EVAL manifest or array admitted'}
    release = {'version': 'fixed-four-TRAIN-eight-updates-release-v1', 'status': 'pending_root_release',
        'expected_head': head, 'output': OUTPUT, 'TRAIN': list(TRAIN), 'closed_roles': CLOSED,
        'SELECT_EVAL_execution': False, 'attempts': 1, 'automatic_retry': False,
        'parent_seconds': PARENT_SECONDS, 'supervision_bytes': SUPERVISION_BYTES,
        'cohort_limits': limits,
        'limits': {k:v for k,v in limits.items() if k not in ('output_bytes', 'checkpoint_bytes')},
        'learning_protocol': protocol, 'learning_protocol_hash': semantic(protocol),
        'public_manifest_index': {'path': PUBLIC_INDEX, 'sha256': PUBLIC_SHA},
        'source_index': {'path': relative(HERE/'source-index.json'), 'sha256': None},
        'retention_mode': MODE,
        'one_update_baseline': BASELINE_FILES,
        'completion': 'all four complete teachers, eight shared IL updates, eight shared RL updates, both fixed checkpoints, eight complete TRAIN greedy native replays, exact initial and first-update numerical control',
        'only_learning_protocol_change': 'updates_per_method:1->8; same seed, architecture, loss, task, four subjects, search and sampling schedule',
        'teacher_gate': 'all complete teachers precede all-no-positive refusal; no capped teacher, skipped negative case, substitute patient or retry',
        'cost_interpretation': 'canonical preview/forward limits are conservative bounds; costs.json and owned receipt contain measured usage',
        'selection': 'fixed eight-update endpoint; SELECT/EVAL closed and require separate future admission',
        'claim': 'bounded numerical TRAIN pilot only; no held-out, physical or clinical performance claim'}
    validate_release({**release, 'status': 'released_one_attempt'})
    # The template itself must fail execution admission.
    try: validate_release(release)
    except ValueError: pass
    else: raise AssertionError('Non-executable template admitted')
    def write(path, value):
        with path.open('x') as stream: json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')
    write(HERE/'source-index.json', index)
    release['source_index']['sha256'] = sha(HERE/'source-index.json')
    write(HERE/'release-template.json', release)
    print(json.dumps({'status': 'pending_root_release', 'head': head,
        'source_count': len(bindings), 'metadata_count': len(metadata),
        'learning_protocol_hash': release['learning_protocol_hash'],
        'pins': {name: sha(HERE/name) for name in ('pilot_contract.py','cohort_worker.py',
            'run_owned.py','freeze-runtime.py','source-index.json','release-template.json')}}, indent=2))


if __name__ == '__main__': main()
