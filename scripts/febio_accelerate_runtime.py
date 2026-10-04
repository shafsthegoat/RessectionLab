#!/usr/bin/env python3
"""One isolated patched FEBio build; no downloads, fixtures or solver launches.

The root release supplies the exact v2 declaration SHA. Two fixed, one-attempt
stages reuse the reviewed supervisor and retain failures. A build identity is
not numerical validation and is published only after parent acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT/'manifests/experiments/febio-accelerate-csc-runtime-v2.json'
V1_SHA = '7517aa3bd9a40cce9ce2864e5118c52185448a1a3eb1bab24b42ae9e9e4026b2'
HELPER_SHA = '679594d7f3759f5485b9fb868e7e5543ebb242bccd24d112f9ca862bb6d2a01e'
_helper_path = ROOT/'scripts/febio_runtime.py'
if hashlib.sha256(_helper_path.read_bytes()).hexdigest() != HELPER_SHA:
    raise ValueError('Reviewed runtime helper changed before import')
_helper_spec = importlib.util.spec_from_file_location('reviewed_febio_runtime', _helper_path)
rt = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(rt)
PREFIX = ROOT/'data/optional-runtimes/febio-4.13-accelerate-csc-v1'
OLD_PREFIX = ROOT/'data/optional-runtimes/febio-4.13'
OUTPUT = ROOT/'artifacts/febio-accelerate-csc-runtime-v2'
PATCHED_SHA = '60e5a3f2350826af1b95376ab93c8a476f2312158763d63d82eb80ec54af79c2'
STAGE_FILES = {
    'configure': ('result.json', 'supervision.json', 'source-inventory.json', 'verified-cache.json',
                  'source-preparation.json', 'commands.json'),
    'build': ('result.json', 'supervision.json', 'installed-inventory.json', 'linkage.json',
              'runtime-identity-candidate.json', 'commands.json'),
}


def pin(binding):
    path = ROOT/binding['path']  # Absolute compiler/SDK paths are deliberately pinned.
    if path.stat().st_size != binding['bytes'] or rt.sha(path) != binding['sha256']:
        raise ValueError('Pinned input changed: '+str(path))
    return path


def load_declaration(expected_sha):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha) or rt.sha(DECLARATION) != expected_sha:
        raise ValueError('Exact released v2 declaration required')
    spec = json.loads(DECLARATION.read_bytes())
    if rt.sha(Path(__file__)) != spec['driver_sha256'] or rt.sha(Path(rt.__file__)) != HELPER_SHA:
        raise ValueError('Driver or reviewed helper changed')
    v1_path = pin(spec['basis'])
    if spec['basis']['sha256'] != V1_SHA:
        raise ValueError('Original runtime-v1 declaration must remain exact')
    v1 = json.loads(v1_path.read_bytes())
    if (v1['configure']['source'] != str(PREFIX/'source')
            or v1['configure']['build'] != str(PREFIX/'build')
            or v1['build']['install_prefix'] != str(PREFIX/'install')
            or v1['build']['parallel_jobs'] != 2
            or v1['caps']['configure_seconds'] != 120
            or v1['caps']['build_and_install_seconds'] != 900
            or v1['caps']['process_family_sampled_rss_bytes'] != 3*1024**3):
        raise ValueError('Frozen layout, commands or caps differ')
    evidence = {name: pin(binding) for name, binding in spec['saved_evidence'].items()}
    review = json.loads(evidence['independent_positive_review'].read_bytes())
    positive = json.loads(evidence['positive_result'].read_bytes())
    supervision = json.loads(evidence['positive_supervision'].read_bytes())
    if (review['status'] != 'independent_static_patch_and_positive_controls_review_passed'
            or positive['status'] != 'six_positive_controls_passed'
            or positive['original_code_executed'] is not False
            or positive['all_pins_unchanged'] is not True
            or supervision['status'] != 'completed'
            or review['patched_source_sha256'] != PATCHED_SHA
            or review['positive_result_sha256'] != spec['saved_evidence']['positive_result']['sha256']):
        raise ValueError('Successful positive-only review is required')
    return spec, v1


def verify_original(v1):
    """Rehash all 6164 acquisition entries without scanning old build products."""
    expected = json.loads(pin(v1['upstream']['original_input_inventory']).read_bytes())
    identity = json.loads(pin(v1['upstream']['original_runtime_identity']).read_bytes())
    roots = {name.split('/')[0] for name in expected}
    if len(expected) != 6164 or roots != {'source', 'tools', 'downloads', 'openmp', 'zstandard-helper'}:
        raise ValueError('Unexpected original acquisition inventory')
    if {p.name for p in OLD_PREFIX.iterdir()} - roots - {'build', 'install', 'state'}:
        raise ValueError('Unexpected file in original private prefix')
    actual = {folder+'/'+name: record for folder in sorted(roots)
              for name, record in rt.inventory(OLD_PREFIX/folder).items()}
    if actual != expected:
        raise ValueError('Original source/tool/acquisition entry changed')
    for binding in v1['tools'].values():
        pin(binding)
    original_receipts = ROOT/'artifacts/febio-runtime-build-v1'
    for relative, digest in identity['receipts_sha256'].items():
        if rt.sha(original_receipts/relative) != digest:
            raise ValueError('Original runtime receipt changed: '+relative)
    installed = json.loads((original_receipts/'build-01/installed-inventory.json').read_bytes())
    if rt.inventory(OLD_PREFIX/'install') != installed:
        raise ValueError('Original installed runtime changed')
    for relative, digest in identity['libraries'].items():
        if rt.sha(OLD_PREFIX/relative) != digest:
            raise ValueError('Original binary changed: '+relative)
    return {'acquisition_entries_verified': len(actual),
            'input_inventory_sha256': v1['upstream']['original_input_inventory']['sha256'],
            'original_runtime_identity_sha256': v1['upstream']['original_runtime_identity']['sha256'],
            'original_install_entries_verified': len(installed)}


def expected_source(v1, *, patched):
    acquisition = json.loads(pin(v1['upstream']['original_input_inventory']).read_bytes())
    expected = {name.removeprefix('source/'): record for name, record in acquisition.items()
                if name.startswith('source/')}
    if patched:
        path = PREFIX/'source'/v1['patch']['relative_source']
        expected[v1['patch']['relative_source']] = {'bytes': path.stat().st_size,
                                                   'sha256': v1['patch']['patched_sha256']}
    return expected


def verify_source(v1, *, patched=True):
    actual = rt.inventory(PREFIX/'source')
    if actual != expected_source(v1, patched=patched):
        raise ValueError('Unexpected extracted/patched source difference')
    return actual


def exact_single_hunk(original, patch, relative):
    """Apply the exact reviewed unified hunk, with no fuzz or external patch tool."""
    lines = patch.splitlines(keepends=True)
    if lines[:2] != [f'--- a/{relative}\n'.encode(), f'+++ b/{relative}\n'.encode()]:
        raise ValueError('Patch names differ')
    match = re.fullmatch(rb'@@ -(\d+),(\d+) \+(\d+),(\d+) @@\n', lines[2])
    if match is None or any(line[:1] not in (b' ', b'+', b'-') for line in lines[3:]):
        raise ValueError('Exactly one ordinary patch hunk required')
    start, old_count, new_start, new_count = map(int, match.groups())
    old = [line[1:] for line in lines[3:] if line[:1] != b'+']
    new = [line[1:] for line in lines[3:] if line[:1] != b'-']
    source = original.splitlines(keepends=True)
    if (start != new_start or len(old) != old_count or len(new) != new_count
            or source[start-1:start-1+old_count] != old):
        raise ValueError('Patch context/position differs; no fuzzy application')
    return b''.join(source[:start-1]+new+source[start-1+old_count:])


def prepare_source(v1, output):
    PREFIX.mkdir(parents=True, exist_ok=False)
    with pin(v1['upstream']['archive']).open('rb') as stream:
        expansion = rt.extract_tar(stream, PREFIX/'source',
            expanded_limit=v1['caps']['source_expanded_bytes'],
            strip_root='FEBio-'+v1['upstream']['commit'])
    verify_source(v1, patched=False)
    patch = v1['patch']
    source = PREFIX/'source'/patch['relative_source']
    if rt.sha(source) != patch['original_sha256']:
        raise ValueError('Original patch target changed')
    marker = PREFIX/'patch-attempt.json'
    write_once(marker, {'patch_sha256': patch['file']['sha256'], 'attempts': 1})
    updated = exact_single_hunk(source.read_bytes(), pin(patch['file']).read_bytes(), patch['relative_source'])
    if patch['patched_sha256'] != PATCHED_SHA or hashlib.sha256(updated).hexdigest() != PATCHED_SHA:
        raise ValueError('Unexpected patched source bytes')
    source.write_bytes(updated)
    rt.write_json(output/'source-inventory.json', verify_source(v1))
    rt.write_json(output/'source-preparation.json', {'expansion': expansion,
        'changed_files': [patch['relative_source']], 'patch': patch['file'],
        'original_sha256': patch['original_sha256'], 'patched_sha256': PATCHED_SHA})


def verify_cache(v1):
    cache = {}
    for line in (PREFIX/'build/CMakeCache.txt').read_text().splitlines():
        if line and not line.startswith(('#', '//')) and '=' in line and ':' in line.split('=', 1)[0]:
            key, value = line.split('=', 1)
            cache[key.split(':', 1)[0]] = value
    for key, expected in v1['configure']['cache_options'].items():
        if cache.get(key) != expected:
            raise ValueError('Configured cache mismatch: '+key)
    if cache.get('CMAKE_GENERATOR') != v1['configure']['generator']:
        raise ValueError('Generator differs')
    return {key: cache[key] for key in v1['configure']['cache_options']}


def commands(stage, v1):
    if stage not in ('configure', 'build'):
        raise ValueError('Only configure/build stages exist')
    cmake = str(ROOT/v1['tools']['cmake']['path'])
    if stage == 'configure':
        return [[cmake, '-S', str(PREFIX/'source'), '-B', str(PREFIX/'build'),
                 '-G', v1['configure']['generator']]+[
                     '-D'+key+'='+value for key, value in v1['configure']['cache_options'].items()]]
    return [[cmake, '--build', str(PREFIX/'build'), '--target', 'febio4', '--parallel', '2'],
            [cmake, '--install', str(PREFIX/'build'), '--prefix', str(PREFIX/'install')]]


def write_once(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def accepted_config(v1, declaration_sha):
    output = OUTPUT/'configure-01'
    accepted = json.loads((output/'acceptance.json').read_bytes())
    if (accepted['status'] != 'completed' or accepted['declaration_sha256'] != declaration_sha
            or accepted['driver_sha256'] != rt.sha(Path(__file__))
            or set(accepted['artifacts']) != set(STAGE_FILES['configure'])):
        raise ValueError('Accepted matching configure stage required')
    for name, digest in accepted['artifacts'].items():
        if rt.sha(output/name) != digest:
            raise ValueError('Configure acceptance artifact changed')
    result = json.loads((output/'result.json').read_bytes())
    supervision = json.loads((output/'supervision.json').read_bytes())
    if (result['status'] != 'completed' or result['stage'] != 'configure'
            or result['declaration_sha256'] != declaration_sha
            or result['driver_sha256'] != rt.sha(Path(__file__))
            or result['solver_executed'] is not False
            or supervision['status'] != 'completed' or supervision['exit_code'] != 0
            or supervision['kill_reason'] is not None or supervision['cleanup_error'] is not None
            or supervision['wall_cap_seconds'] != v1['caps']['configure_seconds']
            or supervision['rss_cap_bytes'] != v1['caps']['process_family_sampled_rss_bytes']):
        raise ValueError('Configure worker/supervision did not complete the declared stage')
    if result['cache_sha256'] != rt.sha(PREFIX/'build/CMakeCache.txt'):
        raise ValueError('Accepted configure cache changed')
    if json.loads((output/'source-inventory.json').read_bytes()) != verify_source(v1):
        raise ValueError('Accepted source inventory changed')
    verify_cache(v1)


def linkage_details(dependencies, load_commands, expected_names, expected_rpaths):
    names = [line.strip().split(' (', 1)[0] for line in dependencies.splitlines()[1:] if line.strip()]
    rpaths = re.findall(r'cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.*?) \(offset \d+\)', load_commands)
    if not names or not set(names) <= set(expected_names) or set(rpaths) != set(expected_rpaths):
        raise ValueError('Undeclared dynamic dependency or RPATH')
    return {'dependencies': names, 'rpaths': rpaths}


def resolve_linkage(path, details, v1):
    """Inspect the declared dyld search order; do not load the executable."""
    resolved = {}
    executable_dir = PREFIX/'install/bin'
    omp = ROOT/v1['tools']['openmp_library']['path']
    for name in details['dependencies']:
        if name.startswith(('/usr/lib/', '/System/Library/')):
            resolved[name] = 'system_install_name_not_loaded'
            continue
        if not name.startswith('@rpath/'):
            raise ValueError('Only declared rpath or system install names allowed')
        basename = name.removeprefix('@rpath/')
        expected = omp if basename == 'libomp.dylib' else PREFIX/'install/lib'/basename
        candidates = []
        for entry in details['rpaths']:
            entry = entry.replace('@loader_path', str(path.parent)).replace('@executable_path', str(executable_dir))
            candidates.append(Path(entry)/basename)
        found = next((candidate.resolve() for candidate in candidates if candidate.exists()), None)
        if found is None or found != expected.resolve():
            raise ValueError('Dynamic library resolves outside declared new runtime/private OpenMP')
        resolved[name] = {'path': str(found), 'sha256': rt.sha(found)}
    return resolved


def installed_macho_names(installed):
    magic = {bytes.fromhex(code) for code in ('feedface', 'cefaedfe', 'feedfacf', 'cffaedfe',
                                             'cafebabe', 'bebafeca', 'cafebabf', 'bfbafeca')}
    result = set()
    for relative in installed:
        path = PREFIX/'install'/relative
        with path.open('rb') as stream:
            if stream.read(4) in magic:
                result.add('install/'+relative)
    return result


def private_environment(v1):
    environment = rt.private_environment(v1)
    for key in list(environment):
        if key.startswith('PYTHON') or key in ('__PYVENV_LAUNCHER__', 'LD_PRELOAD', 'LD_LIBRARY_PATH'):
            environment.pop(key)
    return environment


def inspect_install(v1, output):
    installed = rt.inventory(PREFIX/'install')
    old = json.loads(pin(v1['upstream']['original_runtime_identity']).read_bytes())
    expected = set(old['libraries'])
    actual = installed_macho_names(installed)
    if actual != expected:
        raise ValueError('Unexpected installed binary/library set')
    rows = []
    for relative in sorted(actual):
        path = PREFIX/relative
        rows.append({'path': relative, 'sha256': rt.sha(path), 'commands': []})
        for argv in (['/usr/bin/lipo', '-archs', str(path)], ['/usr/bin/otool', '-L', str(path)],
                     ['/usr/bin/otool', '-l', str(path)]):
            result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=False)
            rows[-1]['commands'].append({'argv': argv, 'exit_code': result.returncode,
                                         'stdout': result.stdout, 'stderr': result.stderr})
            rt.write_json(output/'linkage.json', {'status': 'inspecting', 'files': rows})
            if result.returncode:
                raise ValueError('Installed Mach-O inspection failed')
        if rows[-1]['commands'][0]['stdout'].strip() != 'arm64':
            raise ValueError('Installed architecture is not exclusively arm64')
        rows[-1].update(linkage_details(rows[-1]['commands'][1]['stdout'], rows[-1]['commands'][2]['stdout'],
            old['dependency_install_names'], v1['configure']['cache_options']['CMAKE_INSTALL_RPATH'].split(';')))
        rows[-1]['resolved_dependencies'] = resolve_linkage(path, rows[-1], v1)
    rt.write_json(output/'installed-inventory.json', installed)
    rt.write_json(output/'linkage.json', {'status': 'static_linkage_passed', 'files': rows,
        'solver_executed': False, 'dynamic_symbol_resolution_tested': False})
    return {row['path']: row['sha256'] for row in rows}


def worker(stage, declaration_sha):
    output = OUTPUT/(stage+'-01')
    result = {'status': 'running', 'stage': stage, 'declaration_sha256': declaration_sha,
              'driver_sha256': rt.sha(Path(__file__)), 'solver_executed': False}
    v1 = None
    rt.write_json(output/'result.json', result)
    try:
        spec, v1 = load_declaration(declaration_sha)
        marker = json.loads((OUTPUT/(stage+'-attempt.json')).read_bytes())
        if marker != {'stage': stage, 'driver_sha256': result['driver_sha256'],
                      'declaration_sha256': declaration_sha}:
            raise ValueError('Matching parent attempt marker required')
        result['original_before'] = verify_original(v1)
        if stage == 'configure':
            prepare_source(v1, output)
        else:
            accepted_config(v1, declaration_sha)
            if (PREFIX/'install').exists():
                raise FileExistsError('Fresh install required; no retry')
        environment = private_environment(v1)
        argv_list = commands(stage, v1)
        rt.write_json(output/'commands.json', {'commands': argv_list, 'cwd': str(ROOT),
            'thread_environment': v1['caps']['thread_environment'],
            'sanitizer': 'reviewed private_environment plus isolated Python/LD startup removals'})
        for argv in argv_list:
            subprocess.run(argv, cwd=ROOT, env=environment, check=True)
        verify_source(v1)
        if stage == 'configure':
            rt.write_json(output/'verified-cache.json', verify_cache(v1))
            result['cache_sha256'] = rt.sha(PREFIX/'build/CMakeCache.txt')
        else:
            verify_cache(v1)
            libraries = inspect_install(v1, output)
            identity = {'schema_version': 2, 'status': 'candidate_pending_parent_build_acceptance',
                'source_commit': v1['upstream']['commit'], 'source_archive': v1['upstream']['archive'],
                'source_patched': True, 'patch': v1['patch']['file'], 'patched_source_sha256': PATCHED_SHA,
                'source_patch_sha256': v1['patch']['file']['sha256'],
                'declaration_sha256': declaration_sha, 'driver_sha256': result['driver_sha256'],
                'source_inventory_sha256': rt.sha(OUTPUT/'configure-01/source-inventory.json'),
                'source_inventory_path': str(OUTPUT/'configure-01/source-inventory.json'),
                'source_inventory': {'path': str(OUTPUT/'configure-01/source-inventory.json'),
                                     'sha256': rt.sha(OUTPUT/'configure-01/source-inventory.json')},
                'installed_inventory': {'path': str(output/'installed-inventory.json'),
                                        'sha256': rt.sha(output/'installed-inventory.json')},
                'linkage': {'path': str(output/'linkage.json'), 'sha256': rt.sha(output/'linkage.json')},
                'configure_acceptance_sha256': rt.sha(OUTPUT/'configure-01/acceptance.json'),
                'executable': str(PREFIX/'install/bin/febio4'),
                'executable_sha256': libraries['install/bin/febio4'], 'libraries': libraries,
                'private_openmp': {**v1['tools']['openmp_library'],
                                   'path': str(ROOT/v1['tools']['openmp_library']['path'])},
                'architecture': 'arm64', 'runtime_version': '4.13.0',
                'runtime_version_evidence': 'pinned_source_version_only_not_executed',
                'version_executable_launched': False,
                'controls': spec['saved_evidence']['independent_positive_review'],
                'solver_executed': False, 'models_executed': 0, 'numerically_validated': False,
                'global_environment_changed': False}
            rt.write_json(output/'runtime-identity-candidate.json', identity)
        load_declaration(declaration_sha)
        result['status'] = 'completed'
    except BaseException as error:
        result.update(status='failed_or_incomplete', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if v1 is not None:
            try:
                result['original_after'] = verify_original(v1)
                if result.get('original_before') != result['original_after']:
                    raise ValueError('Original preservation check differs')
            except BaseException as error:
                result.update(status='failed_or_incomplete', preservation_error=str(error))
        rt.write_json(output/'result.json', result)
    return 0 if result['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('configure', 'build'))
    parser.add_argument('--declaration-sha256', required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args.stage, args.declaration_sha256)
    _, v1 = load_declaration(args.declaration_sha256)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_once(OUTPUT/(args.stage+'-attempt.json'), {'stage': args.stage,
        'driver_sha256': rt.sha(Path(__file__)), 'declaration_sha256': args.declaration_sha256})
    output = OUTPUT/(args.stage+'-01')
    command = [sys.executable, '-I', '-S', '-B', str(Path(__file__).resolve()), args.stage,
               '--declaration-sha256', args.declaration_sha256, '--worker']
    receipt = rt.supervise(command, output, cwd=ROOT, environment=private_environment(v1),
        seconds=v1['caps']['configure_seconds' if args.stage == 'configure' else 'build_and_install_seconds'],
        rss_bytes=v1['caps']['process_family_sampled_rss_bytes'])
    accepted, error, artifacts, preservation = False, None, {}, None
    try:
        preservation = verify_original(v1)  # Also runs if the worker was killed before its finally block.
        load_declaration(args.declaration_sha256)
        result = json.loads((output/'result.json').read_bytes())
        accepted = (receipt['status'] == result['status'] == 'completed'
            and result['driver_sha256'] == rt.sha(Path(__file__))
            and result['declaration_sha256'] == args.declaration_sha256
            and result['stage'] == args.stage and result['solver_executed'] is False
            and result['original_before'] == result['original_after'] == preservation)
        artifacts = {name: rt.sha(output/name) for name in STAGE_FILES[args.stage]}
    except (OSError, ValueError, KeyError) as failure:
        accepted, error = False, str(failure)
    acceptance = {'status': 'completed' if accepted else 'failed_or_incomplete', 'error': error,
        'stage': args.stage, 'driver_sha256': rt.sha(Path(__file__)),
        'declaration_sha256': args.declaration_sha256, 'artifacts': artifacts, 'solver_executed': False,
        'parent_original_after': preservation}
    rt.write_json(output/'acceptance.json', acceptance)
    if accepted and args.stage == 'build':
        identity = json.loads((output/'runtime-identity-candidate.json').read_bytes())
        identity.update(status='isolated_patched_runtime_built_pending_numerical_controls',
                        build_acceptance_sha256=rt.sha(output/'acceptance.json'),
                        build_acceptance_path=str(output/'acceptance.json'),
                        build_acceptance={'path': str(output/'acceptance.json'),
                                          'sha256': rt.sha(output/'acceptance.json')},
                        receipts_sha256={f'build-01/{name}': digest for name, digest in artifacts.items()})
        write_once(OUTPUT/'runtime-identity.json', identity)
    return 0 if accepted else 1


if __name__ == '__main__':
    raise SystemExit(main())
