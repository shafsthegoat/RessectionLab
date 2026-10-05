"""One explicit repaired direct-solver profile; no solver or response access.

Runtime/control bindings are supplied by a separate actual release. Missing
receipts cannot be replaced with a prospective profile or a successful build.
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

PROFILE_ID = 'accelerate_csc_v1'
PREFIX = 'data/optional-runtimes/febio-4.13-accelerate-csc-v1'
UPSTREAM_COMMIT = '32ae206ff4881dfb54f62296cd1558e58ed9fcc6'
PATCH_SHA = '67a1858f2d55046796d7ccb46758ca06e5bfb673e75a539d158b1a17652f3340'
PATCHED_SOURCE_SHA = '476ac8471ea681a99c298352a50aba2a7a5baa71635e1678155a7f58b72ba04a'
PATCH_IDENTITY = {'path': 'artifacts/febio-accelerate-lifecycle-v1/patch-identity.json',
                  'sha256': '54198366eccb90a9fcdf4fcbf2a5055919a5658a1db0fdd7db8cfeede3b49eee'}
REUSED_OPENMP = 'data/optional-runtimes/febio-4.13/openmp/lib/libomp.dylib'
REUSED_OPENMP_SHA = '38f6afed27bf1d3bd52779547c7ab53aeda9c06c3473263b5cd11ed4f04b41f8'
SKYLINE_XML = '<linear_solver type="skyline" />'
ACCELERATE_XML = ('<linear_solver type="accelerate"><iterative>0</iterative>'
                  '<factorization>4</factorization><order_method>0</order_method>'
                  '<print_condition_number>0</print_condition_number></linear_solver>')
HEX_CASES = {'zero', 'translation', 'finite_stretch', 'shear', 'shear_double_stiffness'}
TET_CASES = {'tet10_affine', 'mpc_translation', 'mpc_nonrigid'}
CASE_ORDER = ('zero', 'translation', 'finite_stretch', 'shear', 'shear_double_stiffness',
              'tet10_affine', 'mpc_translation', 'mpc_nonrigid')
INSTALLED_FILES = {'install/bin/febio4'} | {
    'install/lib/lib'+name+'.dylib' for name in
    ('feamr', 'febiofluid', 'febiolib', 'febiomech', 'febiomix', 'febioopt',
     'febioplot', 'febiorve', 'febioxml', 'fecore', 'feimglib', 'numcore')}
BUILD_FILES = {'result.json', 'supervision.json', 'installed-inventory.json', 'linkage.json',
               'runtime-identity-candidate.json', 'commands.json'}
CHECKER_PINS = {'hex8': 'b3f373743d6d107de929bf3283c167c6a3e0a68dfe74396ea0f1f8ad7cce2be5',
                'tet10': '04119660cf0a230428fe862d7897ea17953c51cf8756f87fa3152b426e303c66'}
CHECKER_FILES = {'hex8': 'mechanics_febio_verification.py',
                 'tet10': 'mechanics_patient_constraints.py'}


def transform_deck(original):
    """Replace only the canonical existing solver token; all other bytes stay."""
    text = original.decode('utf-8') if isinstance(original, bytes) else original
    tree = ET.fromstring(text)
    solvers = tree.findall('./Control/solver/linear_solver')
    if (len(tree.findall('.//linear_solver')) != 1 or len(solvers) != 1
            or solvers[0].attrib != {'type': 'skyline'} or len(solvers[0])
            or (solvers[0].text or '').strip() or text.count(SKYLINE_XML) != 1):
        raise ValueError('Exactly one original canonical Skyline subtree required')
    return text.replace(SKYLINE_XML, ACCELERATE_XML, 1)


def verify_deck(original, adapted):
    text = adapted.decode('utf-8') if isinstance(adapted, bytes) else adapted
    if transform_deck(original) != text:
        raise ValueError('Backend deck changed content outside the exact solver subtree')


def _accepted_build(root, identity, bound):
    """Consume the runtime builder's actual final six-artifact acceptance schema."""
    def internal(record):
        path = Path(record['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Runtime build evidence escapes repository')
        return {'path': str(path.relative_to(root)), 'sha256': record['sha256']}
    accepted_binding = internal(identity['build_acceptance'])
    accepted = bound(accepted_binding)
    accepted_path = root/accepted_binding['path']
    if (identity.get('build_acceptance_sha256') != accepted_binding['sha256']
            or identity.get('build_acceptance_path') != str(accepted_path)
            or accepted.get('status') != 'completed' or accepted.get('stage') != 'build'
            or accepted.get('driver_sha256') != identity.get('driver_sha256')
            or accepted.get('declaration_sha256') != identity.get('declaration_sha256')
            or set(accepted.get('artifacts', {})) != BUILD_FILES):
        raise ValueError('Actual parent-accepted repaired build required')
    records = {name: bound({'path': str((accepted_path.parent/name).relative_to(root)), 'sha256': digest})
               for name, digest in accepted['artifacts'].items()}
    if identity.get('receipts_sha256') != {'build-01/'+name: digest for name, digest in accepted['artifacts'].items()}:
        raise ValueError('Runtime receipt inventory differs from accepted build')
    result, supervision = records['result.json'], records['supervision.json']
    if (result.get('status') != 'completed' or result.get('stage') != 'build'
            or result.get('driver_sha256') != accepted['driver_sha256']
            or result.get('declaration_sha256') != accepted['declaration_sha256']
            or result.get('original_before') != result.get('original_after')
            or result.get('original_after') != accepted.get('parent_original_after')
            or supervision.get('status') != 'completed' or type(supervision.get('exit_code')) is not int
            or supervision['exit_code'] != 0 or supervision.get('kill_reason') is not None):
        raise ValueError('Runtime build execution or preservation did not complete')
    candidate = dict(records['runtime-identity-candidate.json'])
    if candidate.get('status') != 'candidate_pending_parent_build_acceptance':
        raise ValueError('Unrecognized accepted-build candidate')
    candidate.update({key: identity[key] for key in ('status', 'build_acceptance', 'build_acceptance_sha256',
                                                    'build_acceptance_path', 'receipts_sha256')})
    if candidate != identity:
        raise ValueError('Final runtime identity changed accepted candidate contents')
    adapter_controls = identity.get('adapter_controls', {})
    if set(adapter_controls) != {'declaration', 'acceptance', 'result', 'supervision', 'independent_review'}:
        raise ValueError('Accepted runtime must retain its repaired-only adapter evidence')
    for record in adapter_controls.values():
        bound({'path': record['path'], 'sha256': record['sha256']}, json_value=False)
    for key, filename in [('installed_inventory', 'installed-inventory.json'), ('linkage', 'linkage.json')]:
        record = internal(identity[key])
        if record != {'path': str((accepted_path.parent/filename).relative_to(root)),
                      'sha256': accepted['artifacts'][filename]}:
            raise ValueError('Installed identity evidence differs from accepted build')
    inventory, linkage = records['installed-inventory.json'], records['linkage.json']
    for name, digest in identity['libraries'].items():
        if inventory.get(name.removeprefix('install/'), {}).get('sha256') != digest:
            raise ValueError('Installed library inventory differs')
    rows = linkage.get('files', [])
    if (linkage.get('status') != 'static_linkage_passed' or len(rows) != len(INSTALLED_FILES)
            or {row.get('path'): row.get('sha256') for row in rows} != identity['libraries']):
        raise ValueError('Complete installed linkage evidence required')
    source_binding = internal(identity['source_inventory'])
    source = bound(source_binding)
    if (source_binding['sha256'] != identity.get('source_inventory_sha256')
            or identity.get('source_inventory_path') != str(root/source_binding['path'])
            or source.get('NumCore/AccelerateSparseSolver.cpp', {}).get('sha256') != PATCHED_SOURCE_SHA):
        raise ValueError('Accepted patched-source inventory differs')


def verify_runtime_binding(root, binding):
    """Check private runtime bytes only; this never establishes mechanics acceptance."""
    # Access imports this module too; shared path/hash helpers are loaded late.
    from scripts import mechanics_hbe_access as access
    root = Path(root).resolve()
    inputs = {}
    def bound(record, *, json_value=True):
        value = access.verify_binding(root, record, maximum_bytes=64*1024**2, read_json=json_value)
        inputs[str(access.local_path(root, record['path']))] = record['sha256']
        return value
    identity = bound(binding)
    prefix = root/PREFIX
    executable = prefix/'install/bin/febio4'
    if (identity.get('status') != 'isolated_patched_runtime_built_pending_numerical_controls'
            or identity.get('source_commit') != UPSTREAM_COMMIT
            or identity.get('runtime_version') != '4.13.0' or identity.get('architecture') != 'arm64'
            or identity.get('source_patch_sha256') != PATCH_SHA
            or identity.get('patched_source_sha256') != PATCHED_SOURCE_SHA
            or identity.get('executable') != str(executable)
            or not access.HEX64.fullmatch(identity.get('executable_sha256', ''))
            or identity.get('libraries', {}).get('install/bin/febio4') != identity.get('executable_sha256')):
        raise ValueError('Runtime does not identify the reviewed private repaired build')
    if set(identity['libraries']) != INSTALLED_FILES:
        raise ValueError('Complete thirteen-file installed runtime inventory required')
    _accepted_build(root, identity, bound)
    for name, digest in identity['libraries'].items():
        path = (prefix/name).resolve()
        if not path.is_relative_to(prefix.resolve()):
            raise ValueError('Runtime library escapes repaired private prefix')
        bound({'path': str(path.relative_to(root)), 'sha256': digest}, json_value=False)
    omp = identity['private_openmp']
    path = Path(omp['path']).resolve()
    declared_reuse = (path == (root/REUSED_OPENMP).resolve() and omp['sha256'] == REUSED_OPENMP_SHA)
    if not path.is_relative_to(prefix.resolve()) and not declared_reuse:
        raise ValueError('OpenMP escapes repaired private prefix')
    bound({'path': str(path.relative_to(root)), 'sha256': omp['sha256']}, json_value=False)
    bound({'path': str((prefix/'source/NumCore/AccelerateSparseSolver.cpp').relative_to(root)),
           'sha256': PATCHED_SOURCE_SHA}, json_value=False)
    return {'runtime_identity': binding, 'runtime': identity, 'prefix': str(prefix), 'inputs': inputs}


def _verify_controls(root, profile, groups, runtime, bound):
    """Replay the existing eight tiny checkers from retained, hash-bound output."""
    from scripts import mechanics_hbe_access as access
    evidence = groups[0].get('evidence', {})
    if set(evidence) != {'execution', 'results', 'baseline'} or groups[1].get('evidence') != evidence:
        raise ValueError('Both control groups must bind one actual eight-case attempt')
    execution, results, baseline = (bound(evidence[key]) for key in ('execution', 'results', 'baseline'))
    attempt = Path(baseline.get('attempt_directory', '')).resolve()
    source_directory = Path(baseline.get('source_directory', '')).resolve()
    if not attempt.is_relative_to(root) or not source_directory.is_relative_to(root):
        raise ValueError('Control attempt/source directory escapes repository')
    for key, filename in [('execution', 'execution.json'), ('results', 'results.json'), ('baseline', 'execution-baseline.json')]:
        if access.local_path(root, evidence[key]['path']) != attempt/filename:
            raise ValueError('Control evidence belongs to another attempt directory')
    identity = profile['runtime_identity']['sha256']
    caps = {'aggregate_seconds': 60, 'process_group_rss_bytes': 3*1024**3,
            'numerical_threads': 1, 'maximum_cases': 8, 'time_steps': 4, 'retries': 0}
    supervision = execution.get('supervision', {})
    elapsed = supervision.get('elapsed_seconds')
    if (execution.get('status') != 'completed' or execution.get('worker_started') is not True
            or execution.get('no_retry') is not True or supervision.get('status') != 'completed'
            or type(supervision.get('exit_code')) is not int or supervision['exit_code'] != 0
            or supervision.get('kill_reason') is not None or supervision.get('wall_cap_seconds') != 60
            or supervision.get('rss_cap_bytes') != caps['process_group_rss_bytes']
            or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or not 0 <= elapsed <= 60
            or supervision.get('sampled_peak_process_group_rss_bytes', math.inf) > caps['process_group_rss_bytes']
            or execution.get('thread_environment') != {'OMP_NUM_THREADS': '1', 'OMP_DYNAMIC': 'FALSE',
                'VECLIB_MAXIMUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            or baseline.get('runtime_identity_sha256') != identity or baseline.get('caps') != caps
            or baseline.get('case_order') != list(CASE_ORDER)
            or baseline.get('release', {}).get('runtime_identity') != profile['runtime_identity']
            or results.get('status') != 'completed' or results.get('solver_invocations') != 8
            or results.get('no_retry') is not True or results.get('solver_backend') != 'accelerate'
            or results.get('runtime_identity_sha256') != identity
            or [row.get('case') for row in results.get('cases', [])] != list(CASE_ORDER)):
        raise ValueError('Successful bounded same-runtime eight-case execution required')
    baseline_inputs = baseline.get('input_hashes', {})
    if not baseline_inputs:
        raise ValueError('Actual control source/input baseline required')
    for path, digest in baseline_inputs.items():
        absolute = Path(path).resolve()
        if not absolute.is_relative_to(root):
            raise ValueError('Control baseline input escapes repository')
        bound({'path': str(absolute.relative_to(root)), 'sha256': digest}, json_value=False)
    declaration_path = source_directory/'artifacts/mechanics-accelerate-controls-v1/declaration.json'
    if str(declaration_path) not in baseline_inputs:
        raise ValueError('Original control declaration is not a bound baseline input')
    declaration = bound({'path': str(declaration_path.relative_to(root)), 'sha256': baseline_inputs[str(declaration_path)]})
    if (declaration != baseline.get('declaration') or declaration.get('case_order') != list(CASE_ORDER)
            or set(declaration.get('cases', {})) != set(CASE_ORDER)
            or any(group.get('source_commit') != baseline.get('source_commit') or group.get('execution_accepted') is not True for group in groups)):
        raise ValueError('Control summaries differ from their source-bound eight-case declaration')
    for audit in (execution.get('inputs_after'), execution.get('final_inputs_after'), results.get('inputs_after')):
        if not isinstance(audit, dict) or set(audit) != set(baseline_inputs) or not all(row.get('unchanged') is True for row in audit.values()):
            raise ValueError('Control execution did not preserve its complete bound inputs')
    # The tet10 checker imports its hex8 sibling by filename. Authenticate that
    # entire two-file closure before importing either source, including origin.
    checker_paths = {group: source_directory/'scripts'/name for group, name in CHECKER_FILES.items()}
    for group, path in checker_paths.items():
        if baseline_inputs.get(str(path)) != CHECKER_PINS[group]:
            raise ValueError('Original sibling checker is absent from bound source inputs')
        bound({'path': str(path.relative_to(root)), 'sha256': CHECKER_PINS[group]}, json_value=False)
    checked_rows, raw_text, checkers = {}, {}, {}
    summary_rows = {row['case']: row for group, field in zip(groups, ('rows', 'case_rows')) for row in group[field]}
    for result_row in results['cases']:
        name = result_row['case']; row = summary_rows[name]
        if (any(row.get(key) != value for key, value in result_row.items())
                or row.get('status') != 'passed' or row.get('checker_passed') is not True
                or type(row.get('solver_exit_code')) is not int or row['solver_exit_code'] != 0):
            raise ValueError('Control summary differs from actual completed case')
        records = row.get('evidence', {})
        if set(records) != {'original_deck', 'executed_deck', 'console', 'nodes', 'elements', 'solver_log', 'checked', 'checker_source'}:
            raise ValueError('Each control requires actual deck, output, execution and checker bindings')
        for record in records.values():
            bound(record, json_value=False)
        def text(key):
            path = access.local_path(root, records[key]['path'])
            if path.stat().st_size > 4*1024**2:
                raise ValueError('Tiny control input exceeds fixed reader allowance')
            return path.read_text(encoding='utf-8')
        original, adapted = text('original_deck'), text('executed_deck')
        verify_deck(original, adapted)
        directory = access.local_path(root, records['executed_deck']['path']).parent
        declared = declaration['cases'][name]
        original_path = (source_directory/declared['original_path']).resolve()
        template_path = (source_directory/declared['adapted_path']).resolve()
        if (directory != attempt/name or access.local_path(root, records['original_deck']['path']) != original_path
                or records['original_deck']['sha256'] != declared['original_sha256']
                or records['executed_deck']['sha256'] != declared['adapted_sha256']
                or baseline_inputs.get(str(original_path)) != declared['original_sha256']
                or baseline_inputs.get(str(template_path)) != declared['adapted_sha256']):
            raise ValueError('Executed/original control deck differs from the bound declaration and attempt')
        if row.get('command') != [runtime['runtime']['executable'], '-noconfig', '-no_title', '-i', name+'.feb', '-o', name+'.log']:
            raise ValueError('Control executed an unexpected solver command')
        for key, filename in [('executed_deck', name+'.feb'), ('console', 'console.txt'), ('nodes', name+'.nodes.log'),
                              ('elements', name+'.elements.log'), ('solver_log', name+'.log'), ('checked', 'checked.json')]:
            if access.local_path(root, records[key]['path']) != directory/filename:
                raise ValueError('Control output paths differ from executed case directory')
        selections = [line.strip() for line in text('console').splitlines() if 'selecting linear solver' in line.lower()]
        expected_evidence = {'solver_backend': 'accelerate', 'actual_selection_lines': selections,
            'solver_xml': ACCELERATE_XML, 'solver_only_change_verified': True,
            'runtime_identity_sha256': identity, 'executed_deck_sha256': records['executed_deck']['sha256']}
        if (not selections or any(not re.search(r'\bselecting linear solver accelerate\b', line, re.I) for line in selections)
                or row.get('backend_evidence') != expected_evidence or row.get('deck_sha256') != records['executed_deck']['sha256']):
            raise ValueError('Actual control backend selection differs')
        group = 'hex8' if name in HEX_CASES else 'tet10'
        checker_binding = records['checker_source']
        if (checker_binding['sha256'] != CHECKER_PINS[group]
                or access.local_path(root, checker_binding['path']) != checker_paths[group]):
            raise ValueError('Original numerical checker source changed')
        if group not in checkers:
            path = access.local_path(root, checker_binding['path'])
            spec = importlib.util.spec_from_file_location('hbe_verified_'+group, path)
            checker = importlib.util.module_from_spec(spec); spec.loader.exec_module(checker)
            checkers[group] = checker
        raw = (text('nodes'), text('elements'), text('solver_log'))
        checked = checkers[group].check_outputs(name, *raw)
        saved_check = bound(records['checked'])
        if checked.get('passed') is not True or access.canonical_json(checked) != access.canonical_json(saved_check):
            raise ValueError('Control did not reproduce the retained original checker result')
        checked_rows[name], raw_text[name] = checked, raw
    scaling = checkers['hex8'].check_stiffness_scaling(*raw_text['shear'][:2], *raw_text['shear_double_stiffness'][:2])
    if (scaling.get('passed') is not True or access.canonical_json(scaling) != access.canonical_json(groups[0]['stiffness_scaling'])
            or access.canonical_json(scaling) != access.canonical_json(results.get('stiffness_scaling'))):
        raise ValueError('Original stiffness-scaling check did not reproduce')


def _verify_patch_identity(binding, bound):
    """Bind the committed combined repair; this is not a runtime pass receipt."""
    if binding != PATCH_IDENTITY:
        raise ValueError('Exact committed combined repair identity required')
    patch = bound(binding)
    if (patch.get('one_source_file_only') != 'NumCore/AccelerateSparseSolver.cpp'
            or patch.get('patched_source', {}).get('sha256') != PATCHED_SOURCE_SHA
            or patch.get('patch', {}).get('sha256') != PATCH_SHA):
        raise ValueError('Combined repair source or patch differs')
    for key in ('patch', 'patched_source'):
        record = patch[key]
        bound({'path': record['path'], 'sha256': record['sha256']}, json_value=False)


def verify_profile(root, binding):
    """Verify actual same-runtime controls, repair provenance and private bytes."""
    from scripts import mechanics_hbe_access as access
    root = Path(root).resolve()
    inputs = {}
    def bound(record, *, json_value=True):
        value = access.verify_binding(root, record, maximum_bytes=64*1024**2, read_json=json_value)
        inputs[str(access.local_path(root, record['path']))] = record['sha256']
        return value
    profile = bound(binding)
    if (set(profile) != {'schema', 'profile_id', 'runtime_identity', 'hex8_controls',
                         'tet10_mpc_controls', 'patch_identity'}
            or profile['schema'] != 'hbe-solver-backend-v1' or profile['profile_id'] != PROFILE_ID):
        raise ValueError('Explicit complete repaired-Accelerate profile required')
    runtime = verify_runtime_binding(root, profile['runtime_identity'])
    inputs.update(runtime['inputs'])
    _verify_patch_identity(profile['patch_identity'], bound)
    groups = []
    for key, status, field, cases in (
            ('hex8_controls', 'passed_all_five_fixed_patch_controls', 'rows', HEX_CASES),
            ('tet10_mpc_controls', 'three_actual_fixed_software_controls_passed', 'case_rows', TET_CASES)):
        receipt = bound(profile[key])
        groups.append(receipt)
        rows = receipt.get(field, [])
        if (receipt.get('status') != status or receipt.get('solver_backend') != 'accelerate'
                or receipt.get('runtime_identity_sha256') != profile['runtime_identity']['sha256']
                or type(receipt.get('solver_invocations')) is not int or receipt['solver_invocations'] != len(cases)
                or len(rows) != len(cases) or {row.get('case') for row in rows} != cases
                or not all(row.get('passed') is True for row in rows)):
            raise ValueError('Complete same-runtime actual analytical controls required: '+key)
        if key == 'hex8_controls' and receipt.get('stiffness_scaling', {}).get('passed') is not True:
            raise ValueError('Actual same-runtime stiffness scaling control required')
    _verify_controls(root, profile, groups, runtime, bound)
    return {'profile_id': PROFILE_ID, 'binding': binding, 'runtime_identity': profile['runtime_identity'],
            'runtime': runtime['runtime'], 'hex8_controls': profile['hex8_controls'],
            'tet10_mpc_controls': profile['tet10_mpc_controls'], 'prefix': runtime['prefix'], 'inputs': inputs}


def verify_prepared_decks(root, binding, original_cases):
    """Authenticate the eighteen transformed inputs without regenerating meshes."""
    from scripts import mechanics_hbe_access as access
    root = Path(root).resolve()
    manifest = access.verify_binding(root, binding, maximum_bytes=1024**2, read_json=True)
    expected = {name for name, row in access.expected_runs().items() if row[-1] != 'fitted'}
    if (set(original_cases) != expected or manifest.get('schema') != 'hbe-backend-decks-v1' or manifest.get('profile_id') != PROFILE_ID
            or manifest.get('transform_source_sha256') != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            or set(manifest.get('cases', {})) != set(original_cases)):
        raise ValueError('Complete source-bound backend deck inventory required')
    cases, sources = {}, {}
    inputs = {str(access.local_path(root, binding['path'])): binding['sha256']}
    for run_id, original in original_cases.items():
        row = manifest['cases'][run_id]
        if set(row) != {'original', 'adapted'} or row['original'] != original:
            raise ValueError('Backend preparation changed original case binding')
        adapted = row['adapted']
        if set(adapted) != {'mesh', 'deck', 'loading'} or adapted['mesh'] != original['mesh']:
            raise ValueError('Backend preparation changed mesh or primitive contract')
        for value in [*original.values(), *adapted.values()]:
            access.verify_binding(root, value)
            inputs[str(access.local_path(root, value['path']))] = value['sha256']
        verify_deck(access.local_path(root, original['deck']['path']).read_bytes(),
                    access.local_path(root, adapted['deck']['path']).read_bytes())
        before = access.verify_binding(root, original['loading'], read_json=True)
        after = access.verify_binding(root, adapted['loading'], read_json=True)
        before['deck_sha256'] = adapted['deck']['sha256']
        if before != after:
            raise ValueError('Backend preparation changed loading metadata beyond deck hash')
        cases[run_id], sources[run_id] = adapted, original['deck']
    return cases, sources, inputs


def prepare_decks(root, original_cases, directory):
    """Write new solver-only input copies, never a release or runtime claim."""
    from scripts import mechanics_hbe_access as access
    root, directory = Path(root).resolve(), Path(directory).resolve()
    expected = {name for name, row in access.expected_runs().items() if row[-1] != 'fitted'}
    if set(original_cases) != expected or not directory.is_relative_to(root):
        raise ValueError('Exactly the eighteen fixed reference/scaling cases required')
    directory.mkdir(parents=True, exist_ok=False)
    def record(path):
        return {'path': str(path.relative_to(root)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {'schema': 'hbe-backend-decks-v1', 'profile_id': PROFILE_ID,
                'transform_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'cases': {}}
    for run_id in access.expected_runs():
        if run_id not in original_cases:
            continue
        original = original_cases[run_id]
        if set(original) != {'mesh', 'deck', 'loading'}:
            raise ValueError('Original prepared input set differs')
        for value in original.values():
            access.verify_binding(root, value)
        target = directory/run_id.replace(':', '-')
        target.mkdir()
        xml = transform_deck(access.local_path(root, original['deck']['path']).read_bytes())
        with (target/'specimen.feb').open('x', encoding='utf-8', newline='') as stream:
            stream.write(xml)
        adapted = {'mesh': original['mesh'], 'deck': record(target/'specimen.feb')}
        loading = access.verify_binding(root, original['loading'], read_json=True)
        loading['deck_sha256'] = adapted['deck']['sha256']
        access.exclusive_json(target/'loading.json', loading)
        adapted['loading'] = record(target/'loading.json')
        manifest['cases'][run_id] = {'original': original, 'adapted': adapted}
    access.exclusive_json(directory/'manifest.json', manifest)
    binding = record(directory/'manifest.json')
    verify_prepared_decks(root, binding, original_cases)
    return binding
