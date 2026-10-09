"""One release-gated HBE v5 remaining-row native call and bounded replay.

The tracked preparation has no release. This module never initiates a native
call without an explicit, separately reviewed one-row release. A failed row
consumes its one attempt and blocks the ordered chain; there is no retry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import time

from scripts import mechanics_hbe_v5_n8_one_shot as io


ROOT = Path(__file__).resolve().parents[1]
PREPARATION = 'manifests/experiments/hbe-v5-remaining-one-shot-preparation-v1.json'
OUTPUT_ROOT = 'outputs/mechanics/hbe-v5-remaining-one-shot-v1'
V5 = 'manifests/experiments/hbe-01-03-branch-calibration-v5.json'
SOURCE_MAP = 'manifests/experiments/hbe-01-03-v5-source-deck-bindings-v1.json'
N8_PATH = 'outputs/mechanics/hbe-v5-n8-one-shot-v1/attempt-01/receipt.json'
N8_SHA = '8099c438caadf120d7269c14025fb92932fdda12a05f5ac8e8ade981cb20d45d'
N8_SOURCE_COMMIT = '65c6d0b8b54514ae078bc72164a4d8dea416951c'
N8_RELEASE_SHA = '417f34c234ed5c79eeaab4f9e060cd7e3a49b67acbe1de63730f018aa3afaad3'
N8_DECK_SHA = '3cf4156b19191918841498490bc448f88a61b656b566d5d6a3d36db817d0a724'
N8_NATIVE_SECONDS = 4.570366250118241
N8_RECORDED_OUTPUT_BYTES = 15488457
N8_OUTPUT_BYTES = 16075117
ORDER = (
    'compression:N8:S60:reference', 'compression:N12:S60:reference',
    'compression:N16:S60:reference', 'compression:N24:S60:reference',
    'compression:N32:S60:reference', 'compression:N36:S60:reference',
    'compression:N36:S120:reference', 'tension:N8:S60:reference',
    'tension:N12:S60:reference', 'tension:N16:S60:reference',
    'tension:N24:S60:reference', 'tension:N24:S120:reference',
)
NATIVE_WALL = (0, 90, 420, 600, 1800, 2400, 2400, 90, 90, 420, 600, 600)
OUTPUT_MIB = (0, 64, 256, 384, 768, 1024, 2048, 64, 64, 128, 384, 768)
NATIVE_RSS = READOUT_RSS = 3 * 1024**3
READOUT_WALL = 600
PREP_WALL = 150
AGGREGATE = {'maximum_native_calls_all_12': 12,
             'native_wall_seconds_all_12': 9600,
             'remaining_readout_wall_seconds': 6600,
             'remaining_prep_wall_seconds': 1800,
             'known_stage_total_wall_seconds': 18000,
             'active_output_bytes_all_12': 6 * 1024**3,
             'storage_free_reserve_bytes': 2 * 1024**3}
SOURCE_PATHS = tuple(dict.fromkeys((
    'scripts/mechanics_hbe_v5_remaining_one_shot.py',
    *io.SOURCE_PATHS,
)))
SELECTION = '* Selecting linear solver accelerate                                    *'
HEX = re.compile(r'[0-9a-f]{64}\Z')
COMMIT = re.compile(r'[0-9a-f]{40}\Z')
NATIVE_FILES = {'specimen.feb', 'nodes.log', 'elements.log', 'solver.log',
                'console.txt'}
FINAL_FILES = NATIVE_FILES | {'receipt.json', 'readout-work-order.json',
                              'readout-console.txt', 'readout.json'}
SCAN_ENTRY_CAP = 256
SCAN_SECONDS = 1.


def output_directory(index: int) -> str:
    if type(index) is not int or index not in range(1, 12):
        raise ValueError('Only frozen remaining row indices 1–11 are executable')
    return f'{OUTPUT_ROOT}/{index:02d}-{ORDER[index].replace(":", "-")}/attempt-01'


def receipt_path(index: int) -> str:
    return N8_PATH if index == 0 else output_directory(index) + '/receipt.json'


def caps(index: int) -> dict:
    if index not in range(1, 12):
        raise ValueError('Undeclared remaining index')
    return {'native_wall_seconds': NATIVE_WALL[index],
            'native_sampled_process_group_rss_bytes': NATIVE_RSS,
            'readout_wall_seconds': READOUT_WALL,
            'readout_sampled_process_group_rss_bytes': READOUT_RSS,
            'prep_wall_seconds': PREP_WALL,
            'active_output_bytes': OUTPUT_MIB[index] * 1024**2,
            'numerical_threads': 1, 'native_calls': 1, 'attempts': 1}


def validate_preparation(value: dict) -> None:
    if (not isinstance(value, dict)
            or value.get('schema') != 'hbe-v5-remaining-one-shot-preparation-v1'
            or value.get('status') != 'prepared_not_released_or_executed'
            or value.get('release') is not None
            or value.get('study') != V5 or value.get('source_map') != SOURCE_MAP
            or value.get('output_root') != OUTPUT_ROOT or value.get('first_index') != 1
            or value.get('aggregate') != AGGREGATE
            or value.get('per_row_sampled_process_group_rss_bytes') != NATIVE_RSS
            or value.get('per_row_readout_wall_seconds') != READOUT_WALL
            or value.get('per_row_readout_sampled_rss_bytes') != READOUT_RSS
            or value.get('per_row_prep_wall_seconds') != PREP_WALL
            or value.get('numerical_threads') != 1 or value.get('attempts_per_row') != 1
            or value.get('automatic_retry') is not False
            or value.get('phase_gates') != {
                'native_execution': False, 'automatic_reference_release': False,
                'fit': False, 'held_out_torque_access': False,
                'physical_validation': False}):
        raise ValueError('Remaining-row preparation, resource caps or phase gates changed')
    expected = [{'index': i, 'run_id': ORDER[i], 'native_wall_seconds': NATIVE_WALL[i],
                 'active_output_mib': OUTPUT_MIB[i]} for i in range(1, 12)]
    if value.get('per_row') != expected or value.get('fixed_n8_predecessor') != {
            'run_id': ORDER[0], 'path': N8_PATH, 'sha256': N8_SHA,
            'source_commit': N8_SOURCE_COMMIT, 'release_sha256': N8_RELEASE_SHA,
            'adapted_deck_sha256': N8_DECK_SHA,
            'native_elapsed_seconds': N8_NATIVE_SECONDS,
            'recorded_final_active_output_bytes': N8_RECORDED_OUTPUT_BYTES,
            'actual_saved_output_bytes': N8_OUTPUT_BYTES,
            'readout_elapsed_seconds': None, 'prep_elapsed_seconds': None}:
        raise ValueError('Frozen row order, N8 predecessor or per-row cap differs')
    if (sum(NATIVE_WALL[1:]) != 9510 or sum(OUTPUT_MIB[1:]) != 5952
            or NATIVE_WALL[1:].count(90) != 3):
        raise ValueError('Prospective aggregate arithmetic differs')


def audit_imports(*, root: Path = ROOT) -> None:
    allowed = set(SOURCE_PATHS)
    base = root.resolve()
    for name, module in tuple(sys.modules.items()):
        if not name.startswith('scripts.'):
            continue
        origin = getattr(module, '__file__', None)
        if origin is None:
            continue
        path = Path(origin).resolve()
        if not path.is_relative_to(base / 'scripts') or str(path.relative_to(base)) not in allowed:
            raise ValueError('Unbound executing scripts import: ' + name)
    fixture = sys.modules.get('scripts.mechanics_patient_constraints')
    if fixture is not None:
        patch = getattr(fixture, 'patch', None)
        if patch is None or Path(patch.__file__).resolve() != base / 'scripts/mechanics_febio_verification.py':
            raise ValueError('Unbound dynamically loaded checker')


def source_hashes(release: dict, *, root: Path = ROOT,
                  require_head: bool = True) -> tuple[dict[str, str], str]:
    commit = release.get('source_commit')
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        raise ValueError('Full committed source revision required')
    if set(release.get('source_bindings', {})) != set(SOURCE_PATHS):
        raise ValueError('Complete executing source closure required')
    head = subprocess.run(['/usr/bin/git', 'rev-parse', 'HEAD'], cwd=root,
                          env=io.git_env(), capture_output=True, check=True,
                          timeout=10).stdout.decode().strip()
    if require_head and head != commit:
        raise ValueError('Checkout changed from one-call source release')
    result = {}
    for relative in SOURCE_PATHS:
        data = io.bound(release['source_bindings'][relative], relative, root=root)
        name = f'{commit}:{relative}'
        size = subprocess.run(['/usr/bin/git', 'cat-file', '-s', name], cwd=root,
                              env=io.git_env(), capture_output=True,
                              check=True, timeout=10).stdout
        if int(size) > 1024**2:
            raise ValueError('Committed source exceeds one MiB')
        original = subprocess.run(['/usr/bin/git', 'cat-file', 'blob', name], cwd=root,
                                  env=io.git_env(), capture_output=True,
                                  check=True, timeout=10).stdout
        if len(original) > 1024**2 or io.sha(original) != io.sha(data):
            raise ValueError('Released file differs from committed source: ' + relative)
        result[relative] = release['source_bindings'][relative]['sha256']
    return result, head


def active_bytes(directory: Path, byte_cap: int) -> int:
    """Bounded metadata scan; count all attempt files and reject links."""
    total, count = 0, 0
    deadline = time.monotonic() + SCAN_SECONDS
    stack = [(directory, 0)]
    while stack:
        base, depth = stack.pop()
        if depth > 16:
            raise ValueError('Active output depth cap exceeded')
        with os.scandir(base) as entries:
            for entry in entries:
                count += 1
                if count > SCAN_ENTRY_CAP or time.monotonic() > deadline:
                    raise ValueError('Active output scan work cap exceeded')
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(metadata.st_mode):
                    total += metadata.st_size
                elif stat.S_ISDIR(metadata.st_mode):
                    stack.append((Path(entry.path), depth+1))
                else:
                    raise ValueError('Output symlink or special file refused')
                if total > byte_cap:
                    return total
    return total


def _regular_output_binding(path: Path, *, root: Path, limit: int) -> dict:
    relative = str(path.relative_to(root))
    digest = io.file_hash(io.local(relative, root=root), maximum=limit)
    size = path.stat().st_size
    if not 0 < size <= limit:
        raise ValueError('Native output absent or above bound')
    return {'path': relative, 'sha256': digest, 'bytes': size}


def _check_backend(directory: Path) -> None:
    for name in ('console.txt', 'solver.log'):
        path = directory / name
        if path.stat().st_size > 32 * 1024**2:
            raise ValueError('Backend selection log exceeds bounded reader')
        text = path.read_text(encoding='ascii', errors='strict')
        selections = [line.strip() for line in text.splitlines()
                      if 'selecting linear solver' in line.lower()]
        forbidden = re.search(r'\b(?:fallback|fall\s+back|switching\s+(?:linear\s+)?solver)\b',
                              text, re.I)
        if len(selections) != 1 or not io.BACKEND.fullmatch(selections[0]) or forbidden:
            raise ValueError('Exactly one actual Accelerate selection required in both logs')


def _receipt_binding(root: Path, relative: str, digest: str) -> dict:
    if not isinstance(digest, str) or not HEX.fullmatch(digest):
        raise ValueError('Full receipt SHA256 required')
    return {'path': relative, 'sha256': digest}


def validate_prior_chain(bindings: list[dict], index: int, *, root: Path = ROOT,
                         expected_profile: dict | None = None,
                         expected_runtime: dict | None = None) -> dict:
    """Rehash every earlier receipt/output and preserve N8 as row zero."""
    if type(index) is not int or index not in range(1, 12) or not isinstance(bindings, list) or len(bindings) != index:
        raise ValueError('Complete ordered predecessor chain required')
    hashes = []
    native = N8_NATIVE_SECONDS
    readout = prep = 0.
    output = N8_OUTPUT_BYTES
    for ordinal, item in enumerate(bindings):
        expected = receipt_path(ordinal)
        if (not isinstance(item, dict) or set(item) != {'run_id', 'path', 'sha256'}
                or item['run_id'] != ORDER[ordinal] or item['path'] != expected
                or ordinal == 0 and item['sha256'] != N8_SHA):
            raise ValueError('Predecessor order/path/hash differs')
        document = json.loads(io.bound(_receipt_binding(root, expected, item['sha256']),
                                       expected, root=root, maximum=8*1024**2))
        if (document.get('run_id') != ORDER[ordinal]
                or document.get('status') != 'passed_numerical_software_only'
                or document.get('native_calls_attempted') != 1
                or document.get('no_retry') is not True
                or document.get('saved_numerical_readout', {}).get('numerical_passed') is not True):
            raise ValueError('Predecessor did not pass frozen numerical decision')
        if ordinal == 0:
            if (document.get('schema') != 'hbe-v5-n8-one-call-receipt-v1'
                    or document.get('source_commit') != N8_SOURCE_COMMIT
                    or document.get('release_sha256') != N8_RELEASE_SHA
                    or document.get('adapted_deck_sha256') != N8_DECK_SHA
                    or document.get('elapsed_seconds') != N8_NATIVE_SECONDS
                    or document.get('final_active_output_bytes') != N8_RECORDED_OUTPUT_BYTES
                    or document.get('exit_code') != 0 or document.get('kill_reason') is not None
                    or document.get('caps') != io.CAPS):
                raise ValueError('Fixed independently reviewed N8 predecessor differs')
            names = NATIVE_FILES
            records = document.get('native_output_bindings')
            if document['saved_numerical_readout'].get('frame_count') != 61:
                raise ValueError('N8 saved-frame count differs')
        else:
            if (document.get('schema') != 'hbe-v5-remaining-one-call-receipt-v1'
                    or document.get('ordinal') != ordinal
                    or document.get('caps') != caps(ordinal)
                    or document.get('source_commit') is None
                    or not COMMIT.fullmatch(document['source_commit'])
                    or document.get('native_stage', {}).get('exit_code') != 0
                    or document.get('native_stage', {}).get('status') != 'completed_within_caps'
                    or document.get('native_stage', {}).get('kill_reason') is not None
                    or document.get('readout_stage', {}).get('exit_code') != 0
                    or document.get('readout_stage', {}).get('status') != 'completed_within_caps'
                    or document.get('readout_stage', {}).get('kill_reason') is not None
                    or document.get('readout_calls_attempted') != 1
                    or document.get('prior_receipt_sha256') != hashes
                    or document.get('prior_native_wall_seconds') != native
                    or document.get('prior_readout_wall_seconds') != readout
                    or document.get('prior_prep_wall_seconds') != prep
                    or document.get('prior_active_output_bytes') != output):
                raise ValueError('Continuation predecessor release, caps or ledger differs')
            ns = document.get('native_stage', {}).get('elapsed_seconds')
            rs = document.get('readout_stage', {}).get('elapsed_seconds')
            ps = document.get('prep_elapsed_seconds')
            if (any(type(value) not in (int, float) or not math.isfinite(value) or value < 0
                    for value in (ns, rs, ps))
                    or not (ns < caps(ordinal)['native_wall_seconds']
                            and rs < READOUT_WALL and ps < PREP_WALL)):
                raise ValueError('Predecessor stage timing or cap differs')
            native += ns
            readout += rs
            prep += ps
            if (document.get('aggregate_native_wall_seconds') != native
                    or document.get('aggregate_readout_wall_seconds') != readout
                    or document.get('aggregate_prep_wall_seconds') != prep
                    or document.get('aggregate_native_calls') != ordinal+1):
                raise ValueError('Predecessor cumulative resource ledger differs')
            names = FINAL_FILES - {'receipt.json'}
            records = document.get('output_bindings')
        if not isinstance(records, dict) or set(records) != names:
            raise ValueError('Predecessor output inventory differs')
        directory = io.local(str(Path(expected).parent), root=root)
        if {p.name for p in directory.iterdir()} != names | {'receipt.json'}:
            raise ValueError('Predecessor output directory changed')
        for name in names:
            record = records[name]
            if (not isinstance(record, dict) or set(record) != {'path', 'sha256', 'bytes'}
                    or record['path'] != str(Path(expected).parent / name)
                    or type(record['bytes']) is not int or record['bytes'] <= 0
                    or io.file_hash(io.local(record['path'], root=root), maximum=2*1024**3)
                       != record['sha256']
                    or io.local(record['path'], root=root).stat().st_size != record['bytes']):
                raise ValueError('Predecessor native or readout file changed: ' + name)
        if ordinal > 0:
            summary = document['saved_numerical_readout']
            if (summary.get('run_id') != ORDER[ordinal]
                    or summary.get('frame_count') != (121 if ':S120:' in ORDER[ordinal] else 61)
                    or summary.get('readout_sha256') != records['readout.json']['sha256']
                    or summary.get('native_output_observed') is not True
                    or summary.get('physical_validation_pass') is not None
                    or document.get('native_output_bindings') != {
                        name: records[name] for name in NATIVE_FILES}):
                raise ValueError('Predecessor saved readout or native binding differs')
            release_path = document.get('release_path')
            if (not isinstance(release_path, str)
                    or not Path(release_path).is_absolute()
                    or not Path(release_path).is_relative_to(root.resolve())
                    or expected_profile is None or expected_runtime is None):
                raise ValueError('Later predecessor release or runtime ancestry absent')
            release_raw, _ = io.read_release(Path(release_path))
            if io.sha(release_raw) != document.get('release_sha256'):
                raise ValueError('Later predecessor release bytes changed')
            prior_release = json.loads(release_raw)
            prior_source_bindings = prior_release.get('source_bindings', {})
            prior_work_order = json.loads(io.bound(
                {'path': records['readout-work-order.json']['path'],
                 'sha256': records['readout-work-order.json']['sha256']},
                records['readout-work-order.json']['path'], root=root,
                maximum=1024**2))
            if (prior_release.get('schema') != 'hbe-v5-remaining-one-call-release-v1'
                    or prior_release.get('status') != 'root_released_one_native_call'
                    or prior_release.get('ordinal') != ordinal
                    or prior_release.get('run_id') != ORDER[ordinal]
                    or prior_release.get('output_directory') != output_directory(ordinal)
                    or prior_release.get('caps') != caps(ordinal)
                    or prior_release.get('aggregate_caps') != AGGREGATE
                    or prior_release.get('prior_receipts') != bindings[:ordinal]
                    or prior_release.get('source_commit') != document.get('source_commit')
                    or prior_release.get('runtime_identity') != expected_runtime
                    or prior_release.get('backend_profile') != expected_profile
                    or prior_release.get('adapted_deck_sha256') !=
                       document.get('adapted_deck_sha256')
                    or records['specimen.feb']['sha256'] !=
                       document.get('adapted_deck_sha256')
                    or not isinstance(prior_source_bindings, dict)
                    or set(prior_source_bindings) != set(SOURCE_PATHS)
                    or {name: binding.get('sha256') for name, binding in
                        prior_source_bindings.items() if isinstance(binding, dict)} !=
                       document.get('source_hashes_before')
                    or document.get('source_hashes_after') !=
                       document.get('source_hashes_before')
                    or document.get('observed_head_preflight') !=
                       document.get('source_commit')
                    or not isinstance(document.get('observed_head_after'), str)
                    or not COMMIT.fullmatch(document['observed_head_after'])
                    or document.get('runtime_profile_verified_after') is not True
                    or not isinstance(document.get('readout_token_sha256'), str)
                    or not HEX.fullmatch(document['readout_token_sha256'])
                    or prior_work_order.get('readout_token_sha256') !=
                       document['readout_token_sha256']
                    or prior_release['runtime_identity'].get('sha256') !=
                       document.get('runtime_identity_sha256')
                    or prior_release['backend_profile'].get('sha256') !=
                       document.get('backend_profile_sha256')
                    or prior_release['preparation'].get('sha256') !=
                       document.get('preparation_sha256')
                    or prior_release['v5_declaration'].get('sha256') !=
                       document.get('v5_declaration_sha256')
                    or prior_release['source_map'].get('sha256') !=
                       document.get('source_map_sha256')
                    or prior_release['old_source_deck'].get('sha256') !=
                       document.get('old_source_deck_sha256')
                    or prior_release['native_mesh'].get('sha256') !=
                       document.get('native_mesh_sha256')):
                raise ValueError('Later predecessor release/source/runtime/deck differs')
        if io.file_hash(directory / 'receipt.json', maximum=8*1024**2) != item['sha256']:
            raise ValueError('Predecessor receipt changed during verification')
        closed_total = active_bytes(directory, 64*1024**2 if ordinal == 0
                                    else caps(ordinal)['active_output_bytes'])
        if ordinal == 0:
            if closed_total != N8_OUTPUT_BYTES:
                raise ValueError('Actual N8 closed-directory total differs')
        else:
            if closed_total > caps(ordinal)['active_output_bytes']:
                raise ValueError('Predecessor actual closed directory exceeds row output cap')
            output += closed_total
        hashes.append(item['sha256'])
    if (native > AGGREGATE['native_wall_seconds_all_12']
            or readout > AGGREGATE['remaining_readout_wall_seconds']
            or prep > AGGREGATE['remaining_prep_wall_seconds']
            or native+readout+prep > AGGREGATE['known_stage_total_wall_seconds']
            or output > AGGREGATE['active_output_bytes_all_12']):
        raise ValueError('Predecessor aggregate resource cap exceeded')
    current_caps = caps(index)
    if (native + current_caps['native_wall_seconds'] > AGGREGATE['native_wall_seconds_all_12']
            or readout + READOUT_WALL > AGGREGATE['remaining_readout_wall_seconds']
            or prep + PREP_WALL > AGGREGATE['remaining_prep_wall_seconds']
            or native + readout + prep + current_caps['native_wall_seconds']
               + READOUT_WALL + PREP_WALL > AGGREGATE['known_stage_total_wall_seconds']
            or output + current_caps['active_output_bytes'] > AGGREGATE['active_output_bytes_all_12']):
        raise ValueError('Remaining aggregate resource budget cannot cover one call')
    return {'sha256': hashes, 'native_seconds': native, 'readout_seconds': readout,
            'prep_seconds': prep, 'output_bytes': output, 'native_calls': index}


def validate_release(release: dict, *, root: Path = ROOT) -> dict:
    """Read-only one-row release and storage preflight, before output reservation."""
    keys = {'schema', 'status', 'ordinal', 'run_id', 'source_commit',
            'source_bindings', 'preparation', 'v5_declaration', 'source_map',
            'old_source_deck', 'native_mesh', 'adapted_deck_sha256',
            'adapter_receipt', 'backend_profile', 'runtime_identity',
            'output_directory', 'caps', 'aggregate_caps', 'prior_receipts'}
    if (not isinstance(release, dict) or set(release) != keys
            or release['schema'] != 'hbe-v5-remaining-one-call-release-v1'
            or release['status'] != 'root_released_one_native_call'
            or type(release['ordinal']) is not int
            or release['ordinal'] not in range(1, 12)
            or release['run_id'] != ORDER[release['ordinal']]
            or release['output_directory'] != output_directory(release['ordinal'])
            or release['caps'] != caps(release['ordinal'])
            or release['aggregate_caps'] != AGGREGATE
            or not isinstance(release['adapted_deck_sha256'], str)
            or not HEX.fullmatch(release['adapted_deck_sha256'])):
        raise ValueError('Missing or altered one-row HBE v5 release')
    index = release['ordinal']
    preparation = json.loads(io.bound(release['preparation'], PREPARATION, root=root))
    validate_preparation(preparation)
    from scripts import mechanics_hbe_branch_calibration_v5 as v5
    from scripts import mechanics_hbe_v5_source_bindings as sources
    from scripts import mechanics_hbe_backend as backend
    if (release['v5_declaration'] != {'path': V5, 'sha256': v5.DECLARATION_SHA256}
            or release['source_map'] != {'path': SOURCE_MAP, 'sha256': sources.BINDING_SHA256}):
        raise ValueError('Frozen v5 declaration or source map differs')
    io.bound(release['v5_declaration'], V5, root=root)
    source_map = json.loads(io.bound(release['source_map'], SOURCE_MAP, root=root))
    study, prior_study = v5.validate_preparation(root)
    if tuple(row['run_id'] for row in study['ordered_reference_runs']) != ORDER:
        raise ValueError('Frozen twelve-row order differs')
    sources.validate_binding_manifest(root, inspect_sources=True)
    key = source_map['run_source_keys'][release['run_id']]
    source_binding = source_map['source_decks'][key]
    mesh_binding = source_map['meshes'][source_binding['mesh']]
    source_exact = {name: source_binding[name] for name in ('path', 'sha256')}
    if (release['old_source_deck'] != source_exact
            or release['native_mesh'] != mesh_binding):
        raise ValueError('Frozen source deck or mesh mapping differs')
    old = sources._read_bound(root, source_exact, maximum=sources.MAX_SOURCE_BYTES)
    mesh = sources._read_bound(root, mesh_binding, maximum=sources.MAX_MESH_BYTES)
    sources.validate_topology_bc(json.loads(mesh), old,
                                 native_domain=v5.run_spec(study, prior_study, release['run_id'])['native_domain'])
    _, adapted, adapter_receipt = v5.adapt_deck(study, prior_study, release['run_id'], old)
    deck = adapted.encode('utf-8')
    if (io.sha(deck) != release['adapted_deck_sha256']
            or release['adapter_receipt'] != adapter_receipt
            or adapter_receipt['adapted_deck_sha256'] != release['adapted_deck_sha256']):
        raise ValueError('Released exact adapted deck or full schedule differs')
    profile_binding = prior_study['preserved_v3_fields']['backend_profile']
    runtime_binding = prior_study['preserved_v3_fields']['runtime_identity']
    previous = validate_prior_chain(release['prior_receipts'], index, root=root,
                                    expected_profile=profile_binding,
                                    expected_runtime=runtime_binding)
    source_bound, observed_head = source_hashes(release, root=root)
    if (release['backend_profile'] != profile_binding
            or release['runtime_identity'] != runtime_binding):
        raise ValueError('Repaired runtime/profile release differs from frozen ancestry')
    profile = backend.verify_profile(root, profile_binding)
    audit_imports(root=root)
    if (profile['profile_id'] != 'accelerate_csc_v1'
            or profile['runtime_identity'] != runtime_binding
            or profile['runtime']['executable_sha256'] !=
               profile['runtime']['libraries']['install/bin/febio4']):
        raise ValueError('Reviewed same-runtime Accelerate controls differ')
    directory = io.local(release['output_directory'], root=root)
    if directory.exists() or directory.is_symlink():
        raise ValueError('One-shot output directory already exists')
    free = shutil.disk_usage(root).free
    required = (AGGREGATE['active_output_bytes_all_12'] - previous['output_bytes']
                + AGGREGATE['storage_free_reserve_bytes'])
    if free < required:
        raise ValueError('Free storage cannot cover remaining aggregate output plus reserve')
    return {'index': index, 'run_id': release['run_id'], 'deck': deck,
            'native_domain': v5.run_spec(study, prior_study, release['run_id'])['native_domain'],
            'v5_declaration_sha256': release['v5_declaration']['sha256'],
            'old_source': source_exact, 'mesh': mesh_binding,
            'adapter_receipt': adapter_receipt,
            'source_hashes': source_bound, 'observed_head_preflight': observed_head,
            'profile': profile,
            'previous': previous, 'preflight_free_bytes': free,
            'preflight_required_free_bytes': required}


def _check_aggregate(previous: dict, native: float, readout: float,
                     prep: float, output: int, calls: int) -> dict:
    values = {'aggregate_native_wall_seconds': previous['native_seconds'] + native,
              'aggregate_readout_wall_seconds': previous['readout_seconds'] + readout,
              'aggregate_prep_wall_seconds': previous['prep_seconds'] + prep,
              'aggregate_output_bytes': previous['output_bytes'] + output,
              'aggregate_native_calls': previous['native_calls'] + calls}
    if (values['aggregate_native_wall_seconds'] > AGGREGATE['native_wall_seconds_all_12']
            or values['aggregate_readout_wall_seconds'] > AGGREGATE['remaining_readout_wall_seconds']
            or values['aggregate_prep_wall_seconds'] > AGGREGATE['remaining_prep_wall_seconds']
            or sum(values[name] for name in ('aggregate_native_wall_seconds',
                                             'aggregate_readout_wall_seconds',
                                             'aggregate_prep_wall_seconds')) >
               AGGREGATE['known_stage_total_wall_seconds']
            or values['aggregate_output_bytes'] > AGGREGATE['active_output_bytes_all_12']
            or values['aggregate_native_calls'] > AGGREGATE['maximum_native_calls_all_12']):
        raise ValueError('Aggregate remaining HBE v5 resource cap exceeded')
    return values


def supervise_stage(stage: str, command: list[str], directory: Path,
                    receipt: dict, *, cwd: Path, environment: dict[str, str],
                    wall_cap: int, rss_cap: int, output_cap: int,
                    rss_observer=None, popen=None, sleep=time.sleep) -> dict:
    """Bound one process family; native and readout each have one stage attempt."""
    if stage not in {'native', 'readout'} or not command or wall_cap <= 0:
        raise ValueError('Declared native or readout stage required')
    if rss_observer is None:
        from scripts.febio_runtime import process_group_rss
        rss_observer = process_group_rss
        audit_imports()
    if popen is None:
        popen = subprocess.Popen
    result = {'status': 'starting', 'command': command, 'exit_code': None,
              'kill_reason': None, 'wall_cap_seconds': wall_cap,
              'sampled_process_group_rss_cap_bytes': rss_cap,
              'active_output_cap_bytes': output_cap,
              'peak_sampled_process_group_rss_bytes': 0,
              'peak_sampled_active_output_bytes': 0}
    receipt[stage + '_stage'] = result
    receipt[stage + '_calls_attempted'] = 1
    io.durable_json(directory / 'receipt.json', receipt)
    log_name = 'console.txt' if stage == 'native' else 'readout-console.txt'
    started = time.monotonic()
    last_persisted = started
    process = None
    try:
        with (directory / log_name).open('xb') as console:
            process = popen(command, cwd=cwd, env=environment, stdout=console,
                            stderr=subprocess.STDOUT, start_new_session=True)
            result['pid'] = process.pid
            io.durable_json(directory / 'receipt.json', receipt)
            while True:
                now = time.monotonic()
                elapsed = now - started
                if elapsed >= wall_cap:
                    result['kill_reason'] = 'wall_cap'
                    break
                code = process.poll()
                rss, members = rss_observer(process.pid,
                                            timeout_seconds=min(1., wall_cap-elapsed))
                active = active_bytes(directory, output_cap)
                result['peak_sampled_process_group_rss_bytes'] = max(
                    result['peak_sampled_process_group_rss_bytes'], rss)
                result['peak_sampled_active_output_bytes'] = max(
                    result['peak_sampled_active_output_bytes'], active)
                if rss > rss_cap:
                    result['kill_reason'] = 'process_group_rss_cap'
                elif active > output_cap:
                    result['kill_reason'] = 'active_output_cap'
                elif code is not None:
                    result['exit_code'] = code
                    if members:
                        result['kill_reason'] = 'descendants_outlived_stage'
                    break
                elif not members:
                    if process.poll() is None:
                        raise RuntimeError('Running process family RSS unavailable')
                    continue
                if result['kill_reason']:
                    break
                if time.monotonic() - started >= wall_cap:
                    result['kill_reason'] = 'wall_cap'
                    break
                result['elapsed_seconds'] = elapsed
                if now - last_persisted >= 5.:
                    io.durable_json(directory / 'receipt.json', receipt)
                    last_persisted = now
                sleep(.2)
    except BaseException as error:
        result['kill_reason'] = result['kill_reason'] or 'supervision_exception'
        result['supervision_error'] = {'type': type(error).__name__,
                                       'message': str(error)}
    finally:
        if process is not None:
            try:
                if result['kill_reason'] or process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                result['exit_code'] = process.wait(timeout=2)
            except BaseException as error:
                result['kill_reason'] = result['kill_reason'] or 'cleanup_exception'
                result['cleanup_error'] = {'type': type(error).__name__,
                                           'message': str(error)}
        result['elapsed_seconds'] = time.monotonic() - started
        result['status'] = ('completed_within_caps' if result['exit_code'] == 0
                            and result['kill_reason'] is None
                            and result['elapsed_seconds'] < wall_cap
                            else 'failed_or_incomplete')
        io.durable_json(directory / 'receipt.json', receipt)
    return result


def inspect_native(directory: Path, context: dict, *, root: Path = ROOT) -> dict:
    limit = caps(context['index'])['active_output_bytes']
    if {p.name for p in directory.iterdir()} != NATIVE_FILES | {'receipt.json'}:
        raise ValueError('Native output file inventory differs')
    if active_bytes(directory, limit) > limit:
        raise ValueError('Native output exceeds per-row active byte cap')
    records = {name: _regular_output_binding(directory / name, root=root,
                                              limit=limit)
               for name in sorted(NATIVE_FILES)}
    if records['specimen.feb']['sha256'] != io.sha(context['deck']):
        raise ValueError('Executed adapted deck changed')
    _check_backend(directory)
    return records


def _work_order(context: dict, receipt: dict, records: dict,
                token: str, *, root: Path = ROOT) -> dict:
    return {'schema': 'hbe-v5-supervised-readout-work-order-v1',
            'ordinal': context['index'], 'run_id': context['run_id'],
            'release_sha256': receipt['release_sha256'],
            'source_commit': receipt['source_commit'],
            'readout_token_sha256': io.sha(token.encode()),
            'bindings': {'source_deck': context['old_source'],
                         'mesh': context['mesh'],
                         'adapted_deck': {key: records['specimen.feb'][key]
                                          for key in ('path', 'sha256')},
                         'nodes': {key: records['nodes.log'][key]
                                   for key in ('path', 'sha256')},
                         'elements': {key: records['elements.log'][key]
                                      for key in ('path', 'sha256')},
                         'solver': {key: records['solver.log'][key]
                                    for key in ('path', 'sha256')}},
            'adapter_receipt': context['adapter_receipt']}


def readout_worker(work_order_path: str, *, root: Path = ROOT) -> int:
    """Replay bound saved logs in a separately supervised child, never FEBio."""
    path = io.local(work_order_path, root=root)
    if not work_order_path.startswith(OUTPUT_ROOT + '/') or path.name != 'readout-work-order.json':
        raise ValueError('Bound remaining-row work order required')
    if path.stat().st_size > 1024**2:
        raise ValueError('Readout work order too large')
    order = json.loads(path.read_bytes())
    keys = {'schema', 'ordinal', 'run_id', 'release_sha256', 'source_commit',
            'readout_token_sha256', 'bindings', 'adapter_receipt'}
    token = os.environ.pop('HBE_V5_READOUT_TOKEN', None)
    if (not isinstance(order, dict) or set(order) != keys
            or order['schema'] != 'hbe-v5-supervised-readout-work-order-v1'
            or type(order['ordinal']) is not int or order['ordinal'] not in range(1, 12)
            or order['run_id'] != ORDER[order['ordinal']]
            or path.parent != io.local(output_directory(order['ordinal']), root=root)
            or not isinstance(token, str) or len(token) != 64
            or io.sha(token.encode()) != order['readout_token_sha256']):
        raise ValueError('Separately supervised one-time readout context required')
    from scripts import mechanics_hbe_v5_stream as stream
    audit_imports(root=root)
    result = stream.read_bound_run(root, order['run_id'], order['bindings'],
                                   order['adapter_receipt'])
    audit_imports(root=root)
    if (result.get('numerical_passed') is not True
            or result.get('frame_count') != (121 if ':S120:' in order['run_id'] else 61)
            or result.get('provenance', {}).get('output_origin') != 'unverified_saved_stream'):
        raise ValueError('Complete saved native numerical stream failed')
    io.durable_json(path.parent / 'readout.json', result)
    return 0


def inspect_readout(directory: Path, context: dict, native_records: dict,
                    receipt: dict,
                    *, root: Path = ROOT) -> tuple[dict, dict]:
    """Bind child readout to the one native attempt without pretending it is independent."""
    limit = caps(context['index'])['active_output_bytes']
    if {p.name for p in directory.iterdir()} != FINAL_FILES:
        raise ValueError('Readout output file inventory differs')
    if active_bytes(directory, limit) > limit:
        raise ValueError('Readout exceeded per-row active output cap')
    records = {name: _regular_output_binding(directory / name, root=root,
                                              limit=limit)
               for name in sorted(FINAL_FILES - {'receipt.json'})}
    if any(records[name] != native_records[name] for name in NATIVE_FILES):
        raise ValueError('Native input changed during separate readout stage')
    if records['readout.json']['bytes'] > 16*1024**2:
        raise ValueError('Compact saved readout exceeds bounded JSON reader')
    if records['readout-work-order.json']['bytes'] > 1024**2:
        raise ValueError('Readout work order exceeds bounded JSON reader')
    readout = json.loads((directory / 'readout.json').read_bytes())
    expected_frames = 121 if ':S120:' in context['run_id'] else 61
    half = context['native_domain'] == 'lower_half_reconstructed'
    provenance = readout.get('provenance', {})
    order = json.loads((directory / 'readout-work-order.json').read_bytes())
    expected_order = _work_order(context, receipt, native_records, '', root=root)
    expected_order['readout_token_sha256'] = receipt.get('readout_token_sha256')
    if (not isinstance(expected_order['readout_token_sha256'], str)
            or not HEX.fullmatch(expected_order['readout_token_sha256'])
            or order != expected_order):
        raise ValueError('Readout work order differs from supervised native inputs')
    if (readout.get('schema') != 'hbe-v5-complete-stream-v1'
            or readout.get('run_id') != context['run_id']
            or readout.get('frame_count') != expected_frames
            or readout.get('steps') != expected_frames - 1
            or readout.get('representation') != ('reconstructed_full' if half
                                                else 'full_native_fixture')
            or readout.get('numerical_passed') is not True
            or readout.get('solver', {}).get('passed') is not True
            or readout.get('full_energy_work', {}).get('passed') is not True
            or (readout.get('native_energy_work', {}).get('passed') is not True
                if half else readout.get('native_energy_work') is not None)
            or provenance.get('output_origin') != 'unverified_saved_stream'
            or provenance.get('source_binding_checked') is not True
            or provenance.get('native_output_observed') is not None
            or provenance.get('generated_fixture_only') is not None
            or provenance.get('physical_validation_pass') is not None
            or provenance.get('measured_response_accessed') is not False
            or provenance.get('patient_data_accessed') is not False
            or provenance.get('adapted_deck_sha256') != io.sha(context['deck'])
            or provenance.get('v5_declaration_sha256') !=
               context['v5_declaration_sha256']
            or provenance.get('source_deck_sha256') != context['old_source']['sha256']
            or provenance.get('primitive_bindings') != order['bindings']
            or provenance.get('reconstruction_provenance') !=
               ('reflected_native_half_not_native_full' if half else 'full_native')):
        raise ValueError('Saved readout identity or numerical decision differs')
    summary = {'schema': readout['schema'], 'run_id': context['run_id'],
               'frame_count': expected_frames, 'numerical_passed': True,
               'readout_sha256': records['readout.json']['sha256'],
               'native_output_observed': True,
               'generated_fixture_only': False,
               'physical_validation_pass': None,
               'measured_response_accessed': False,
               'patient_data_accessed': False,
               'interpretation': 'One supervised specimen numerical software result only.'}
    return records, summary


def execute(release_path: Path, *, root: Path = ROOT) -> dict:
    """Spend one released native call and one bounded saved-output replay."""
    started = time.monotonic()
    raw, release_identity = io.read_release(release_path)
    release = json.loads(raw)
    context = validate_release(release, root=root)
    index = context['index']
    row_caps = caps(index)
    directory = io.local(output_directory(index), root=root)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': 'hbe-v5-remaining-one-call-receipt-v1',
               'status': 'reserved', 'ordinal': index, 'run_id': context['run_id'],
               'release_path': str(release_path.resolve()), 'release_sha256': io.sha(raw),
               'source_commit': release['source_commit'],
               'source_hashes_before': context['source_hashes'],
               'observed_head_preflight': context['observed_head_preflight'],
               'preparation_sha256': release['preparation']['sha256'],
               'v5_declaration_sha256': release['v5_declaration']['sha256'],
               'source_map_sha256': release['source_map']['sha256'],
               'old_source_deck_sha256': release['old_source_deck']['sha256'],
               'native_mesh_sha256': release['native_mesh']['sha256'],
               'adapted_deck_sha256': release['adapted_deck_sha256'],
               'runtime_identity_sha256': release['runtime_identity']['sha256'],
               'backend_profile_sha256': release['backend_profile']['sha256'],
               'prior_receipt_sha256': context['previous']['sha256'],
               'prior_native_wall_seconds': context['previous']['native_seconds'],
               'prior_readout_wall_seconds': context['previous']['readout_seconds'],
               'prior_prep_wall_seconds': context['previous']['prep_seconds'],
               'prior_active_output_bytes': context['previous']['output_bytes'],
               'caps': row_caps, 'aggregate_caps': AGGREGATE,
               'preflight_free_bytes': context['preflight_free_bytes'],
               'preflight_required_free_bytes': context['preflight_required_free_bytes'],
               'native_calls_attempted': 0, 'readout_calls_attempted': 0,
               'no_retry': True,
               'sampling_limit': 'Brief between-sample RSS/output peaks may be missed.',
               'known_timing_gap': 'Historical N8 readout and preparation wall time were not recorded.',
               'interpretation': 'Numerical HBE specimen fixture only; no measured force or patient validation.'}
    io.durable_json(directory / 'receipt.json', receipt)
    try:
        deck_path = directory / 'specimen.feb'
        with deck_path.open('xb') as stream:
            stream.write(context['deck'])
            stream.flush()
            os.fsync(stream.fileno())
        if io.file_hash(deck_path, maximum=row_caps['active_output_bytes']) != io.sha(context['deck']):
            raise ValueError('Copied adapted deck differs before native launch')
        prep_so_far = time.monotonic() - started
        if (prep_so_far >= PREP_WALL
                or context['previous']['prep_seconds'] + prep_so_far >
                   AGGREGATE['remaining_prep_wall_seconds']):
            raise ValueError('Preparation wall cap exhausted before native launch')
        native_command = [context['profile']['runtime']['executable'], '-noconfig',
                          '-no_title', '-i', 'specimen.feb', '-o', 'solver.log']
        native = supervise_stage('native', native_command, directory, receipt,
            cwd=directory, environment=io.private_environment(),
            wall_cap=row_caps['native_wall_seconds'], rss_cap=NATIVE_RSS,
            output_cap=row_caps['active_output_bytes'])
        if native['status'] != 'completed_within_caps':
            raise ValueError('One native call failed or exceeded a frozen cap')
        _check_aggregate(context['previous'], native['elapsed_seconds'], 0.,
                         time.monotonic()-started-native['elapsed_seconds'],
                         active_bytes(directory, row_caps['active_output_bytes']), 1)
        native_records = inspect_native(directory, context, root=root)
        receipt['native_output_bindings'] = native_records
        token = secrets.token_hex(32)
        order = _work_order(context, receipt, native_records, token, root=root)
        receipt['readout_token_sha256'] = order['readout_token_sha256']
        order_path = directory / 'readout-work-order.json'
        io.durable_json(order_path, order)
        worker_command = [sys.executable, '-B', '-m',
                          'scripts.mechanics_hbe_v5_remaining_one_shot',
                          '--readout-worker',
                          '--work-order', str(order_path.relative_to(root))]
        worker_env = io.private_environment()
        worker_env['HBE_V5_READOUT_TOKEN'] = token
        readout = supervise_stage('readout', worker_command, directory, receipt,
            cwd=root, environment=worker_env, wall_cap=READOUT_WALL,
            rss_cap=READOUT_RSS, output_cap=row_caps['active_output_bytes'])
        token = ''
        if readout['status'] != 'completed_within_caps':
            raise ValueError('Bounded saved-output readout failed or exceeded a cap')
        records, summary = inspect_readout(directory, context, native_records,
                                           receipt, root=root)
        receipt['output_bindings'] = records
        receipt['saved_numerical_readout'] = summary
        source_after, observed_head_after = source_hashes(
            release, root=root, require_head=False)
        if (io.read_release(release_path) != (raw, release_identity)
                or source_after != context['source_hashes']
                or validate_prior_chain(release['prior_receipts'], index, root=root,
                    expected_profile=release['backend_profile'],
                    expected_runtime=release['runtime_identity']) !=
                   context['previous']):
            raise ValueError('Release, current committed source or predecessor changed')
        receipt['source_hashes_after'] = source_after
        receipt['observed_head_after'] = observed_head_after
        for field, path in [('preparation', PREPARATION), ('v5_declaration', V5),
                            ('source_map', SOURCE_MAP),
                            ('old_source_deck', release['old_source_deck']['path']),
                            ('native_mesh', release['native_mesh']['path'])]:
            io.bound(release[field], path, root=root, maximum=128*1024**2)
        from scripts import mechanics_hbe_backend as backend
        after = backend.verify_profile(root, release['backend_profile'])
        audit_imports(root=root)
        if (after['runtime_identity'] != release['runtime_identity']
                or after['inputs'] != context['profile']['inputs']):
            raise ValueError('Repaired runtime or prior analytical controls changed')
        receipt['runtime_profile_verified_after'] = True
        if any(_regular_output_binding(directory / name, root=root,
                                       limit=row_caps['active_output_bytes']) != binding
               for name, binding in records.items()):
            raise ValueError('Native/readout output changed after post-run validation')
        receipt['status'] = 'passed_numerical_software_only'
    except BaseException as error:
        receipt['status'] = 'failed_or_incomplete'
        receipt['failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        native_seconds = receipt.get('native_stage', {}).get('elapsed_seconds', 0.)
        readout_seconds = receipt.get('readout_stage', {}).get('elapsed_seconds', 0.)
        prep_seconds = max(0., time.monotonic()-started-native_seconds-readout_seconds)
        receipt['prep_elapsed_seconds'] = prep_seconds
        try:
            if (prep_seconds >= PREP_WALL
                    or active_bytes(directory, row_caps['active_output_bytes']) >
                       row_caps['active_output_bytes']):
                raise ValueError('Preparation wall or active output cap exceeded')
            ledger = _check_aggregate(context['previous'], native_seconds,
                readout_seconds, prep_seconds,
                active_bytes(directory, row_caps['active_output_bytes']),
                receipt['native_calls_attempted'])
            receipt.update({key: value for key, value in ledger.items()
                            if key != 'aggregate_output_bytes'})
        except BaseException as error:
            receipt['status'] = 'failed_or_incomplete'
            receipt['final_resource_failure'] = {'type': type(error).__name__,
                                                  'message': str(error)}
        io.durable_json(directory / 'receipt.json', receipt)
        # A receipt cannot contain its own final hash or exact byte count.
        # The next release and independent audit charge the closed directory.
        for _ in range(2):
            try:
                closed_bytes = active_bytes(directory, row_caps['active_output_bytes'])
                if (closed_bytes <= row_caps['active_output_bytes']
                        and context['previous']['output_bytes'] + closed_bytes <=
                            AGGREGATE['active_output_bytes_all_12']):
                    break
                raise ValueError('Closed output byte cap exceeded')
            except BaseException as error:
                receipt['status'] = 'failed_or_incomplete'
                receipt['final_resource_failure'] = {'type': type(error).__name__,
                                                     'message': str(error)}
                io.durable_json(directory / 'receipt.json', receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', type=Path,
                        help='Separate reviewed release for exactly one next row')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--readout-worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--work-order', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.readout_worker:
        if args.execute or args.release or not args.work_order:
            raise ValueError('Readout worker only accepts its bounded work order')
        return readout_worker(args.work_order)
    if not args.execute or args.release is None or args.work_order is not None:
        raise ValueError('Explicit --execute and separate one-row release required')
    result = execute(args.release)
    print(json.dumps({'status': result['status'], 'run_id': result['run_id'],
                      'receipt': output_directory(result['ordinal']) + '/receipt.json'}))
    return 0 if result['status'] == 'passed_numerical_software_only' else 1


if __name__ == '__main__':
    raise SystemExit(main())
