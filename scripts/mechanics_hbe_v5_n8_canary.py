"""Source-only first HBE v5 N8 canary preparation; never launches FEBio.

The manifest is a frozen proposal, not an execution release. Only the declared
old compression N8 source/mesh pair is opened; the other eleven rows are not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts import mechanics_hbe_backend as backend
from scripts import mechanics_hbe_branch_calibration_v5 as v5
from scripts import mechanics_hbe_v5_source_bindings as sources


RUN_ID = 'compression:N8:S60:reference'
SOURCE_KEY = 'compression:N8'
MESH_KEY = 'full:N8'
BASELINE_COMMIT = '2f8f53f18111c2e839f677ac9c371e98f2f5175c'
MANIFEST_PATH = 'manifests/experiments/hbe-v5-n8-canary-source-only-v1.json'
MANIFEST_SHA256 = '2d74b090e9cef22a74d4daaf74309a26391f66f12bd7f872c6615c72e974f0b7'
OUTPUT_PREFIX = 'outputs/mechanics/hbe-v5-n8-canary-preparation-v1'
SOURCE_MODULES = {
    'scripts/mechanics_hbe_backend.py': 'b9ea307055ee147271a745ade05b7a65ba51070b74d48f8b85e86b1f78df97ae',
    'scripts/mechanics_hbe_branch_calibration_v5.py': '1f183287e65d008095f004815a1a2344a74f71d935cedddc7388d42d1dbdc016',
    'scripts/mechanics_hbe_v5_source_bindings.py': '1ab6ee2c46aa93304efffdc88f3e4b6d57ade0a083d0907c3294708b6d093208',
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _record(path: str, data: bytes) -> dict:
    return {'path': path, 'sha256': sha256(data)}


def expected_preparation(root: Path) -> tuple[dict, dict[str, bytes]]:
    """Derive the exact proposal from pinned source bytes; no writes or launches."""
    root = Path(root).resolve()
    study, prior = v5.validate_preparation(root)
    sources.validate_binding_manifest(root, inspect_sources=False)
    bound = json.loads(sources._read_bound(
        root, {'path': sources.BINDING_PATH, 'sha256': sources.BINDING_SHA256},
        maximum=64 * 1024))
    if bound['run_source_keys'][RUN_ID] != SOURCE_KEY:
        raise ValueError('N8 canary source mapping changed')
    source_binding = bound['source_decks'][SOURCE_KEY]
    mesh_binding = bound['meshes'][MESH_KEY]
    if source_binding['mesh'] != MESH_KEY:
        raise ValueError('N8 canary full-native mesh mapping changed')
    source = sources._read_bound(root, {'path': source_binding['path'],
                                       'sha256': source_binding['sha256']},
                                 maximum=sources.MAX_SOURCE_BYTES)
    if sha256(source) != source_binding['sha256']:
        raise ValueError('N8 source bytes differ from old-deck binding')
    mesh_raw = sources._read_bound(root, mesh_binding, maximum=sources.MAX_MESH_BYTES)
    topology = sources.validate_topology_bc(json.loads(mesh_raw), source,
                                            native_domain='full_native')
    if (topology['node_count'] != 1045 or topology['hex8_count'] != 768
            or topology['native_domain'] != 'full_native'):
        raise ValueError('N8 native topology differs from declared mesh')
    skyline, accelerate, adapter_receipt = v5.adapt_deck(study, prior, RUN_ID, source)
    schedule = v5.schedule(study, prior, RUN_ID)
    if (adapter_receipt['times'] != schedule['times']
            or adapter_receipt['full_load_coordinates_m'] != schedule['full_coordinates_m']
            or adapter_receipt['native_boundary_coordinates_m'] != schedule['native_boundary_coordinates_m']
            or adapter_receipt['source_deck_sha256'] != source_binding['sha256']
            or adapter_receipt['endpoint_float64_hex'] != '-0x1.82895cb58d960p-11'
            or adapter_receipt['native_boundary_factor'] != 1.0
            or len(schedule['times']) != 61):
        raise ValueError('Exact extended-endpoint N8 schedule differs')
    profile_binding = prior['preserved_v3_fields']['backend_profile']
    runtime_binding = prior['preserved_v3_fields']['runtime_identity']
    profile = backend.verify_profile(root, profile_binding)
    if (profile['profile_id'] != backend.PROFILE_ID
            or profile['runtime_identity'] != runtime_binding):
        raise ValueError('Repaired runtime profile differs from frozen ancestry')
    modules = {path: _record(path, sources._read_bound(
        root, {'path': path, 'sha256': digest}, maximum=1024 * 1024))
        for path, digest in SOURCE_MODULES.items()}
    decks = {
        'new_endpoint_skyline': skyline.encode('utf-8'),
        'new_endpoint_accelerate': accelerate.encode('utf-8'),
    }
    manifest = {
        'schema': 'hbe-v5-n8-canary-source-only-v1',
        'status': 'prepared_source_only_not_execution_ready',
        'specimen': 'HBE_01_03',
        'baseline_source_commit': BASELINE_COMMIT,
        'run_id': RUN_ID,
        'run': v5.run_spec(study, prior, RUN_ID),
        'v5_declaration': {'path': v5.DECLARATION_PATH, 'sha256': v5.DECLARATION_SHA256},
        'source_binding_manifest': {'path': sources.BINDING_PATH,
                                    'sha256': sources.BINDING_SHA256},
        'old_source_deck': {'path': source_binding['path'], 'sha256': source_binding['sha256']},
        'full_native_mesh': mesh_binding,
        'topology_and_fixture_bc': topology,
        'source_module_bindings': modules,
        'backend_profile': profile_binding,
        'runtime_identity': runtime_binding,
        'solver_backend': 'accelerate_csc_v1',
        'schedule': schedule,
        'deck_adapter_receipt': adapter_receipt,
        'new_endpoint_decks': {
            name: _record(f'{OUTPUT_PREFIX}/{name}.feb', data)
            for name, data in decks.items()
        },
        'phase_gates': {'native_execution': False, 'automatic_retry': False,
                        'fit': False, 'held_out_torque_access': False},
        'release': None,
        'interpretation': ('Pure source and deck preparation only. No FEBio call, '
                           'saved-frame numerical check, measured-force fit, or physical validation.'),
    }
    return manifest, decks


def verify_preparation(root: Path, *, require_deck_files: bool = True) -> dict:
    """Read-only verification; refuses manifest, source, runtime or deck drift."""
    if not MANIFEST_SHA256:
        raise ValueError('No independently pinned canary manifest SHA256')
    root = Path(root).resolve()
    raw = sources._read_bound(root, {'path': MANIFEST_PATH, 'sha256': MANIFEST_SHA256},
                              maximum=128 * 1024)
    manifest = json.loads(raw)
    expected, decks = expected_preparation(root)
    if manifest != expected:
        raise ValueError('Canary proposal differs from exact source-derived declaration')
    if require_deck_files:
        for name, binding in manifest['new_endpoint_decks'].items():
            actual = sources._read_bound(root, binding, maximum=sources.MAX_SOURCE_BYTES)
            if actual != decks[name]:
                raise ValueError('Materialized adapted deck differs from pure adapter')
    return {'status': 'source_only_verified_not_execution_ready',
            'run_id': RUN_ID, 'frames_declared': 61,
            'adapted_deck_sha256': manifest['new_endpoint_decks']['new_endpoint_accelerate']['sha256'],
            'release': None}


def materialize_decks_once(root: Path) -> dict:
    """Write only absent ignored deck files after validating the pinned proposal."""
    verification = verify_preparation(root, require_deck_files=False)
    root = Path(root).resolve()
    manifest = json.loads((root / MANIFEST_PATH).read_text())
    _, decks = expected_preparation(root)
    paths = {}
    for name, binding in manifest['new_endpoint_decks'].items():
        path = root / binding['path']
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root and parent.is_relative_to(root)):
            raise ValueError('Symlinked materialization path refused')
        if path.exists():
            if path.stat().st_size != len(decks[name]) or path.read_bytes() != decks[name]:
                raise ValueError('Existing deck differs; no overwrite permitted')
        paths[name] = path
    for name, path in paths.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(decks[name])
    verify_preparation(root)
    return verification


def require_execution_ready(*args, **kwargs):
    raise ValueError('N8 canary has no one-shot supervisor or native execution release')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--materialize-decks', action='store_true')
    args = parser.parse_args()
    result = (materialize_decks_once(args.root) if args.materialize_decks
              else verify_preparation(args.root))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
