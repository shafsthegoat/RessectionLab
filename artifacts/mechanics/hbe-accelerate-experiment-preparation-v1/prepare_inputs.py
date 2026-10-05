"""One fixed, source-bound preparation; no solver or measured-member reader."""
from pathlib import Path
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FROZEN = ROOT/'outputs/validation/hbe-accelerate-6486dbc/frozen-source'
sys.dont_write_bytecode = True
sys.path.insert(0, str(FROZEN))
from scripts import mechanics_hbe_backend as backend
from scripts import mechanics_hbe_access as access


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def binding(path):
    path = Path(path).resolve()
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def save(name, value):
    access.exclusive_json(OUT/name, value)
    return binding(OUT/name)


def main():
    state = {'status': 'preparing', 'solver_invocations': 0, 'measured_members_read': 0,
             'preparation_driver': binding(__file__)}
    try:
        archive = json.loads((OUT/'source-archive.json').read_text())
        for name, record in archive['members'].items():
            if sha(FROZEN/name) != record['sha256']:
                raise ValueError('Frozen source changed: '+name)
        for module in (backend, access):
            if Path(module.__file__).resolve() != FROZEN/'scripts'/Path(module.__file__).name:
                raise ValueError('Preparation module imported from wrong source')
        review = {'path': 'artifacts/mechanics-accelerate-controls-saved-review-v1/review.json',
                  'sha256': 'b410fbe9cd8cbb9be5492a77f402d44844da01042259ecfcbccc0d04fef83fba'}
        accepted = access.verify_binding(ROOT, review, read_json=True)
        if accepted['status'] != 'eight_saved_numerical_controls_independently_passed':
            raise ValueError('Saved numerical controls not independently accepted')
        attempt = ROOT/'outputs/mechanics/accelerate-controls-v1/attempt-01'
        identity = binding(ROOT/'artifacts/febio-accelerate-csc-runtime-v3/runtime-identity.json')
        if identity['sha256'] != '13c4f60cbae8ad89232996e4f7d773bafbd2489f23a669c355f5ac09e417cc57':
            raise ValueError('Actual repaired runtime identity changed')
        profile = {'schema': 'hbe-solver-backend-v1', 'profile_id': backend.PROFILE_ID,
                   'runtime_identity': identity, 'hex8_controls': binding(attempt/'hex8-summary.json'),
                   'tet10_mpc_controls': binding(attempt/'tet10_mpc-summary.json'),
                   'patch_identity': backend.PATCH_IDENTITY}
        profile_binding = save('backend-profile.json', profile)
        verified = backend.verify_profile(ROOT, profile_binding)
        save('profile-verification.json', {'status': 'accepted_actual_runtime_and_saved_eight_controls',
             'profile': profile_binding, 'independent_control_review': review,
             'inputs': verified['inputs'], 'source': binding(backend.__file__),
             'solver_invocations': 0, 'measured_members_read': 0})
        raw = ROOT/'outputs/mechanics/hbe-01-03-poc-v1'
        original_cases = {}
        for run_id, (branch, level, steps, role) in access.expected_runs().items():
            if role == 'fitted':
                continue
            directory = raw/f'mesh-preparation/N{level}/generated'
            receipt = json.loads((directory/'receipt.json').read_text())
            name = f'{branch}-{steps}-{role}'
            original_cases[run_id] = {
                'mesh': {'path': str((directory/'mesh.json').relative_to(ROOT)), 'sha256': receipt['mesh_sha256']},
                'deck': {'path': str((directory/name/'specimen.feb').relative_to(ROOT)), 'sha256': receipt['decks'][name]['deck_sha256']},
                'loading': {'path': str((directory/name/'loading.json').relative_to(ROOT)), 'sha256': receipt['decks'][name]['loading_sha256']}}
        original_index = save('original-cases.json', original_cases)
        deck_manifest = backend.prepare_decks(ROOT, original_cases, raw/'backend-inputs-accelerate-csc-v1')
        cases, sources, deck_inputs = backend.verify_prepared_decks(ROOT, deck_manifest, original_cases)
        state.update(status='profile_and_eighteen_solver_only_copies_prepared_no_execution',
                     profile=profile_binding, backend_decks=deck_manifest, original_cases=original_index,
                     independent_control_review=review, prepared_cases=len(cases),
                     preserved_original_deck_bindings=sources, prepared_inputs=deck_inputs)
        for name, record in archive['members'].items():
            if sha(FROZEN/name) != record['sha256']:
                raise ValueError('Frozen source changed after preparation')
        for path, expected in verified['inputs'].items():
            if sha(path) != expected:
                raise ValueError('Verified prerequisite changed after preparation')
        state['all_verified_inputs_unchanged'] = True
    except BaseException as error:
        state.update(status='failed_preparation_no_retry', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        save('preparation-result.json', state)


if __name__ == '__main__':
    main()
