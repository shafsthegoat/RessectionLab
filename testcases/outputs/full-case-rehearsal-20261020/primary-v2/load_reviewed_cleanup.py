"""Load reviewed director code without restoring or advancing the live engine."""
from datetime import datetime, timezone
import os

saved_record = load_workspace(destination)[1]['surgical_toolbox']
saved_snapshot = saved_record['branches'][branch_id]
assert saved_snapshot == engine.snapshot()
assert saved_record['fullCase'][branch_id] == director
assert director['lastStateId'] == engine.state_id
assert hashlib.sha256(original.read_bytes()).hexdigest() == source_sha
assert array_digest(scene.labels) == label_hash and array_digest(case.mri) == image_hash
before_state = engine.state_id
before_snapshot = engine.snapshot()
method_names = ('step', 'preview', '_execute', '_substep', '_physical_substep', 'cells', 'occupancy')
method_codes = {name: getattr(type(engine), name).__code__ for name in method_names}
checkpoint = destination.parent / 'checkpoints' / ('primary-before-field-cleanup-director-' + str(len(engine.actions)) + '.ressectionlab')
assert not checkpoint.exists()
os.link(destination, checkpoint)
rehearsal = importlib.reload(rehearsal)
assert before_state == engine.state_id and before_snapshot == engine.snapshot()
assert all(getattr(type(engine), name).__code__ is code for name, code in method_codes.items())
assert saved_record['fullCase'][branch_id] == director
proof = {'utc': datetime.now(timezone.utc).isoformat(), 'loadedOnly': ['full_case_rehearsal'],
    'lazyProposalHelper': 'full_case_cleanup', 'codeCommit': '3371a848',
    'actions': len(engine.actions), 'stateIdBefore': before_state, 'stateIdAfter': engine.state_id,
    'nativeBoundaryEqualsLiveSnapshot': True, 'directorUnchanged': True,
    'engineMethodCodeIdentitiesUnchanged': list(method_names), 'noPhysicsReplay': True,
    'sourceUnchanged': True, 'checkpoint': str(checkpoint),
    'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'scope': 'Director-only reload. Patient cleanup has not run; endpoint, independent canonical review and GUI acceptance remain pending.'}
(destination.parent / 'field-cleanup-director-reload-proof.json').write_text(json.dumps(proof, indent=2) + '\n')
receipt_value = receipt()
print(json.dumps({'cleanupDirectorReloadPassed': True, 'actions': len(engine.actions), 'stateId': engine.state_id,
                  'stage': director['stage'], 'targetRemovedMm3': engine.metrics()['targetRemovedMm3']}), flush=True)
