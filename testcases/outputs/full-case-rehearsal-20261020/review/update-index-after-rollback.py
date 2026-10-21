import json
from datetime import datetime, timezone
from pathlib import Path

root = Path(__file__).resolve().parents[1]
path = root / 'episode-index.json'
index = json.loads(path.read_text())
proof = json.loads((root / 'primary-v2/exact-rollback-copy-live-reload-proof.json').read_text())
receipt = json.loads((root / 'primary-v2/execution-receipt.json').read_text())
interval = proof['firstActualPostReloadBatch']
assert receipt['actions'] == interval['endPosition']
assert receipt['stateId'] == interval['endStateId']
assert proof['sourceUnchanged'] and receipt['sourceUnchanged']
index['updatedUtc'] = datetime.now(timezone.utc).isoformat()
index['codeCommit'] = proof['codeCommit']
primary = index['episodes']['primary']
primary.update(recordedStateId=receipt['stateId'], recordedActionCount=receipt['actions'],
    directorStatus=receipt['director']['status'], recordedTargetRemovedMm3=receipt['metrics']['targetRemovedMm3'],
    ownedHealthyRemovedMm3=receipt['accounting']['healthyRemovedMm3'],
    ownedUnknownRemovedMm3=receipt['accounting']['unknownRemovedMm3'],
    recordedOwnedRemovedByMaterialMm3={'soft_tissue': 0., 'tumor': receipt['metrics']['targetRemovedMm3'],
        'bone': receipt['metrics']['removedByLabelMm3']['bone'], 'vessel': 0.})
primary.pop('recordedNativeSha256', None)
primary['recordedBoundaryEvidence'] = 'primary-v2/exact-rollback-copy-live-reload-proof.json (firstActualPostReloadBatch)'
index['evidence'].update(exactRollbackCopyReload='primary-v2/exact-rollback-copy-live-reload-proof.json',
    exactRollbackCopyKernel='review/rollback-copy-installed-kernel-20261020T180136Z.json')
runtime = index['runtime']
runtime.pop('pendingRollbackCopyOptimization', None)
runtime.update(activeCommand='advance(4000), ordinary continuation after verified rollback-copy1486→1488 interval',
    pendingDirectorReload='0c4045cf',
    recordedCleanupLoader='primary-v2/load_reviewed_recorded_cleanup.py (requires actual withdrawn endpoint; source bound to0c4045cf)',
    liveRollbackCopyCommit=proof['codeCommit'], exactRollbackCopyReloadBoundary=proof['actions'],
    exactRollbackCopyFirstInterval=interval,
    boundaryPauseFlagPresent=(root / 'primary-v2/pause-after-batch').exists())
temp = path.with_suffix('.json.new')
temp.write_text(json.dumps(index, indent=2) + '\n')
temp.replace(path)
print(json.dumps({'updatedIndex': str(path), 'recordedActions': receipt['actions'],
    'recordedTargetMm3': receipt['metrics']['targetRemovedMm3'], 'status': 'in-progress'}))
