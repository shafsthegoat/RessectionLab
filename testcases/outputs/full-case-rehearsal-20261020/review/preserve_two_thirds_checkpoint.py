"""Preserve one saved execution milestone; never construct or replay an engine."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
from resectionlab.core import array_digest
from resectionlab.surgical_environment import load_environment
from resectionlab.workspace_bundle import load_workspace

base = ROOT / 'testcases/outputs/full-case-rehearsal-20261020'
source = ROOT / 'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--name', default='primary-over-two-thirds-target')
parser.add_argument('--minimum-fraction', type=float, default=2. / 3.)
args = parser.parse_args()
if (not args.name or any(char not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for char in args.name)
        or not math.isfinite(args.minimum_fraction) or not 0. <= args.minimum_fraction <= 1.):
    parser.error('Use a lowercase checkpoint name and a fraction in [0,1]')
checkpoint = base / ('primary-v2/checkpoints/' + args.name + '.ressectionlab')
if not checkpoint.exists():
    os.link(base / 'primary-v2/primary.ressectionlab', checkpoint)
case, artifacts, _ = load_workspace(checkpoint)
original_case, _, _ = load_workspace(source)
scene = load_environment(checkpoint)
original_scene = load_environment(source)
source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_sha == 'ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'
assert scene.identity == original_scene.identity
assert case.semantic_hash == original_case.semantic_hash
np.testing.assert_array_equal(scene.labels, original_scene.labels)
np.testing.assert_array_equal(scene.affine, original_scene.affine)
np.testing.assert_array_equal(case.mri, original_case.mri)
assert case.compartments.keys() == original_case.compartments.keys()
for name, mask in case.compartments.items():
    np.testing.assert_array_equal(mask, original_case.compartments[name])
record = artifacts['surgical_toolbox']
branch = record['branchId']
snapshot = record['branches'][branch]
state = snapshot['checkpoint']['state']
removed = np.asarray(state['removed'], dtype=np.int64)
assert len(removed) == len(np.unique(removed))
target_ids = [r['id'] for r in scene.regions if r.get('target')]
target = np.isin(scene.labels, target_ids)
initial = int(np.count_nonzero(target))
removed_target = int(np.count_nonzero(target.ravel()[removed]))
assert initial == 43573 and removed_target / initial >= args.minimum_fraction
target.ravel()[removed] = False
_, regions = label(target)
owned = {}
for region in scene.regions:
    material = region['material']
    owned[material] = owned.get(material, 0.) + float(np.count_nonzero(scene.labels.ravel()[removed] == region['id'])) * scene.voxel_mm3
report = {
    'utc': datetime.now(timezone.utc).isoformat(), 'checkpoint': str(checkpoint),
    'checkpointSha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    'originalSha256': source_sha, 'caseSemanticHash': case.semantic_hash,
    'sceneId': scene.identity, 'labelArrayHash': array_digest(scene.labels),
    'branchId': branch, 'actions': len(snapshot['actions']), 'stateId': snapshot['stateId'],
    'originalOwnedTargetMm3': initial * scene.voxel_mm3,
    'removedOwnedTargetMm3': removed_target * scene.voxel_mm3,
    'removedOwnedTargetPercent': 100. * removed_target / initial,
    'remainingOwnedTargetMm3': (initial - removed_target) * scene.voxel_mm3,
    'remainingSixConnectedRegions': int(regions), 'removedByOwnedMaterialMm3': owned,
    'removedRawAnnotationMembershipCells': {name: int(np.count_nonzero(mask.ravel()[removed])) for name, mask in case.compartments.items()},
    'directorStage': record['fullCase'][branch]['stage'],
    'method': 'Hardlink of atomically saved native archive; counts independently recomputed from original source ownership and recorded removed IDs. No engine construction or replay.',
    'qualification': 'Raw annotation memberships overlap. Zero owned soft_tissue removal does not establish zero raw brain-mask membership or clinical healthy-brain injury.',
    'scope': 'Execution milestone only; cleanup, final canonical acceptance and full Electron playback remain pending.',
}
checkpoint.with_suffix('.provenance.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
