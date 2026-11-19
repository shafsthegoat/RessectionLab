"""Shared CT brain/skull preparation and explicit zero-exit worker recovery."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np

from resectionlab.anatomy_model_registry import ct_intensities
from resectionlab.data_root import sha256_file


def recover_ct_worker(root, collection, row, receipt):
    """Validate finished masks from a zero-exit model attempt with a leaked log pipe."""
    from resectionlab import poc_models
    from resectionlab.anatomy_model_identity import poc_worker_identity
    from resectionlab.poc_worker_supervision import MEMORY_LIMIT_BYTES

    model = 'totalsegmentator-skull'
    receipt = receipt.resolve()
    output = receipt.parent
    if (receipt.name != 'inference.json' or not output.is_relative_to(root / 'cache' / 'poc-model-results')
            or receipt.stat().st_size > 1024**2):
        raise ValueError('Choose an existing supervised model receipt on the guarded cache')
    receipt_sha = sha256_file(receipt)
    record = json.loads(receipt.read_text())
    request = json.loads((output / 'request.json').read_text())
    worker_path = output / 'worker.json'
    worker = json.loads(worker_path.read_text())
    source = collection.path(row['original']['path'])
    source_sha = sha256_file(source)
    launcher_sha = sha256_file(Path(poc_models.__file__))
    if (record.get('schema') != 'resectionlab-poc-inference-v1' or record.get('status') != 'failed'
            or record.get('failure') != 'worker_output_not_closed' or record.get('exit_code') != 0
            or row['modality'] not in {'CT', 'CTA'} or record.get('model_id') != model or request.get('model_id') != model
            or request.get('modality') != row['modality'] or request.get('image') != str(source)
            or request.get('case') != record.get('case') or request.get('device') != record.get('device')
            or request.get('output') != str(output) or record['input'].get('path') != str(source)
            or record['input'].get('modality') != row['modality']
            or source_sha != record['input'].get('sha256') or source_sha != row['original']['sha256']
            or record.get('launcher_sha256') != launcher_sha
            or record['identity'].get('runner_sha256') != poc_worker_identity(launcher_sha)
            or record.get('model') != poc_models.MODEL_CONFIG[model]
            or request.get('parameters') != poc_models.MODEL_CONFIG[model]['parameters']
            or record.get('worker') != worker or not 0 < worker.get('compute_seconds', 0) <= record['elapsed_seconds']
            or not 0 < record.get('peak_process_family_rss_bytes_sampled', 0) <= MEMORY_LIMIT_BYTES):
        raise ValueError('Model completion, source binding or runtime evidence does not match')
    weights = record['weights']
    if len(weights) != 1 or weights != request.get('weights'):
        raise ValueError('Worker must identify the same single pinned checkpoint')
    weight_path = Path(weights[0]['path']).resolve()
    if (not weight_path.is_relative_to(root / 'cache' / 'models')
            or weight_path.stat().st_size != weights[0]['bytes']
            or sha256_file(weight_path) != weights[0]['sha256']):
        raise ValueError('Retained model checkpoint differs from the worker evidence')
    records = []
    for name in ('brain_mask.nii.gz', 'skull_mask.nii.gz'):
        path = output / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f'Missing completed worker artifact {name}')
        image = nib.load(path)
        mask = np.asarray(image.dataobj)
        voxels = int(np.count_nonzero(mask))
        if (list(image.shape) != row['data']['shape'] or not np.allclose(image.affine, row['affine'], atol=1e-5)
                or not voxels or not np.isin(mask, (0, 1)).all()):
            raise ValueError(f'Retained worker artifact failed spatial or binary QC: {name}')
        records.append({'path': str(path), 'name': name, 'sha256': sha256_file(path),
                        'shape': list(image.shape), 'affine_ras_mm': image.affine.tolist(),
                        'foreground_voxels': voxels, 'provenance': 'model_inference_unreviewed'})
        del mask
    if sha256_file(receipt) != receipt_sha:
        raise ValueError('Prior worker receipt changed during validation')
    return {**record, 'outputs': records,
            'recovery': {'status': 'validated_after_log_pipe_cleanup_failure',
                         'original_receipt_sha256': receipt_sha, 'original_receipt': str(receipt.relative_to(root)),
                         'worker_completion_sha256': sha256_file(worker_path), 'new_model_runs': 0}}


def estimate_anatomy(root, collection, series_id, recovery=None):
    from resectionlab import anatomy_models

    model = 'totalsegmentator-skull'
    kinds = {row['kind'] for row in collection.document['layers']
             if row['source_series'] == series_id and row.get('provenance', {}).get('model') == model}
    if not {'brain', 'skull'} <= kinds:
        if recovery:
            row = collection.series(series_id)
            inference = recover_ct_worker(root, collection, row, recovery)
            if not ct_intensities(collection, row):
                raise ValueError('CT-derived reconstruction requires a supported intensity scale')
            for output in inference['outputs']:
                kind = output['name'].removesuffix('_mask.nii.gz')
                if kind == 'skull':
                    output = anatomy_models.restrict_ct_skull(collection, row, output)
                if not output['foreground_voxels']:
                    raise ValueError(f'The retained {kind} estimate has no supported voxels')
                record = {
                    'model': model, 'kind': kind, 'source_series': series_id, 'source_modality': row['modality'],
                    'source': row['original']['sha256'], 'inference': inference,
                    'path': output['path'], 'output_sha256': output['sha256'],
                    'version': inference['model']['version'], 'inference_spacing_mm': [3, 3, 3],
                    'parameters': {**inference['model']['parameters'], 'positive_ct_bone_gate_hu': 250},
                    'method': 'image_derived_estimate', 'evidence_use': 'context_only', 'clinical_validation': False,
                    'label_ontology': {'value': 1, 'source_label': {'brain': 90, 'skull': 91}[kind],
                                       'meaning': f'Estimated {kind}'},
                    'coverage_limitation': 'Estimated anatomy in the source CT-family field. Coverage and sensitivity remain unassessed.',
                }
                anatomy_models.import_proposal(collection, series_id, record)
        else:
            record = anatomy_models.reconstruct(collection, series_id, model)
            anatomy_models.import_proposal(collection, series_id, record)
        collection.save('Retain source-bound CT skull and brain estimates')


def inspect_surfaces(collection, series_id):
    from resectionlab.anatomy_geometry import mesh

    summary = []
    for row in collection.document['layers']:
        if row['source_series'] != series_id or row['kind'] not in {'brain', 'skull'}:
            continue
        mask = collection.mask(row)
        geometry = mesh(collection, row)
        counts = int(np.count_nonzero(mask))
        summary.append({
            'id': row['id'], 'kind': row['kind'], 'voxels': counts,
            'volume_mm3': counts * abs(float(np.linalg.det(np.asarray(row['affine'])[:3, :3]))),
            'vertices': len(geometry['positions']), 'triangles': len(geometry['indices']),
            'max_display_displacement_mm': float(geometry['max_displacement_mm']),
            'surface_policy': str(geometry['policy_version']),
        })
    return summary
