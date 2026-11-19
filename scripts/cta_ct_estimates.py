"""Shared CT brain/skull preparation and explicit zero-exit worker recovery."""

import numpy as np

from resectionlab.anatomy_model_registry import ct_intensities


def recover_ct_worker(root, collection, row, receipt):
    """Keep the explicit recovery command as a shortcut to shared validation."""
    from resectionlab.poc_ct_recovery import recover_ct_worker as validate
    from resectionlab.poc_models import MODEL_CONFIG

    source = collection.path(row['original']['path'])
    result = validate(receipt, cache_root=root / 'cache' / 'poc-model-results',
                      image=source, source_hash=row['original']['sha256'], modality=row['modality'],
                      model=MODEL_CONFIG['totalsegmentator-skull'], weight_root=root / 'cache' / 'models')
    if (result['input']['shape'] != row['data']['shape']
            or not np.allclose(result['input']['affine_ras_mm'], row['affine'], atol=1e-5, rtol=0)):
        raise ValueError('Completed worker outputs differ from the study source grid')
    result['recovery']['original_receipt'] = str(receipt.resolve().relative_to(root))
    return result


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
