"""Inspect or revision-commit automatic alignment; keep a reproducible receipt.

Usage: python scripts/verify_anatomy_alignment.py PATIENT_DIR --output report.json
Add --commit to save the candidate transforms after verification.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import numpy as np
from resectionlab.patient_anatomy import PatientImagingCollection, digest
from resectionlab.anatomy_alignment import ensure_alignment, load_grid, probe_points, displacement
from resectionlab.anatomy_intracranial import intracranial_vessels
from resectionlab.anatomy_workspace import read_view, save_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('patient', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--commit', action='store_true')
    parser.add_argument('--privacy-redacted', action='store_true', help='Record user-confirmed intentional privacy removal')
    args = parser.parse_args()
    c = PatientImagingCollection(args.patient)
    before = deepcopy(c.document); old_id = c.identity(); view = read_view(c)
    if args.privacy_redacted:
        c.document['coverage_notes'] = {'privacy_redacted': True, 'source': 'user-reported', 'policy': 'Removed anatomy is not reconstructed'}
    start = time.monotonic()
    changed = ensure_alignment(c, progress=lambda _, message: print(message, flush=True))
    reports = {s: t.get('quality') for s, t in c.document['transforms'].items() if t.get('quality')}
    ct_pairs = {}
    for s, t in c.document['transforms'].items():
        parent = t.get('parent_series')
        if parent and c.series(s)['modality'] == c.series(parent)['modality'] == 'CT':
            points = probe_points(load_grid(c, c.series(s)))
            pair = {'after_p95_mm': displacement(np.array(t['matrix']), np.array(c.document['transforms'][parent]['matrix']), points)}
            if s in before['transforms'] and parent in before['transforms']:
                pair['before_p95_mm'] = displacement(np.array(before['transforms'][s]['matrix']), np.array(before['transforms'][parent]['matrix']), points)
            ct_pairs[s+' / '+parent] = pair
    filters = {}
    for layer in c.document['layers']:
        if layer['kind'] in {'vessels', 'artery', 'vein', 'sinus'}:
            _, filters[layer['id']] = intracranial_vessels(c, layer)
    assert c.document['layers'] == before['layers']
    assert c.document['series'] == before['series']
    assert not ensure_alignment(c), 'Unchanged inputs did not hit alignment cache'
    if args.commit and changed:
        c.save('automatic alignment quality migration', expected=old_id)
        if view:
            view['snapshotId'] = c.identity(); save_view(c, view)
    report = {'patient_id': c.document['patient_id'], 'committed': args.commit and changed,
              'original_snapshot': old_id, 'candidate_identity': c.identity(),
              'source_layers_preserved': True, 'source_series_preserved': True,
              'unchanged_inputs_cached': True, 'ct_reconstruction_disagreement': ct_pairs,
              'alignment': reports, 'vessel_display': filters, 'elapsed_seconds': time.monotonic()-start}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k not in {'alignment'}}, indent=2))


if __name__ == '__main__':
    main()
