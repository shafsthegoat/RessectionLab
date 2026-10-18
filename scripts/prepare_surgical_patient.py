#!/usr/bin/env python3
"""Export existing four-layer patient anatomy as a research simulation derivative.

Example (source unchanged; unreviewed registration remains explicitly unreviewed):
  PYTHONPATH=src .venv/bin/python scripts/prepare_surgical_patient.py \
    /path/to/native-collection outputs/surgical-patient \
    --brain brain --tumor tumor --bone bone --vessels vessels \
    --resolution-mm 1 --allow-unreviewed-registration

Outputs patient.ressectionlab and report.json. No segmentation inference or
training runs. Original acquired files remain external with verified SourceRefs.
"""
import argparse
import json
from resectionlab.surgical_patient import prepare_surgical_patient


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('collection')
    parser.add_argument('output')
    for name in ('brain', 'tumor', 'bone', 'vessels'):
        parser.add_argument('--' + name, required=True, help='Existing native layer ID')
    parser.add_argument('--resolution-mm', type=float, default=1.)
    parser.add_argument('--allow-unreviewed-registration', action='store_true')
    args = parser.parse_args()
    report = prepare_surgical_patient(args.collection, args.output,
        layer_ids={k: getattr(args, k) for k in ('brain', 'tumor', 'bone', 'vessels')},
        resolution_mm=args.resolution_mm, allow_unreviewed_registration=args.allow_unreviewed_registration)
    print(json.dumps({k: report[k] for k in ('casePath', 'caseHash', 'shape', 'ownershipCells', 'warnings')}, indent=2))


if __name__ == '__main__':
    main()
