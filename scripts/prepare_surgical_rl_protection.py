#!/usr/bin/env python3
"""Read existing native estimates into source-bound research exclusion masks."""
import argparse
import json
from resectionlab.imaging import load_case
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_rl_protection import annotation_from_collection, ProtectionSet, save_protection

# Explicit research policy mapping from existing SynthSeg layer IDs, not names.
LAYERS = {'brainstem': 'synthseg-16', 'left_thalamus': 'synthseg-10', 'right_thalamus': 'synthseg-49',
          'left_caudate': 'synthseg-11', 'right_caudate': 'synthseg-50',
          'left_putamen': 'synthseg-12', 'right_putamen': 'synthseg-51',
          'left_pallidum': 'synthseg-13', 'right_pallidum': 'synthseg-52'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--clearance-mm', type=float, default=2.)
    parser.add_argument('--allow-unreviewed', action='store_true')
    parser.add_argument('--include-hippocampus', action='store_true', help='Additional explicit conservative research policy')
    args = parser.parse_args()
    scene = from_case(load_case(args.case))
    layers = dict(LAYERS)
    if args.include_hippocampus:
        layers.update(left_hippocampus='synthseg-17', right_hippocampus='synthseg-53')
    annotations = [annotation_from_collection(scene, args.collection, layer, structure,
                    allow_unreviewed=args.allow_unreviewed) for structure, layer in layers.items()]
    protection = ProtectionSet(scene, annotations, clearance_mm=args.clearance_mm, required=tuple(layers), strict=True)
    save_protection(protection, args.output)
    print(json.dumps(protection.report(), indent=2))


if __name__ == '__main__':
    main()
