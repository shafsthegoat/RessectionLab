"""Package the preinjured readiness control without changing its physical source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from check_surgical_rl_readiness import surrogate_scene
from resectionlab.core import CaseData, SourceRef, semantic_digest
from resectionlab.imaging import save_case
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_scene import from_case


def control():
    source = surrogate_scene(rescue=True, causal_cusa=True)
    case = CaseData(case_id='generated-readiness-preinjured-outlet',
        mri=source.labels.astype(np.float32), affine=source.affine,
        compartments={row['name']: source.labels == row['id'] for row in source.regions},
        source_refs=(SourceRef('readiness-preinjured-outlet', 'synthetic://resectionlab/readiness-preinjured-outlet',
                               native_frame='RAS+', provenance='simulated'),),
        metadata={'generated': True, 'is_synthetic': True, 'surgical_properties': source.properties,
                  'anatomy_layers': {row['name']: {'kind': row['material']} for row in source.regions},
                  'description': 'Generated readiness control. Declared preinjured source in a 27 mm³ tumor cube. No patient imaging or vascular graph.'})
    scene = from_case(case, source.regions)
    assert np.array_equal(scene.labels, source.labels) and np.array_equal(scene.affine, source.affine)
    assert semantic_digest(scene.properties) == semantic_digest(source.properties)
    snapshot = SurgicalEngine(scene).snapshot()
    return case, {'branches': {'main': snapshot}, 'branchId': 'main', 'position': 0,
                  'mapping': scene.regions, 'parents': {'main': None}}


def practice_actions():
    """Declared operator inputs for the exposed-face and bipolar-contact checks."""
    def act(operation, duration=.1, *, z=None, params=None, instrument=None):
        row = {'slot': 'left', 'operation': operation, 'duration': duration}
        if z is not None:
            row['pose'] = {'position': [9., 9., z], 'quaternion': [0., 0., 0., 1.]}
        if params:
            row['parameters'] = params
        if instrument:
            row['instrument'] = instrument
        return row
    return [
        act('inspect', 15.),
        act('exchange', instrument='cusa'),
        act('place', z=-5., params={'radiusMm': 1.5, 'power': 1., 'duty': 1.}),
        act('apply', 3., z=8.8), act('deactivate'), act('withdraw', 1., z=-5.),
        act('exchange', instrument='frazier_suction'),
        act('place', z=-5., params={'radiusMm': 2., 'flowMm3PerSecond': 10.}),
        act('apply', .5, z=8.), act('apply', 1.), act('deactivate'), act('withdraw', 1., z=-5.),
        act('exchange', instrument='bipolar_forceps'),
        act('place', z=-5., params={'radiusMm': 1.5, 'power': 1., 'depthMm': 1., 'jawOpeningMm': .5}),
        act('apply', 1., z=8.5), act('apply', .5), act('deactivate'), act('withdraw', 1., z=-5.),
        act('exchange', instrument='irrigation'),
        act('place', z=-5., params={'depthMm': 1., 'flowMm3PerSecond': 5.}),
        act('move', .5, z=6.), act('apply', 1.), act('deactivate'), act('withdraw', 1., z=-5.),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--actions', type=Path, help='Also save the corresponding desktop input sequence')
    args = parser.parse_args()
    output = args.output.absolute()
    root = Path(__file__).resolve().parents[1]/'testcases'
    for destination in (output, args.actions):
        if destination is not None and ('..' in destination.parts or not destination.absolute().is_relative_to(root) or destination.exists()):
            parser.error('Choose new output files under testcases/')
    if output.suffix != '.ressectionlab' or (args.actions and args.actions.suffix != '.json'):
        parser.error('Use .ressectionlab for the case and .json for the actions')
    case, record = control()
    save_case(case, output, artifacts={'surgical_toolbox': record})
    if args.actions:
        args.actions.write_text(json.dumps(practice_actions(), indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
