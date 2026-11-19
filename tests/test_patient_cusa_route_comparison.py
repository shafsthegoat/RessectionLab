"""Generated-source checks for prospective treatment caps and a common physical root."""
from pathlib import Path

import numpy as np

from test_surgical_task_admission import prepared

def test_prospective_guard_withholds_removal_and_preserves_the_common_root(prepared, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    from compare_patient_cusa_routes import execute

    _, source, record, _ = prepared[0]
    root = record['branches']['main']
    route = {'entry': [18., 18., 2.], 'target': [18., 18., 28.], 'quaternion': [0., 0., 0., 1.]}
    clear = np.zeros(source.scene.shape, dtype=bool)
    engine, allowed = execute(source.scene, root, route, clear)
    guarded, refused = execute(source.scene, root, route, np.logical_not(clear))
    assert allowed['initialStateId'] == refused['initialStateId'] == root['stateId']
    assert allowed['metrics']['nonTargetRemovedMm3'] > 0
    assert refused['stop'] == 'prospective_action_refused'
    assert 'named protection removal' in refused['blocked'][0]['reason']
    assert not guarded.state['removed'] and engine.state['removed']
    assert root['checkpoint']['state']['removed'] == []
    assert refused['residualOriginalTargetMm3'] == refused['initialOriginalTargetMm3']
    assert refused['withdrawal'] == 'recorded'
