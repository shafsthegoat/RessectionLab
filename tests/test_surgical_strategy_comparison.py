from types import SimpleNamespace

import pytest

from resectionlab import surgical_strategy_comparison as comparison


@pytest.mark.parametrize('difference', ['initial', 'caseHash', 'sceneId', 'instrumentCatalogHash',
                                        'physicsHash', 'targetBinding', 'identical actions'])
def test_incompatible_recordings_are_not_ranked(monkeypatch, difference):
    shared = {name: name for name in ('patientId', 'sourceSnapshotId', 'targetLayerId', 'targetBinding',
                                      'caseHash', 'sceneId', 'instrumentCatalogHash', 'physicsHash')}

    def read_arm(collection, target_id, path, request):
        receipt = dict(shared, actionsHash=path)
        arm = {'baselineStateId': 'intact'}
        if path == 'second':
            if difference == 'initial':
                arm['baselineStateId'] = 'already operated'
            elif difference == 'identical actions':
                receipt['actionsHash'] = 'first'
            else:
                receipt[difference] = 'different'
        return arm, receipt

    monkeypatch.setattr(comparison, '_read_arm', read_arm)
    with pytest.raises(ValueError, match='same study|same recorded action'):
        comparison.compare_recordings(None, 'target', ['first', 'second'],
                                      SimpleNamespace(check=lambda: None), lambda *args: None)
