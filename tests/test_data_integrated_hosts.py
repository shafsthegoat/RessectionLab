"""Target-free baseline restores anatomy without transferring task authority."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from resectionlab.core import CaseData, SourceRef
from resectionlab.surgical_scene import from_case
from resectionlab.workspace_bundle import load_workspace, save_workspace, view_state

spec = importlib.util.spec_from_file_location(
    'integrated_hosts', Path(__file__).parents[1] / 'scripts/data_integrated_hosts.py')
hosts = importlib.util.module_from_spec(spec)
with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(Path(__file__).parents[1] / 'scripts'))
    spec.loader.exec_module(hosts)


@pytest.fixture
def parent():
    order = ('brain', hosts.TARGET, *hosts.ORDER[1:])
    masks = {name: np.zeros((4, 4, 4), dtype=bool) for name in order}
    masks['brain'][:, 0, 0] = True
    masks[hosts.TARGET][0, 0, 0] = True
    masks['estimated_skull'][1:3, 0, 0] = True
    masks[order[-2]][2:4, 0, 0] = True
    masks[order[-1]][3, 0, 0] = True
    return CaseData(case_id='fixture-synthetic', mri=np.zeros((4, 4, 4)), affine=np.eye(4),
                    compartments=masks, source_refs=(
                        SourceRef('original', 'urn:fixture:source', 'a' * 64),
                        SourceRef('synthetic-target', 'urn:fixture:target', 'b' * 64, provenance='simulated'),
                    ), metadata={
                        'surgical_adapter': {'ownershipPrecedence': order, 'reviewedTaskDomainV1': {'test': True}},
                        'integrated_rehearsal': {'stage': 'conditional'},
                        'anatomy_layers': {name: {'kind': kind} for name, kind in zip(
                            order, ('brain', 'tumor', 'skull', 'vessels', 'vessels'))},
                        'data_case': {'components': {'brain': {}, 'skull': {}, 'tumor': {}},
                                      'recipe': {'test': True}, 'family_id': 'test', 'role': 'TRAIN'}
                    })


def test_baseline_has_underlying_brain_and_retains_overlap_order(parent, tmp_path):
    candidate = hosts.make_baseline(parent, 'baseline', {})
    path = tmp_path / 'baseline.ressectionlab'
    save_workspace(candidate, path, sources=[], imaging=view_state(candidate, []), episode=None,
                   selection=None, artifacts={})
    baseline, artifacts, workspace = load_workspace(path)
    assert not artifacts and workspace['episode'] is None
    assert baseline.semantic_hash == candidate.semantic_hash
    assert set(baseline.compartments) == set(baseline.source_compartments) == set(hosts.ORDER)
    scene = from_case(baseline)
    assert [region['name'] for region in scene.regions] == list(hosts.ORDER)
    assert [region['target'] for region in scene.regions] == [False] * 4
    assert scene.labels[:, 0, 0].tolist() == [1, 2, 3, 4]
    assert scene.labels[0, 0, 1] == 0
    assert 'reviewedTaskDomainV1' not in scene.source['adapter']
    assert 'integrated_rehearsal' not in baseline.metadata
    assert 'recipe' not in baseline.metadata['data_case']
    assert tuple(ref.source_id for ref in baseline.source_refs) == ('original',)


def test_target_without_brain_support_is_refused(parent):
    masks = dict(parent.source_compartments)
    masks['brain'] = masks['brain'].copy()
    masks['brain'][masks[hosts.TARGET]] = False
    invalid = replace(parent, compartments=masks, source_compartments=masks)
    with pytest.raises(ValueError, match='fully backed by brain'):
        hosts.make_baseline(invalid, 'baseline', {})
