"""Alternate skull displays require exact subject review and zero new overlaps."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / 'scripts/data_integrated_skull_display.py'
with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location('data_integrated_skull_display', SCRIPT)
    skull = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skull)


def test_additional_subject_must_have_independent_review(tmp_path):
    with pytest.raises(ValueError, match='eligible'):
        skull.build(tmp_path, 'v1', subject='004', authority='fake.json')
    with pytest.raises(ValueError, match='require --authority'):
        skull.build(tmp_path, 'v1', subject='012')


def test_exact_skull_panel_is_required_and_conflicting_candidate_is_refused(tmp_path):
    def write(relative, data):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return skull.binding(tmp_path, path)

    host = {'path': 'host.ressectionlab', 'sha256': 'a' * 64}
    recipe = {'path': 'recipe.json', 'sha256': 'b' * 64}
    source = {name: {'path': name, 'sha256': 'c' * 64}
              for name in ('cta', 'supplied_vessels', 'skull_model', 'local_skull', 'brain')}
    proposal = {'sources': [source['cta'], source['supplied_vessels']],
                'inputs': {'model_skull': source['skull_model']},
                'outputs': {'observed_skull': source['local_skull'], 'brain_estimate': source['brain']}}
    panel = write('panel.png', {})
    row = {'person': 'TopCoW:012', 'role': 'TRAIN', 'source_proposal': recipe, 'source_inputs': source,
           'added_voxels': 5, 'added_conflicts_by_existing_membership': {'brain': 0, 'arterial': 0, 'venous_sinus': 0}}
    review_path = tmp_path / 'authority.json'

    def prepare_review(arterial):
        row['added_conflicts_by_existing_membership']['arterial'] = arterial
        screen = write(skull.SCREEN, {'schema': 'readonly-source-bound-CTA-skull-coverage-screen-v1',
                                      'formula': skull.SCREEN_FORMULA, 'rows': [row]})
        write(skull.PANELS, {'schema': 'CTA-ct-skull-source-comparisons-v1',
                            'numerical_screen_sha256': screen['sha256'],
                            'rows': [{'subject': '012', **panel, 'arterial_conflict_voxels': arterial,
                                      'native_planes': []}]})
        write('authority.json', {'schema': 'independent-CTA-skull-display-preparation-v1',
                                 'decision': 'accepted_for_exact_display_only_native_export', 'reviewer': 'fixture',
                                 'family_id': 'TopCoW:012', 'subject': '012', 'screen': screen,
                                 'source_panel': panel, 'core_host': host, 'added_voxels': 5})

    prepare_review(0)
    skull.reviewed_screen(tmp_path, '012', review_path, host, recipe, proposal)
    (tmp_path / panel['path']).write_text('different bytes')
    with pytest.raises(ValueError, match='checksum differs'):
        skull.reviewed_screen(tmp_path, '012', review_path, host, recipe, proposal)
    prepare_review(279)
    with pytest.raises(ValueError, match='retained-anatomy conflict'):
        skull.reviewed_screen(tmp_path, '012', review_path, host, recipe, proposal)
