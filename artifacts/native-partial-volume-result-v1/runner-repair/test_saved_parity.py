"""Tiny scalar/JSON controls only: no engine, history replay, or clipping."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('saved_partial_parity', HERE / 'repair_saved_parity.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def frozen_functions():
    raw = R.bound_inputs()
    return R.functions_from_frozen_runner(raw['runner'], json.loads(raw['config']))


def test_original_numpy_failure_tip_false_negative_and_exact_json_repair():
    equivalent = frozen_functions()['equivalent']
    original = [0., 1.8, .9375]
    live = tuple(np.asarray(original, dtype=np.float64))
    assert all(type(v) is np.float64 for v in live)
    assert not equivalent(original, live)
    assert equivalent(original, json.loads(json.dumps(live, allow_nan=False)))
    assert R.canonical(original) == R.canonical(live)


@pytest.mark.parametrize('replacement', [[0., 1.8, .937500000001], [False, 1.8, .9375], [0., 1.8]])
def test_saved_exact_projection_does_not_accept_changed_value_type_or_length(replacement):
    assert R.canonical([0., 1.8, .9375]) != R.canonical(replacement)


def test_actual_saved_equality_and_original_false_verdict_preserved():
    result = R.calculate()
    assert result['all_18_saved_projections_exactly_equal']
    assert sum(not r['original_parity'] for r in result['rows']) == 5
    assert not result['original_criterion']['meets_predeclared_useful_criterion']
    assert result['corrected_criterion']['meets_predeclared_useful_criterion']
    assert result['new_native_calls'] == result['new_independent_geometry_replays'] == result['new_plane_clipping_calls'] == 0


def test_resealed_reward_change_remains_negative_and_other_failures_are_not_erased():
    raw = R.bound_inputs()
    current, prior, default = (json.loads(raw[key]) for key in ('current', 'prior', 'default'))
    functions = frozen_functions()
    changed = deepcopy(current)
    changed['rows'][0]['actions'][0]['frozen_reward'] += 1e-12
    changed['rows'][0]['current_legacy_projection_sha256'] = R.digest(functions['projection'](changed['rows'][0]))
    result = R.corrected_comparison(changed, prior, default, functions)
    assert not result['all_18_saved_projections_exactly_equal']
    assert not result['corrected_criterion']['meets_predeclared_useful_criterion']
    changed = deepcopy(current)
    changed['rows'][0]['partial_evacuation']['parent_target_mass_enclosure_pass'] = False
    changed['useful_criterion'] = functions['criterion'](default, prior, changed)
    result = R.corrected_comparison(changed, prior, default, functions)
    assert not result['corrected_criterion']['all_physical_bounds_respected']
    assert not result['corrected_criterion']['meets_predeclared_useful_criterion']
