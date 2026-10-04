"""Independent orchestration failure controls; no solver or measured responses."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from scripts import mechanics_hbe_experiment as x


def owner_fixture(tmp_path, monkeypatch):
    source = Path(__file__).with_name('test_mechanics_hbe_experiment.py')
    spec = importlib.util.spec_from_file_location('hbe_orchestration_analytic_fixture', source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.fixture(tmp_path, monkeypatch)


@pytest.mark.parametrize('stage', ['calibration', 'held_out'])
def test_access_interruption_keeps_actual_attempt_unknown_and_never_retries(tmp_path, monkeypatch, stage):
    plan, events = owner_fixture(tmp_path, monkeypatch)
    study_type = x.access.ReleasedStudy
    method_name = 'read_calibration' if stage == 'calibration' else 'evaluate_held_out'
    original = getattr(study_type, method_name)
    calls = []

    def interrupted(self, *args, **kwargs):
        # Let the analytical double verify preexisting source/chronology first,
        # then emulate an interruption after access starts but before it returns.
        original(self, *args, **kwargs)
        calls.append(stage)
        state = json.loads((tmp_path/'experiment/state.json').read_text())
        assert state[stage+'_access_attempted'] is True
        assert state[stage+'_responses_accessed'] is None
        raise OSError('review control: response access interrupted after attempt')

    monkeypatch.setattr(study_type, method_name, interrupted)
    result = x.experiment_worker(tmp_path, plan)
    durable = json.loads((tmp_path/'experiment/state.json').read_text())
    assert result == durable
    assert result['status'] == 'failed_or_incomplete'
    assert result[stage+'_access_attempted'] is True
    assert result[stage+'_responses_accessed'] is None
    assert calls == [stage]
    assert result['solver_invocations'] == (18 if stage == 'calibration' else 20)
    if stage == 'calibration':
        assert 'freeze' not in events and 'holdout' not in events
        assert all(row['status'] == 'not_executed' for key, row in result['runs'].items() if key.endswith(':fitted'))
    else:
        assert events.index('compare20') < events.index('freeze') < events.index('holdout')
        assert (tmp_path/'experiment/parameter-prediction-freeze.json').is_file()
        assert not (tmp_path/'experiment/held-out-metrics.json').exists()


def test_late_input_mutation_cannot_preserve_success_label(tmp_path, monkeypatch):
    plan, events = owner_fixture(tmp_path, monkeypatch)
    source = tmp_path/'pinned-source';source.write_text('frozen source')
    plan['inputs'] = {str(source): x.runtime.sha(source)}
    original = x.access.ReleasedStudy.evaluate_held_out

    def change_after_access(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        source.write_text('changed after predictions and response access')
        return result

    monkeypatch.setattr(x.access.ReleasedStudy, 'evaluate_held_out', change_after_access)
    result = x.experiment_worker(tmp_path, plan)
    assert result['status'] == 'failed_or_incomplete'
    assert result['original_inputs_unchanged'] is False
    assert result['held_out_responses_accessed'] is True
    assert result['solver_invocations'] == 20
    assert 'Bound input changed' in result['integrity_error']
    assert (tmp_path/'experiment/parameter-prediction-freeze.json').is_file()
    assert (tmp_path/'experiment/held-out-metrics.json').is_file()
    assert events.count('holdout') == 1


def test_cleanup_crossing_case_deadline_is_recorded_failure_even_after_zero_exit(tmp_path, monkeypatch):
    clock = [0.]
    waits = []

    class CompletedChild:
        pid = 424242

        def wait(self, timeout):
            waits.append(timeout)
            clock[0] = .4 if len(waits) == 1 else .6
            return 0

        def poll(self):
            return 0

        def kill(self):
            pytest.fail('An exited analytical child must not be killed')

    monkeypatch.setattr(x.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(x.subprocess, 'Popen', lambda *args, **kwargs: CompletedChild())
    with pytest.raises(TimeoutError, match='Final accounted'):
        x.solve(['analytical-child-never-launched'], tmp_path, .5)
    record = json.loads((tmp_path/'solver-execution.json').read_text())
    assert record['exit_code'] == 0 and record['status'] == 'failed'
    assert record['timed_out'] and record['elapsed_seconds'] == .6
    assert record['cleanup'] == {'kill_sent': False, 'reaped': True}
    assert len(waits) == 2
