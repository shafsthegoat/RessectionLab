"""Unpassed native export must never turn a positive task label into a failure."""
import importlib.util
from pathlib import Path

import pytest


def _exporter():
    path=Path(__file__).resolve().parents[1]/'scripts/export_patient_cusa_episode.py'
    spec=importlib.util.spec_from_file_location('patient_cusa_unpassed_export',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unpassed_export_refuses_success_or_task_gate_passing_labels():
    exporter=_exporter()
    for episode in (
        {'status':'success','taskGateMet':True},
        {'status':'success','taskGateMet':False},
        {'status':'abstained','taskGateMet':True},
    ):
        with pytest.raises(ValueError,match='refuses successful'):
            exporter.require_unpassed(episode)


def test_unpassed_export_names_both_failed_stroke_and_target_gates():
    exporter=_exporter()
    episode={'status':'abstained','taskGateMet':False,
        'deltaTargetMm3':10.,'admittedNonStopActions':1,
        'cleanupComplete':True,'stopReason':'policy_stop'}
    assert exporter.failure_reasons(episode)==[
        'target_progress_below_declared_30_mm3_goal',
        'fewer_than_two_admitted_non_stop_actions']
