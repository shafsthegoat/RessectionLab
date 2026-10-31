"""Independent adversarial review of saved planning comparisons and applications."""
import copy
from pathlib import Path

import pytest

from resectionlab.core import semantic_digest
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_interactive_planning import (
    generate_interactive_approaches, interactive_planning_availability,
    materialize_interactive_approach, validate_saved_interactive_report,
)
from resectionlab.surgical_interactive_workspace import (
    install_application, install_report, validate_workspace,
)


@pytest.fixture(scope='module')
def source():
    from resectionlab.imaging import read_case_artifacts
    from resectionlab.surgical_environment import load_environment
    path = Path(__file__).parents[1]/'testcases/cases/UCSF-PDGM-0007-operative-demo.ressectionlab'
    if not path.exists():
        pytest.skip('Ignored acquired case is unavailable; no substitute source is generated')
    engine = SurgicalEngine(load_environment(path))
    record = read_case_artifacts(path)['surgical_toolbox']
    for action in record['branches']['original-exposure']['actions']:
        assert engine.step(action)['status'] == 'accepted'
    report = generate_interactive_approaches(engine, interactive_planning_availability(engine)['defaults'])
    return engine, record, report


def reseal(value, field):
    value.pop(field, None)
    value[field] = semantic_digest(value)


def test_saved_comparison_rejects_resealed_entry_outside_requested_region(source):
    engine, _, report = source
    forged = copy.deepcopy(report)
    forged['routes'][0]['entryPose']['position'][0] += 100.
    reseal(forged, 'reportSeal')
    with pytest.raises(ValueError):
        validate_saved_interactive_report(engine.scene, engine.snapshot(), forged)


def test_saved_comparison_rejects_fabricated_clinical_probability(source):
    engine, _, report = source
    forged = copy.deepcopy(report)
    forged['routes'][0]['metrics']['clinicalRisk'] = 0.
    reseal(forged, 'reportSeal')
    with pytest.raises(ValueError):
        validate_saved_interactive_report(engine.scene, engine.snapshot(), forged)


def test_saved_application_rejects_other_valid_native_suffix(source):
    engine, original, report = source
    record = copy.deepcopy(original)
    record.update(branchId='original-exposure', position=len(engine.actions))
    item = install_report(engine, record, 'review-case', report)
    approach, receipt = materialize_interactive_approach(engine, report, report['routes'][0]['id'])
    record['branches']['review-application'] = approach.snapshot()
    install_application(record, item, 'review-application', approach, report['routes'][0]['id'], receipt)
    validate_workspace(engine.scene, record, 'review-case')
    # A canonically replayable native branch still must match the checked plan.
    approach.step({'slot': 'left', 'operation': 'inspect', 'duration': .1})
    changed = approach.snapshot()
    record['branches']['review-application'] = changed
    application = record['interactiveApproachApplications'][0]
    application['snapshotHash'] = semantic_digest(changed)
    application['receipt']['approachStateId'] = changed['stateId']
    reseal(application, 'applicationSeal')
    with pytest.raises(ValueError):
        validate_workspace(engine.scene, record, 'review-case')


def test_saved_application_rejects_forged_withdrawal_receipt(source):
    engine, original, report = source
    record = copy.deepcopy(original)
    record.update(branchId='original-exposure', position=len(engine.actions))
    item = install_report(engine, record, 'review-case', report)
    approach, receipt = materialize_interactive_approach(engine, report, report['routes'][0]['id'])
    record['branches']['review-application'] = approach.snapshot()
    install_application(record, item, 'review-application', approach, report['routes'][0]['id'], receipt)
    application = record['interactiveApproachApplications'][0]
    application['receipt']['withdrawalStateId'] = 'sha256:'+'0'*64
    reseal(application, 'applicationSeal')
    with pytest.raises(ValueError):
        validate_workspace(engine.scene, record, 'review-case')
