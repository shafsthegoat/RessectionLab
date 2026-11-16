"""Function-level ranking controls, separate from physical demonstrations."""
import copy

from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_consequences import consequence_vector, compare_samples
from resectionlab.surgical_procedure_policy import procedure_comparison_policy
from test_surgical_procedure_policy import source


def comparison(costs, *, neural_limit=1., bone_limit=200.):
    scene = source(neural_limit=neural_limit)
    baseline = SurgicalEngine(scene)
    vector = consequence_vector(baseline, baseline)
    policy = procedure_comparison_policy(scene)
    for constraint in policy['constraints']:
        if constraint['metric'] == 'boneRemovedMm3':
            constraint['value'] = bone_limit
    rows = []
    for branch, target, neural, bone in [('STOP', 0., 0., 0.), *costs]:
        sample = copy.deepcopy(vector)
        for name, value in [('targetRemovedMm3', target), ('neuralRemovedMm3', neural), ('boneRemovedMm3', bone)]:
            sample['dimensions'][name]['cumulative'] = sample['dimensions'][name]['incremental'] = value
        rows.append(dict(branchId=branch, scenarioId='nominal', vector=sample))
    return compare_samples(rows, policy, stop_branch='STOP', scenario_ids=['nominal'],
        branch_ids=[row['branchId'] for row in rows], selection_rule='max_treatment_within_harm_bounds')


def test_100_versus_101_treatment_changes_selection_only_inside_declared_neural_budget():
    costs = [('less-treatment', 100., 0., 1.), ('more-treatment', 101., 1., 0.)]
    tight = comparison(costs, neural_limit=0.)
    assert tight['selectedBranchId'] == 'less-treatment'
    assert tight['violations']['more-treatment'][0]['metric'] == 'neuralRemovedMm3'
    relaxed = comparison(costs, neural_limit=1.)
    assert relaxed['selectedBranchId'] == 'more-treatment'
    assert set(relaxed['pairedScenarioParetoFront']) == {'less-treatment', 'more-treatment'}
    assert relaxed['reason'] == 'max_treatment_within_harm_bounds'


def test_bone_savings_cannot_compensate_for_equal_treatment_neural_removal():
    result = comparison([('bone-expensive', 100., 0., 101.), ('neural-expensive', 100., 1., 0.)])
    assert result['selectedBranchId'] == 'bone-expensive'
    assert set(result['pairedScenarioParetoFront']) == {'bone-expensive', 'neural-expensive'}


def test_no_progress_is_not_a_successful_continuation_and_stop_retains_residual():
    result = comparison([('passive-observation', 0., 0., 0.)])
    assert result['selectedBranchId'] == 'STOP'
    assert result['stopExecutable'] and not result['taskSatisfied']
    assert result['violations']['passive-observation'][0]['reason'] == 'no_additional_target_treatment'


def test_versioned_comparison_remeasures_structure_counts_when_saved_report_is_resealed():
    import pytest
    from resectionlab.core import semantic_digest
    from resectionlab.surgical_consequences import seal_report, validate_saved_comparison
    from test_surgical_procedure_policy import prepared, action
    scene = source(neural_limit=125.)
    baseline = SurgicalEngine(scene)
    candidate = prepared(scene)
    assert candidate.step(action('apply', [55., 40., 100.], duration=1.))['status'] == 'accepted'
    assert candidate.step(action('withdraw', [55., 40., 65.], duration=1.))['status'] == 'accepted'
    samples = [dict(branchId=branch, scenarioId='nominal', vector=consequence_vector(engine, baseline))
        for branch, engine in [('candidate', candidate), ('STOP', baseline)]]
    policy = procedure_comparison_policy(scene)
    declaration = dict(sourceArchiveSha256='0'*64, sourceSceneId=scene.identity,
        sourceGeometryHash=samples[0]['vector']['sourceGeometryHash'], prefixBranch='main', prefixActions=0,
        alternativeBranches=['candidate', 'STOP'], stopBranch='STOP',
        scenarios=[dict(id='nominal', relaxationTimeScale=1.)], policies=[policy])
    scenarios = [dict(id='nominal', relaxationTimeScale=1., sourceSceneId=scene.identity,
        prefixStateId=baseline.state_id, profileHash=semantic_digest(None))]
    record = {'branches': {'main': baseline.snapshot(), 'STOP': baseline.snapshot(), 'candidate': candidate.snapshot()}}
    def report(rows):
        return seal_report(scene, rows, scenarios, [policy], baseline_branch='main', stop_branch='STOP',
            source_archive_sha256='0'*64, declaration=declaration)
    saved = report(samples)
    validate_saved_comparison(scene, record, saved, declaration=declaration)
    assert saved['decisions'][0]['selectedBranchId'] == 'candidate'
    # These are function-level corruptions, not simulated patient effects.
    forged = copy.deepcopy(samples)
    forged[0]['vector']['dimensions']['neuralRemovedMm3'].update(cumulative=1., incremental=1.)
    with pytest.raises(ValueError, match='direct structure consequence'):
        validate_saved_comparison(scene, record, report(forged), declaration=declaration)
