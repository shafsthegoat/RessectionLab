"""Fixed-source domain comparisons retain refusals and distinct integration scopes."""
import copy
from pathlib import Path
import runpy

import pytest

api = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/qualify_continuum_domain.py'))
compare = api['compare_domains']


def experiment():
    p = {'branches':['short','long'], 'patchRadiiMm':[4.,8.,12.], 'probeActions':17, 'stability':{
        'lastTwoRadiiMm':[8.,12.], 'maximumCenterDisplacementRmsDifferenceMm':.005,
        'maximumReactionRelativeDifference':.05, 'maximumFixedProbeStrainSquaredRelativeDifference':.05,
        'maximumFixedProbeStressSquaredRelativeDifference':.05}}
    rows = [{'branchId':branch,'patchRadiusMm':radius,'status':'completed',
        'acceptedActions':17,'exactCanonicalReplay':True,
        'fixedProbeCellsHash':'fixed source cells','removal':{'sourceIdsSha256':'same removal'},
        'probeActionHash':branch,'probeCenterDisplacementMm':[[.1,0.,0.],[.02,0.,0.]],
        'reactionNormN':.001,'fixedProbeStrainSquaredMm3':.1,'fixedProbeStressSquaredPa2Mm3':10.,
        'wholePatchCumulativeExposureBySourceLabel':{'arbitrary':radius}}
        for branch in p['branches'] for radius in p['patchRadiiMm']]
    return rows,p


def test_equal_fields_are_radius_stable_without_comparing_different_cumulative_domains():
    rows,p=experiment(); result=compare(rows,p)
    assert all(r['radiusStable'] for r in result)
    assert all(all(v==0 for v in r['differences'].values()) for r in result)


@pytest.mark.parametrize('index', [0, 1, 2])
def test_refused_domain_is_not_estimable_and_never_zero_filled(index):
    rows,p=experiment(); rows[index]={'branchId':'short','patchRadiusMm':rows[index]['patchRadiusMm'],'status':'refused'}
    result=compare(rows,p)
    assert result[0]['status']=='not_estimable' and not result[0]['radiusStable']
    assert 'differences' not in result[0] and result[1]['radiusStable']


@pytest.mark.parametrize('change', [lambda rows:rows.pop(), lambda rows:rows.append(copy.deepcopy(rows[0]))])
def test_missing_or_duplicated_domains_cannot_shrink_the_experiment(change):
    rows,p=experiment(); change(rows)
    with pytest.raises(ValueError,match='Every declared'):compare(rows,p)


@pytest.mark.parametrize('field', ['fixedProbeCellsHash','probeActionHash'])
def test_changed_probe_support_or_history_refuses_comparison(field):
    rows,p=experiment(); rows[2][field]='changed'
    with pytest.raises(ValueError,match='fixed source'):compare(rows,p)


def test_vector_field_difference_uses_euclidean_mm_not_per_component_rms():
    rows,p=experiment(); rows[2]['probeCenterDisplacementMm']=[[.106,0.,0.],[.026,0.,0.]]
    r=compare(rows,p)[0]
    assert r['differences']['centerDisplacementRmsDifferenceMm']==pytest.approx(.006)
    assert not r['radiusStable'] and not r['checks']['centerDisplacementRmsDifferenceMm']


def test_relative_changes_have_a_symmetric_denominator_and_keep_zero():
    f=api['relative_difference']
    assert f(0.,0.)==0 and f(1.,0.)==1 and f(1.,2.)==f(2.,1.)==.5
    for a,b in [(True,1.),(-1.,2.),(float('nan'),1.)]:
        with pytest.raises(ValueError): f(a,b)


def test_missing_or_nonfinite_probe_components_refuse_scoring():
    rows,p=experiment(); rows[2]['probeCenterDisplacementMm'][0][0]=float('inf')
    with pytest.raises(ValueError,match='finite source-center'):compare(rows,p)


def test_short_or_unverified_completed_probe_cannot_pass_stability():
    rows,p=experiment(); rows[0]['acceptedActions']=16
    with pytest.raises(ValueError,match='canonically replayed'):compare(rows,p)
    rows,p=experiment(); rows[0]['exactCanonicalReplay']=False
    with pytest.raises(ValueError,match='canonically replayed'):compare(rows,p)
