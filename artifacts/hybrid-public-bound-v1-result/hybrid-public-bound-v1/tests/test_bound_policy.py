"""Real generated native tasks; fixture truth never enters policy features."""
from dataclasses import replace
import numpy as np
import pytest
from hybrid_proposal import GeneratedSparseTask, RidgePriority, FEATURES, Costs
from fallback_policy import NoPositiveFallbackMixin
from bound_policy import PublicBoundFallbackMixin
from resectionlab.public_reward_bound import PublicTargetBoundIndex
from resectionlab.public_target_context import VERSION
from resectionlab.native_spatial_task import NativeSpatialCase
from resectionlab.native_proposals import NominalCavityProposalConfig
from resectionlab.geometry import AccessWindow, ToolGeometry


def fixture(target=False):
    support=np.zeros((13,13,10),bool);support[4:9,4:9,2:4]=True
    nominal=support if target else np.zeros(support.shape,bool)
    return NativeSpatialCase(np.where(support,.8,0.).astype(np.float32), support,support,np.eye(4),
        AccessWindow((6.,6.,1.5),(0.,0.,1.),4.,'generated-bound'),
        (ToolGeometry('small',1.3,.7,8.,45.,1.5),),
        track='annotation_assisted',support_source_kind='derived_from_scan',
        support_derivation='GENERATED public-bound fixture',nominal_target=nominal,
        target_source_kind='supplied_annotation',target_derivation='invented public target',
        public_target_context_variant=VERSION,public_target_domain=np.ones(support.shape,bool),
        proposal_mode='nominal_cavity_v1',proposal_config=NominalCavityProposalConfig(
            offsets_source_voxels=tuple((a,b) for a in (-1,0,1) for b in (-1,0,1)),
            max_candidates=120,intermediate_opening_mm=1.,tool_footprint_opening=True))


class GeneratedPublicTask(GeneratedSparseTask):
    # Test-only explicit generated provenance, not an acquired admission API.
    def _require_source(self,case):
        assert case.support_derivation=='GENERATED public-bound fixture'
        assert case.observed_support.shape==(13,13,10)

class Baseline(NoPositiveFallbackMixin,GeneratedPublicTask):pass
class Bounded(PublicBoundFallbackMixin,GeneratedPublicTask):pass

def task(cls,case,costs=None):
    model=RidgePriority(np.zeros(FEATURES),np.ones(FEATURES),np.zeros(FEATURES),0.)
    return cls(case,mode='ridge_top8',ranker=model,max_steps=2,costs=costs)


def test_actual_native_zero_target_STOP_prunes_without_repeated_queries():
    case=fixture();old=task(Baseline,case);new=task(Bounded,case)
    a=old.candidate_inventory()['shortlist'];b=new.candidate_inventory()['shortlist']
    assert a['raw_count']==b['raw_count']>8
    assert a['queried_count']==a['raw_count'] and b['queried_count']==8
    assert b['bound_pruned_count']==a['raw_count']-8
    assert new.costs.counts['native_previews']==8
    assert old.observed_one_step_search(seconds=5)[0]==new.observed_one_step_search(seconds=5)[0]==('STOP',)
    counts=new.costs.counts.copy();new.observation();new.candidate_inventory()
    assert counts==new.costs.counts
    replay=new.fresh();replay.step('STOP')
    assert replay.terminated and replay.independent_geometry_check().feasible


def test_positive_native_winner_and_private_permutation_remain_exact():
    case=fixture(True);baseline=task(Baseline,case)
    bounded=task(Bounded,case);private=task(Bounded,replace(case,reference_target=np.zeros(case.reference_target.shape)))
    expected=baseline.observed_one_step_search(seconds=5)[0]
    assert expected[0]!='STOP'
    assert bounded.observed_one_step_search(seconds=5)[0]==expected
    assert private.observed_one_step_search(seconds=5)[0]==expected
    assert bounded.candidate_inventory()==private.candidate_inventory()
    assert bounded.observation().fingerprint==private.observation().fingerprint
    assert bounded.costs.counts==private.costs.counts


def test_cancelled_bound_keeps_paid_top8_and_never_becomes_STOP(monkeypatch):
    def cancelled(*args,**kwargs):raise TimeoutError('generated bound cap')
    monkeypatch.setattr(PublicTargetBoundIndex,'bound',cancelled)
    events=[];costs=Costs(inventory_callback=events.append)
    broken=Bounded.__new__(Bounded)
    model=RidgePriority(np.zeros(FEATURES),np.ones(FEATURES),np.zeros(FEATURES),0.)
    with pytest.raises(TimeoutError,match='generated bound cap'):
        broken.__init__(fixture(),mode='ridge_top8',ranker=model,max_steps=2,costs=costs)
    assert costs.counts['native_previews']==8 and costs.counts['public_bound_queries']==1
    assert broken._inventory is None and not events
    counts=costs.counts.copy()
    with pytest.raises(RuntimeError,match='cannot be resumed'):broken._prepare_inventory()
    assert costs.counts==counts
