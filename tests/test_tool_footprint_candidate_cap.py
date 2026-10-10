"""Generated proposal-budget controls; no patient, model, search or step calls."""
from dataclasses import asdict, replace
import numpy as np
import pytest
from resectionlab.core import array_digest
from resectionlab.geometry import AccessWindow, GENERIC_TOOLS
from resectionlab.native_proposals import (DEFAULT_COLUMN_OFFSETS, NominalCavityProposalConfig,
    PreparedNominalCavityProposer)
from resectionlab.native_resection import NativeResectionConfig, NativeResectionEngine
from resectionlab.native_spatial_task import MAX_PRIMITIVES


@pytest.mark.parametrize('intermediate', [None, 1.])
@pytest.mark.parametrize('cap', [97, 118, 120])
def test_expanded_cap_requires_explicit_footprint(intermediate, cap):
    with pytest.raises(ValueError):
        NominalCavityProposalConfig(max_candidates=cap, intermediate_opening_mm=intermediate)
    config=NominalCavityProposalConfig(max_candidates=cap,
        intermediate_opening_mm=intermediate, tool_footprint_opening=True)
    assert config.max_candidates==cap
    assert config.fingerprint!=replace(config,max_candidates=96).fingerprint


@pytest.mark.parametrize('cap', [0, 121, 128, True, 120., None])
def test_expanded_cap_remains_exact_bounded_integer(cap):
    with pytest.raises(ValueError):
        NominalCavityProposalConfig(max_candidates=cap,tool_footprint_opening=True)


@pytest.mark.parametrize('footprint', [False, True])
@pytest.mark.parametrize('intermediate', [None, 1.])
def test_default_stays_96_and_explicit_default_is_identical(footprint, intermediate):
    original=NominalCavityProposalConfig(tool_footprint_opening=footprint,
        intermediate_opening_mm=intermediate)
    explicit=replace(original,max_candidates=96)
    assert original.max_candidates==96 and original.fingerprint==explicit.fingerprint
    assert asdict(original)==asdict(explicit)
    assert MAX_PRIMITIVES==96


def _batch(cap, offsets=DEFAULT_COLUMN_OFFSETS):
    # Three distinct original endpoints plus the intermediate endpoint per tool.
    # Full native geometry is deliberately not evaluated by a proposal-budget test.
    support=np.zeros((16,23,23),bool);support[2:14]=True
    target=np.zeros(support.shape,np.float32);target[6:11]=1
    affine=np.diag([.5,.5,.5,1.])
    access=AccessWindow(np.array([.75,5.5,5.5]),np.array([1.,0.,0.]),6.)
    native=NativeResectionConfig(support,np.zeros(support.shape,np.int16),affine,access,
        GENERIC_TOOLS,'generated-cap-budget','generated occupancy only')
    engine=NativeResectionEngine(native)
    provider=PreparedNominalCavityProposer(native,target,nominal_provenance={
        'source_hash':native.source_hash,'nominal_target_hash':array_digest(target),
        'source_kind':'supplied_annotation','derivation':'generated cap fixture'},
        config=NominalCavityProposalConfig(offsets_source_voxels=offsets,max_candidates=cap,
            intermediate_opening_mm=1.,tool_footprint_opening=True))
    return provider.propose(engine),engine


def test_larger_cap_retains_physical_prefix_and_all_dispositions_without_preview(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('proposal cap test called native preview')
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    old,first=_batch(96);new,second=_batch(120)
    physical=lambda row:(row.family,row.column_index,row.tool_id,row.voxel,row.entry_mm,row.tip_mm)
    assert len(old.proposals)==96 and 96<len(new.proposals)<=120
    assert [physical(r) for r in old.proposals]==[physical(r) for r in new.proposals[:96]]
    assert old.model_hash!=new.model_hash
    assert len(old.ledger)==len(new.ledger)==130
    assert sum(r.reason=='CANDIDATE_CAP' for r in new.ledger)<sum(r.reason=='CANDIDATE_CAP' for r in old.ledger)
    assert not first.history and not second.history
    assert not first._preview_records and not second._preview_records


def test_120_is_a_shared_emission_cap_not_claim_of_catalog_completeness(monkeypatch):
    def forbidden(*args,**kwargs): raise AssertionError('proposal cap test called native preview')
    monkeypatch.setattr(NativeResectionEngine,'preview_stroke',forbidden)
    offsets=tuple((x,y) for x in (-3,-1,1,3) for y in (-3,-1,1,3))
    batch,engine=_batch(120,offsets)
    assert len(batch.proposals)==120
    assert len(batch.ledger)==len(offsets)*2*5==160
    assert any(r.reason=='CANDIDATE_CAP' for r in batch.ledger)
    assert not engine._preview_records and not engine.history
