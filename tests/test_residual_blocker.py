"""Generated controls for one combined residual/aperture selection correction.

No acquired payloads. Synthetic sidecars are explicitly metadata-only controls;
the causal route control obtains actual native sidecars and full-tool results.
"""
from dataclasses import asdict, replace
import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from resectionlab.core import thaw_json
from resectionlab.geometry import AccessWindow, GENERIC_TOOLS
from resectionlab.native_proposals import (NominalCavityProposalConfig,
    RESIDUAL_OBSTRUCTION_RULE, _residual_aperture_test)
from resectionlab.native_resection import NativeResectionEngine
from resectionlab.paged_obstruction import PagedObstructionTask
from test_obstruction_opening_proposals import source, prepared, synthetic_sidecar

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/resectionlab').is_dir())
BASELINE = Path(os.environ.get('RESIDUAL_BLOCKER_BASELINE_DIR',
    str(ROOT/'build/select013-residual-blocker-v1/before')))


def case_with_rule(case, rule=RESIDUAL_OBSTRUCTION_RULE):
    return replace(case, proposal_config=replace(case.proposal_config,
        obstruction_page_size=8, obstruction_selection_rule=rule))


def old_module():
    path = BASELINE/'native_proposals.py'
    if not path.is_file():
        pytest.skip('Optional exact pre-residual archive unavailable; set RESIDUAL_BLOCKER_BASELINE_DIR')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == 'ec35077fc63e19dff422d2072c7fbe705f798ae46696ae5978f040db4b0e73a8'
    spec = importlib.util.spec_from_file_location('resectionlab._residual_before', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_default_records_fingerprints_and_sidecar_extension_exact():
    old = old_module()
    options = ({}, {'intermediate_opening_mm': 1.},
        {'tool_footprint_opening': True, 'max_candidates': 120},
        {'tool_footprint_opening': True, 'obstruction_opening': True, 'max_candidates': 120},
        {'tool_footprint_opening': True, 'obstruction_opening': True,
         'obstruction_page_size': 8, 'max_candidates': 120})
    for kwargs in options:
        a, b = old.NominalCavityProposalConfig(**kwargs), NominalCavityProposalConfig(**kwargs)
        assert a.to_record() == b.to_record() and a.fingerprint == b.fingerprint
        assert a.version == b.version and 'obstruction_selection_rule' not in b.to_record()
    case = case_with_rule(source(cap=120), None)
    engine, new, batch, evidence = prepared(case)
    prior = old.PreparedNominalCavityProposer(case._native_config, new._nominal,
        nominal_provenance=new._provenance, config=old.NominalCavityProposalConfig(**case.proposal_config.to_record()),
        index_affine=new._index_affine, index_frame_record=new._frame_record)
    old_base = prior.propose(engine)
    assert asdict(old_base) == asdict(batch)
    left, right = prior.append_obstruction_openings(old_base, engine, evidence), new.append_obstruction_openings(batch, engine, evidence)
    assert asdict(left) == asdict(right)
    assert left.obstruction_accounting == right.obstruction_accounting


def test_explicit_rule_round_trip_and_rejections():
    config = case_with_rule(source(cap=120)).proposal_config
    assert config.to_record()['obstruction_selection_rule'] == RESIDUAL_OBSTRUCTION_RULE
    assert NominalCavityProposalConfig(**config.to_record()).fingerprint == config.fingerprint
    assert replace(config, obstruction_selection_rule=None).fingerprint != config.fingerprint
    for bad in (True, 0, 'residual', ''):
        with pytest.raises(ValueError): replace(config, obstruction_selection_rule=bad)
    with pytest.raises(ValueError): replace(config, obstruction_page_size=None)
    with pytest.raises(ValueError): replace(config, obstruction_opening=False)


@pytest.mark.parametrize('sign,rotated', [(1, False), (-1, True)])
def test_actual_residual_blocker_cut_then_target_full_replay(sign, rotated):
    # Original suction cannot fully contain an off-axis cell. Aspirator's
    # optimistic radial envelope can, but that tool cannot fit this aperture.
    # The old filter therefore suppresses the useful centered suction attempt.
    case = source(cap=120, sign=sign, rotated=rotated)
    blocker = (4 if sign > 0 else 11, 10, 9)
    goal = (9 if sign > 0 else 6, 9, 9)
    support = np.zeros(case.observed_support.shape, bool)
    support[blocker] = support[goal] = True
    target = np.zeros(support.shape, np.float32); target[goal] = 1
    case = replace(case, observed_support=support, nominal_target=target, reference_target=target,
        access=AccessWindow(case.access.center_mm, case.access.normal_inward, 2.5))
    previous = PagedObstructionTask(case_with_rule(case, None), max_steps=3).planning_clone()
    old_pages = list(previous.planning_action_pages())
    old_rows = old_pages[0][0].candidate_inventory()['ledger']
    suppressed = [r for r in old_rows if tuple(r.get('blocker_voxel') or ()) == blocker]
    assert len(suppressed) == 2
    assert all(r['proposal_reason'] == 'ORIGINAL_AXIS_RADIAL_NONCONTAINMENT_NOT_ESTABLISHED' for r in suppressed)
    assert not any(ids for _, ids in old_pages)
    task = PagedObstructionTask(case_with_rule(case), max_steps=3)
    planner = task.planning_clone(); pages = list(planner.planning_action_pages())
    rows = [r for p, _ in pages for r in p.candidate_inventory()['ledger']
            if r['verification'] == 'NATIVE_VERIFIED']
    candidate = next(r for r in rows if r['family'] == 'obstruction_opening'
        and tuple(r['voxel']) == blocker and r['tool_id'] == 'generic_suction' and r['feasible'])
    first = candidate['action_id']
    task.step(first)
    assert task.metrics()['history'][0]['reward'] < 0
    assert task.metrics()['history'][0]['removed_indices_native'] == [list(blocker)]
    successors = list(task.planning_clone().planning_action_pages())
    successor_rows = [r for p, _ in successors for r in p.candidate_inventory()['ledger']]
    target_ray = next(r for r in successor_rows if r['verification'] == 'NATIVE_VERIFIED'
        and tuple(r['voxel']) == goal and r['tool_id'] == 'generic_suction' and r['feasible'])
    # The exposed row is emitted first; coincident nominal rows are explicit
    # geometry aliases, not separately certified actions.
    assert any(r['family'] in ('proximal_nominal', 'distal_nominal')
        and r['proposal_reason'] == 'DUPLICATE_GEOMETRY'
        and r['action_id'] == target_ray['action_id'] for r in successor_rows)
    task.step(target_ray['action_id']); task.step('STOP')
    replay = task.fresh()
    for action in (first, target_ray['action_id'], 'STOP'): replay.step(action)
    assert replay.metrics()['history'] == task.metrics()['history']
    assert task.metrics()['target_removed_mm3'] == pytest.approx(case._native_config.voxel_volume_mm3)
    from resectionlab.native_spatial_evaluation import evaluate_native_spatial_episode
    assert evaluate_native_spatial_episode(replay)['accepted']
    assert planner._steps == 0 and planner._engine.history == []


def test_aperture_pair_filter_is_conservative_and_tool_specific():
    access = AccessWindow([0, 0, 0], [1, 0, 0], 6)
    suction, aspirator = GENERIC_TOOLS
    entry, tip = [0, 4, 0], [4, 4, 0]
    assert not _residual_aperture_test(access, suction, entry, tip)['excluded']
    assert _residual_aperture_test(access, aspirator, entry, tip)['excluded']
    for offset in (4.6-1e-7, 4.6, 4.6+1e-7):
        assert not _residual_aperture_test(access, suction, [0, offset, 0], [4, offset, 0])['excluded']
    assert _residual_aperture_test(access, suction, [0, 4.61, 0], [4, 4.61, 0])['excluded']
    # No prefilter inference outside its bounded numerical scope.
    assert not _residual_aperture_test(access, suction, [0, 10001, 0], [4, 10001, 0])['excluded']
    assert not _residual_aperture_test(access, suction, entry, entry)['excluded']


def test_impossible_cells_do_not_consume_budget_all_dispositions_retained():
    impossible = [(4, 0, z) for z in range(16)]
    available = [(4, y, z) for y in range(7, 12) for z in range(7, 12)]
    cells = impossible+available
    case = case_with_rule(source(cap=120, extra=cells))
    engine = NativeResectionEngine(case._native_config); provider = case._nominal_proposer
    batch = provider.propose(engine)
    chunks = [cells[i:i+16] for i in range(0, len(cells), 16)]
    assert len(batch.proposals) >= len(chunks)
    evidence = [synthetic_sidecar(case, ray, chunk) for ray, chunk in zip(batch.proposals, chunks)]
    evidence += [None]*(len(batch.proposals)-len(evidence))
    extended = provider.append_obstruction_openings(batch, engine, evidence)
    rows = extended.ledger[len(batch.ledger):]
    assert len(rows) == 2*len(cells)
    assert all(r.reason == 'FULL_TOOL_APERTURE_PREFILTER' for r in rows if r.blocker_voxel in impossible)
    assert extended.obstruction_accounting['selected_cells'] == 16
    assert extended.obstruction_accounting['selection_omitted_cells'] == 9
    assert sum(r.reason == 'OBSTRUCTION_SELECTION_CAP' for r in rows) == 18
    assert len(extended.proposals)-len(batch.proposals) <= 32
    assert extended.proposals[:len(batch.proposals)] == batch.proposals
    assert not any(r.reason == 'ORIGINAL_AXIS_RADIAL_NONCONTAINMENT_NOT_ESTABLISHED' for r in rows)


def test_no_extra_discovery_preview_private_reference_effect_or_stale_reuse(monkeypatch):
    case = case_with_rule(source(cap=120))
    engine, provider, batch, evidence = prepared(case)
    def forbidden(*a, **k): raise AssertionError('Discovery must reuse paid sidecars')
    monkeypatch.setattr(NativeResectionEngine, 'preview_stroke', forbidden)
    extended = provider.append_obstruction_openings(batch, engine, evidence)
    provider.validate_batch(extended, engine, obstruction_results=evidence)
    with pytest.raises(InterruptedError):
        provider.append_obstruction_openings(batch, engine, evidence, cancelled=lambda: True)
    with pytest.raises(ValueError): provider.append_obstruction_openings(extended, engine, evidence)
    monkeypatch.undo()
    other = replace(case, reference_target=np.ones(case.reference_target.shape, np.float32))
    left = list(PagedObstructionTask(case).planning_clone().planning_action_pages())
    right = list(PagedObstructionTask(other).planning_clone().planning_action_pages())
    assert [p.candidate_inventory() for p, _ in left] == [p.candidate_inventory() for p, _ in right]
    assert [a for _, a in left] == [a for _, a in right]
