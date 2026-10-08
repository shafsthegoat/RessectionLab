"""Pure controls and hash-pinned saved-geometry/deck checks; no response fixtures."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import mechanics_hbe_halfheight_global_n36_temporal as core
from scripts import mechanics_hbe_halfheight_global_n36_temporal_readout as reader
from scripts import mechanics_hbe_halfheight_global_n36_temporal_experiment as runner


@pytest.fixture(autouse=True)
def no_native_or_response_access(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Native execution, phase execution and response parsing forbidden in controls')
    for module, name in ((runner.subprocess, 'Popen'), (runner.old, 'solve'), (runner.runtime, 'supervise'),
                         (runner, 'launch'), (runner, 'worker'), (runner, 'prepare_deck'),
                         (core.previous, 'generate_full_mesh'), (reader, '_records'),
                         (reader.spatial_reader, 'replay_prior_runs'),
                         (runner.previous_runner, 'replay_P1'), (runner.previous_runner, 'replay_N32')):
        monkeypatch.setattr(module, name, forbidden)


@pytest.fixture(scope='module')
def study():
    return core.declaration(ROOT, {'path': core.DECLARATION_PATH, 'sha256': core.DECLARATION_SHA256})


def test_immutable_twenty_module_closure_and_review(study):
    assert len(study['inherited_source_sha256']) == 20
    for key, binding in study['baseline']['source_bindings'].items():
        assert hashlib.sha256((ROOT/'scripts'/Path(binding['path']).name).read_bytes()).hexdigest() == study['inherited_source_sha256'][key]
    review = core.access.verify_binding(ROOT, study['independent_design_review'], read_json=True)
    assert review['blocking_findings'] == []
    assert review['proposal'] == study['reviewed_proposal']
    assert study['budgets']['preparation_nested_in_aggregate'] is True
    changed = deepcopy(study); changed['budgets']['each_solver_seconds'] += 1
    with pytest.raises(ValueError, match='frozen'):
        core.require_study(changed)


@pytest.mark.parametrize('deadline,cap,now,expected', [(2500.,2100.,100.,2100.), (2500.,2100.,1000.,1500.),
                                                     (160.,60.,100.,60.), (159.,60.,100.,59.)])
def test_native_and_preparation_allocations_use_remaining_aggregate(deadline, cap, now, expected):
    assert runner.remaining_seconds(deadline, cap, now=now) == expected


@pytest.mark.parametrize('deadline,cap,now,error', [(10.,60.,10.,TimeoutError), (9.,60.,10.,TimeoutError),
    (math.nan,60.,1.,ValueError), (10.,0.,1.,ValueError), (10.,math.inf,1.,ValueError)])
def test_exhausted_or_invalid_deadlines_refuse(deadline,cap,now,error):
    with pytest.raises(error):
        runner.remaining_seconds(deadline,cap,now=now)


def test_unreleased_and_incomplete_source_fail_closed(tmp_path, study):
    target = tmp_path/core.DECLARATION_PATH
    target.parent.mkdir(parents=True)
    target.write_bytes((ROOT/core.DECLARATION_PATH).read_bytes())
    release = {'schema':'hbe-halfheight-global-n36-temporal-release-v1','authorized':False,
               'phase':'solve','study':{'path':core.DECLARATION_PATH,'sha256':core.DECLARATION_SHA256}}
    p=tmp_path/'release.json'; p.write_text(json.dumps(release))
    with pytest.raises(ValueError, match='explicit temporal'):
        runner.preflight(tmp_path,release['study'],{'path':'release.json','sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    with pytest.raises(ValueError, match='twenty-three-module'):
        runner.source_inventory(ROOT,{'source_bindings':study['baseline']['source_bindings']},study)
    assert not (tmp_path/'outputs').exists()


@pytest.mark.parametrize('kind', ['nodes','elements'])
def test_exact_streaming_parser_dispatch_without_log_fixture(monkeypatch,study,kind):
    seen={}
    marker=object()
    def delegate(lines,**kwargs):
        seen.update(kwargs);return marker
    monkeypatch.setattr(reader.outputs,'_iter_data_records',delegate)
    spec=study['parser'][kind]
    assert reader.iter_global_records(None,study=study,kind=kind,item_count=spec['items'],
        field_count=spec['fields'],record_name=spec['name'],expected_times=core.TIMES,expected_run_id=core.RUN_ID) is marker
    assert len(seen['expected_times'])==121 and seen['maximum_items']==spec['items']
    assert seen['maximum_bytes']==(768 if kind=='nodes' else 512)*1024**2
    for times,run_id in ((np.linspace(0.,1.,61),core.RUN_ID),(core.TIMES,'compression:N36:S60:reference')):
        with pytest.raises(ValueError,match='S120 parser'):
            reader.iter_global_records(None,study=study,kind=kind,item_count=spec['items'],
                field_count=spec['fields'],record_name=spec['name'],expected_times=times,expected_run_id=run_id)
    with pytest.raises(ValueError,match='item/field'):
        reader.iter_global_records(None,study=study,kind=kind,item_count=spec['items']+1,
            field_count=spec['fields'],record_name=spec['name'],expected_times=core.TIMES,expected_run_id=core.RUN_ID)
    assert reader.primitive_limit(kind)==seen['maximum_bytes']


def test_exact_common_grid_and_signed_direction(study):
    assert np.array_equal(core.TIMES[::2],reader.inherited.TIMES)
    margin=study['classification_fragility']['observed_S60_endpoint_margin']
    # Scalar arithmetic about a predeclared sensitivity interval, not response data.
    positive=reader.signed_margin_report(1e-6,margin)
    negative=reader.signed_margin_report(-1e-6,margin)
    assert positive['inside_fixed_other_state_scalar_interval'] is True
    assert negative['inside_fixed_other_state_scalar_interval'] is False
    assert positive['absolute_shift_over_minimum_margin']>1
    assert positive['force_error_bound'] is positive['used_as_acceptance_gate'] is False
    with pytest.raises(ValueError):reader.signed_margin_report(math.nan,margin)


@pytest.fixture(scope='module')
def actual_geometry_and_decks(study):
    # Only immutable accepted geometry and XML; no primitive response logs/readouts.
    protocol, extraction, wrapper=core.baseline_geometry(ROOT,study)
    skyline,deck,loading=core.case_contents(ROOT,study,protocol,extraction)
    return protocol,extraction,wrapper,skyline,deck,loading


def test_actual_saved_geometry_s120_deck_preserves_physics(study,actual_geometry_and_decks):
    protocol,extraction,wrapper,skyline,deck,loading=actual_geometry_and_decks
    assert wrapper['verification']==extraction['verification']
    assert extraction['verification']['half_nodes']==39610
    assert extraction['verification']['half_elements']==34992
    baseline_loading=core.access.verify_binding(ROOT,study['baseline']['primitive_bindings']['loading'],read_json=True)
    assert np.array_equal(np.asarray(loading['load_coordinate'])[::2],baseline_loading['load_coordinate'])
    assert loading['steps']==120 and loading['mu_Pa']==1000.
    assert loading['mesh_sha256']==study['baseline']['primitive_bindings']['mesh']['sha256']
    tree=ET.fromstring(deck)
    assert tree.findtext('Control/time_steps')=='120'
    assert tree.findtext('Control/time_stepper/max_retries')=='0'
    assert tree.findtext('Control/output_stride')=='1'
    assert len(tree.findall("Boundary/bc[@type='prescribed displacement']"))==3964
    assert not any(b.findtext('dof') in ('x','y') for b in tree.findall('Boundary/bc') if b.attrib['node_set'].startswith('top_node_'))


@pytest.mark.parametrize('path,value', [('Material/material/k','1000'),('MeshDomains/SolidDomain/laugon','1'),
    ('Control/solver/dtol','1e-5'),('Control/output_stride','2'),('Boundary/bc/value','0.25')])
def test_deck_control_rejects_non_temporal_edits(study,actual_geometry_and_decks,path,value):
    _,_,_,_,deck,_=actual_geometry_and_decks
    old=core.access.local_path(ROOT,study['baseline']['primitive_bindings']['deck']['path']).read_bytes()
    changed=ET.fromstring(deck);changed.find(path).text=value
    with pytest.raises(ValueError,match='Non-temporal'):
        core.verify_schedule_only_change(old,ET.tostring(changed))


def test_missing_odd_load_state_refused(study,actual_geometry_and_decks):
    _,_,_,_,deck,_=actual_geometry_and_decks
    old=core.access.local_path(ROOT,study['baseline']['primitive_bindings']['deck']['path']).read_bytes()
    changed=ET.fromstring(deck);points=changed.find('LoadData/load_controller/points');points.remove(points[1])
    with pytest.raises(ValueError,match='Complete S60 and S120'):
        core.verify_schedule_only_change(old,ET.tostring(changed))


def test_incomplete_comparison_refused(study):
    with pytest.raises(ValueError,match='six authenticated'):
        reader.comparison_report({}, {}, study, {})
