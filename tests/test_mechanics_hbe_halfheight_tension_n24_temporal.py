"""Metadata, scalar clocks and saved geometry only; no native/response fixtures."""
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import mechanics_hbe_halfheight_temporal_common as common
from scripts import mechanics_hbe_halfheight_tension_n24_temporal as case
from scripts import mechanics_hbe_halfheight_tension_n24_temporal_experiment as runner


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('No native, phase execution or response parsing in controls')
    for module,name in ((common.subprocess,'Popen'),(runner.old,'solve'),(runner.runtime,'supervise'),
        (runner,'launch'),(runner,'worker'),(runner,'prepare'),(common,'read_frames'),
        (common.receipts.core,'generate_full_mesh'),(common.receipts,'replay_P1'),(common.receipts,'replay_N32')):
        monkeypatch.setattr(module,name,forbidden)


@pytest.fixture(scope='module')
def study():return case.declaration(ROOT,{'path':case.DECLARATION_PATH,'sha256':case.DECLARATION_SHA256})


def test_frozen_source_and_proposal(study):
    assert len(study['inherited_source_sha256'])==21
    for key,sha in study['inherited_source_sha256'].items():
        assert hashlib.sha256((ROOT/'scripts'/study['inherited_source_filenames'][key]).read_bytes()).hexdigest()==sha
    proposal=case.access.verify_binding(ROOT,study['reviewed_proposal'],read_json=True)
    assert proposal['case']==study['case'] and proposal['authorized'] is False
    changed=deepcopy(study);changed['budgets']['each_solver_seconds']=421
    with pytest.raises(ValueError,match='frozen'):case.require_study(changed)


@pytest.mark.parametrize('deadline,cap,now,expected',[(600.,420.,0.,420.),(600.,420.,250.,350.),(60.,60.,0.,60.),(55.,60.,10.,45.)])
def test_nested_remaining_time(deadline,cap,now,expected):
    assert common.remaining_seconds(deadline,cap,now=now)==expected


@pytest.mark.parametrize('deadline,cap,now,error',[(60.,60.,60.,TimeoutError),(59.,60.,60.,TimeoutError),
    (600.,math.inf,0.,ValueError),(math.nan,420.,0.,ValueError),(600.,0.,0.,ValueError)])
def test_clock_refusal(deadline,cap,now,error):
    with pytest.raises(error):common.remaining_seconds(deadline,cap,now=now)


def test_release_and_source_refusals(tmp_path,study):
    p=tmp_path/case.DECLARATION_PATH;p.parent.mkdir(parents=True);p.write_bytes((ROOT/case.DECLARATION_PATH).read_bytes())
    r={'schema':'hbe-tension-n24-temporal-release-v1','authorized':False,'phase':'solve',
       'study':{'path':case.DECLARATION_PATH,'sha256':case.DECLARATION_SHA256}}
    q=tmp_path/'release.json';q.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='Explicit separate'):
        runner.preflight(tmp_path,r['study'],{'path':'release.json','sha256':hashlib.sha256(q.read_bytes()).hexdigest()})
    with pytest.raises(ValueError,match='Exact24-module'):
        runner.source_inventory(ROOT,{'source_bindings':study['baseline']['source_bindings']},study,None)
    assert not (tmp_path/'outputs').exists()


def test_accepted_tension_metadata_retains_global_failure(study):
    # Compact receipts are parsed; large primitive files are only chunk-hashed.
    def bound(b,*,json_value=True,maximum_bytes=16*1024**2):
        return case.access.verify_binding(ROOT,b,read_json=json_value,maximum_bytes=maximum_bytes)
    runner.baseline_metadata(ROOT,study,{'runtime_identity':study['baseline']['runtime_identity']},bound)
    bad=deepcopy(study)
    bad['baseline']['run_id']='compression:N24:S60:reference'
    with pytest.raises(ValueError,match='origin differs'):
        runner.baseline_metadata(ROOT,bad,{'runtime_identity':study['baseline']['runtime_identity']},bound)


@pytest.mark.parametrize('count,fields,name',[(12439,9,'mechanics_nodes_si'),(10368,8,'mechanics_elements_si')])
def test_public_streaming_dispatch_without_response_fixture(monkeypatch,count,fields,name):
    seen={};marker=object()
    class Handle:
        def open(self,**kwargs):return io.StringIO('')
    monkeypatch.setattr(common.access,'local_path',lambda *args:Handle())
    def delegate(lines,**kwargs):seen.update(kwargs);return marker
    monkeypatch.setattr(common.outputs,'iter_data_records',delegate)
    with ExitStack() as stack:assert common.records(stack,ROOT,{'path':'unused'},count,fields,name) is marker
    assert seen['item_count']==count and seen['field_count']==fields
    assert len(seen['expected_times'])==121 and seen['maximum_bytes']==256*1024**2
    assert np.array_equal(common.TIMES[::2],common.inherited.TIMES)


@pytest.fixture(scope='module')
def saved_geometry(study):
    protocol,extraction,wrapper,full=case.geometry(ROOT,study)
    return protocol,extraction,wrapper,full,case.contents(ROOT,study,protocol,extraction)


def test_actual_geometry_and_tension_endpoint(study,saved_geometry):
    protocol,extraction,wrapper,full,(skyline,deck,loading)=saved_geometry
    assert len(full['rest_nodes_m'])==23101 and len(full['elements_hex8'])==20736
    assert len(extraction['mesh']['rest_nodes_m'])==12439 and len(extraction['mesh']['elements_hex8'])==10368
    tree=ET.fromstring(deck)
    assert tree.findtext('Control/time_steps')=='120' and tree.findtext('Control/time_stepper/max_retries')=='0'
    assert len(tree.findall('Boundary/bc'))==1780
    assert loading['load_coordinate'][-1]==pytest.approx(.15*common.H)
    old=case.access.verify_binding(ROOT,study['baseline']['primitive_bindings']['loading'],read_json=True)
    assert np.array_equal(np.asarray(loading['load_coordinate'])[::2],old['load_coordinate'])
    for bc in tree.findall('Boundary/bc'):
        if bc.attrib['node_set'].startswith('top_node_'):
            assert bc.findtext('dof')=='z' and bc.findtext('value')=='0.5'
    assert loading['mesh_sha256']==study['baseline']['primitive_bindings']['mesh']['sha256']


@pytest.mark.parametrize('path,value',[('Material/material/k','1'),('MeshDomains/SolidDomain/laugon','1'),
    ('Control/solver/dtol','1e-5'),('Control/output_stride','2'),('Boundary/bc/value','0.1')])
def test_non_temporal_deck_edits_refused(study,saved_geometry,path,value):
    deck=saved_geometry[-1][1];tree=ET.fromstring(deck);tree.find(path).text=value
    old=case.access.local_path(ROOT,study['baseline']['primitive_bindings']['deck']['path']).read_bytes()
    with pytest.raises(ValueError,match='Non-temporal'):
        common.verify_schedule_only_change(old,ET.tostring(tree))


def test_missing_odd_loading_state_and_partial_comparison_refused(study,saved_geometry):
    tree=ET.fromstring(saved_geometry[-1][1]);points=tree.find('LoadData/load_controller/points');points.remove(points[1])
    old=case.access.local_path(ROOT,study['baseline']['primitive_bindings']['deck']['path']).read_bytes()
    with pytest.raises(ValueError,match='Complete S60 and S120'):
        common.verify_schedule_only_change(old,ET.tostring(tree))
    with pytest.raises(ValueError,match='four authenticated'):case.compare({}, {},study,{})


def test_preserved_strict_trend_operator():
    assert not case.metric_pass({'actual':1.,'limit':1.,'comparison':'lt'})
    assert case.metric_pass({'actual':1.,'limit':1.})
