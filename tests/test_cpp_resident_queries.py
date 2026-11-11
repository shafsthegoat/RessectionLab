"""Native query ownership, deformation precedence, and real resident batching."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from resectionlab import native_backend as backend
from resectionlab.geometry import GeometryScene, _capsule_voxel_indices_reference
from resectionlab.surgical_effects import _current_points
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_native_engine import create
from resectionlab.surgical_scene import SurgicalScene


CPP_CHECK = r'''
#include <cassert>
#include <iomanip>
#include <iostream>
#include "simulation_deformation_native.hpp"
using namespace resectionlab::sim;
void integers(const std::vector<std::int64_t>& values) {
    std::cout << '['; bool comma=false;
    for(auto value:values) {if(comma)std::cout<<',';comma=true;std::cout<<value;}
    std::cout << ']';
}
void points(const std::vector<Point>& values) {
    std::cout << '['; bool comma=false;
    for(const auto& p:values) {if(comma)std::cout<<',';comma=true;std::cout<<'['<<p[0]<<','<<p[1]<<','<<p[2]<<']';}
    std::cout << ']';
}
int main(int argc,char** argv) {
    assert(argc==2);const int mode=std::stoi(argv[1]);
    auto scene=std::make_shared<Scene>();scene->shape={3,4,8};scene->labels.resize(96);
    scene->regions=Value::array({Value::object({{"id",2},{"material","tumor"}}),
        Value::object({{"id",1},{"material","soft_tissue"}}),Value::object({{"id",3},{"material","bone"}})});
    for(auto cell:{4,9,11,20})scene->labels[cell]=1;
    for(auto cell:{10,12,49})scene->labels[cell]=2;
    if(mode==2)scene->labels[12]=3;
    if(mode==8)scene->labels[12]=1;
    if(mode!=5)scene->properties.set("tissuePhysicsV2",Value::object({{"enabled",true}}));
    scene->rebuild_region_index();State state(scene);
    state.state=Value::object({{"removed",Value::array({49})},{"holds",Value::object()},
        {"deformation",Value::object({{"11",Value::array({0.,0.,1.})},{"10",Value::array({0.,0.,2.})},
            {"49",Value::array({0.,0.,999.})}})}});
    if(mode==3)state.state.at("deformation").set("10",Value::array({0.,-5.,0.}));
    if(mode==1||mode==4||mode==6||mode==7) {
        state.state.set("toolMechanicsRevision",mode==6?2:1);
        state.state.at("holds").set("left",Value::object({{"cells",Value::array({4})},{"offset",Value::array({1,0,0})}}));
        state.state.set("detachedFragments",Value::array({Value::object({{"cells",Value::array({9})},{"offset",Value::array({0,1,0})}})}));
        state.state.at("deformation").set("4",Value::array({0.,0.,.25}));
    }
    if(mode==4)state.state.at("holds").set("right",Value::object({{"cells",Value::array({4})},{"offset",Value::array({0,0,0})}}));
    if(mode==7)state.state.at("removed").append(4);
    StopToken stop;
    try {
        const auto owner=occupancy(state,stop);
        std::vector<std::int64_t> all;for(int i=0;i<96;++i)all.push_back(i);
        const std::vector<std::int64_t> selected{10,10,11,4,9,49};
        const std::vector<Capsule> queries{{{0.,1.,4.},{0.,1.,4.},.1,"working"},
            {{1.,0.,4.},{1.,0.,4.},.2,"working"},{{0.,1.,4.},{1.,1.,4.},.3,"shaft"}};
        const auto batch=touched_many(state,queries,stop);
        for(std::size_t i=0;i<queries.size();++i) {
            const auto& q=queries[i];assert(batch[i]==touched(state,q.a,q.b,q.radius,stop));
        }
        assert(touched_many(state,{},stop).empty());
        std::vector<Capsule> bounded(64,queries.front());
        const auto full=touched_many(state,bounded,stop);assert(full.size()==64);
        for(const auto& cells:full)assert(cells==batch.front());
        bounded.push_back(queries.front());bool oversized=false;
        try {touched_many(state,bounded,stop);}catch(const std::invalid_argument& e){oversized=std::string(e.what())=="Native geometry batch exceeds 64 items";}assert(oversized);
        StopToken cancelled;cancelled.cancel();bool interrupted=false;
        try {touched_many(state,queries,cancelled);}catch(const Cancelled&){interrupted=true;}assert(interrupted);
        StopToken expired;expired.deadline(StopToken::Clock::now());bool deadline=false;
        try {current_points(state,selected,expired);}catch(const DeadlineExceeded&){deadline=true;}assert(deadline);
        auto detached=owner;detached.moved[0]=96;assert(owner.moved.find(0)==owner.moved.end());
        assert(scene->labels[10]==2&&state.state.at("deformation").at("10").at(2).as_number()==2.);
        std::cout<<std::setprecision(17)<<"{\"owners\":";integers(owner.labels(all,stop));
        std::cout<<",\"materials\":";integers(owner.material_ids(all,stop));
        std::cout<<",\"points\":";points(current_points(state,selected,stop));
        std::cout<<",\"cusaPoints\":";points(current_points(state,selected,stop,false));
        std::cout<<",\"touched\":[";for(std::size_t i=0;i<batch.size();++i){if(i)std::cout<<',';integers(batch[i]);}
        std::cout<<"],\"memberships\":[";bool comma=false;
        for(auto cell:all){const auto found=owner.memberships.find(cell);if(found==owner.memberships.end())continue;
            if(comma)std::cout<<',';comma=true;std::cout<<'['<<cell<<',';integers(found->second);std::cout<<']';}
        std::cout<<"]}";
    }catch(const std::exception& e){std::cout<<"{\"error\":\""<<e.what()<<"\"}";}
}
'''


@pytest.fixture(scope='module')
def native_query_check(tmp_path_factory):
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++17 compiler required for native query unit check')
    folder = tmp_path_factory.mktemp('resident-queries')
    source = folder / 'queries.cpp'
    binary = folder / 'queries'
    source.write_text(CPP_CHECK)
    include = Path(__file__).resolve().parents[1] / 'src' / 'resectionlab'
    subprocess.run([compiler, '-std=c++17', '-O1', '-ffp-contract=off',
                    '-I', str(include), str(source), '-o', str(binary)], check=True,
                   capture_output=True, text=True)
    return binary


def query_fixture(mode):
    labels = np.zeros((3,4,8), np.uint16)
    labels.ravel()[[4,9,11,20]] = 1
    labels.ravel()[[10,12,49]] = 2
    if mode == 2:
        labels.ravel()[12] = 3
    if mode == 8:
        labels.ravel()[12] = 1
    regions = [dict(id=2,material='tumor'),dict(id=1,material='soft_tissue'),dict(id=3,material='bone')]
    scene = SimpleNamespace(labels=labels,shape=labels.shape,regions=regions,
        affine=np.eye(4),inverse=np.eye(4),properties={} if mode==5 else {'tissuePhysicsV2':{'enabled':True}})
    scene.points = MethodType(SurgicalScene.points, scene)
    state = dict(removed=[49],holds={},deformation={'11':[0.,0.,1.],
        '10':[0.,0.,2.],'49':[0.,0.,999.]})
    if mode == 3:
        state['deformation']['10'] = [0.,-5.,0.]
    if mode in (1,4,6,7):
        state.update(toolMechanicsRevision=2 if mode==6 else 1,
            holds={'left':dict(cells=[4],offset=[1,0,0])},
            detachedFragments=[dict(cells=[9],offset=[0,1,0])])
        state['deformation']['4'] = [0.,0.,.25]
    if mode == 4:
        state['holds']['right'] = dict(cells=[4],offset=[0,0,0])
    if mode == 7:
        state['removed'].append(4)
    return SimpleNamespace(scene=scene,state=state,by_label={r['id']:r for r in regions},
        _removed_ids_proof=None,_occupancy_cache=None)


@pytest.mark.parametrize('mode',range(9))
def test_native_deformation_queries_match_original_functions(native_query_check,mode,monkeypatch):
    monkeypatch.setenv('RESECTIONLAB_NUMERICAL_BACKEND','reference')
    actual = json.loads(subprocess.run([str(native_query_check),str(mode)],check=True,
        capture_output=True,text=True).stdout)
    engine = query_fixture(mode)
    before = copy.deepcopy(engine.state)
    try:
        owner = SurgicalEngine.occupancy(engine)
    except ValueError as error:
        assert actual == {'error':str(error)}
        assert engine.state == before
        return
    assert 'error' not in actual, actual
    all_ids = np.arange(engine.scene.labels.size)
    assert actual['owners'] == owner[all_ids].tolist()
    assert actual['materials'] == owner.material_ids(all_ids)
    assert actual['memberships'] == [[key,owner.memberships[key]] for key in sorted(owner.memberships)]
    selected = [10,10,11,4,9,49]
    assert actual['points'] == _current_points(engine,selected).tolist()
    # CUSA retains continuous displacement after its holds-only offsets.
    points = engine.scene.points(selected)
    locations = {cell:index for index,cell in enumerate(selected)}
    for hold in engine.state['holds'].values():
        for cell in hold['cells']:
            if cell in locations:
                points[locations[cell]] += engine.scene.affine[:3,:3] @ hold['offset']
    for index,cell in enumerate(selected):
        displacement = engine.state['deformation'].get(str(cell))
        if displacement is not None:
            points[index] = engine.scene.points([cell])[0] + np.asarray(displacement)
    assert actual['cusaPoints'] == points.tolist()
    queries = [([0.,1.,4.],[0.,1.,4.],.1),([1.,0.,4.],[1.,0.,4.],.2),([0.,1.,4.],[1.,1.,4.],.3)]
    grid = GeometryScene(np.zeros(engine.scene.shape,bool),engine.scene.affine)
    expected = []
    for query in queries:
        cover = _capsule_voxel_indices_reference(grid,*query)
        expected.append(owner.material_ids(np.ravel_multi_index(cover.T,grid.shape)))
    assert actual['touched'] == expected
    assert engine.state == before


def test_complete_resident_step_submits_actual_multiitem_geometry_batch(monkeypatch):
    compiled = backend._module
    if compiled is None:
        pytest.skip('resident C++ simulator required')
    reference = SurgicalEngine(SurgicalScene(np.zeros((8,8,8),np.uint16),np.eye(4),[],{},{}))
    native, reason = create(reference)
    assert reason is None
    action = dict(slot='left',operation='inspect',duration=.1)
    with monkeypatch.context() as patch:
        patch.setattr(backend,'_module',None)
        expected = reference.step(copy.deepcopy(action))
    compiled.simulation_stats(reset=True)
    assert compiled.simulation_step(native,action,compiled.SimulationStopToken()) == expected
    assert compiled.simulation_data(native,'state') == reference.state
    stats = compiled.simulation_stats()
    assert stats['geometryBatches'] >= 1
    assert stats['geometryBatchItems'] >= 2
    assert stats['geometryBatchMaxItems'] >= 2
