"""Independent mesh/deck controls; never import Gmsh or launch a solver."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('hbe_mesh_independent_target', ROOT/'scripts/mechanics_hbe_mesh.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def two_hexes(*, same_side=False):
    nodes = np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],
                      [-1,-1,2],[1,-1,2],[1,1,2],[-1,1,2]], dtype=float)
    cells = np.array([[1,2,3,4,5,6,7,8],
                      [1,2,3,4,9,10,11,12] if same_side else [5,6,7,8,9,10,11,12]])
    bottom = [[5,6,7,8]] if same_side else [[1,2,3,4]]
    sides = [cell[face].tolist() for cell in cells for face in m.HEX_FACES[2:]]
    boundary = {name: {'faces_quad4':faces,'node_ids':sorted({i for face in faces for i in face})}
                for name, faces in [('bottom',bottom),('top',[[9,10,11,12]]),('side',sides)]}
    return nodes, cells, boundary


def test_interior_faces_require_opposite_cyclic_orientation_not_just_same_ids():
    X, cells, boundary = two_hexes()
    assert np.all(m.rest_jacobians(X,cells)[0]>0)
    valid = m.boundary_topology(cells,boundary)
    assert valid['connected_cell_count']==2 and valid['interior_face_count']==1
    # Rotating a canonical cell by 90 degrees changes face start but not geometry.
    rotated = cells.copy()
    rotated[1] = rotated[1,[1,2,3,0,5,6,7,4]]
    assert m.boundary_topology(rotated,boundary)['interior_face_count']==1
    X, overlapping, boundary = two_hexes(same_side=True)
    assert np.all(m.rest_jacobians(X,overlapping)[0]>0)
    with pytest.raises(ValueError):
        m.boundary_topology(overlapping,boundary)


def test_mesh_child_environment_drops_loader_and_python_overrides_only_in_child(tmp_path,monkeypatch):
    protocol=m.read_protocol(ROOT/'manifests/experiments/hbe-01-03-mechanics-poc-v1.json')
    monkeypatch.setenv('DYLD_INSERT_LIBRARIES','/unrelated/injected.dylib')
    monkeypatch.setenv('DYLD_FALLBACK_LIBRARY_PATH','/unrelated/lib')
    monkeypatch.setenv('PYTHONPATH','/unrelated/python')
    monkeypatch.setenv('PYTHONHOME','/unrelated/runtime')
    before=dict(os.environ)
    captured={}
    class Process:
        pid=123456
        def poll(self):return 0
        def wait(self,timeout):return 0
    def popen(*args,**kwargs):
        captured.update(kwargs)
        return Process()
    monkeypatch.setattr(m.subprocess,'Popen',popen)
    monkeypatch.setitem(sys.modules,'febio_runtime',SimpleNamespace(
        process_group_rss=lambda *args,**kwargs:(0,[])))
    result=m.supervise_level(['mocked; no process launched'],tmp_path/'attempt',tmp_path,protocol,5.)
    assert result['status']=='complete'
    child=captured['env']
    assert not any(name.startswith('DYLD_') for name in child)
    assert 'PYTHONPATH' not in child and 'PYTHONHOME' not in child
    assert child.get('HOME')==before.get('HOME')
    assert child.get('CODEX_HOME')==before.get('CODEX_HOME')
    assert captured['start_new_session'] is True
    assert dict(os.environ)==before


@pytest.mark.parametrize('steps',[60,120])
def test_specimen_deck_reuses_verified_schema_but_prescribes_only_end_faces(steps):
    protocol=m.read_protocol(ROOT/'manifests/experiments/hbe-01-03-mechanics-poc-v1.json')
    X,cells,boundary=two_hexes()
    R,H=(protocol['geometry'][key] for key in ('radius_m','height_m'))
    X*=np.array([R/np.sqrt(2),R/np.sqrt(2),H/2])
    fixture={'rest_nodes_m':X.tolist(),'elements_hex8':cells.tolist(),'boundaries':boundary}
    xml,record=m.specimen_deck(fixture,'torsion_neg',steps,1000.,protocol)
    deck=ET.fromstring(xml)
    # Compare the material/domain/log contract with an actually solved patch deck.
    patch=ET.parse(ROOT/'artifacts/mechanics-febio-patch-run-v1/shear/shear.feb').getroot()
    for path in ('Module','Control/solver/linear_solver','Control/solver/qn_method'):
        assert deck.find(path).attrib==patch.find(path).attrib
    for key in ('c1','m1','c2','m2','k','pressure_model'):
        assert float(deck.findtext('Material/material/'+key))==float(patch.findtext('Material/material/'+key))
    assert deck.find('MeshDomains/SolidDomain').get('type')==patch.find('MeshDomains/SolidDomain').get('type')
    assert deck.findtext('MeshDomains/SolidDomain/laugon')==patch.findtext('MeshDomains/SolidDomain/laugon')
    for kind in ('node_data','element_data'):
        for key in ('data','name','delim'):
            assert deck.find('Output/logfile/'+kind).get(key)==patch.find('Output/logfile/'+kind).get(key)
    nodesets={row.get('name'):{int(i) for i in row.text.split(',')} for row in deck.findall('Mesh/NodeSet')}
    curves={row.get('id'):np.array([[float(v) for v in pt.text.split(',')]
             for pt in row.findall('points/pt')]) for row in deck.findall('LoadData/load_controller')}
    constrained={}
    for bc in deck.findall('Boundary/bc'):
        values=curves[bc.find('value').get('lc')]
        assert values.shape==(steps+1,2)
        axis='xyz'.index(bc.findtext('dof'))
        for node in nodesets[bc.get('node_set')]:
            assert (node,axis) not in constrained
            constrained[node,axis]=values[:,1]*float(bc.findtext('value'))
    assert {node for node,_ in constrained}==set(range(1,5))|set(range(9,13))
    assert len(constrained)==24  # Side-only middle-layer nodes remain free.
    angle=np.array(record['load_coordinate'])
    for node in range(9,13):
        actual=X[node-1]+np.column_stack([constrained[node,axis] for axis in range(3)])
        complex_position=(X[node-1,0]+1j*X[node-1,1])*np.exp(1j*angle)
        np.testing.assert_allclose(actual[:,0]+1j*actual[:,1],complex_position,atol=2e-18,rtol=0.)
        np.testing.assert_array_equal(actual[:,2],np.full(steps+1,H))
