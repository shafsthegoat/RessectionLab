"""Source-owned graph/support inheritance and stable I/E lineage controls."""
import numpy as np
import pytest

from resectionlab.surgical_continuum import voxel_mesh
from resectionlab.surgical_continuum_events import SourceHistoryLedger
from resectionlab.surgical_continuum_topology import SourceDomain
from resectionlab.surgical_scene import SurgicalScene


def coupon(cuts=()):
    scene = SurgicalScene(np.ones((3,2,2),np.uint16),np.eye(4),
        [{'id':1,'name':'Generated coupon','material':'soft_tissue'}],{'kind':'generated topology control'}, {})
    mesh = voxel_mesh(scene,range(12),cuts)
    free = np.repeat(mesh.nodes[:,0] != -.5,3)
    g = dict(cells=np.asarray(mesh.cells),shape=np.asarray(scene.shape),nodes=mesh.nodes,
        elementNodes=mesh.element_nodes,affine=scene.affine,freeDofs=free,
        muPa=np.arange(12)+1000.,nu=np.full(12,.3))
    original_free = np.flatnonzero(free)
    core_nodes = np.unique(mesh.element_nodes[:4])
    selected = np.intersect1d(original_free,(3*core_nodes[:,None]+np.arange(3)).ravel())
    return g,np.searchsorted(original_free,selected)


def test_removal_keeps_original_core_descendants_and_inherits_all_material_support_owners():
    g,core = coupon();domain = SourceDomain(g,core);result = domain.after([0,1,2,3])
    new = result['geometry']; lineage = result['originalFreeDofForNew']; new_core = result['coreFreeDofs']
    assert set(new['cells']) == set(range(4,12))
    # No remaining element is a declared original core cell, but interface
    # descendants stay in I instead of silently moving into the E factor.
    assert len(new_core)>0 and np.isin(lineage[new_core],core).all()
    np.testing.assert_array_equal(new['muPa'],g['muPa'][4:])
    full = (3*result['originalNodeForNew'][:,None]+np.arange(3)).ravel()
    np.testing.assert_array_equal(new['freeDofs'],g['freeDofs'][full])
    assert domain.after([3,2,1,0])['removedSourceIds']==result['removedSourceIds']


def test_core_order_is_explicit_and_repeated_split_descendants_keep_lineage_order():
    g,core = coupon();core = core[::-1];domain = SourceDomain(g,core)
    result = domain.after()
    np.testing.assert_array_equal(result['originalFreeDofForNew'][result['coreFreeDofs']],core)
    split = domain.after(additional_cut_faces=[(i,i+4) for i in range(4)])
    rank = {int(dof):i for i,dof in enumerate(core)}
    ranks = [rank[int(dof)] for dof in split['originalFreeDofForNew'][split['coreFreeDofs']]]
    assert ranks==sorted(ranks) and len(set(ranks))<len(ranks)


def test_cut_splits_inherit_full_history_and_baseline_cut_declaration_is_required():
    g,core = coupon();domain = SourceDomain(g,core)
    cuts = [(i,i+4) for i in range(4)]
    result = domain.after(additional_cut_faces=cuts);new = result['geometry']
    assert len(new['nodes'])>len(g['nodes'])
    u = np.arange(g['nodes'].size,dtype=float)*1e-5;u[~g['freeDofs']]=0.
    z = np.asarray([.2*u,.7*u]);transferred = SourceHistoryLedger().transfer(g,new,u,z,[.5,.2])
    full = (3*result['originalNodeForNew'][:,None]+np.arange(3)).ravel()
    np.testing.assert_array_equal(transferred['displacement'],u[full])
    np.testing.assert_array_equal(transferred['memories'],z[:,full])
    assert transferred['event']['splitParents']>0
    with pytest.raises(ValueError,match='SOURCE_JOIN|BASELINE_GRAPH'):SourceDomain(new,result['coreFreeDofs'])
    # An explicit full baseline graph reproduces all sectors and cannot heal.
    cut_domain = SourceDomain(new,result['coreFreeDofs'],cut_faces=cuts)
    assert len(cut_domain.after()['geometry']['nodes'])==len(new['nodes'])


@pytest.mark.parametrize('owners',[[12],[-1],[True],[.5]])
def test_undeclared_removals_refuse(owners):
    g,core = coupon()
    with pytest.raises(ValueError,match='REMOV'):SourceDomain(g,core).after(owners)


@pytest.mark.parametrize('cuts',[[(0,11)],[(0,12)],[(False,True)],[(0.,4.)]])
def test_nonface_or_undeclared_cuts_refuse(cuts):
    g,core = coupon()
    with pytest.raises(ValueError,match='CUT'):SourceDomain(g,core).after(additional_cut_faces=cuts)
