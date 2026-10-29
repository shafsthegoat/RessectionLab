"""Generated end-to-end pilot/export contract before a source-sized factor run."""
from pathlib import Path
import json
import runpy
import sys

import numpy as np
import pytest

from resectionlab.surgical_continuum import voxel_mesh
from resectionlab.surgical_continuum_exterior import source_stiffness
from resectionlab.surgical_continuum_partition import source_partition
from resectionlab.surgical_continuum_topology import SourceDomain
from resectionlab.surgical_scene import SurgicalScene


@pytest.mark.parametrize('split_refusal',[False,True])
def test_generated_two_block_removal_release_exports_pass_independent_quadrature(tmp_path,split_refusal):
    root=Path(__file__).resolve().parents[1];scripts=root/'scripts';sys.path.insert(0,str(scripts))
    try:api=runpy.run_path(str(scripts/'profile_continuum_retained.py'))
    finally:sys.path.remove(str(scripts))
    sha=api['sha'];original=json.loads(api['PROTOCOL'].read_text())
    def bound(path):return {'path':str(path.relative_to(root)),'sha256':sha(path)}
    scene=SurgicalScene(np.ones((5,5,5),np.uint16),np.eye(4),
        [{'id':1,'name':'Generated coupon','material':'soft_tissue'}],{'kind':'generated pilot control'}, {})
    def geometry(cells):
        mesh=voxel_mesh(scene,cells);fixed=np.any(np.isclose(mesh.nodes,-.5)|np.isclose(mesh.nodes,4.5),axis=1)
        return dict(cells=np.asarray(mesh.cells),shape=np.asarray(scene.shape),nodes=mesh.nodes,
            elementNodes=mesh.element_nodes,affine=scene.affine,freeDofs=np.repeat(~fixed,3),
            muPa=np.full(len(mesh.cells),1000.),nu=np.full(len(mesh.cells),.3))
    cell=lambda xyz:int(np.ravel_multi_index(xyz,scene.shape))
    g=geometry(range(125));cg=geometry(range(75));_,partition=source_partition(cg,g)
    free=np.flatnonzero(g['freeDofs']);core_full=(3*partition['coreNodesInReference'][:,None]+np.arange(3)).ravel()
    core=np.searchsorted(free,core_full[g['freeDofs'][core_full]])
    # Separated owners have bijective surviving sectors. The separate compact
    # cluster intentionally splits a sector and exercises the refusal path.
    removal_coordinates=([(2,1,1),(2,1,2),(2,2,1),(3,2,2),(1,2,2),(2,2,2),(2,3,2),(1,3,2)]
        if split_refusal else [(1,0,0),(1,0,2),(1,0,4),(4,4,4),(1,2,0),(1,2,4),(1,4,0),(1,4,4)])
    removals=[cell(v) for v in removal_coordinates]
    contacts=np.array([cell(v) for v in [(0,1,1),(0,1,2),(0,1,3),(0,2,1),(0,2,2),(0,2,3),(0,3,1)]])
    alpha=np.array([.5]);times=np.array([.5]);spring=np.full(21,.02/7);target=np.tile([.001,0.,0.],7)
    extra=dict(alpha=alpha,times=times,spring=spring,contactCells=contacts,commonCells=np.array(sorted(set(range(125))-set(removals))))
    reference=tmp_path/'reference.npz';np.savez_compressed(reference,**g,**extra)
    corefile=tmp_path/'core.npz';np.savez_compressed(corefile,**cg)
    partfile=tmp_path/'partition.npz';np.savez_compressed(partfile,**partition)
    c=api['contact_matrix'](g,contacts)[:,free];k=source_stiffness(g,source_order=g['cells'])[free][:,free].toarray()
    u=np.linalg.solve(1.5*k+c.T@np.diag(spring)@c,c.T@(spring*target));z=np.asarray([.25*u])
    seed=tmp_path/'initial.npz';np.savez_compressed(seed,freeIndices=free,displacement=u,memories=z,target=target,h=1/120)
    domain=SourceDomain(g,core);states=[]
    for index in range(9):
        mask=sorted(removals[:index]);derived=domain.after(mask)
        path=tmp_path/f'geometry-{index}.npz'
        np.savez_compressed(path,**derived['geometry'],coreFreeDofs=derived['coreFreeDofs'],
            originalFreeDofForNew=derived['originalFreeDofForNew'])
        states.append({'removedSourceIds':mask,'newExteriorGroup':index in (0,4),
            'geometry':{'path':path.name,'sha256':sha(path)}})
    topofile=tmp_path/'topology.json';topofile.write_text(json.dumps({'exactFactorReuseCandidate':True,'uniqueTissueStates':9,'states':states}))
    p={**original,'geometry':{'core':bound(corefile),'reference':bound(reference)},'partition':bound(partfile),
        'baselineWorld':bound(seed),'topologyReport':bound(topofile),'expectedRemovalOrder':removals}
    protocol=tmp_path/'protocol.json';protocol.write_text(json.dumps(p));launch=tmp_path/'launch.json'
    launch.write_text(json.dumps({'protocolSha256':sha(protocol),'commit':'generated-control','sourceHashes':{}}))
    api['worker'].__globals__['PROTOCOL']=protocol
    if split_refusal:
        # This distinct generated deletion splits a retained source sector.
        # Preserve the fast-path refusal rather than weakening its bijection.
        with pytest.raises(ValueError,match='EXTERIOR_RETAINED_FREE_LINEAGE'):
            api['worker'](tmp_path/'result',launch)
        partial=json.loads((tmp_path/'result/report.json').read_text())
        assert partial['status']=='invalid_or_incomplete' and len(partial['states'])==4
        return
    result=api['worker'](tmp_path/'result',launch)
    assert result['fullExteriorBuilds']==2 and result['retainedCoreBuilds']==7
    assert result['states'][4]['exteriorResetRefusal']=='EXTERIOR_RETAINED_EXTERIOR_COEFFICIENTS'
    assert result['states'][4]['oldExteriorOperatorReleased']
    checker=runpy.run_path(str(scripts/'check_continuum_retained.py'))
    independent=checker['check'](tmp_path/'result',protocol)
    assert independent['status']=='passed' and independent['samples']==11
    assert independent['removedOwners']==sorted(removals)
    assert independent['states'][9]['contactModelEnergyJumpNmm']<0
