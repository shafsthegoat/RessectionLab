"""Offline source-owned history and instantaneous model-change accounting.

Geometry is a declared source-cell/corner graph. Transfer never averages or
zeros retained fields and never re-equilibrates them. These numerical event
identities are not cutting-device work, healing, injury or live admission.
"""
from __future__ import annotations

import hashlib
import numpy as np

from .surgical_continuum import CORNERS, element_operators
from .surgical_continuum_partition import _validate


def _geometry(data):
    cells, shape, nodes, elements, affine, free = _validate(data)
    mu = np.asarray(data['muPa'],float); nu = np.asarray(data['nu'],float)
    expected = (np.column_stack(np.unravel_index(cells,tuple(shape)))[:,None,:]+CORNERS-.5)@affine[:3,:3].T+affine[:3,3]
    if (np.max(np.abs(expected-nodes[elements]),initial=0.) > 1e-10
            or not np.isfinite(expected).all() or np.linalg.det(affine[:3,:3]) == 0):
        raise ValueError('EVENT_SOURCE_AFFINE_CORNERS')
    if (mu.shape != cells.shape or nu.shape != cells.shape or not np.isfinite(mu).all()
            or not np.isfinite(nu).all() or np.any(mu <= 0) or np.any(nu < 0) or np.any(nu > .45)):
        raise ValueError('EVENT_SOURCE_MATERIAL')
    return cells,shape,nodes,elements,affine,free,mu,nu


def _fields(geometry, displacement, memories, alpha):
    n = geometry[2].size; u=np.asarray(displacement); z=np.asarray(memories); a=np.asarray(alpha)
    if (a.dtype.kind not in 'fiu' or a.ndim!=1 or len(a)>2 or not np.isfinite(a).all() or np.any(a<0)
            or u.dtype.kind not in 'fiu' or u.shape!=(n,) or not np.isfinite(u).all()
            or z.dtype.kind not in 'fiu' or z.shape!=(len(a),n) or not np.isfinite(z).all()):
        raise ValueError('EVENT_FIELD_SHAPE')
    if np.any(u[~geometry[5]]!=0.) or np.any(z[:,~geometry[5]]!=0.):
        raise ValueError('EVENT_NONZERO_CONSTRAINED_HISTORY')
    return u.astype(float,copy=False),z.astype(float,copy=False),a.astype(float,copy=False)


def stored_energy_by_source(data, displacement, memories, alpha):
    g=_geometry(data); u,z,a=_fields(g,displacement,memories,alpha)
    cells,_,_,en,affine,_,mu,nu=g; shear,lame,_=element_operators(affine)
    energy=np.zeros(len(cells))
    for start in range(0,len(cells),256):
        nodes=en[start:start+256]; m=mu[start:start+256]*1e-6; v=nu[start:start+256]
        k=m[:,None,None]*shear+(2*m*v/(1-2*v))[:,None,None]*lame
        for coefficient,field in [(1.,u),*[(aj,u-zj) for aj,zj in zip(a,z)]]:
            local=field.reshape((-1,3))[nodes].reshape((-1,24))
            energy[start:start+len(nodes)] += .5*coefficient*np.einsum('ei,eij,ej->e',local,k,local)
    if not np.isfinite(energy).all() or np.any(energy < -1e-12):
        raise ValueError('EVENT_STORED_ENERGY')
    return energy


def source_operator_identity(data, alpha, times, *, core_free_dofs, removed_source_ids=(), cut_faces=()):
    """Bind the complete domain/support/declaration, including outside removals."""
    g=_geometry(data); a=np.asarray(alpha,float); t=np.asarray(times,float)
    if (a.ndim!=1 or t.shape!=a.shape or len(a)>2 or not np.isfinite(a).all()
            or not np.isfinite(t).all() or np.any(a<0) or np.any(t<=0)):
        raise ValueError('EVENT_SPECTRUM')
    core=np.asarray(core_free_dofs)
    if (core.ndim!=1 or core.dtype.kind not in 'iu' or not 0<len(core)<int(g[5].sum())
            or len(core)>6000 or len(np.unique(core))!=len(core) or np.any(core<0) or np.any(core>=g[5].sum())):
        raise ValueError('EVENT_CORE_FREE_PARTITION')
    removed=np.asarray(removed_source_ids)
    if removed.size==0:removed=np.empty(0,dtype=np.int64)
    cuts=np.asarray(cut_faces)
    if cuts.size==0:cuts=np.empty((0,2),dtype=np.int64)
    if (removed.ndim!=1 or removed.dtype.kind not in 'iu' or cuts.ndim!=2
            or cuts.shape[1]!=2 or cuts.dtype.kind not in 'iu'):
        raise ValueError('EVENT_SOURCE_IDENTITY')
    size=int(np.prod(g[1]))
    if np.any(removed<0) or np.any(removed>=size) or np.any(cuts<0) or np.any(cuts>=size):
        raise ValueError('EVENT_SOURCE_IDENTITY')
    digest=hashlib.sha256(b'resectionlab.exterior-source-system.v1')
    for value in [*g,a,t,core,np.unique(removed),np.unique(np.sort(cuts,axis=1),axis=0)]:
        arr=np.ascontiguousarray(value)
        digest.update(str((arr.dtype.str,arr.shape)).encode());digest.update(arr.tobytes())
    return digest.hexdigest()


class SourceHistoryLedger:
    """Transactional removal ledger; no retained episode field is stored here."""
    def __init__(self):
        self.removed_source_ids=set(); self.removed_stored_energy_nmm=0.; self.events=[]

    def transfer(self, old, new, displacement, memories, alpha):
        og=_geometry(old); ng=_geometry(new); u,z,a=_fields(og,displacement,memories,alpha)
        oc,shape,on,oe,affine,_,omu,onu=og; nc,nshape,nn,ne,na,nf,nmu,nnu=ng
        if not np.array_equal(shape,nshape) or not np.array_equal(affine,na):
            raise ValueError('EVENT_DIFFERENT_SOURCE_GRID')
        lookup={int(cell):i for i,cell in enumerate(oc)}
        if any(int(cell) not in lookup for cell in nc):
            raise ValueError('EVENT_UNSUPPORTED_SOURCE_ACTIVATION')
        rows=np.asarray([lookup[int(cell)] for cell in nc])
        if not np.array_equal(omu[rows],nmu) or not np.array_equal(onu[rows],nnu):
            raise ValueError('EVENT_UNDECLARED_MATERIAL_CHANGE')
        removed=sorted(set(map(int,oc))-set(map(int,nc)))
        if self.removed_source_ids.intersection(removed):
            raise ValueError('EVENT_DUPLICATE_REMOVAL_EXPORT')
        if self.removed_source_ids.intersection(map(int,oc)) or self.removed_source_ids.intersection(map(int,nc)):
            raise ValueError('EVENT_REMOVED_SOURCE_RESURRECTION')
        # Canonical first owner is independent of the supplied element-row order.
        order=np.argsort(nc,kind='stable'); target=ne[order].ravel(); source=oe[rows[order]].ravel()
        first=np.full(len(nn),len(target),dtype=np.int64)
        np.minimum.at(first,target,np.arange(len(target)))
        if np.any(first==len(target)):raise ValueError('EVENT_ORPHAN_NODE')
        local_u=u.reshape((-1,3))[source]; local_z=z.reshape((len(a),len(on),3))[:,source]
        after=local_u[first].copy(); memories_after=local_z[:,first].copy()
        if (not np.array_equal(local_u.view(np.uint64),after[target].view(np.uint64))
                or not np.array_equal(local_z.view(np.uint64),memories_after[:,target].view(np.uint64))):
            raise ValueError('EVENT_UNEQUAL_JOINED_HISTORY')
        after=after.ravel(); memories_after=memories_after.reshape((len(a),nn.size))
        if np.any(after[~nf]!=0.) or np.any(memories_after[:,~nf]!=0.):
            raise ValueError('EVENT_NEW_SUPPORT_NEEDS_WORK_OR_TRANSFER')
        old_energy=stored_energy_by_source(old,u,z,a); new_energy=stored_energy_by_source(new,after,memories_after,a)
        exported=float(sum(old_energy[lookup[cell]] for cell in removed))
        balance=float(new_energy.sum()-old_energy.sum()+exported)
        if abs(balance)>1e-10:raise ValueError('EVENT_TISSUE_ENERGY_BALANCE')
        ancestors=[set() for _ in range(len(nn))]
        for node,parent in zip(target,source):ancestors[int(node)].add(int(parent))
        descendants={}
        for node,parents in enumerate(ancestors):
            for parent in parents:descendants.setdefault(parent,set()).add(node)
        row={'removedSourceIds':removed,'removedElementCount':len(removed),
            'removedStoredEnergyNmm':exported,'oldTissueEnergyNmm':float(old_energy.sum()),
            'transferredTissueEnergyNmm':float(new_energy.sum()),'tissueEventBalanceResidualNmm':balance,
            'splitParents':sum(len(nodes)>1 for nodes in descendants.values()),
            'joinedNodes':sum(len(parents)>1 for parents in ancestors)}
        # Commit only after all owner/support/energy checks succeed.
        self.removed_source_ids.update(removed);self.removed_stored_energy_nmm+=exported;self.events.append(row)
        return {'displacement':after,'memories':memories_after,'event':row}


def contact_binding_jump(old_solver, new_solver, old_u, new_u, old_target, new_target):
    """Signed spring-model energy change, separate from removal and BE loss."""
    before=old_solver.contact_energy(old_u,old_target)
    after=new_solver.contact_energy(new_u,new_target)
    return {'oldContactEnergyNmm':before,'newContactEnergyNmm':after,
            'contactModelEnergyJumpNmm':after-before,
            'scope':'Declared instantaneous contact-model change; no measured work or dissipation assigned'}
