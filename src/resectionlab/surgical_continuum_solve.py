"""Opt-in proportional-matrix solve with full distributed Maxwell history.

This changes linear algebra, not the tissue law or artificial boundary. Cached
operators contain no episode history. Source removal/cuts rebuild them through
the existing topology signature; corner history is transferred by source owner.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import splu

PROPERTY='continuumLinearSolveV1'
SCHEMA='resectionlab.continuum-linear-solve.v1'


def profile(method='proportional-history-compliance'):
    if method not in ('detailed','proportional-history-compliance'):raise ValueError('Unknown continuum linear solver')
    return {'schema':SCHEMA,'method':method,'supportIdentity':'current-retained-source-interface',
        'scope':'Research computation only; same constitutive law, boundaries, physical guards and evidence limits.'}


def alpha(materials):
    rows=[np.asarray(m['relaxationShearPa'],float)/m['shearPa'] for m in materials.values()]
    if not rows or any(not np.allclose(r,rows[0],rtol=64*np.finfo(float).eps,atol=0.) for r in rows[1:]):
        raise ValueError('CONTINUUM_CONTACT_NONPROPORTIONAL: distributed branch stiffness needs the detailed solver')
    return rows[0]


def validate(scene,continuum_profile):
    declared=scene.properties.get(PROPERTY)
    if declared is None:return
    if continuum_profile is None:raise ValueError('CONTINUUM_CONTACT_REQUIRES_CONTINUUM')
    if not isinstance(declared,dict) or declared.get('method') not in ('detailed','proportional-history-compliance') or declared!=profile(declared['method']):
        raise ValueError('COMPLETE_CONTINUUM_LINEAR_SOLVE_DECLARATION_REQUIRED')
    if declared['method']=='proportional-history-compliance':alpha(continuum_profile['materials'])


class ProportionalHistorySolve:
    def __init__(self,matrices,free,contacts,ratios,materials):
        self.k0=matrices[0];self.free=np.asarray(free,bool);self.n=len(free)
        self.alpha=alpha(materials);self.ratios=np.asarray(ratios);self.beta=1+float(self.ratios@self.alpha)
        for a,k in zip(self.alpha,matrices[1:]):
            residual=k-a*self.k0
            scale=max(float(np.max(np.abs(k.data),initial=0.)),float(np.max(np.abs((a*self.k0).data),initial=0.)))
            if np.max(np.abs(residual.data),initial=0.)>64*np.finfo(float).eps*scale:
                raise ValueError('CONTINUUM_CONTACT_MATRIX_PROPORTIONALITY')
        self.contacts=tuple((key,cell,tuple(map(int,nodes)),float(k)) for key,cell,nodes,k in contacts)
        rows=[];cols=[];spring=[]
        for i,(_,_,nodes,k) in enumerate(contacts):
            for axis in range(3):
                rows.extend([3*i+axis]*8);cols.extend(3*np.asarray(nodes)+axis);spring.append(k)
        self.c=sparse.coo_matrix((np.full(len(rows),1/8.),(rows,cols)),shape=(3*len(contacts),self.n)).tocsr()
        self.spring=np.asarray(spring);self._factor=None
        self.basis=np.zeros((self.n,len(spring)))
        if len(spring) and self.free.any():self.basis[self.free]=self._equilibrium_factor().solve(self.c[:,self.free].T.toarray())
        self.compliance=np.asarray(self.c@self.basis)
        self._small=(lu_factor(self.beta*np.eye(len(spring))+self.spring[:,None]*self.compliance) if len(spring) else None)
        self._body_force=None;self._body_field=np.zeros(self.n)
        residual=self.k0[self.free][:,self.free]@self.basis[self.free]-self.c[:,self.free].T.toarray()
        if not np.isfinite(self.basis).all() or np.max(np.abs(residual),initial=0.)>1e-9:
            raise ValueError('CONTINUUM_CONTACT_BASIS_RESIDUAL')

    def _equilibrium_factor(self):
        if self._factor is None:
            self._factor=splu(self.k0[self.free][:,self.free].tocsc(),permc_spec='MMD_AT_PLUS_A',options={'SymmetricMode':True})
        return self._factor

    def advance(self,memories,gravity,targets):
        gravity=np.asarray(gravity,float)
        if gravity.shape!=(self.n,) or not np.isfinite(gravity).all() or len(memories)!=len(self.alpha):
            raise ValueError('CONTINUUM_CONTACT_HISTORY_SHAPE')
        for z in memories:
            if np.asarray(z).shape!=(self.n,) or not np.isfinite(z).all():raise ValueError('CONTINUUM_CONTACT_HISTORY_SHAPE')
            if np.any(np.asarray(z)[~self.free]!=0.):
                raise ValueError('CONTINUUM_CONTACT_FIXED_HISTORY: nonzero constrained memory requires the detailed solver')
        if self._body_force is None or not np.array_equal(self._body_force,gravity):
            self._body_field=np.zeros(self.n)
            if self.free.any() and np.any(gravity[self.free]):
                self._body_field[self.free]=self._equilibrium_factor().solve(gravity[self.free])
            self._body_force=gravity.copy()
        base=self._body_field.copy()
        for r,a,z in zip(self.ratios,self.alpha,memories):base+=r*a*z
        base/=self.beta
        if not self.contacts:return base
        if len(targets)!=len(self.contacts):raise ValueError('CONTINUUM_CONTACT_TARGET_SHAPE')
        target=np.asarray([entry[3] for entry in targets],float).ravel()
        if target.shape!=self.spring.shape or not np.isfinite(target).all():raise ValueError('CONTINUUM_CONTACT_TARGET_SHAPE')
        rhs=self.spring*(target-self.c@base)
        q=lu_solve(self._small,rhs)
        if np.max(np.abs(self.beta*q+self.spring*(self.compliance@q)-rhs),initial=0.)>1e-9:
            raise ValueError('CONTINUUM_CONTACT_SCHUR_RESIDUAL')
        return base+self.basis@q
