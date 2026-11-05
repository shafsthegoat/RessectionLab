"""Finite-face nonadhesive contact over the existing continuum/history law.

This first research boundary is frictionless and fixed in orientation. It is
not a measured human blade law. Loaded entry through a finite pad edge is
refused: the edge contact normal is not represented by a flat-face constraint.
Unloading, separation, interior sliding and departure retain their true forces.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .surgical_contact_complementarity import solve_unilateral,GAP_TOLERANCE_MM,FORCE_TOLERANCE_N,ENERGY_TOLERANCE_NMM


def step(matrices,free,before,memories,times,h,body,mesh,surface,old_blade,blade,*,inverse=None,check=lambda:None):
    """Advance distributed Maxwell memory and exact sampled normal contact.

    Source matrices/supports and history are supplied by the ordinary solver.
    No contact potential/spring is stored. Moving-blade work is lambda*Delta b;
    lambda*old_gap is explicit BE contact-activation dissipation, not injury.
    """
    check();u0=np.asarray(before,float);z0=np.asarray(memories,float);times=np.asarray(times,float);free=np.asarray(free,bool);body=np.asarray(body,float)
    n=len(u0);count=len(times);matrices=tuple(sparse.csr_matrix(k) for k in matrices)
    if count==0 and z0.size==0:z0=z0.reshape(0,n)
    if (u0.shape!=(n,) or z0.shape!=(count,n) or free.shape!=(n,) or body.shape!=(n,) or len(matrices)!=count+1
            or any(k.shape!=(n,n) for k in matrices) or not np.isfinite(h) or h<=0 or not np.isfinite(times).all() or np.any(times<=0)
            or not all(np.isfinite(v).all() for v in (u0,z0,body)) or np.any(u0[~free]!=0) or np.any(z0[:,~free]!=0)):
        raise ValueError('RETRACTOR_CONTACT_HISTORY_OR_SUPPORT')
    if blade is not None and old_blade is None:raise ValueError('RETRACTOR_CONTACT_INITIAL_POSE_REQUIRED')
    if blade is not None and np.max(np.abs(old_blade.matrix-blade.matrix),initial=0.)>1e-10:raise ValueError('RETRACTOR_CONTACT_ROTATION_UNQUALIFIED')
    if blade is not None and (old_blade.half_width_mm,old_blade.depth_mm,old_blade.half_thickness_mm)!=(blade.half_width_mm,blade.depth_mm,blade.half_thickness_mm):
        raise ValueError('RETRACTOR_CONTACT_GEOMETRY_CHANGED')
    ratio=times/(times+h);effective=matrices[0].copy();rhs=body.copy()
    for r,k,z in zip(ratio,matrices[1:],z0):effective=effective+r*k;rhs+=r*(k@z)
    a=effective[free][:,free].tocsc();f=rhs[free]
    if not free.any():raise ValueError('RETRACTOR_CONTACT_REQUIRES_EXPLICIT_FREE_BODY')
    if inverse is None:
        factor=splu(a,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
        inverse=factor.solve
    rest,_=surface.geometry(mesh.nodes,np.zeros_like(mesh.nodes));old_points,_=surface.geometry(mesh.nodes,u0)
    old_local=old_blade.local(old_points) if blade is not None else np.zeros_like(old_points)
    side=np.where(old_local[:,1]>=0,1.,-1.)
    normals=side[:,None]*blade.matrix[:,1] if blade is not None else np.zeros_like(old_points)
    # A body inverse is reusable; contact rows are rebuilt from the current
    # finite footprint, not from the original hold's cell-center membership.
    trial=u0.copy();last=None;result=None
    for iteration in range(24):
        check();points,outward=surface.geometry(mesh.nodes,trial)
        local=blade.local(points) if blade is not None else np.zeros_like(points)
        footprint=((np.abs(local[:,0])<=blade.half_width_mm+1e-9)&(local[:,2]>=-blade.depth_mm-1e-9)&(local[:,2]<=1e-9)
            if blade is not None else np.zeros(len(points),bool))
        facing=np.einsum('pi,pi->p',outward,normals)<-1e-10;indices=np.flatnonzero(footprint&facing)
        if len(indices)>512:raise ValueError('RETRACTOR_CONTACT_POINT_LIMIT')
        c=surface.rows(normals[indices],indices)
        b=np.einsum('pi,pi->p',normals[indices],blade.position-rest[indices])+blade.half_thickness_mm if blade is not None else np.empty(0)
        old_b=np.einsum('pi,pi->p',normals[indices],old_blade.position-rest[indices])+old_blade.half_thickness_mm if blade is not None else np.empty(0)
        old_gap=c@u0-old_b
        if np.min(old_gap,initial=0.)<-GAP_TOLERANCE_MM:
            raise ValueError('RETRACTOR_CONTACT_INITIAL_PENETRATION_OR_LOADED_EDGE_ENTRY')
        result=solve_unilateral(a,f,c[:,free],b,inverse,check=check);u=np.zeros(n);u[free]=result['displacement']
        current=tuple(map(int,indices))
        if current==last:break
        last=current;trial=u
    else:raise ValueError('RETRACTOR_CONTACT_FOOTPRINT_NOT_CONVERGED')
    # One final geometry check ensures the converged solve uses its own surface.
    points,outward=surface.geometry(mesh.nodes,u)
    local=blade.local(points) if blade is not None else np.zeros_like(points)
    actual=(np.flatnonzero((np.abs(local[:,0])<=blade.half_width_mm+1e-9)&(local[:,2]>=-blade.depth_mm-1e-9)&(local[:,2]<=1e-9)&(np.einsum('pi,pi->p',outward,normals)<-1e-10))
        if blade is not None else np.empty(0,int))
    if not np.array_equal(actual,indices):raise ValueError('RETRACTOR_CONTACT_FOOTPRINT_CHANGED')
    z=(times[:,None]*z0+h*u)/(times[:,None]+h);delta=u-u0
    def energy(x,history):
        return float(.5*x@(matrices[0]@x)+sum(.5*(x-v)@(k@(x-v)) for k,v in zip(matrices[1:],history)))
    E0=energy(u0,z0);E=energy(u,z);viscous=0.;numerical=float(.5*delta@(matrices[0]@delta))
    for tau,k,v,old in zip(times,matrices[1:],z,z0):
        dz=v-old;dv=(u-v)-(u0-old);viscous+=float(tau/h*dz@(k@dz));numerical+=float(.5*dv@(k@dv))
    force=result['normalForcesN'];work=float(force@(b-old_b));contact_loss=float(force@old_gap);body_work=float(body@delta)
    balance=body_work+work-(E-E0)-viscous-numerical-contact_loss
    residual=matrices[0]@u-body-c.T@force
    for k,v in zip(matrices[1:],z):residual+=k@(u-v)
    maximum=float(np.max(np.abs(residual[free]),initial=0.))
    if (maximum>FORCE_TOLERANCE_N or abs(balance)>ENERGY_TOLERANCE_NMM
            or min(viscous,numerical,contact_loss)<-ENERGY_TOLERANCE_NMM):raise ValueError('RETRACTOR_CONTACT_FORCE_OR_ENERGY')
    loaded=force>FORCE_TOLERANCE_N;touching=result['gapMm']<=GAP_TOLERANCE_MM
    tissue_force=np.sum(force[:,None]*normals[indices],axis=0)
    return {'displacement':u,'memories':z,'pointIndices':indices,'normalForcesN':force,'gapMm':result['gapMm'],
        'contactSourceIds':sorted(set(map(int,surface.source_ids[indices[touching]]))),
        'loadedSourceIds':sorted(set(map(int,surface.source_ids[indices[loaded]]))),
        'sampledTouchAreaMm2':float(surface.reference_area_mm2[indices[touching]].sum()),
        'forceOnTissueN':tissue_force,'reactionOnInstrumentN':-tissue_force,'summedNormalForceN':float(force.sum()),
        'energyNmm':E,'oldEnergyNmm':E0,'bodyWorkNmm':body_work,'toolWorkNmm':work,
        'dissipatedEnergyNmm':viscous,'numericalDissipationNmm':numerical,'contactActivationDissipationNmm':contact_loss,
        'energyBalanceResidualNmm':balance,'maxFreeResidualN':maximum,'maxComplementarityNmm':result['maxComplementarityNmm'],
        'footprintIterations':iteration+1,'edgeContactValidated':False,'patientForceValidated':False}
