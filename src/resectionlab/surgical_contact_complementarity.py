"""Compressive-only, frictionless linear contact using an existing body inverse.

The body operator/material/history stays with its caller. Constraints are
surface normal gaps C*u-b >= 0 in mm; multipliers are nonnegative forces in N.
No spring stiffness, adhesion, force clipping or contact-based support is added.
"""
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

GAP_TOLERANCE_MM=1e-9
FORCE_TOLERANCE_N=1e-10
ENERGY_TOLERANCE_NMM=1e-10


def solve_unilateral(matrix,rhs,contact,target,inverse,*,check=lambda:None):
    """Solve a convex contact QP and certify its full physical KKT residuals.

    A positive-definite material/support operator is required before this call;
    contact cannot rescue a floating body. Rank-deficient contact quadrature is
    permitted. Multipliers/point pressures need not be unique; displacement and
    aggregate nodal force are the supported quantities.
    """
    check();a=sparse.csr_matrix(matrix);c=sparse.csr_matrix(contact);f=np.asarray(rhs,float);b=np.asarray(target,float)
    n=a.shape[0];m=c.shape[0]
    if (a.shape!=(n,n) or not 0<n<=24000 or c.shape!=(m,n) or not 0<=m<=512 or f.shape!=(n,) or b.shape!=(m,)
            or any(not np.isfinite(v).all() for v in (a.data,c.data,f,b))):raise ValueError('CONTACT_SYSTEM_SHAPE_OR_RESOURCE_LIMIT')
    scale=float(np.max(np.abs(a.data),initial=0.));asym=a-a.T
    if np.max(np.abs(asym.data),initial=0.)>64*np.finfo(float).eps*max(scale,1e-300):raise ValueError('CONTACT_BODY_SYMMETRY')
    base=np.asarray(inverse(f),float)
    if base.shape!=(n,) or not np.isfinite(base).all():raise ValueError('CONTACT_BODY_INVERSE')
    if np.max(np.abs(a@base-f),initial=0.)>FORCE_TOLERANCE_N:raise ValueError('CONTACT_BODY_INVERSE_RESIDUAL')
    if not m:
        return {'displacement':base,'normalForcesN':np.empty(0),'gapMm':np.empty(0),'maxFreeResidualN':float(np.max(np.abs(a@base-f),initial=0.)),
            'maxComplementarityNmm':0.,'minimumGapMm':0.,'optimizerIterations':0,'optimizerStatus':0,'activeSetPolished':False}
    response=np.empty((n,m))
    for start in range(0,m,64):
        check();columns=np.asarray(inverse(c[start:start+64].T.toarray()),float)
        if columns.shape!=(n,min(64,m-start)) or not np.isfinite(columns).all():raise ValueError('CONTACT_RESPONSE_SHAPE')
        response[:,start:start+64]=columns
    if np.max(np.abs(a@response-c.T.toarray()),initial=0.)>FORCE_TOLERANCE_N:raise ValueError('CONTACT_RESPONSE_RESIDUAL')
    compliance=np.asarray(c@response);magnitude=float(np.max(np.abs(compliance),initial=0.));tol=64*np.finfo(float).eps*max(m,1)*max(magnitude,1e-300)
    if np.max(np.abs(compliance-compliance.T),initial=0.)>tol:raise ValueError('CONTACT_COMPLIANCE_SYMMETRY')
    compliance=(compliance+compliance.T)/2
    if np.linalg.eigvalsh(compliance).min(initial=0.)<-tol:raise ValueError('CONTACT_NONPOSITIVE_COMPLIANCE')
    deficit=b-c@base
    zero=np.max(np.abs(compliance),axis=1)<=tol
    if np.any(zero&(deficit>GAP_TOLERANCE_MM)):raise ValueError('CONTACT_INFEASIBLE_FIXED_POINT')
    if np.max(deficit,initial=0.)<=0:
        multipliers=np.zeros(m);iterations=0;status=0;polished=False
    else:
        def objective(x):
            check();g=compliance@x-deficit;return float(.5*x@(compliance@x)-x@deficit),g
        result=minimize(objective,np.zeros(m),method='L-BFGS-B',jac=True,bounds=[(0.,None)]*m,
            options={'gtol':GAP_TOLERANCE_MM/10,'ftol':0.,'maxiter':1000,'maxfun':5000,'maxls':50})
        multipliers=np.asarray(result.x);iterations=int(result.nit);status=int(result.status);polished=False
        active=np.flatnonzero(multipliers>0.)
        if len(active):
            # One bounded rank-revealing refinement; no modified constraints.
            values=np.linalg.lstsq(compliance[np.ix_(active,active)],deficit[active],rcond=64*np.finfo(float).eps*len(active))[0]
            candidate=np.zeros(m);candidate[active]=values;gap=compliance@candidate-deficit
            if (np.min(candidate,initial=0.)>=0 and np.min(gap,initial=0.)>=-GAP_TOLERANCE_MM
                    and np.max(np.abs(candidate*gap),initial=0.)<=ENERGY_TOLERANCE_NMM):multipliers=candidate;polished=True
    u=base+response@multipliers;gap=c@u-b;residual=a@u-f-c.T@multipliers
    maximum=float(np.max(np.abs(residual),initial=0.));complementarity=float(np.max(np.abs(multipliers*gap),initial=0.));minimum=float(np.min(gap,initial=0.))
    if (not np.isfinite(u).all() or not np.isfinite(multipliers).all() or np.min(multipliers,initial=0.)<0
            or minimum<-GAP_TOLERANCE_MM or maximum>FORCE_TOLERANCE_N or complementarity>ENERGY_TOLERANCE_NMM):
        raise ValueError('CONTACT_COMPLEMENTARITY_OR_EQUILIBRIUM_REFUSAL')
    check()
    return {'displacement':u,'normalForcesN':multipliers,'gapMm':gap,'maxFreeResidualN':maximum,
        'maxComplementarityNmm':complementarity,'minimumGapMm':minimum,'optimizerIterations':iterations,
        'optimizerStatus':status,'activeSetPolished':polished}
