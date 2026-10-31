"""Independent affine Q1 export quadrature, with no app imports or solve.

Functions preserved from frozen export checker SHA256
68e049b54508c46aa189337ed00b7c1acb5fd8f1b923f5e1cc9b5ad0e433a355.
Eight Gauss points integrate; eight additional corners guard gradients only.
"""
from itertools import product
import numpy as np
CORNERS=np.array(list(product((0.,1.),repeat=3)))

def fail(condition,message):
    if not condition:raise ValueError(message)


def finite(value,shape,name):
    a=np.asarray(value)
    fail(a.shape==shape and a.dtype.kind in 'fiu' and np.isfinite(a).all(),'Invalid finite field: '+name)
    return np.asarray(a,float)


def scalar(value,name):return float(finite(value,(),name))


def near(a,b,tolerance,name):
    a,b=np.asarray(a,float),np.asarray(b,float)
    fail(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid comparison: '+name)
    error=float(np.max(np.abs(a-b),initial=0.));fail(error<=tolerance,'Comparison exceeds tolerance: '+name+' '+str(error));return error


def quadrature(affine):
    jac=affine[:3,:3]/2.;fail(np.linalg.det(jac)!=0,'Singular affine')
    inverse=np.linalg.inv(jac);weight=abs(np.linalg.det(jac));signs=2*CORNERS-1.;gradients=[]
    for point in product((-1/np.sqrt(3),1/np.sqrt(3)),repeat=3):
        derivative=np.empty((8,3))
        for j in range(3):
            other=[i for i in range(3) if i!=j]
            derivative[:,j]=signs[:,j]*np.prod(1+signs[:,other]*np.asarray(point)[other],axis=1)/8.
        gradients.append(derivative@inverse)
    return gradients,weight


def elastic_fields(corners,mu,nu,gradients,weight):
    """Per-cell integrated internal nodal force and elastic strain energy."""
    force=np.zeros_like(corners);energy=np.zeros(len(corners));m=mu*1e-6;lam=2*m*nu/(1-2*nu)
    for gradient in gradients:
        gu=np.einsum('eai,aj->eij',corners,gradient);strain=(gu+gu.transpose(0,2,1))/2.;trace=np.trace(strain,axis1=1,axis2=2)
        stress=2*m[:,None,None]*strain+lam[:,None,None]*trace[:,None,None]*np.eye(3)
        force+=weight*np.einsum('eij,aj->eai',stress,gradient)
        energy+=weight*(m*np.sum(strain*strain,axis=(1,2))+.5*lam*trace*trace)
    return force,energy


def field_probe_metrics(g,u,z):
    en=g['elementNodes'];gradients,weight=quadrature(g['affine']);mu=g['muPa'];nu=g['nu'];lam=2*mu*nu/(1-2*nu)
    local=u[en];strain_sum=np.zeros(len(en));stress_sum=np.zeros(len(en));maximum=0.
    # The eight Gauss gradients integrate rates. Corners are guard points only.
    for grad in gradients:
        gu=np.einsum('eai,aj->eij',local,grad);eps=(gu+gu.transpose(0,2,1))/2
        maximum=max(maximum,float(np.linalg.norm(gu,axis=(1,2)).max()))
        stress=2*mu[:,None,None]*eps+lam[:,None,None]*np.trace(eps,axis1=1,axis2=2)[:,None,None]*np.eye(3)
        for a,memory in zip(g['alpha'],z):
            gg=np.einsum('eai,aj->eij',(u-memory)[en],grad);ee=(gg+gg.transpose(0,2,1))/2
            stress+=a*(2*mu[:,None,None]*ee+lam[:,None,None]*np.trace(ee,axis1=1,axis2=2)[:,None,None]*np.eye(3))
        strain_sum+=weight*np.sum(eps*eps,axis=(1,2));stress_sum+=weight*np.sum(stress*stress,axis=(1,2))
    signs=2*CORNERS-1;inv=np.linalg.inv(g['affine'][:3,:3]/2)
    for point in product((-1.,1.),repeat=3):
        derivative=np.empty((8,3))
        for axis in range(3):
            other=[i for i in range(3) if i!=axis]
            derivative[:,axis]=signs[:,axis]*np.prod(1+signs[:,other]*np.asarray(point)[other],axis=1)/8
        gu=np.einsum('eai,aj->eij',local,derivative@inv)
        maximum=max(maximum,float(np.linalg.norm(gu,axis=(1,2)).max()))
    rows=g['commonRows'];rates=np.array([strain_sum[rows].sum(),stress_sum[rows].sum()]);displacement=float(np.linalg.norm(local.mean(axis=1),axis=1).max())
    return rates,maximum,displacement

