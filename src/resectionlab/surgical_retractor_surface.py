"""Source-owned exposed Q1 faces for a finite, frictionless planar blade.

Contact quadrature is a numerical surface representation, not observed anatomy
or calibrated pressure. Artificial patch interfaces and retained solid/material
interfaces are never exposed faces. Out-of-image faces remain unknown.
"""
from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy import sparse
from scipy.spatial.transform import Rotation

from .surgical_continuum import CORNERS,SOFT


def _owned(value,dtype=None):
    result=np.array(value,dtype=dtype,copy=True);result.flags.writeable=False;return result


@dataclass(frozen=True)
class SurfaceQuadrature:
    source_ids: np.ndarray
    axes: np.ndarray
    signs: np.ndarray
    point_indices: np.ndarray
    element_nodes: np.ndarray
    weights: np.ndarray
    derivatives: np.ndarray
    orientation: np.ndarray
    reference_area_mm2: np.ndarray
    node_count: int
    unknown_fov_faces: tuple

    def geometry(self,nodes,displacement):
        nodes=np.asarray(nodes,float);u=np.asarray(displacement,float)
        if u.shape==(self.node_count*3,):u=u.reshape(-1,3)
        if nodes.shape!=(self.node_count,3) or u.shape!=nodes.shape or not np.isfinite(nodes).all() or not np.isfinite(u).all():
            raise ValueError('RETRACTOR_SURFACE_FIELD')
        corners=(nodes+u)[self.element_nodes]
        points=np.einsum('pa,pai->pi',self.weights,corners)
        tangent=np.einsum('pja,pai->pji',self.derivatives,corners)
        normal=np.cross(tangent[:,0],tangent[:,1])*self.orientation[:,None]
        length=np.linalg.norm(normal,axis=1)
        if np.any(length<=1e-12) or not np.isfinite(points).all() or not np.isfinite(normal).all():
            raise ValueError('RETRACTOR_SURFACE_DEGENERATE_FACE')
        return points,normal/length[:,None]

    def rows(self,normal,point_indices):
        indices=np.asarray(point_indices,int);normal=np.asarray(normal,float)
        if (normal.shape!=(len(indices),3) or not np.isfinite(normal).all() or np.any(indices<0) or np.any(indices>=len(self.source_ids))
                or np.max(np.abs(np.linalg.norm(normal,axis=1)-1),initial=0.)>1e-10):
            raise ValueError('RETRACTOR_SURFACE_ROWS')
        values=self.weights[indices,:,None]*normal[:,None,:]
        columns=(3*self.element_nodes[indices,:,None]+np.arange(3)).reshape(len(indices),24)
        rows=np.repeat(np.arange(len(indices)),24)
        result=sparse.coo_matrix((values.ravel(),(rows,columns.ravel())),shape=(len(indices),3*self.node_count)).tocsr()
        result.eliminate_zeros();return result


def exposed_quadrature(scene,mesh,removed=(),cut_faces=(),*,order=2):
    """Integrate declared free/cut faces, including source-sector node aliases."""
    scene.require_cache_source()
    if type(order) is not int or not 1<=order<=4:raise ValueError('RETRACTOR_SURFACE_QUADRATURE_ORDER')
    cells=tuple(mesh.cells);en=np.asarray(mesh.element_nodes);nodes=np.asarray(mesh.nodes,float)
    if (cells!=tuple(sorted(set(cells))) or en.shape!=(len(cells),8) or en.dtype.kind not in 'iu'
            or nodes.ndim!=2 or nodes.shape[1]!=3 or not np.isfinite(nodes).all() or np.any(en<0) or np.any(en>=len(nodes))
            or any(isinstance(c,(bool,np.bool_)) or not isinstance(c,(int,np.integer)) or not 0<=c<scene.labels.size for c in cells)):
        raise ValueError('RETRACTOR_SURFACE_MESH')
    gone=set(map(int,removed));cuts={tuple(sorted(map(int,e))) for e in cut_faces};labels=scene.labels.ravel()
    if gone.intersection(cells) or any(c<0 or c>=labels.size for c in gone):raise ValueError('RETRACTOR_SURFACE_REMOVED_OWNERS')
    by_label={r['id']:r for r in scene.regions};A=scene.affine[:3,:3];det=float(np.linalg.det(A))
    if abs(det)<1e-9:raise ValueError('RETRACTOR_SURFACE_AFFINE')
    xi,w=np.polynomial.legendre.leggauss(order);xi=(xi+1)/2;w=w/2
    source=[];axes=[];signs=[];pi=[];element=[];weights=[];derivatives=[];orientation=[];areas=[];unknown=[]
    for row,c in enumerate(cells):
        if c in gone or int(labels[c]) not in by_label or by_label[int(labels[c])]['material'] not in SOFT:
            raise ValueError('RETRACTOR_SURFACE_SOURCE_MATERIAL')
        ijk=np.array(np.unravel_index(c,scene.shape));expected=(ijk+CORNERS-.5)@A.T+scene.affine[:3,3]
        if np.max(np.abs(nodes[en[row]]-expected),initial=0.)>1e-10:raise ValueError('RETRACTOR_SURFACE_CORNER_OWNERSHIP')
        for axis in range(3):
            others=[k for k in range(3) if k!=axis];area=float(np.linalg.norm(np.cross(A[:,others[0]],A[:,others[1]])))
            for sign in (-1,1):
                neighbor=ijk.copy();neighbor[axis]+=sign
                if np.any(neighbor<0) or np.any(neighbor>=scene.shape):unknown.append((c,axis,sign));continue
                other=int(np.ravel_multi_index(neighbor,scene.shape));label=int(labels[other])
                # A retained brain/tumor, vessel, membrane, bone or unknown
                # neighbor is a material interface, not a free blade surface.
                exposed=other in gone or label==0 or tuple(sorted((c,other))) in cuts
                if not exposed:continue
                for point,(j,k) in enumerate(product(range(order),repeat=2)):
                    ref=np.empty(3);ref[axis]=(sign+1)/2;ref[others]=[xi[j],xi[k]]
                    factors=np.where(CORNERS,ref,1-ref);shape=factors.prod(axis=1);d=[]
                    for a in others:
                        remaining=[b for b in range(3) if b!=a]
                        d.append((2*CORNERS[:,a]-1)*factors[:,remaining].prod(axis=1))
                    source.append(c);axes.append(axis);signs.append(sign);pi.append(point);element.append(en[row]);weights.append(shape);derivatives.append(d)
                    orientation.append(sign*(-1)**axis*np.sign(det));areas.append(area*w[j]*w[k])
    count=len(source)
    return SurfaceQuadrature(_owned(source,np.int64),_owned(axes,np.int8),_owned(signs,np.int8),_owned(pi,np.int16),
        _owned(np.asarray(element).reshape(count,8),np.int64),_owned(np.asarray(weights).reshape(count,8)),
        _owned(np.asarray(derivatives).reshape(count,2,8)),_owned(orientation),_owned(areas),len(nodes),tuple(unknown))


@dataclass(frozen=True)
class PlanarBlade:
    position: np.ndarray
    matrix: np.ndarray
    half_width_mm: float
    depth_mm: float
    half_thickness_mm: float=.4

    @classmethod
    def create(cls,position,quaternion,half_width_mm,depth_mm,half_thickness_mm=.4):
        p=np.asarray(position,float);q=np.asarray(quaternion,float)
        if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all() or np.linalg.norm(q)<1e-12:
            raise ValueError('RETRACTOR_BLADE_POSE')
        for value in (half_width_mm,depth_mm,half_thickness_mm):
            if isinstance(value,(bool,np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value<=0:raise ValueError('RETRACTOR_BLADE_SIZE')
        return cls(_owned(p),_owned(Rotation.from_quat(q).as_matrix()),float(half_width_mm),float(depth_mm),float(half_thickness_mm))

    def local(self,points):return (np.asarray(points,float)-self.position)@self.matrix

    def candidates(self,surface,nodes,displacement,*,tolerance_mm=1e-9):
        if not np.isscalar(tolerance_mm) or not np.isfinite(tolerance_mm) or not 0<=tolerance_mm<=1e-9:raise ValueError('RETRACTOR_GEOMETRY_TOLERANCE')
        points,outward=surface.geometry(nodes,displacement);local=self.local(points)
        side=np.where(local[:,1]>=0,1.,-1.);normals=side[:,None]*self.matrix[:,1]
        footprint=(np.abs(local[:,0])<=self.half_width_mm+tolerance_mm)&(local[:,2]>=-self.depth_mm-tolerance_mm)&(local[:,2]<=tolerance_mm)
        facing=np.einsum('pi,pi->p',outward,normals)<-1e-10
        ids=np.flatnonzero(footprint&facing);gap=np.abs(local[ids,1])-self.half_thickness_mm
        return {'pointIndices':ids,'side':side[ids],'normal':normals[ids],'gapMm':gap,
            'positionsMm':points[ids],'sourceIds':surface.source_ids[ids],
            'scope':'Sampled exposed-face flat-blade contact; finite edge entry and continuous penetration need separate checks'}
