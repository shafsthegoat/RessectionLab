"""Bounded conditional geometry witnesses for displaced source-owned Q1 cells.

Q1 images lie in the convex hull of their eight displaced corners. The hull's
AABB can therefore exclude contact; overlap alone cannot establish contact.
Explicit reference-lattice material points give positive numerical witnesses.
No engine action, material effect, solver admission or clinical claim occurs.
"""
from itertools import product

import numpy as np

from .independent_geometry_batch import segment_box_distances_sq

GEOMETRY_TOLERANCE_MM=1e-9
CORNERS=np.array(list(product((0.,1.),repeat=3)))
REFERENCE_POINTS=np.array(list(product((0.,.5,1.),repeat=3)))
WEIGHTS=np.prod(np.where(CORNERS[None,:,:]>0,REFERENCE_POINTS[:,None,:],1-REFERENCE_POINTS[:,None,:]),axis=2)


class Q1ContactWitness:
    def __init__(self,cells,shape,affine,corner_displacement_mm):
        cells=np.asarray(cells);shape=np.asarray(shape);affine=np.asarray(affine,float);u=np.asarray(corner_displacement_mm,float)
        if (cells.ndim!=1 or cells.dtype.kind not in 'iu' or not 0<len(cells)<=16000 or len(np.unique(cells))!=len(cells)
                or shape.shape!=(3,) or shape.dtype.kind not in 'iu' or np.any(shape<=0)
                or np.any(cells<0) or np.any(cells>=np.prod(shape))):raise ValueError('WITNESS_SOURCE_IDS')
        if (affine.shape!=(4,4) or not np.isfinite(affine).all() or not np.array_equal(affine[3],[0.,0.,0.,1.])
                or abs(np.linalg.det(affine[:3,:3]))<1e-9):raise ValueError('WITNESS_AFFINE')
        if u.shape!=(len(cells),8,3) or not np.isfinite(u).all():raise ValueError('WITNESS_CORNER_FIELD')
        ijk=np.column_stack(np.unravel_index(cells,tuple(shape)))
        with np.errstate(over='raise',invalid='raise'):
            try:corners=(ijk[:,None,:]+CORNERS-.5)@affine[:3,:3].T+affine[:3,3]+u
            except FloatingPointError as error:raise ValueError('WITNESS_GEOMETRY_OVERFLOW') from error
        if not np.isfinite(corners).all():raise ValueError('WITNESS_GEOMETRY_OVERFLOW')
        self.cells=cells.astype(np.int64,copy=True);self.corners=corners
        self.shape=shape.astype(np.int64,copy=True);self.affine=affine.copy()
        self.lower=np.min(corners,axis=1);self.upper=np.max(corners,axis=1)
        for value in (self.cells,self.corners,self.lower,self.upper,self.shape,self.affine):value.flags.writeable=False

    def capsule(self,start,end,radius_mm,*,check=lambda:None,batch_size=256):
        start,end=(np.array(value,dtype=float,copy=True) for value in (start,end))
        if any(value.shape!=(3,) or not np.isfinite(value).all() for value in (start,end)):
            raise ValueError('WITNESS_CAPSULE')
        if (isinstance(radius_mm,(bool,np.bool_)) or not np.isscalar(radius_mm)
                or not np.isfinite(radius_mm) or radius_mm<0):raise ValueError('WITNESS_CAPSULE')
        if type(batch_size) is not int or not 1<=batch_size<=4096:raise ValueError('WITNESS_BATCH')
        check();radius=float(radius_mm);tol=GEOMETRY_TOLERANCE_MM
        with np.errstate(over='raise',invalid='raise'):
            try:threshold=float(np.float64(radius+tol)*np.float64(radius+tol))
            except FloatingPointError as error:raise ValueError('WITNESS_DISTANCE_OVERFLOW') from error
        distances=segment_box_distances_sq(start,end,self.lower,self.upper,batch_size=batch_size,
            cancelled=lambda:(check(),False)[1])
        possible=np.flatnonzero(distances<=threshold)
        witnessed=[];uncertain=[];points=[];minimum=[];reference=[];direction=end-start;length=float(direction@direction)
        for begin in range(0,len(possible),batch_size):
            check();rows=possible[begin:begin+batch_size]
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                try:
                    material=np.einsum('pa,eai->epi',WEIGHTS,self.corners[rows],optimize=True)
                    offset=material-start
                    t=np.clip(np.sum(offset*direction,axis=2)/length,0.,1.) if length else np.zeros(offset.shape[:2])
                    d=np.linalg.norm(offset-t[:,:,None]*direction,axis=2);chosen=np.argmin(d,axis=1)
                    best=d[np.arange(len(rows)),chosen]
                except FloatingPointError as error:raise ValueError('WITNESS_DISTANCE_OVERFLOW') from error
            if not np.isfinite(best).all():raise ValueError('WITNESS_DISTANCE_OVERFLOW')
            for local,(row,point,distance) in enumerate(zip(rows,chosen,best)):
                if radius>tol and distance<radius-tol:
                    witnessed.append(int(row));points.append(material[local,point].tolist())
                    minimum.append(float(distance));reference.append(REFERENCE_POINTS[point].tolist())
                else:uncertain.append(int(row))
        check()
        return {'witnessedRows':np.asarray(witnessed,dtype=np.int64),'indeterminateRows':np.asarray(uncertain,dtype=np.int64),
            'separatedRows':np.flatnonzero(distances>threshold),'witnessPointsMm':points,
            'referencePoints':reference,'witnessDistancesMm':minimum,'toleranceMm':tol,
            'scope':'Numerical Q1 point-contact / enclosing-AABB separation witnesses; overlap without a point witness is indeterminate'}

    def obstacle_points(self,scene,removed,*,check=lambda:None,batch_size=256):
        """Strict interior point witnesses in declared blocking source cells.

        No statement is made about unsampled intersections or unannotated
        anatomy. Source cube boundaries are kept indeterminate at 1e-9 mm.
        """
        scene.require_cache_source()
        if not np.array_equal(scene.shape,self.shape) or not np.array_equal(scene.affine,self.affine):raise ValueError('WITNESS_SCENE_FRAME')
        if type(batch_size) is not int or not 1<=batch_size<=4096:raise ValueError('WITNESS_BATCH')
        blocked={r['id'] for r in scene.regions if r['material'] in ('bone','vessel','unknown')}
        removed_array=np.asarray(removed)
        if removed_array.size==0:removed_array=np.empty(0,dtype=np.int64)
        if (removed_array.ndim!=1 or removed_array.dtype.kind not in 'iu' or np.any(removed_array<0)
                or np.any(removed_array>=scene.labels.size)):raise ValueError('WITNESS_REMOVED_SOURCE_IDS')
        removed=set(map(int,removed_array))
        if removed.intersection(map(int,self.cells)):raise ValueError('WITNESS_REMOVED_SOURCE_GEOMETRY')
        witnesses=[];outside=[]
        # A world-space tolerance conservatively maps to each affine grid axis.
        margin=GEOMETRY_TOLERANCE_MM*np.linalg.norm(scene.inverse[:3,:3],axis=1)
        for start in range(0,len(self.cells),batch_size):
            check();points=np.einsum('pa,eai->epi',WEIGHTS,self.corners[start:start+batch_size],optimize=True)
            voxel=points@scene.inverse[:3,:3].T+scene.inverse[:3,3]
            out=np.any((voxel<-.5-margin)|(voxel>np.asarray(scene.shape)-.5+margin),axis=(1,2))
            outside.extend(self.cells[start:start+len(points)][out].tolist())
            index=np.floor(voxel+.5).astype(np.int64)
            valid=np.all((index>=0)&(index<np.asarray(scene.shape))&(np.abs(voxel-index)<.5-margin),axis=2)
            rows,pi=np.nonzero(valid)
            if len(rows) and blocked:
                cell=np.ravel_multi_index(index[rows,pi].T,scene.shape)
                hit=np.isin(scene.labels.ravel()[cell],list(blocked))&~np.isin(cell,list(removed))
                rows,pi,cell=rows[hit],pi[hit],cell[hit]
                _,first=np.unique(rows,return_index=True)
                for k in first:
                    witnesses.append({'sourceCell':int(self.cells[start+rows[k]]),'blockingSourceCell':int(cell[k]),
                        'referencePoint':REFERENCE_POINTS[pi[k]].tolist(),'pointMm':points[rows[k],pi[k]].tolist()})
        check();return {'blockingPointWitnesses':witnesses,'outsideGridSourceCells':sorted(set(outside)),
            'unknownAnatomyValidated':False,'scope':'Declared labels and sampled strict interior material points only'}
