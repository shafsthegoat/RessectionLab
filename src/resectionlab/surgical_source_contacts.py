"""Phase-neutral declared source geometry; no engine/action/solver admission."""
import numpy as np

from .geometry import GeometryScene,capsule_voxel_indices
from .surgical_contact_witness import Q1ContactWitness
from .surgical_geometry_view import raster_view


class SourceContactGeometry:
    def __init__(self,scene,grid,removed,cells,corner_displacement_mm,*,holds,detached_fragments,check=lambda:None):
        scene.require_cache_source()
        if detached_fragments or any('target' not in hold for hold in holds.values()):
            raise ValueError('SOURCE_CONTACT_RIGID_STATE_UNSUPPORTED')
        if (not isinstance(grid,GeometryScene) or tuple(grid.shape)!=tuple(scene.shape)
                or not np.array_equal(grid.affine,scene.affine) or not np.array_equal(grid._inverse,scene.inverse)):
            raise ValueError('CONTACT_SURVEY_GRID_FRAME')
        self.scene=scene;self.grid=grid;self.check=check
        self.field=Q1ContactWitness(cells,scene.shape,scene.affine,corner_displacement_mm)
        if np.any(scene.labels.ravel()[self.field.cells]==0):raise ValueError('CONTACT_SURVEY_UNOWNED_FIELD')
        self.u=np.array(corner_displacement_mm,copy=True);self.u.flags.writeable=False
        self.lookup={int(cell):row for row,cell in enumerate(self.field.cells)}
        self.removed=set(removed);self.labels=scene.labels.ravel();self.by_label={r['id']:r for r in scene.regions}
        self.occupancy,self.raster_pairs=raster_view(scene,removed,self.field.cells,self.u)

    def grid_cover(self,a,b,radius):
        self.check();ijk=capsule_voxel_indices(self.grid,a,b,radius);self.check()
        return np.ravel_multi_index(ijk.T,self.scene.shape)

    def capsule(self,a,b,radius):
        cover=self.grid_cover(a,b,radius);raster=set(self.occupancy.material_ids(cover))
        result=self.field.capsule(a,b,radius,check=self.check)
        witnessed={int(self.field.cells[i]) for i in result['witnessedRows']}
        uncertain={int(self.field.cells[i]) for i in result['indeterminateRows']}
        points=[{'sourceCell':int(self.field.cells[i]),'pointMm':point,'referencePoint':reference,'distanceMm':distance}
            for i,point,reference,distance in zip(result['witnessedRows'],result['witnessPointsMm'],result['referencePoints'],result['witnessDistancesMm'])]
        outside=sorted({int(i) for i in cover if self.labels[i] and i not in self.lookup and i not in self.removed})
        if outside:
            rest=Q1ContactWitness(np.array(outside,np.int64),self.scene.shape,self.scene.affine,np.zeros((len(outside),8,3)))
            extra=rest.capsule(a,b,radius,check=self.check)
            witnessed.update(int(rest.cells[i]) for i in extra['witnessedRows'])
            uncertain.update(int(rest.cells[i]) for i in extra['indeterminateRows'])
            points.extend({'sourceCell':int(rest.cells[i]),'pointMm':point,'referencePoint':reference,'distanceMm':distance}
                for i,point,reference,distance in zip(extra['witnessedRows'],extra['witnessPointsMm'],extra['referencePoints'],extra['witnessDistancesMm']))
        if witnessed&uncertain:raise ValueError('CONTACT_SURVEY_OWNER_OVERLAP')
        return {'rasterSourceIds':sorted(raster),'witnessedSourceIds':sorted(witnessed),
            'indeterminateSourceIds':sorted(uncertain),'positivePoints':points,
            'outsideFieldCandidateSourceIds':outside,'completeDeclaredSourceCoverage':True,
            'unannotatedAnatomyValidated':False}

    def cusa_order(self,position):
        p=np.asarray(position,float);ids=self.occupancy.material_ids(self.grid_cover(p,p,1.5))
        ids=[i for i in ids if i not in self.removed and self.by_label[int(self.labels[i])]['material'] in ('soft_tissue','tumor')]
        centers={int(i):point for i,point in zip(ids,self.scene.points(ids))}
        for i in ids:
            if i in self.lookup:centers[i]=self.scene.points([i])[0]+self.u[self.lookup[i]].mean(axis=0)
        return sorted(ids,key=lambda i:(float(np.sum((centers[i]-p)**2)),i))
