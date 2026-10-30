"""Conditional endpoint contact descriptions, never public action admission.

Every retained source owner is represented: a displaced local Q1 cell, or its
declared resting affine cube outside the field. No tissue effect is applied.
"""
import copy
import numpy as np

from .geometry import GeometryScene,capsule_voxel_indices
from .surgical_contact_witness import Q1ContactWitness
from .surgical_geometry_view import raster_view
from .surgical_motion import envelopes,pair_conflict
from .surgical_tools import TOOLS


def require_supported_context(frame):
    if frame.get('detachedFragments'):raise ValueError('CONTACT_SURVEY_FRAGMENT_UNSUPPORTED')
    if any('target' not in hold for hold in frame['holds'].values()):
        raise ValueError('CONTACT_SURVEY_RIGID_HOLD_UNSUPPORTED')
    if not frame.get('mechanics') or frame['withinTick']<0 or frame['withinTick']>=frame['mechanics']['nativeFixedSteps']:
        raise ValueError('CONTACT_SURVEY_PHASE')
    if frame['atCommittedTickEnd']!=(frame['withinTick']+1==frame['mechanics']['nativeFixedSteps']):
        raise ValueError('CONTACT_SURVEY_PHASE')


def require_sample_join(frame,sample,row):
    if any(frame[a]!=sample[b] for a,b in [('branch','branch'),('coupledStep','step'),
            ('sampleFieldsFile','file'),('sampleFieldsSha256','sha256'),('tissueState','tissueState')]):
        raise ValueError('CONTACT_SURVEY_SAMPLE_JOIN')
    if any(frame[a]!=row[b] for a,b in [('branch','branch'),('coupledStep','step'),('withinTick','withinTick'),
            ('fixedStepSeconds','h'),('actionIndex','actionIndex'),('operation','operation'),('slot','slot'),
            ('tissueState','tissueState'),('sourceOperatorIdentity','sourceOperatorIdentity')]):
        raise ValueError('CONTACT_SURVEY_PHASE_JOIN')
    require_supported_context(frame)


class ConditionalContactSurvey:
    def __init__(self,scene,grid,frame,cells,corner_displacement_mm,*,check=lambda:None):
        require_supported_context(frame);scene.require_cache_source()
        if (not isinstance(grid,GeometryScene) or tuple(grid.shape)!=tuple(scene.shape)
                or not np.array_equal(grid.affine,scene.affine) or not np.array_equal(grid._inverse,scene.inverse)):
            raise ValueError('CONTACT_SURVEY_GRID_FRAME')
        frame=copy.deepcopy(frame);self.scene=scene;self.grid=grid;self.frame=frame;self.check=check
        self.field=Q1ContactWitness(cells,scene.shape,scene.affine,corner_displacement_mm)
        if np.any(scene.labels.ravel()[self.field.cells]==0):raise ValueError('CONTACT_SURVEY_UNOWNED_FIELD')
        self.u=np.array(corner_displacement_mm,copy=True);self.u.flags.writeable=False
        self.lookup={int(cell):row for row,cell in enumerate(self.field.cells)}
        self.removed=set(frame['removedSourceIds']);self.labels=scene.labels.ravel()
        self.by_label={r['id']:r for r in scene.regions}
        self.occupancy,self.raster_pairs=raster_view(scene,frame['removedSourceIds'],self.field.cells,self.u)

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
        # Outside the declared local field, retained source cubes remain at rest.
        # Grid cover bounds the candidates; its overlap alone is never a positive.
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

    def cusa_order(self,slot):
        p=np.asarray(self.frame['slots'][slot]['position'],float)
        ids=self.occupancy.material_ids(self.grid_cover(p,p,1.5))
        ids=[i for i in ids if i not in self.removed and self.by_label[int(self.labels[i])]['material'] in ('soft_tissue','tumor')]
        centers={int(i):point for i,point in zip(ids,self.scene.points(ids))}
        for i in ids:
            if i in self.lookup:centers[i]=self.scene.points([i])[0]+self.u[self.lookup[i]].mean(axis=0)
        return sorted(ids,key=lambda i:(float(np.sum((centers[i]-p)**2)),i))

    def run(self):
        self.check();capsules=[];all_envelopes={};protected=set(self.frame['protectedSourceIds'])
        for key,slot in self.frame['slots'].items():
            tool=TOOLS[slot['instrument']];held=set(self.frame['holds'].get(key,{}).get('cells',[]))
            all_envelopes[key]=envelopes(tool,self.frame['parametersBySlot'][key],slot['position'],slot['quaternion'])
            for index,(a,b,radius,part) in enumerate(all_envelopes[key]):
                contact=self.capsule(a,b,radius);positive=set(contact['witnessedSourceIds']);unknown=set(contact['indeterminateSourceIds'])
                raster=set(contact['rasterSourceIds'])
                row={'slot':key,'instrument':slot['instrument'],'envelopeIndex':index,'part':part,
                    'startMm':a.tolist(),'endMm':b.tolist(),'radiusMm':radius,**contact,
                    'protectedWitnessedSourceIds':sorted(positive&protected),'protectedIndeterminateSourceIds':sorted(unknown&protected)}
                if part=='shaft':
                    row.update(ownHeldExemptionSourceIds=sorted(held),forbiddenWitnessedSourceIds=sorted(positive-held),
                        forbiddenIndeterminateSourceIds=sorted(unknown-held),rasterForbiddenSourceIds=sorted(raster-held))
                else:
                    compatible={i for i in positive if self.by_label[int(self.labels[i])]['material'] in tool['materials']}
                    row.update(ownHeldWitnessedSourceIds=sorted(positive&held),compatibleWitnessedSourceIds=sorted(compatible-held),
                        otherMaterialWitnessedSourceIds=sorted(positive-compatible-held),workingContactLegalityValidated=False)
                capsules.append(row)
        keys=list(all_envelopes);pairs=[{'slots':[a,b],'endpointEnvelopeConflict':pair_conflict(all_envelopes[a],all_envelopes[b])}
            for i,a in enumerate(keys) for b in keys[i+1:]]
        return {'capsules':capsules,'endpointToolPairs':pairs,'rasterMovedSourceCells':sum(a!=b for a,b in self.raster_pairs),
            'instantaneousPostFieldCusaCandidateOrders':{key:self.cusa_order(key) for key,slot in self.frame['slots'].items() if slot['instrument']=='cusa'},
            'tissueObstaclePointSurvey':self.field.obstacle_points(self.scene,sorted(self.removed),check=self.check),
            'completeActionChecks':False,'cusaEffectValidated':False,'protectedConfigurationSourceIds':sorted(protected),
            'unconfiguredCriticalAnatomyValidated':False,'largerDomainPublicActionsAdmitted':False}
