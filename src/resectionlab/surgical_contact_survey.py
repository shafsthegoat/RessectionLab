"""Conditional endpoint contact descriptions, never public action admission.

Every retained source owner is represented: a displaced local Q1 cell, or its
declared resting affine cube outside the field. No tissue effect is applied.
"""
import copy
import numpy as np

from .surgical_source_contacts import SourceContactGeometry
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


class ConditionalContactSurvey(SourceContactGeometry):
    def __init__(self,scene,grid,frame,cells,corner_displacement_mm,*,check=lambda:None):
        require_supported_context(frame);self.frame=copy.deepcopy(frame)
        super().__init__(scene,grid,frame['removedSourceIds'],cells,corner_displacement_mm,
            holds=frame['holds'],detached_fragments=frame.get('detachedFragments',[]),check=check)

    def cusa_order(self,slot):
        return super().cusa_order(self.frame['slots'][slot]['position'])

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
