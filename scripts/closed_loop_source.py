"""Source-bound seed loader; no solve or state-driven outcome is prescribed."""
from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
import gzip
import json

import numpy as np

from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_scene import from_case
from resectionlab.surgical_environment import standardize
from resectionlab.geometry import GeometryScene
from resectionlab.surgical_continuum_topology import SourceDomain
from resectionlab.surgical_continuum_program import field_hash
from resectionlab.surgical_pre_effect_witness import pre_event_fields
from resectionlab import surgical_cusa_v3 as cusa
from survey_continuum_pre_effects import require_join
from profile_continuum_exterior import ROOT,bound,sha,load_geometry,free_partition


def seed(p,inputs,check):
    def frozen(entry):
        path=bound(entry);inputs[str(path.relative_to(ROOT))]=entry['sha256'];return path
    def nested(directory,name,digest):
        path=(directory/name).resolve()
        if path.parent!=directory.resolve():raise ValueError('CLOSED_LOOP_SEED_PATH')
        return frozen({'path':str(path.relative_to(ROOT)),'sha256':digest})
    check();full_path=frozen(p['fullProgramReport']);full=json.loads(full_path.read_text())
    if full['status']!='completed' or not full['deterministicAllStepReplayExact']:raise ValueError('Missing complete prescribed field qualification')
    samples=json.loads(nested(full_path.parent,'samples.json',full['samplesSha256']).read_text())
    metrics=json.loads(gzip.decompress(nested(full_path.parent,full['metricsFile'],full['metricsSha256']).read_bytes()))
    name=p['seed']['branch'];step=p['seed']['coupledStep'];row=next(r for r in metrics if (r['branch'],r['step'])==(name,step))
    sample=next(s for s in samples if (s['branch'],s['step'])==(name,step));entries={s['tissueState']:s['geometry'] for s in samples}
    capture_path=frozen(p['preEffectCaptureReport']);capture=json.loads(capture_path.read_text())
    if capture['status']!='completed' or not capture['nativeAllPredictionsExact']:raise ValueError('Missing native pre-effect oracle')
    frames=json.loads(gzip.decompress(nested(capture_path.parent,capture['framesFile'],capture['framesSha256']).read_bytes()))
    frame=next(f for f in frames if (f['branch'],f['coupledStep'])==(name,step))
    native=json.loads(gzip.decompress(frozen(p['nativeInput']).read_bytes()));prefix=sum(t['actionIndex']<p['commonActions'] for t in native['ticks'])
    require_join(frame,row,sample,native,prefix)
    survey_path=frozen(p['preEffectSurveyReport']);survey=json.loads(survey_path.read_text())
    event=next(e for e in survey['events'] if (e['branch'],e['step'])==(name,step))
    witness=json.loads(nested(survey_path.parent,event['file'],event['sha256']).read_text())
    if survey['status']!='completed' or witness['comparison']['allCusaDecisionsExact']:raise ValueError('Seed must preserve the observed conditional discrepancy')
    root,summary,partition=load_geometry(p);_,core=free_partition(root,partition)
    for entry in [*p['geometry'].values(),p['partition']]:frozen(entry)
    domain=SourceDomain(root,core)
    data={}
    for stage in (row['tissueState']-1,row['tissueState']):
        with np.load(frozen(entries[stage]),allow_pickle=False) as f:
            data[stage]={key:f[key] for key in ('cells','shape','nodes','elementNodes','affine','freeDofs','muPa','nu')}
    old=data[row['tissueState']-1];current=data[row['tissueState']]
    stage=domain.after(sorted(set(frame['removedSourceIds'])&set(map(int,root['cells']))))
    if any(not np.array_equal(stage['geometry'][key],old[key]) for key in old):raise ValueError('Seed current domain graph/support/material mismatch')
    with np.load(nested(full_path.parent,sample['file'],sample['sha256']),allow_pickle=False) as f:
        payload={key:f[key] for key in ('freeIndices','beforeDisplacement','beforeMemories','orphanOldNodeIndices','orphanOldDisplacement','orphanOldMemories')}
    corners,memories=pre_event_fields(current,old,payload,row);en=old['elementNodes']
    u=np.full(old['nodes'].shape,np.nan);z=np.full((len(memories),len(u),3),np.nan)
    u[en.ravel()]=corners.reshape((-1,3));z[:,en.ravel()]=memories.reshape((len(z),-1,3))
    if (not np.array_equal(u[en].view(np.uint64),corners.view(np.uint64))
            or not np.array_equal(z[:,en].view(np.uint64),memories.view(np.uint64))
            or field_hash(u.ravel(),z.reshape((len(z),-1)))!=row['previousArrayHash']):raise ValueError('Seed full old history hash/owner mismatch')
    case,artifacts,_=load_workspace(frozen(p['nativeCase']));record=artifacts['surgical_toolbox']
    scene=standardize(from_case(case,record['mapping']),physics_profile=record['physicsProfile'])
    if scene.identity!=p['sourceSceneId']:raise ValueError('Seed source scene mismatch')
    with np.load(frozen(p['geometry']['reference']),allow_pickle=False) as f:alpha,times,common=(f[k] for k in ('alpha','times','commonCells'))
    h=frame['mechanics']['fixedStepSeconds']
    if h!=p['stepSeconds'] or frame['mechanics']['bodyForce']!='zero':raise ValueError('Seed clock/body mismatch')
    initial={'displacement':u.ravel(),'memories':z.reshape((len(z),-1)),'removedSourceIds':frame['removedSourceIds'],
        'holds':frame['holds'],'slots':frame['slots'],'parametersBySlot':frame['parametersBySlot'],'cusaBefore':frame['cusaBefore'],
        'elapsedSeconds':frame['elapsedSecondsBefore'],'remainderSeconds':frame['nativeRemainderBeforeSeconds'],
        'toolMechanicsRevision':frame['toolMechanicsRevision'],'protectedSourceIds':frame['protectedSourceIds'],
        'guardReport':frame['guardReport'],'cutFaces':frame['cutFaces'],'detachedFragments':frame['detachedFragments'],'bodyForce':'zero'}
    commands=[{'slot':frame['slot'],'operation':'apply','effectiveActive':True,'durationSeconds':p['tickSeconds'],
        'pose':frame['requestedPose']} for _ in range(p['cusaTicks'])]
    commands+=[{'slot':'left','operation':'release','effectiveActive':False,'durationSeconds':p['tickSeconds'],'pose':'current'}]
    commands+=[{'slot':frame['slot'],'operation':'inspect','effectiveActive':False,'durationSeconds':p['tickSeconds'],'pose':'current'} for _ in range(p['relaxTicks'])]
    if commands!=p['commands']:raise ValueError('Seed intervention differs from frozen physical microticks')
    pending=sum((Fraction(*value) for value in initial['cusaBefore']['progressExact'].values()),Fraction(0))
    params=initial['parametersBySlot'][frame['slot']];profile=cusa.profile(SimpleNamespace(scene=scene,state={'toolMechanicsRevision':2}))
    budget=profile['capacityMm3PerSecond']*params['power']*params['duty']*p['tickSeconds']
    removal_bound=int((pending+p['cusaTicks']*Fraction.from_float(budget))//Fraction.from_float(float(scene.voxel_mm3)))
    if removal_bound!=p['maximumNewRemovals']:raise ValueError('Exact initial-pending/command capacity bound changed')
    metadata={'branch':name,'coupledStep':step,'previousArrayHash':row['previousArrayHash'],'nativeSeedMask':stage['removedSourceIds'],
        'rootCells':len(root['cells']),'rootTotalDofs':root['nodes'].size,'rootFreeDofs':int(root['freeDofs'].sum()),
        'seedCells':len(old['cells']),'seedTotalDofs':old['nodes'].size,'seedFreeDofs':int(old['freeDofs'].sum()),
        'totalRemovedSourceOwnersAtSeed':len(frame['removedSourceIds']),'elapsedSeconds':initial['elapsedSeconds'],
        'remainderSeconds':initial['remainderSeconds'],'commands':commands,'nativeOrderAtSeed':witness['predictions']['native8']['orderedSourceIds'],
        'maximumNewRemovalsFromExactCapacity':removal_bound,
        'coupledOrderAtSeed':witness['predictions']['coupled20']['orderedSourceIds'],
        'initialInterpretation':'Declared coupled pre-event fields plus captured native pending-work ledger; not an original20mm episode'}
    check();return scene,GeometryScene(np.zeros(scene.shape,bool),scene.affine),domain,initial,alpha,times,h,common,commands,metadata
