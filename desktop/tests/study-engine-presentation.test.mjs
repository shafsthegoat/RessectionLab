import assert from 'node:assert/strict';
import test from 'node:test';
import {recordedCorridorView,recordedMicroscope,recordedOverviewRoute,recordingMatchesStudy,recordingSelectionKey} from '../src/study-engine-presentation.ts';

const recordedApproach={id:'a',name:'Recorded approach',entry:[-10,2,8],target:[-10,3,3],radiusMm:1.5,
  parameterHash:'p',measurementHash:'m'};
const receipt={caseHash:'c',patientId:'one',sourceSnapshotId:'r',targetLayerId:'tumor',recordedApproach};
const route={...recordedApproach,sourceSnapshotId:'r',targetLayerId:'tumor',assessmentStatus:'current'};
const selection={collectionId:'open',patientId:'one',snapshotId:'r',targetLayerId:'tumor',route};

test('source-bound episode checks the current route, measured geometry and source at display time',()=>{
  assert(recordingMatchesStudy(receipt,'c',selection));
  for(const patch of [{id:'b'},{assessmentStatus:'stale'},{parameterHash:'changed'},{measurementHash:'changed'},
    {entry:[0,0,0]},{target:[0,0,0]},{radiusMm:.5},{targetLayerId:'another'},{sourceSnapshotId:'other'}])
    assert.equal(recordingMatchesStudy(receipt,'c',{...selection,route:{...route,...patch}}),false);
  for(const patch of [{snapshotId:'changed'},{patientId:'changed'},{targetLayerId:'other'},{route:null}])
    assert.equal(recordingMatchesStudy(receipt,'c',{...selection,...patch}),false);
  assert.equal(recordingMatchesStudy(receipt,'other',selection),false);
  assert.notEqual(recordingSelectionKey(selection),recordingSelectionKey({...selection,route:{...route,measurementHash:'new'}}));
  assert.notEqual(recordingSelectionKey(selection),recordingSelectionKey({...selection,collectionId:'other'}));
  assert.equal(recordingMatchesStudy({...receipt,recordedApproach:undefined},'c',{...selection,route:null}),true);
});

test('cold overview geometry and initial optical axis derive from the recorded route; saved microscope settings take precedence',()=>{
  const overview=recordedOverviewRoute(recordedApproach);
  assert.deepEqual(overview.entry_mm,route.entry);assert.deepEqual(overview.target_mm,route.target);
  assert.equal(overview.tool.shaft_radius_mm,route.radiusMm);
  const settings=recordedMicroscope(recordedApproach,{});
  assert.deepEqual(settings.microscope.target,route.target.map((n,i)=>n+.65*(route.entry[i]-n)));
  assert.deepEqual(settings.camera.target,settings.microscope.target);
  assert.equal(settings.operativeView,'surgeon');
  const saved={...settings,microscope:{...settings.microscope,angleDeg:11}};
  assert.equal(recordedMicroscope(recordedApproach,saved),saved);
});

test('longitudinal inspection contains both recorded endpoints and clips the camera-facing side',()=>{
  const original={operativeView:'external',camera:{target:[1,2,3],position:[1,2,80],up:[0,1,0]}};
  const section=recordedCorridorView(recordedApproach,original,true);
  const {normal,offset}=section.obliqueClip;
  const distance=p=>normal.reduce((value,n,i)=>value+n*p[i],offset);
  assert(Math.abs(distance(recordedApproach.entry))<1e-9);
  assert(Math.abs(distance(recordedApproach.target))<1e-9);
  assert(distance(section.camera.position)<0);
  assert.equal(section.microscope.inspectionCutaway,true);
  const exterior=recordedCorridorView(recordedApproach,section,false);
  assert.equal(exterior.obliqueClip,undefined);
  assert.equal(exterior.microscope.inspectionCutaway,false);
  assert.deepEqual(exterior.externalCamera,original.camera);
  assert.equal(exterior.microscope.fieldMm,recordedApproach.radiusMm*13);
});
