import assert from 'node:assert/strict';
import {test} from 'node:test';
import {checkedPassiveRecordedTools} from './recordedTool.ts';

// Generated software fixture: no patient image, physical measurement or surgery.
const hash = c => `sha256:${c.repeat(64)}`;
const volume = {caseHash:hash('a'),frame:'RAS+',shape:[8,8,8],
  affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]};
const phases = ['ready','held','inspected','withdrawn','both_withdrawn','stopped'];
function tool(frameIndex,index) {
  const held = index === 0 ? [1,2].includes(frameIndex) : frameIndex < 4;
  return {scope:'generated-persistent-passive-episode',caseHash:volume.caseHash,
    episodeId:hash('b'),frameIndex,toolId:index === 0 ? 'moving' : 'other',
    mode:index === 0 ? 'probe' : 'aspirate',
    tipRasMm:[2+index*2,3,held ? 4 : 2],axis:[0,0,1],
    workingLengthMm:4,tipLengthMm:1,shaftRadiusMm:.25,tipRadiusMm:.5,
    retainedAtDepth:held,physicalAbsenceClaim:false,
    poseStatus:held?'held_in_declared_void':'access_pose_recorded',
    activation:index === 0 && held?'passive_only':'off'};
}
function frame(frameIndex) {
  return {scope:'generated-persistent-passive-episode',caseHash:volume.caseHash,
    episodeId:hash('b'),frameIndex,stepCount:5,shape:volume.shape,
    affine:volume.affine,phase:phases[frameIndex],elapsedGeneratedSeconds:frameIndex,
    inspectionStatus:frameIndex<2?'not_acquired':'verified',
    event:frameIndex===2?{kind:'generated_inspection',status:'verified'}:
      frameIndex===5?{kind:'stop',reason:'declared_stop_without_injury_claim'}:null,
    recordedTools:[tool(frameIndex,0),tool(frameIndex,1)],
    recordedEpisodeBounds:{caseHash:volume.caseHash,episodeId:hash('b'),
      boundsRasMm:[[-3,-3,-5],[9,9,9]]},
    primaryPairClearanceChecked:true,independentTissueMotionChecked:true};
}

test('all six generated paired frames keep both complete capsule poses, including access poses',()=>{
  for(let step=0;step<6;step++) {
    const result=checkedPassiveRecordedTools(volume,frame(step));
    assert.equal(result.displays.length,2);
    assert.deepEqual(result.displays.map(p=>p.tip),
      [[2,3,[1,2].includes(step)?4:2],[4,3,step<4?4:2]]);
    assert.equal(result.bounds.episodeId,hash('b'));
  }
});

test('pair clearance and independent tissue-only checks have separate required flags',()=>{
  const pair=frame(1);pair.primaryPairClearanceChecked=false;
  assert.throws(()=>checkedPassiveRecordedTools(volume,pair),/clearance proof/);
  const tissue=frame(1);tissue.independentTissueMotionChecked=false;
  assert.throws(()=>checkedPassiveRecordedTools(volume,tissue),/clearance proof/);
});

test('generated inspection status cannot appear before the acquisition event',()=>{
  for(const step of [0,1]) {
    const premature=frame(step);
    premature.inspectionStatus='verified';
    assert.throws(()=>checkedPassiveRecordedTools(volume,premature),/withheld/);
  }
});

test('paired-v1 refuses a one-tool payload even when its remaining pose is valid',()=>{
  const missing=frame(1);
  missing.recordedTools.pop();
  assert.throws(()=>checkedPassiveRecordedTools(volume,missing),/withheld/);
  missing.stepCount=4;
  assert.throws(()=>checkedPassiveRecordedTools(volume,missing),/withheld/);
});

test('stale source, frame, pair identity, physical absence and camera bounds fail closed',()=>{
  const edits=[
    f=>{f.caseHash=hash('c');},
    f=>{f.affine[0][3]=1;},
    f=>{f.recordedTools[1].frameIndex=0;},
    f=>{f.recordedTools[1].toolId='moving';},
    f=>{f.recordedTools[1].physicalAbsenceClaim=true;},
    f=>{f.recordedTools[1].retainedAtDepth=false;},
    f=>{f.recordedTools[0].axis=[0,0,2];},
    f=>{f.recordedEpisodeBounds.boundsRasMm[1][0]=2;},
    f=>{f.event={kind:'stop',reason:'made up'};},
  ];
  for(const edit of edits) {
    const f=structuredClone(frame(2));edit(f);
    assert.throws(()=>checkedPassiveRecordedTools(volume,f));
  }
});
