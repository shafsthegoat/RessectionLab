import assert from 'node:assert/strict';
import test from 'node:test';
import {checkPracticeReceipt, jogPracticePose, matchesPracticeParent, practiceDrag, practiceJogKey, practiceStartIssue, practiceViewportMode, practiceWithdrawalPose, savedPracticeEpisode} from '../src/practice-input.ts';

const pose = {position:[2,3,4], quaternion:[0,0,0,1]};
test('jogs follow instrument axes, depth, angle and a bounded drag without changing the input pose', () => {
  assert.deepEqual(jogPracticePose(pose, practiceJogKey('PageDown')).position, [2,3,4.5]);
  const tilted = jogPracticePose(pose, practiceJogKey('ArrowRight', true));
  assert.deepEqual(tilted.position, pose.position);
  const advanced = jogPracticePose(tilted, practiceJogKey('PageDown'));
  assert(advanced.position[0] > 2 && advanced.position[2] > 4);
  const {offset} = practiceDrag(500, 600, false);
  assert(Math.abs(Math.hypot(...offset)-4) < 1e-10);
  assert.deepEqual(practiceDrag(0,-80,true).offset, [0,0,4]);
  assert.deepEqual(pose, {position:[2,3,4], quaternion:[0,0,0,1]});
  assert.equal(practiceJogKey('Home'), null);
});

const checkpoint = {branchId:'recorded', position:107, caseHash:'source', view:{scene:{sceneId:'scene'},stateId:'state'},
  practiceAdmission:{available:true,checkpointSeal:'sealed'}};
const child = {...checkpoint, branchId:'practice-unique',
  practiceEpisode:{schema:'resectionlab.surgical-practice.v1',episodeId:'practice-unique',
    parent:{branchId:'recorded',position:107,caseHash:'source',sceneId:'scene',stateId:'state',checkpointSeal:'sealed'}}};
test('recordings, unverified imports, previews and microsteps cannot become practice through renderer input', () => {
  assert.equal(practiceStartIssue(checkpoint,false,false),null);
  for (const args of [[checkpoint,true,false], [checkpoint,false,true], [{...checkpoint,playback:{}},false,false],
    [{...checkpoint,verification:{readOnly:true}},false,false], [{...checkpoint,practiceAdmission:{}},false,false]]) {
    assert(practiceStartIssue(...args));
  }
});
test('child publication requires the bound original source, parent, checkpoint seal and a new editable episode', () => {
  assert(matchesPracticeParent(checkpoint,child));
  for (const patch of [{caseHash:'other'}, {readOnly:true}, {playback:{}}, {position:106}, {branchId:checkpoint.branchId},
    {view:{...child.view,stateId:'unrelated'}}, {view:{...child.view,scene:{sceneId:'other'}}}]) {
    assert.equal(matchesPracticeParent(checkpoint,{...child,...patch}),false);
  }
  for (const patch of [{branchId:'other'},{position:108},{sceneId:'other'},{stateId:'other'},{checkpointSeal:'other'}]) {
    assert.equal(matchesPracticeParent(checkpoint,{...child,practiceEpisode:{...child.practiceEpisode,
      parent:{...child.practiceEpisode.parent,...patch}}}),false);
  }
});

test('saved child navigation is limited to its active branch and cannot grant write authority', () => {
  const branchId=child.branchId;
  assert(savedPracticeEpisode({surgical_toolbox:{branchId,practiceEpisodeId:branchId}}));
  assert(savedPracticeEpisode({surgical_toolbox:{branchId,practiceEpisodes:{[branchId]:child.practiceEpisode}}}));
  assert(!savedPracticeEpisode({surgical_toolbox:{branchId,practiceEpisodeId:'other'}}));
  assert(!savedPracticeEpisode({surgical_toolbox:{branchId:'main'}}));
});
test('observed practice frames retain the source denominator and can report their own state quantities',()=>{
  const accepted={...child,practiceMetrics:{originalTargetVolumeMm3:27,nonTargetBrainRemovedMm3:1}};
  const observed={...accepted,readOnly:true,playback:{},view:{...accepted.view,stateId:'frame'},practiceMetrics:{originalTargetVolumeMm3:27,nonTargetBrainRemovedMm3:0}};
  assert.doesNotThrow(()=>checkPracticeReceipt(accepted,observed));
  for(const patch of [{practiceEpisode:null},{branchId:'sibling'},{practiceMetrics:undefined},{practiceMetrics:{originalTargetVolumeMm3:28}}])
    assert.throws(()=>checkPracticeReceipt(accepted,{...observed,...patch}));
});
test('checked withdrawal uses the exterior pose of this hand and instrument at the shown checkpoint',()=>{
  const row = (slot, operation, more={}, status='accepted') => ({status,action:{slot,operation,...more}});
  const history=[row('left','place',{pose}),row('right','place',{pose:{...pose,position:[20,20,20]}}),
    row('left','exchange',{},'rejected'),row('left','apply',{},'partial'),row('left','exchange'),row('left','place',{path:[pose]})];
  const at = position => ({position,view:{history}});
  assert.deepEqual(practiceWithdrawalPose(at(4),'left'),pose);
  assert.equal(practiceWithdrawalPose(at(5),'left'),undefined);
  assert.deepEqual(practiceWithdrawalPose(at(6),'left'),pose);
  assert.deepEqual(practiceWithdrawalPose(at(6),'right').position,[20,20,20]);
});
test('transient jog handles disappear at publication, including partial endpoints, while chosen gizmos remain available',()=>{
  const draft=jogPracticePose(pose,practiceJogKey('PageDown'));
  assert.equal(practiceViewportMode('navigate',draft,pose,false),'translate');
  assert.equal(practiceViewportMode('navigate',jogPracticePose(pose,practiceJogKey('ArrowRight',true)),pose,false),'translate');
  const partial={...draft,position:[2,3,4.2]};
  assert.equal(practiceViewportMode('navigate',draft,partial,true),'navigate');
  assert.equal(practiceViewportMode('navigate',partial,partial,false),'navigate');
  assert.equal(practiceViewportMode('navigate',{...pose,quaternion:[0,0,0,-1]},pose,false),'navigate');
  for(const mode of ['rotate','translate','draw']) {
    assert.equal(practiceViewportMode(mode,draft,pose,false),mode);
    assert.equal(practiceViewportMode(mode,partial,partial,true),mode);
  }
});
