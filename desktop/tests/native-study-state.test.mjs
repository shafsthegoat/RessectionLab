import test from 'node:test';
import assert from 'node:assert/strict';
import {nextScanState,scanLabel,studyTimepoint,migrateSkullView} from '../src/native-study-state.ts';

test('scan switches retain shared anatomy and appearance while restoring scan-specific contrast',()=>{
  const current={cursor:[1,2,3],visible:{brain:true,tumor:true,vessels:false},settings:{contrast:2,layerOpacity:{brain:.15},layerColor:{tumor:'#aa00ff'}}};
  const saved={cursor:[4,5,6],visible:{brain:false,tumor:false,vessels:true},settings:{contrast:4,volumeRendering:{preset:'ct-bone'}}};
  const next=nextScanState(current,saved);
  assert.deepEqual(next.visible,current.visible);
  assert.deepEqual(next.settings.layerOpacity,current.settings.layerOpacity);
  assert.equal(next.settings.contrast,4);
  assert.equal(next.settings.volumeRendering.preset,'ct-bone');
  assert.deepEqual(next.cursor,saved.cursor);
  next.visible.brain=false;
  assert.equal(current.visible.brain,true);
});
test('study labels use scan names and preserve a distinct postoperative study',()=>{
  assert.equal(scanLabel({id:'ct-brain',modality:'CT'}),'CT BRAIN');
  assert.equal(scanLabel({id:'12345678-abcd-1234-abcd-123456789012',modality:'MRI',label:'T2 FLAIR'}),'T2 FLAIR');
  assert.equal(studyTimepoint({document:{study_timepoint:'postoperative',series:[]}}),'postoperative');
});

test('legacy skull transparency migrates without rewriting explicitly selected modes',()=>{
  const layers=[{id:'skull-ct',kind:'bone'},{id:'vessels',kind:'vessels'}];
  const view={cursor:null,visible:{'skull-ct':true},settings:{layerOpacity:{'skull-ct':.08,vessels:1},clipping:{axis:'z',value:0,enabled:true}}};
  const next=migrateSkullView(view,layers);
  assert.equal(next.settings.skullMode,'exterior');
  assert.equal(next.settings.layerOpacity['skull-ct'],1);
  assert.equal(next.settings.clipping.enabled,false);
  assert.equal(view.settings.layerOpacity['skull-ct'],.08);
  const explicit={...view,settings:{...view.settings,skullMode:'transparent'}};
  assert.equal(migrateSkullView(explicit,layers),explicit);
});
