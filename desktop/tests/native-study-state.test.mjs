import test from 'node:test';
import assert from 'node:assert/strict';
import {nextScanState,scanLabel,studyTimepoint,migrateSkullView,anatomyPresentation} from '../src/native-study-state.ts';
import {checkedViewerSettings} from '../src/workspace-view-settings.ts';

test('scan switches retain shared anatomy and appearance while restoring scan-specific contrast',()=>{
  const current={cursor:[1,2,3],visible:{brain:true,tumor:true,vessels:false},settings:{contrast:2,contextStyle:'shell',sliceFraming:{fit:'anatomy',zoom:2},layerOpacity:{brain:.15},layerColor:{tumor:'#aa00ff'}}};
  const saved={cursor:[4,5,6],visible:{brain:false,tumor:false,vessels:true},settings:{contrast:4,sliceFraming:{fit:'image',zoom:.75},volumeRendering:{preset:'ct-bone'}}};
  const next=nextScanState(current,saved);
  assert.deepEqual(next.visible,current.visible);
  assert.deepEqual(next.settings.layerOpacity,current.settings.layerOpacity);
  assert.equal(next.settings.contrast,4);
  assert.equal(next.settings.volumeRendering.preset,'ct-bone');
  assert.equal(next.settings.contextStyle,'shell');
  assert.deepEqual(next.settings.sliceFraming,{fit:'image',zoom:.75});
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

test('legacy vessel filters migrate once; explicit new choices survive save and reopen',async()=>{
  const {vesselScopeForView}=await import('../src/native-study-state.ts');
  assert.equal(vesselScopeForView(),'all');
  for(const vesselScope of ['brain','intracranial','excluded','all',undefined]){
    const legacy={vesselScope,states:{mri:{settings:{contrast:4}}}};
    const before=structuredClone(legacy);
    assert.equal(vesselScopeForView(legacy),'all');
    assert.deepEqual(legacy,before);
    const saved=JSON.parse(JSON.stringify({...legacy,vesselDisplayVersion:2}));
    assert.equal(vesselScopeForView(saved),vesselScope??'all');
  }
});

test('presentation choices preserve source identities, selected anatomy, camera and quantitative masks',()=>{
  const layers=[{id:'vessels',kind:'vessels',source_series:'t1ce',mask:new Uint8Array([1,0,1])},{id:'tumor',kind:'tumor'},{id:'bone',kind:'bone'},{id:'brain',kind:'brain'}];
  const camera={position:[1,2,3],target:[0,0,0],up:[0,0,1]};
  const settings={camera,contrast:1.7,layerOpacity:{custom:.4},layerColor:{custom:'#123456'},clipping:{enabled:true,axis:'z',value:2},volumeRendering:{preset:'ct-angiography',points:[[0,0,0,0,0],[1,1,0,0,1]]}};
  const original=structuredClone({settings,layers});
  for(const mode of ['segmentation','vascular']){
    const next=anatomyPresentation(settings,layers,mode,'T1CE');
    assert.equal(next.camera,camera);
    assert.equal(next.contrast,1.7);
    assert.equal(next.sliceOverlay,'contour');
    assert.equal(next.contextStyle,'shell');
    assert.deepEqual(next.sliceFraming,{fit:'anatomy',zoom:1});
    assert.equal(next.volumeRendering.preset,'auto');
    assert.equal(next.volumeRendering.points,undefined);
    assert.equal(next.clipping.enabled,false);
    assert.equal(next.layerOpacity.custom,.4);
    assert.ok(next.layerOpacity.vessels>0);
    assert.ok(next.layerOpacity.tumor>.9);
    assert.equal(next.layerColor.custom,'#123456');
    assert.deepEqual(JSON.parse(JSON.stringify(next)).presentation,mode);
  }
  assert.deepEqual({settings,layers},original);
});

test('saved context and physical slice framing reopen detached and reject malformed extents',()=>{
  const source={compartments:[],brainMask:null,metadata:{}};
  for(const fit of ['image','anatomy'])for(const zoom of [.5,1,4])for(const contextStyle of ['shell','uniform']){
    const saved={sliceFraming:{fit,zoom},contextStyle};
    const reopened=checkedViewerSettings(source,saved);
    assert.deepEqual(reopened,saved);
    reopened.sliceFraming.zoom=2;
    assert.equal(saved.sliceFraming.zoom,zoom);
  }
  for(const sliceFraming of [null,[],{}, {fit:'brain',zoom:1},{fit:'image',zoom:true},{fit:'image',zoom:.49},{fit:'anatomy',zoom:4.01},{fit:'image',zoom:NaN},{fit:'image',zoom:1,alterSource:true}])
    assert.throws(()=>checkedViewerSettings(source,{sliceFraming}));
  for(const contextStyle of [null,{},[],'unknown'])assert.throws(()=>checkedViewerSettings(source,{contextStyle}));
});
