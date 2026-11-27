import assert from 'node:assert/strict';
import {after,before,test} from 'node:test';
import * as React from 'react';
import {createServer} from 'vite';
import {mkdir,mkdtemp,rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

let server,cache,useNativeWorkspace;
before(async()=>{
  const output=fileURLToPath(new URL('../../testcases/outputs/integrated-goal-20261026/',import.meta.url));
  await mkdir(output,{recursive:true});
  cache=await mkdtemp(path.join(output,'empty-native-save-vite-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({useNativeWorkspace}=await server.ssrLoadModule('/src/use-native-workspace.ts'));
});
after(async()=>{await server?.close();if(cache)await rm(cache,{recursive:true,force:true});});

test('an empty native study packages without writing a view for a nonexistent scan',async()=>{
  const calls=[];
  const api={
    anatomySaveView:async args=>{calls.push(['view',args]);throw Error('Unknown scan');},
    anatomySavePackage:async args=>{calls.push(['package',args]);return {path:'/testcases/empty.rsworkspace'};},
  };
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const previous=internals.H,slots=[];
  let cursor=0;
  internals.H={
    useState(initial){const index=cursor++;if(!(index in slots))slots[index]=typeof initial==='function'?initial():initial;
      return [slots[index],value=>{slots[index]=typeof value==='function'?value(slots[index]):value;}];},
    useRef(initial){const index=cursor++;if(!(index in slots))slots[index]={current:initial};return slots[index];},
    useEffect(){},
    useCallback(fn){return fn;},
  };
  const render=()=>{cursor=0;return useNativeWorkspace(api);};
  try{
    await render().install({collectionId:'empty',document:{patient_id:'P',snapshot_id:'snapshot',series:[],layers:[],reference_series:null},viewState:null});
    const workspace=render();
    assert.equal(workspace.series,'');
    await workspace.savePackage();
    assert.deepEqual(calls,[['package',{collectionId:'empty',snapshotId:'snapshot'}]]);
    assert.match(render().message,/Patient study file saved/);
  }finally{internals.H=previous;}
});

test('changing a populated study reveals new structures without restoring hidden existing layers',async()=>{
  const initialLayer={id:'old',kind:'tumor',label:'Existing tumor',source_series:'t1',role:'target',color:'#aabbcc'};
  const addedLayers=[{id:'brain',kind:'brain',label:'New brain'},
    {id:'skull',kind:'bone',label:'New skull'},
    {id:'review',kind:'review-region',label:'Review region'}].map(row=>({...row,source_series:'t1',role:'context',color:'#aabbcc'}));
  const document={patient_id:'P',snapshot_id:'snapshot',reference_series:'t1',
    series:[{id:'t1',modality:'T1',grid_spacing_mm:[1,1,1]}],layers:[initialLayer],transforms:{t1:{status:'reference'}}};
  const changed={...document,layers:[initialLayer,...addedLayers]};
  const requests=[];
  const api={
    anatomyRegister:async args=>({document:args.snapshotId==='snapshot'&&requests.length>0?changed:document}),
    anatomyView:async args=>{requests.push(args);return {snapshotId:args.snapshotId,mri:{assetId:'image',byteLength:4,byteOrder:'little',order:'C',sha256:'image'},
      shape:[1,1,1],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],compartments:[],
      registrationStatus:'patient frame',unavailableLayers:[]};},
    readAsset:async()=>new Uint8Array(new Float32Array([0]).buffer),
  };
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const previous=internals.H,previousWorker=globalThis.Worker,slots=[];
  let cursor=0;
  internals.H={
    useState(initial){const index=cursor++;if(!(index in slots))slots[index]=typeof initial==='function'?initial():initial;
      return [slots[index],value=>{slots[index]=typeof value==='function'?value(slots[index]):value;}];},
    useRef(initial){const index=cursor++;if(!(index in slots))slots[index]={current:initial};return slots[index];},
    useEffect(){},
    useCallback(fn){return fn;},
  };
  globalThis.Worker=class {postMessage(){queueMicrotask(()=>this.onmessage({data:{}}));}terminate(){}};
  const render=()=>{cursor=0;return useNativeWorkspace(api);};
  try{
    await render().install({collectionId:'patient',document,
      viewState:{snapshotId:'snapshot',seriesId:'t1',states:{t1:{cursor:null,
        visible:{old:false},settings:{layout:'3d-focus'}}}}});
    await render().change(async()=>({document:changed}));
    assert.deepEqual(render().state.visible,{old:false,brain:true,skull:true,review:false});
    assert.deepEqual(requests.at(-1).layers,['brain','skull']);
  }finally{internals.H=previous;globalThis.Worker=previousWorker;}
});

test('native workspace keeps the shared cutaway through source switches and preserves independent inspection',async()=>{
  const mriShape=[10,10,10];
  const affine=x=>[[1,0,0,x],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const scans=[{id:'mri',modality:'T1'},{id:'ct',modality:'CT'},{id:'scout',modality:'scout'}];
  const transforms={mri:{status:'reference'},ct:{status:'source-provided'}};
  const document={patient_id:'P',snapshot_id:'snapshot',reference_series:'mri',series:scans,layers:[],transforms};
  const sources=scans.map(({id,modality})=>({seriesId:id,modality,alignment:transforms[id]?.status??'unavailable',inPatientBounds:!!transforms[id]}));
  const asset=(assetId,byteLength)=>({assetId,byteLength,byteOrder:'little',order:'C',sha256:assetId});
  const patientContext={sources,boundsRasMm:[[-.5,-.5,-.5],[29.5,9.5,9.5]]};
  const reads=[];
  let saved,packaged=false;
  const api={
    anatomyRegister:async()=>({document}),
    anatomyView:async({seriesId,snapshotId})=>({
      snapshotId,
      mri:asset(seriesId,4000),shape:mriShape,affine:affine({mri:0,ct:20,scout:80}[seriesId]),frame:'RAS+',compartments:[],
      modality:scans.find(scan=>scan.id===seriesId).modality,
      registrationStatus:seriesId==='scout'?'independent source frame':'patient frame',unavailableLayers:[],
      ...(seriesId==='scout'?{}:{patientContext}),
    }),
    readAsset:async id=>{reads.push(id);return new Uint8Array(4000);},
    anatomySaveView:async({viewState})=>{saved=JSON.parse(JSON.stringify(viewState));},
    anatomySavePackage:async()=>{packaged=true;return {path:'/testcases/patient-cutaway.rsworkspace'};},
  };
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const previous=internals.H,previousWorker=globalThis.Worker,slots=[];
  let cursor=0;
  internals.H={
    useState(initial){const index=cursor++;if(!(index in slots))slots[index]=typeof initial==='function'?initial():initial;
      return [slots[index],value=>{slots[index]=typeof value==='function'?value(slots[index]):value;}];},
    useRef(initial){const index=cursor++;if(!(index in slots))slots[index]={current:initial};return slots[index];},
    useEffect(){},
    useCallback(fn){return fn;},
  };
  globalThis.Worker=class {postMessage(){queueMicrotask(()=>this.onmessage({data:{}}));}terminate(){}};
  const render=()=>{cursor=0;return useNativeWorkspace(api);};
  const select=async id=>{
    render().selectScan(id);
    for(let turn=0;turn<20&&render().busy;turn++)await new Promise(setImmediate);
    assert.equal(render().busy,false);
    assert.equal(render().series,id);
  };
  const clipping=(value,enabled=true)=>({axis:'x',value,enabled});
  const cut=()=>render().state.settings.clipping;
  const setCut=clipping=>{const w=render();w.setSettings({...w.state.settings,clipping});};
  try{
    await render().install({collectionId:'study',document});
    assert.equal(render().state.settings.presentation,'segmentation');
    setCut(clipping(20));
    render().present('scan');
    assert.equal(cut(),undefined);
    setCut(clipping(3));
    render().present('segmentation');
    assert.deepEqual(cut(),clipping(20));
    let w=render();
    w.setSettings({...w.state.settings,volumeRendering:{preset:'mri',window:[10,200],density:.3}});
    await select('ct');
    assert.deepEqual(cut(),clipping(20));
    assert.equal(render().state.settings.volumeRendering.window,undefined);
    assert.deepEqual(reads,['mri','ct']);
    setCut(clipping(20,false));
    await select('scout');
    assert.equal(cut(),undefined);
    setCut(clipping(84));
    await render().savePackage();
    assert.ok(packaged);
    assert.equal(saved.seriesId,'scout');
    assert.equal(saved.scalarTransferBasis,'selected-acquisition-v1');
    assert.deepEqual(saved.patientCutaway.clipping,clipping(20,false));
    assert.deepEqual(saved.states.mri.settings.clipping,clipping(3));
    assert.deepEqual(saved.states.scout.settings.clipping,clipping(84));
    assert.equal(saved.states.ct.settings.clipping,undefined);

    await render().install({collectionId:'cold-reopen',document,viewState:saved});
    assert.deepEqual(cut(),clipping(84));
    await select('mri');
    assert.deepEqual(cut(),clipping(20,false));
    render().present('scan');
    assert.deepEqual(cut(),clipping(3));

    await render().change(async()=>({document:{...document,snapshot_id:'realigned'},
      retiredClipping:[{seriesId:null,cutaway:{clipping:clipping(20,false)}}]}),{align:false});
    assert.match(render().message,/cutaway outside the new field was cleared/);
    assert.deepEqual(cut(),clipping(3));
    render().present('segmentation');
    assert.equal(cut(),undefined);

    const legacy=structuredClone(saved);
    delete legacy.patientCutaway;
    legacy.seriesId='mri';
    legacy.states.mri.settings={...legacy.states.mri.settings,clipping:clipping(7),skullMode:'cutaway'};
    await render().install({collectionId:'legacy',document,viewState:legacy});
    assert.deepEqual(cut(),clipping(7));
    assert.deepEqual(render().viewState.patientCutaway.clipping,clipping(7));
    assert.equal(render().viewState.states.mri.settings.clipping,undefined);
  }finally{internals.H=previous;globalThis.Worker=previousWorker;}
});

test('a large study retains interactive source state when exact anatomy is cancelled and retried',async()=>{
  const layer={id:'vessels',kind:'vessels',label:'Vessels',source_series:'tof',role:'context',color:'#cc8899'};
  const document={patient_id:'P',snapshot_id:'snapshot',reference_series:'tof',
    series:[{id:'tof',modality:'TOF',grid_spacing_mm:[1,1,1],data:{shape:[672,672,210]}}],
    layers:[layer],transforms:{tof:{status:'reference'}}};
  const asset=(id,byteLength)=>({assetId:id,byteLength,byteOrder:'little',order:'C',sha256:id});
  const source={snapshotId:'snapshot',caseHash:'case',mri:asset('signal',32),shape:[2,2,2],
    affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],frame:'RAS+',compartments:[],
    registrationStatus:'patient frame',unavailableLayers:[]};
  const full={...source,compartments:[{name:'vessels',kind:'vessels',color:'#cc8899',mask:asset('mask',8),surface:{empty:true}}]};
  const requests=[],cancelled=[],reads=[],answers=[];
  const api={
    anatomyView(args){requests.push(args);return args.layers.length?new Promise(resolve=>answers.push(resolve)):Promise.resolve(source);},
    cancel(id){cancelled.push(id);return Promise.resolve();},
    async readAsset(id){reads.push(id);return new Uint8Array(id==='signal'?32:8);},
  };
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE;
  const previous=internals.H,previousWorker=globalThis.Worker,slots=[];
  let cursor=0;
  internals.H={
    useState(initial){const index=cursor++;if(!(index in slots))slots[index]=typeof initial==='function'?initial():initial;
      return [slots[index],value=>{slots[index]=typeof value==='function'?value(slots[index]):value;}];},
    useRef(initial){const index=cursor++;if(!(index in slots))slots[index]={current:initial};return slots[index];},
    useEffect(){},
    useCallback(fn){return fn;},
  };
  globalThis.Worker=class {postMessage(){queueMicrotask(()=>this.onmessage({data:{}}));}terminate(){}};
  const render=()=>{cursor=0;return useNativeWorkspace(api);};
  try{
    const opening=render().install({collectionId:'study',document,
      viewState:{snapshotId:'snapshot',seriesId:'tof',states:{tof:{cursor:null,visible:{vessels:true},settings:{layout:'3d-focus'}}}}});
    await new Promise(setImmediate);
    const preview=render().volume;
    assert.equal(render().scanPreview,true);
    assert.equal(render().patientFrameReady,false);
    assert.deepEqual(requests.map(request=>request.layers),[[],['vessels']]);
    render().setCursor([0.2,0.3,0.4]);
    const interim=render();
    interim.setSettings({...interim.state.settings,contrast:1.3});
    render().cancel();
    assert.deepEqual(cancelled,[requests[1].requestId]);
    answers[0](full);
    await opening;
    assert.equal(render().volume,preview);
    assert.equal(render().busy,false);
    render().resumeAnatomy();
    await new Promise(setImmediate);
    assert.notEqual(requests[1].requestId,requests[2].requestId);
    answers[1](full);
    await new Promise(setImmediate);
    const exact=render();
    assert.equal(exact.scanPreview,false);
    assert.equal(exact.patientFrameReady,true);
    assert.equal(exact.volume.displayIdentity,preview.displayIdentity);
    assert.equal(exact.volume.mri,preview.mri);
    assert.deepEqual(exact.state.cursor,[0.2,0.3,0.4]);
    assert.equal(exact.state.settings.contrast,1.3);
    assert.deepEqual(reads,['signal','mask']);
  }finally{internals.H=previous;globalThis.Worker=previousWorker;}
});
