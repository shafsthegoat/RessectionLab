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
    anatomyView:async args=>{requests.push(args);return {mri:{assetId:'image',byteLength:4,byteOrder:'little',order:'C',sha256:'image'},
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
