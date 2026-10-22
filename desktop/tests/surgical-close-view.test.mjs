import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import * as React from 'react';
import {createServer} from 'vite';
import {mkdtemp,mkdir,rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
let server,cache,Controls;
before(async()=>{
  const output=fileURLToPath(new URL('../../testcases/outputs/rl-readiness-20261022/',import.meta.url));await mkdir(output,{recursive:true});cache=await mkdtemp(path.join(output,'close-view-vite-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({SurgicalCloseViewControls:Controls}=await server.ssrLoadModule('/src/SurgicalCloseViewControls.tsx'));
});
after(async()=>{await server?.close();if(cache)await rm(cache,{recursive:true,force:true});});
function* walk(node){if(Array.isArray(node)){for(const item of node)yield*walk(item);}else if(node?.props){yield node;yield*walk(node.props.children);}}
const text=node=>Array.isArray(node)?node.map(text).join(''):node?.props?text(node.props.children):typeof node==='string'||typeof node==='number'?String(node):'';
const button=(tree,label)=>[...walk(tree)].find(node=>node.type==='button'&&text(node)===label);
function fixture(){
  const source={caseHash:'case'},settings={camera:{position:[300,-400,500],target:[0,0,0],up:[0,0,1]},layerOpacity:{brain:.3,bone:.2},smoothSurfaces:false};
  const view={stateId:'accepted0',scene:{sceneId:'scene',regions:[{name:'brain',material:'soft_tissue'},{name:'bone',material:'bone'}]},catalog:[{id:'drill',name:'Research drill',radiusMm:2}],slots:{left:{instrument:'drill',inserted:true,active:true,position:[1,2,3],quaternion:[0,0,0,1],parameters:{}},right:{instrument:'drill',inserted:true,active:false,position:[4,5,6],quaternion:[0,0,0,1],parameters:{}}}};
  const result={branchId:'main',view,playback:{effectiveActivation:{left:false,right:true}},verification:{status:'reconstructed-prefix',readOnly:true,originSnapshotHash:'origin',actionLogHash:'actions',implementation:'current'}};
  const calls=[],props={source,settings,cursor:[9,8,7],c:{view,result,slot:'left',pending:false,preview:null,pose:{position:[999,999,999]},interaction:{view:{...view,stateId:'unaccepted-future'}},run:()=>assert.fail('display controls cannot dispatch replay or physics')},onSettings(value){calls.push(['settings',value]);props.settings=value;},onCursor(value){calls.push(['cursor',value]);props.cursor=value;},onFitAnatomy(){calls.push(['fit']);}};
  const slots=[],effects=[];let index=0,dirty=false;const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,previous=internals.H,same=(a,b)=>a&&b&&a.length===b.length&&a.every((value,i)=>Object.is(value,b[i]));
  const dispatcher={useState(value){const i=index++;if(!(i in slots))slots[i]=typeof value==='function'?value():value;return [slots[i],next=>{slots[i]=typeof next==='function'?next(slots[i]):next;dirty=true;}];},useEffect(fn,deps){const i=index++;if(!slots[i]||!same(slots[i].deps,deps)){slots[i]={deps};effects.push(fn);}}};
  const render=({passive=true}={})=>{let tree;do{dirty=false;index=0;internals.H=dispatcher;try{tree=Controls(props);}finally{internals.H=previous;}if(passive)for(const effect of effects.splice(0))effect();}while(dirty);return tree;};
  return {props,calls,view,result,source,settings,render};
}
test('close view is opt-in, pinned to accepted effective tool and restores camera, opacity and linked cursor without dispatch',()=>{
  const f=fixture(),original=structuredClone({settings:f.settings,view:f.view});let tree=f.render();assert.deepEqual(f.calls,[]);assert(button(tree,'Close tool view'));assert.equal(button(tree,'Restore previous view'),undefined);
  button(tree,'Close tool view').props.onClick();tree=f.render();assert.deepEqual(f.props.cursor,[4,5,6]);assert.equal(f.props.settings.layerOpacity.brain,.045);assert(text(tree).includes('right hand'));assert.equal(tree.props['data-pinned-state-id'],'accepted0');assert.equal(f.view, f.props.c.view);
  const pinnedCamera=f.props.settings.camera,newView={...f.view,stateId:'accepted1',slots:{...f.view.slots,right:{...f.view.slots.right,position:[14,15,16]}}};f.props.c.view=newView;f.props.c.result={...f.result,view:newView};tree=f.render();assert.equal(f.props.settings.camera,pinnedCamera,'accepted motion never auto-follows or interpolates the camera');assert.equal(tree.props['data-pinned-state-id'],'accepted0');
  button(tree,'Refocus accepted tool').props.onClick();tree=f.render();assert.deepEqual(f.props.cursor,[14,15,16]);assert.equal(tree.props['data-pinned-state-id'],'accepted1');
  const toggle=[...walk(tree)].find(node=>node.type==='input'&&node.props.type==='checkbox');toggle.props.onChange({target:{checked:false}});tree=f.render();assert.deepEqual(f.props.settings.layerOpacity,f.settings.layerOpacity);assert(text(tree).includes('Original layer opacities restored'));
  button(tree,'Restore previous view').props.onClick();tree=f.render();assert.deepEqual(f.props.settings,f.settings);assert.deepEqual(f.props.cursor,[9,8,7]);assert.equal(button(tree,'Restore previous view'),undefined);assert.deepEqual(f.view,original.view);assert.deepEqual(f.settings,original.settings);assert(!f.calls.some(call=>call[0]==='fit'));
});
test('restore snapshots invalidate synchronously across same-hash source, branch, scene and authority changes',()=>{
  for(const change of [f=>{f.props.source={...f.source};},f=>{f.props.c.result={...f.result,branchId:'other'};},f=>{f.props.c.view={...f.view,scene:{...f.view.scene,sceneId:'new-scene'}};},f=>{f.props.c.result={...f.result,verification:{...f.result.verification,originSnapshotHash:'new-origin'}};}]){
    const f=fixture();button(f.render(),'Close tool view').props.onClick();f.render();const before=f.calls.length;change(f);const tree=f.render({passive:false});assert.equal(button(tree,'Restore previous view'),undefined);assert(button(tree,'Close tool view'));assert.equal(f.calls.length,before,'ownership invalidation never reapplies old settings/cursor');f.render();
  }
});
test('withdrawn/invalid accepted tools and pending/preview views block close framing; a missing prior camera resets whole anatomy explicitly',()=>{
  for(const change of [f=>{for(const slot of Object.values(f.view.slots))slot.inserted=false;},f=>{for(const slot of Object.values(f.view.slots))slot.position=[NaN,0,0];},f=>{f.props.c.pending=true;},f=>{f.props.c.preview={stateId:'preview'};}]){
    const f=fixture();change(f);const tree=f.render(),close=button(tree,'Close tool view');assert.equal(close.props.disabled,true);close.props.onClick();assert.deepEqual(f.calls,[]);
  }
  const f=fixture();delete f.props.settings.camera;button(f.render(),'Close tool view').props.onClick();button(f.render(),'Restore previous view').props.onClick();assert.equal(f.calls.filter(call=>call[0]==='fit').length,1);
});
