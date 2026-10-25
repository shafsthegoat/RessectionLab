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
const checkbox=(tree,label)=>{
  const parent=[...walk(tree)].find(node=>node.type==='label'&&text(node)===label);
  return [...walk(parent)].find(node=>node.type==='input'&&node.props.type==='checkbox');
};
function fixture(){
  const source={caseHash:'case',shape:[21,21,21],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],frame:'RAS+'},settings={camera:{position:[300,-400,500],target:[0,0,0],up:[0,0,1]},layerOpacity:{brain:.3,bone:.2},smoothSurfaces:false};
  const view={stateId:'accepted0',scene:{sceneId:'scene',regions:[{name:'brain',material:'soft_tissue'},{name:'bone',material:'bone'}]},catalog:[{id:'drill',name:'Research drill',radiusMm:2}],slots:{left:{instrument:'drill',inserted:true,active:true,position:[1,2,3],quaternion:[0,0,0,1],parameters:{}},right:{instrument:'drill',inserted:true,active:false,position:[4,5,6],quaternion:[0,0,0,1],parameters:{}}}};
  const result={branchId:'main',view,playback:{effectiveActivation:{left:false,right:true}},verification:{status:'reconstructed-prefix',readOnly:true,originSnapshotHash:'origin',actionLogHash:'actions',implementation:'current'}};
  const calls=[],props={source,settings,cursor:[9,8,7],c:{view,result,slot:'left',pending:false,preview:null,pose:{position:[999,999,999]},interaction:{view:{...view,stateId:'unaccepted-future'}},run:()=>assert.fail('display controls cannot dispatch replay or physics')},onSettings(value){calls.push(['settings',value]);props.settings=value;},onCursor(value){calls.push(['cursor',value]);props.cursor=value;},onFitAnatomy(){calls.push(['fit']);}};
  const slots=[],effects=[];let index=0,dirty=false;const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,previous=internals.H,same=(a,b)=>a&&b&&a.length===b.length&&a.every((value,i)=>Object.is(value,b[i]));
  const dispatcher={useState(value){const i=index++;if(!(i in slots))slots[i]=typeof value==='function'?value():value;return [slots[i],next=>{slots[i]=typeof next==='function'?next(slots[i]):next;dirty=true;}];},useRef(value){const i=index++;if(!(i in slots))slots[i]={current:value};return slots[i];},useEffect(fn,deps){const i=index++;if(!slots[i]||!same(slots[i].deps,deps)){slots[i]={deps};effects.push(fn);}}};
  const render=({passive=true}={})=>{let tree;do{dirty=false;index=0;internals.H=dispatcher;try{tree=Controls(props);}finally{internals.H=previous;}if(passive)for(const effect of effects.splice(0))effect();}while(dirty);return tree;};
  return {props,calls,view,result,source,settings,render};
}
test('close view is opt-in, pinned to accepted effective tool and restores camera, opacity and linked cursor without dispatch',()=>{
  const f=fixture(),original=structuredClone({settings:f.settings,view:f.view});let tree=f.render();assert.deepEqual(f.calls,[]);assert(button(tree,'Close tool view'));assert.equal(button(tree,'Restore previous view'),undefined);
  button(tree,'Close tool view').props.onClick();tree=f.render();assert.deepEqual(f.props.cursor,[4,5,6]);assert.equal(f.props.settings.layerOpacity.brain,.045);assert(text(tree).includes('right hand'));assert.equal(tree.props['data-pinned-state-id'],'accepted0');assert.equal(f.view, f.props.c.view);
  const pinnedCamera=f.props.settings.camera,newView={...f.view,stateId:'accepted1',slots:{...f.view.slots,right:{...f.view.slots.right,position:[14,15,16]}}};f.props.c.view=newView;f.props.c.result={...f.result,view:newView};tree=f.render();assert.equal(f.props.settings.camera,pinnedCamera,'accepted motion never auto-follows or interpolates the camera');assert.equal(tree.props['data-pinned-state-id'],'accepted0');
  button(tree,'Refocus accepted tool').props.onClick();tree=f.render();assert.deepEqual(f.props.cursor,[14,15,16]);assert.equal(tree.props['data-pinned-state-id'],'accepted1');
  const toggle=checkbox(tree,'Context transparency · display only');toggle.props.onChange({target:{checked:false}});tree=f.render();assert.deepEqual(f.props.settings.layerOpacity,f.settings.layerOpacity);assert(text(tree).includes('Original layer opacities restored'));
  button(tree,'Restore previous view').props.onClick();tree=f.render();assert.deepEqual(f.props.settings,f.settings);assert.deepEqual(f.props.cursor,[9,8,7]);assert.equal(button(tree,'Restore previous view'),undefined);assert.deepEqual(f.view,original.view);assert.deepEqual(f.settings,original.settings);assert(!f.calls.some(call=>call[0]==='fit'));
});
test('restore snapshots invalidate synchronously across same-hash source, branch, scene and authority changes',()=>{
  for(const change of [f=>{f.props.source={...f.source};},f=>{f.props.c.result={...f.result,branchId:'other'};},f=>{f.props.c.view={...f.view,scene:{...f.view.scene,sceneId:'new-scene'}};},f=>{f.props.c.result={...f.result,verification:{...f.result.verification,originSnapshotHash:'new-origin'}};}]){
    const f=fixture();button(f.render(),'Close tool view').props.onClick();f.render();const before=f.calls.length;change(f);const tree=f.render({passive:false});assert.equal(button(tree,'Restore previous view'),undefined);assert(button(tree,'Close tool view'));assert.equal(f.calls.length,before,'ownership invalidation never reapplies old settings/cursor');f.render();
  }
});
test('accepted approach tip outside MRI still frames exact 3D pose while retaining linked slices and restore ownership',()=>{
  const f=fixture(),tip=[-10.464996337890625,30.441162109375,-369.9433333333333],cursor=[-15.889134651090231,8.01669131252251,-428.68486645239943];
  Object.assign(f.source,{shape:[220,220,181],affine:[[-1,0,0,109.53500366210938],[0,1,0,-123.558837890625],[0,0,1,-554.25],[0,0,0,1]]});
  f.view.slots.right.position=tip;f.props.cursor=cursor;const before=structuredClone(f.view);
  button(f.render(),'Close tool view').props.onClick();let tree=f.render();
  assert.equal(f.props.cursor,cursor);assert(!f.calls.some(call=>call[0]==='cursor'));
  assert.deepEqual(f.props.settings.camera.target,tip);assert(text(tree).includes('Linked slices keep their previous position'));
  assert.deepEqual(f.view,before);
  f.props.c.view={...f.view,stateId:'accepted-inside',slots:{...f.view.slots,right:{...f.view.slots.right,position:[0,0,-430]}}};
  tree=f.render();
  button(tree,'Refocus accepted tool').props.onClick();tree=f.render();assert.deepEqual(f.props.cursor,[0,0,-430]);
  assert(!text(tree).includes('Linked slices keep their previous position'));
  button(tree,'Restore previous view').props.onClick();assert.deepEqual(f.props.cursor,cursor);assert.deepEqual(f.props.settings,f.settings);
});
test('withdrawn/invalid accepted tools and pending/preview views block close framing; a missing prior camera resets whole anatomy explicitly',()=>{
  for(const change of [f=>{for(const slot of Object.values(f.view.slots))slot.inserted=false;},f=>{for(const slot of Object.values(f.view.slots))slot.position=[NaN,0,0];},f=>{f.props.c.pending=true;},f=>{f.props.c.preview={stateId:'preview'};}]){
    const f=fixture();change(f);const tree=f.render(),close=button(tree,'Close tool view');assert.equal(close.props.disabled,true);close.props.onClick();assert.deepEqual(f.calls,[]);
  }
  const f=fixture();delete f.props.settings.camera;button(f.render(),'Close tool view').props.onClick();button(f.render(),'Restore previous view').props.onClick();assert.equal(f.calls.filter(call=>call[0]==='fit').length,1);
});
test('local cutaway toggles stay pinned across replay and orbit, transparency preserves them, and restore recovers exact prior clipping',()=>{
  const f=fixture();Object.assign(f.settings,{
    clipping:{enabled:true,axis:'x',value:7,flipped:true},
    clipBox:{enabled:false,min:[-100,-90,-80],max:[100,90,80]},
    obliqueClip:{enabled:true,normal:[0,1,0],offset:11},
  });
  const original=structuredClone(f.settings),originalView=structuredClone(f.view),originalCursor=[...f.props.cursor];
  let tree=f.render();assert.equal(checkbox(tree,'Local anatomy cutaway · display only').props.checked,true);assert.deepEqual(f.calls,[]);
  button(tree,'Close tool view').props.onClick();tree=f.render();
  const pinned={clipBox:structuredClone(f.props.settings.clipBox),obliqueClip:structuredClone(f.props.settings.obliqueClip)},pinnedCursor=f.props.cursor;
  assert.equal(f.props.settings.clipping.enabled,false);assert.equal(f.props.settings.clipBox.enabled,true);assert.equal(f.props.settings.obliqueClip.enabled,true);
  const orbit={position:[25,-35,45],target:[4,5,6],up:[0,0,1]};f.props.settings={...f.props.settings,camera:orbit};
  const moved={...f.view,stateId:'accepted-later',slots:{...f.view.slots,right:{...f.view.slots.right,position:[8,9,10],quaternion:[.2,-.3,.4,.5]}}};
  f.props.c.view=moved;f.props.c.result={...f.result,view:moved};const calls=f.calls.length;tree=f.render();
  assert.equal(f.calls.length,calls,'accepted replay motion never applies a new window');assert.equal(tree.props['data-pinned-state-id'],'accepted0');
  checkbox(tree,'Local anatomy cutaway · display only').props.onChange({target:{checked:false}});tree=f.render();
  assert.deepEqual(f.props.settings.clipping,original.clipping);assert.deepEqual(f.props.settings.clipBox,original.clipBox);assert.deepEqual(f.props.settings.obliqueClip,original.obliqueClip);assert.equal(f.props.settings.camera,orbit);assert.equal(f.props.cursor,pinnedCursor);
  checkbox(tree,'Local anatomy cutaway · display only').props.onChange({target:{checked:true}});tree=f.render();
  assert.deepEqual(f.props.settings.clipBox,pinned.clipBox);assert.deepEqual(f.props.settings.obliqueClip,pinned.obliqueClip);assert.equal(f.props.settings.camera,orbit);assert.equal(tree.props['data-pinned-state-id'],'accepted0');
  for(const checked of [false,true]){
    checkbox(tree,'Context transparency · display only').props.onChange({target:{checked}});tree=f.render();
    assert.deepEqual(f.props.settings.clipBox,pinned.clipBox);assert.deepEqual(f.props.settings.obliqueClip,pinned.obliqueClip);assert.equal(f.props.settings.camera,orbit);assert.equal(f.props.cursor,pinnedCursor);assert.equal(tree.props['data-pinned-state-id'],'accepted0');
  }
  button(tree,'Refocus accepted tool').props.onClick();tree=f.render();assert.equal(tree.props['data-pinned-state-id'],'accepted-later');assert.deepEqual(f.props.settings.camera.target,[8,9,10]);assert.notDeepEqual(f.props.settings.clipBox,pinned.clipBox);assert.notDeepEqual(f.props.settings.obliqueClip,pinned.obliqueClip);
  checkbox(tree,'Local anatomy cutaway · display only').props.onChange({target:{checked:false}});tree=f.render();assert.deepEqual(f.props.settings.clipping,original.clipping);assert.deepEqual(f.props.settings.clipBox,original.clipBox);assert.deepEqual(f.props.settings.obliqueClip,original.obliqueClip);
  button(tree,'Restore previous view').props.onClick();tree=f.render();assert.deepEqual(f.props.settings,original);assert.deepEqual(f.props.cursor,originalCursor);assert.deepEqual(f.view,originalView);assert.deepEqual(f.settings,original);
});
test('local cutaway can be disabled before focusing and restores missing clipping keys without resetting orbit',()=>{
  const f=fixture();let tree=f.render();checkbox(tree,'Local anatomy cutaway · display only').props.onChange({target:{checked:false}});tree=f.render();assert.deepEqual(f.calls,[]);
  button(tree,'Close tool view').props.onClick();tree=f.render();
  for(const key of ['clipping','clipBox','obliqueClip'])assert.equal(Object.hasOwn(f.props.settings,key),false);
  const camera=f.props.settings.camera;checkbox(tree,'Local anatomy cutaway · display only').props.onChange({target:{checked:true}});tree=f.render();assert.equal(f.props.settings.clipBox.enabled,true);assert.equal(f.props.settings.camera,camera);
  checkbox(tree,'Local anatomy cutaway · display only').props.onChange({target:{checked:false}});tree=f.render();
  for(const key of ['clipping','clipBox','obliqueClip'])assert.equal(Object.hasOwn(f.props.settings,key),false);
  assert.equal(f.props.settings.camera,camera);
});
