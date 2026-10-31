import test,{before,after} from 'node:test';
import assert from 'node:assert/strict';
import {createServer} from 'vite';
import {mkdtemp,mkdir,rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import * as THREE from 'three';
import {surgicalToolMesh,surgicalToolGeometryKey,updateSurgicalToolMesh,surgicalEffectGroup} from '../src/viewer/surgicalGeometry.ts';
let server,cache,Panel,SurgeonViewControls;
before(async()=>{
  const output=fileURLToPath(new URL('../../testcases/outputs/planar-blade-ui/',import.meta.url));await mkdir(output,{recursive:true});cache=await mkdtemp(path.join(output,'vite-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({PlanarBladeManeuverPanel:Panel}=await server.ssrLoadModule('/src/PlanarBladeManeuverPanel.tsx'));
  ({SurgeonViewControls}=await server.ssrLoadModule('/src/SurgeonViewControls.tsx'));
});
after(async()=>{await server?.close();if(cache)await rm(cache,{recursive:true,force:true});});
function* walk(node){if(Array.isArray(node)){for(const child of node)yield*walk(child);}else if(node?.props){yield node;yield*walk(node.props.children);}}
const text=node=>Array.isArray(node)?node.map(text).join(''):node?.props?text(node.props.children):typeof node==='string'||typeof node==='number'?String(node):'';
const nodes=tree=>[...walk(tree)];
const action={slot:'left',operation:'bladeManeuver',maneuverPhase:'approach',duration:.1};
function controller(){
  const calls=[],data={schema:'resectionlab.planar-blade-maneuver-view.v1',declarationId:'source-owned',phase:'ready',activePhase:null,available:true,refusal:null,inContact:false,normalReactionN:0,contactAreaMm2:0,maxDisplacementMm:0,offsetMm:-.1,nextActions:[{label:'Approach',action}],details:{supportedDomain:'Fixed reference footprint; no friction or edge migration.',energy:{signedWorkNmm:0}},contactSurface:null};
  return {view:{stateId:'accepted',planarBladeManeuver:data},preview:null,result:{},pending:false,calls,run:(command,extra)=>{calls.push({command,...extra});return Promise.resolve(true);}};
}
test('historical views stay unchanged and disclosures/mount never issue native actions',()=>{
  const c=controller();delete c.view.planarBladeManeuver;assert.equal(Panel({c,pause(){}}),null);
  const current=controller(),tree=Panel({c:current,pause(){}});assert.equal(current.calls.length,0);
  const details=nodes(tree).find(n=>n.type==='details');assert.equal(details.props.open,undefined);assert.equal(details.props.onToggle,undefined);
  assert(text(tree).includes('FAIL · 10.08% / 11.41%'));assert(text(tree).includes('Fixed reference footprint'));assert(text(tree).includes('No contact'));
});
test('candidate quantities follow preview while commit uses the exact committed canonical action',()=>{
  const c=controller();c.preview={stateId:'candidate',planarBladeManeuver:{...c.view.planarBladeManeuver,phase:'approached',normalReactionN:.0023456789,maxDisplacementMm:.04,inContact:true,nextActions:[{label:'Indent',action:{...action,maneuverPhase:'indent'}}]}};
  let pauses=0;const tree=Panel({c,pause:()=>pauses++});assert.equal(tree.props['data-planar-blade-state-id'],'candidate');assert.equal(tree.props['data-planar-blade-phase'],'approached');
  const metric=nodes(tree).find(n=>n.props['data-planar-blade-metric']==='normalReactionN');assert.equal(metric.props['data-value'],.0023456789);
  for(const command of ['preview','step']){const button=nodes(tree).find(n=>n.props['data-planar-blade-command']===command);assert.equal(button.props['data-planar-blade-phase'],'approach');button.props.onClick();assert.deepEqual(c.calls.at(-1),{command,action});assert.equal(c.calls.at(-1).action,action);}
  assert.equal(pauses,2);
});
test('read-only, computing and refusal states keep actions blocked and refusal exposed',()=>{
  for(const patch of [{pending:true},{result:{readOnly:true}},{result:{playback:{readOnly:true}}},{result:{verification:{readOnly:true}}}]){
    const c=Object.assign(controller(),patch),tree=Panel({c,pause(){}});assert(nodes(tree).filter(n=>n.type==='button').every(n=>n.props.disabled));
  }
  const c=controller();c.view.planarBladeManeuver.available=false;c.view.planarBladeManeuver.refusal='FIXED_FOOTPRINT_MIGRATION_REFUSED';const tree=Panel({c,pause(){}});
  assert(nodes(tree).some(n=>n.props.role==='alert'&&text(n)==='FIXED_FOOTPRINT_MIGRATION_REFUSED'));assert(nodes(tree).filter(n=>n.type==='button').every(n=>n.props.disabled));assert.equal(c.calls.length,0);
});
test('first microscope entry uses the declared blade optical frame rather than a parked tool pose',()=>{
  const c=controller(),changes=[];c.view.slots={left:{position:[-4,2,-6],quaternion:[0,0,0,1],inserted:false},right:{position:[9,8,7],quaternion:[0,0,0,1],inserted:false}};
  c.view.planarBladeManeuver.opticalFrame={targetMm:[2,1.5,2],outward:[0,-1,0]};
  const before=JSON.stringify(c.view),settings={operativeView:'external',camera:{position:[20,10,30],target:[1,1,1]},clipping:{axis:'z',value:1,enabled:true}};
  const tree=SurgeonViewControls({c,settings,onSettings:value=>changes.push(value)});assert.equal(changes.length,0,'mounting cannot move the viewpoint');
  nodes(tree).find(n=>n.type==='button'&&text(n)==='Surgeon / Microscope').props.onClick();
  assert.equal(changes.length,1);const next=changes[0];assert.equal(next.operativeView,'surgeon');assert.deepEqual(next.microscope.target,[2,1.5,2]);assert.deepEqual(next.microscope.outward,[0,-1,0]);assert.equal(next.microscope.angleDeg,30);assert.equal(next.microscope.fieldMm,12);assert.equal(next.microscope.workingDistanceMm,160);assert.deepEqual(next.camera.target,[2,1.5,2]);assert.equal(next.externalCamera,settings.camera);assert.equal(next.clipping,undefined);
  assert.notEqual(next.microscope.target,c.view.planarBladeManeuver.opticalFrame.targetMm);assert.notEqual(next.microscope.outward,c.view.planarBladeManeuver.opticalFrame.outward);assert.equal(JSON.stringify(c.view),before);
});
test('microscope optics stay fixed across maneuver actions and preserve user adjustments on reentry',()=>{
  const c=controller(),changes=[],microscope={target:[2.5,1.25,2.75],outward:[0,-1,0],angleDeg:22,azimuthDeg:15,workingDistanceMm:150,fieldMm:18,focusOffsetMm:.5,light:3};
  c.view.slots={left:{position:[90,80,70],quaternion:[0,0,0,1],inserted:true}};c.view.planarBladeManeuver={...c.view.planarBladeManeuver,phase:'indented',opticalFrame:{targetMm:[3,4,5],outward:[1,0,0]}};
  const settings={operativeView:'surgeon',microscope},before=JSON.stringify(settings);
  const live=SurgeonViewControls({c,settings,onSettings:value=>changes.push(value)});assert.equal(changes.length,0,'action/render changes cannot dispatch optical updates');
  nodes(live).find(n=>n.type==='button'&&text(n)==='Surgeon / Microscope').props.onClick();assert.equal(changes.length,0,'already enabled entry cannot recenter on the tool or declaration');
  nodes(live).find(n=>n.type==='button'&&text(n)==='External anatomy view').props.onClick();const external=changes.at(-1);assert.equal(external.microscope,microscope);
  c.view.planarBladeManeuver={...c.view.planarBladeManeuver,phase:'withdrawn',opticalFrame:{targetMm:[100,200,300],outward:[0,0,1]}};
  const reentry=SurgeonViewControls({c,settings:external,onSettings:value=>changes.push(value)});const count=changes.length;assert.equal(count,1,'reentry render stays pure');nodes(reentry).find(n=>n.type==='button'&&text(n)==='Surgeon / Microscope').props.onClick();
  const restored=changes.at(-1);assert.equal(restored.microscope,microscope);assert.deepEqual(restored.microscope.target,[2.5,1.25,2.75]);assert.equal(restored.microscope.angleDeg,22);assert.equal(restored.microscope.fieldMm,18);assert.equal(JSON.stringify(settings),before);
});
test('ordinary tool scenes retain the existing microscope entry defaults',()=>{
  const c=controller(),changes=[];delete c.view.planarBladeManeuver;c.view.slots={left:{position:[11,12,13],quaternion:[0,0,0,1],inserted:true}};
  const tree=SurgeonViewControls({c,settings:{operativeView:'external'},onSettings:value=>changes.push(value)});nodes(tree).find(n=>n.type==='button'&&text(n)==='Surgeon / Microscope').props.onClick();
  const m=changes[0].microscope;assert.deepEqual(m.target,[11,12,13]);assert.deepEqual(m.outward,[0,0,-1]);assert.equal(m.angleDeg,0);assert.equal(m.fieldMm,42);
});
const descriptor={sizeMm:[6,.5,9],centerLocalMm:[0,-.25,-4.5],shaft:{a:[0,-.8,-40],b:[0,-.8,-9],radiusMm:.3},inContact:false};
const tool={id:'brain_retractor',radiusMm:3,shaftRadiusMm:.3,workingLengthMm:40};
const slot=()=>({position:[11,12,13],quaternion:new THREE.Quaternion().setFromEuler(new THREE.Euler(.3,.7,-.2)).toArray(),active:false,parameters:{planarBladeContactRevision:1},planarBladeGeometry:structuredClone(descriptor)});
test('rigid pad face exactly matches the finite rectangle in arbitrary RAS pose; shaft stays exterior',()=>{
  const s=slot(),group=surgicalToolMesh(tool,s);group.updateMatrixWorld(true);const pad=group.getObjectByName('instrument:planar-blade-pad');assert(pad);
  assert.deepEqual(pad.geometry.parameters,{width:6,height:.5,depth:9,widthSegments:1,heightSegments:1,depthSegments:1});assert.deepEqual(pad.position.toArray(),[0,-.25,-4.5]);
  const q=new THREE.Quaternion(...s.quaternion),origin=new THREE.Vector3(...s.position),normal=new THREE.Vector3(0,1,0).applyQuaternion(q);
  for(const x of [-3,3])for(const z of [-9,0]){const face=new THREE.Vector3(x,.25,z+4.5).applyMatrix4(pad.matrixWorld),expected=new THREE.Vector3(x,0,z).applyQuaternion(q).add(origin);assert(face.distanceTo(expected)<1e-12);assert(Math.abs(face.clone().sub(origin).dot(normal))<1e-12);}
  const shaft=group.getObjectByName('instrument:shaft');assert.equal(shaft.geometry.parameters.radius,.3);assert(Math.abs(shaft.position.y+.8)<1e-12);assert(shaft.position.y+.3<=-.5+1e-12);
  const ghost=surgicalToolMesh(tool,s,true);assert.deepEqual(ghost.getObjectByName('instrument:planar-blade-pad').geometry.parameters,pad.geometry.parameters);assert(ghost.children.every(m=>m.material.transparent));
});
test('contact color updates without rebuilding pad and historical instruments keep legacy geometry',()=>{
  const s=slot(),group=surgicalToolMesh(tool,s),key=surgicalToolGeometryKey(tool,s),pad=group.getObjectByName('instrument:planar-blade-pad');s.planarBladeGeometry.inContact=true;updateSurgicalToolMesh(group,s);assert.equal(surgicalToolGeometryKey(tool,s),key);assert.equal(group.getObjectByName('instrument:planar-blade-pad'),pad);assert.equal(pad.material.color.getHexString(),'54ded8');
  assert.equal(surgicalToolMesh(tool,{...s,parameters:{}}).getObjectByName('instrument:planar-blade-pad'),undefined);
  assert.equal(surgicalToolMesh({...tool,id:'cusa'},s).getObjectByName('instrument:planar-blade-pad'),undefined);
});
test('Q1 display uses backend deformed RAS corners and indices without altering anatomy or contact markers',()=>{
  const surface={positionsMm:[[1,2,3.04],[2,2,3.08],[2,3,3.07],[1,3,3.03]],triangles:[[0,1,2],[0,2,3]],sourceIds:[8]};
  const view={stateId:'q1',scene:{shape:[3,3,3],resolutionMm:[1,1,1],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},planarBladeManeuver:{contactSurface:surface},contact:[],cuts:[],removed:[],moved:[],thermal:{},fluid:{}};
  const before=JSON.stringify(view),group=surgicalEffectGroup(view),mesh=group.getObjectByName('effect:planar-blade-q1-surface');assert(mesh);assert.deepEqual([...mesh.geometry.index.array],surface.triangles.flat());
  [...mesh.geometry.attributes.position.array].forEach((value,i)=>assert(Math.abs(value-surface.positionsMm.flat()[i])<2e-7));assert.deepEqual(mesh.userData.sourceIds,[8]);assert.equal(mesh.material.opacity,1);assert.equal(JSON.stringify(view),before);
});
