import test from 'node:test';
import assert from 'node:assert/strict';
import {modeledVolume,validateMapping} from '../src/use-surgical.ts';
import {surgicalToolMesh} from '../src/viewer/surgicalGeometry.ts';
const regions=[{id:1,name:'tissue',color:'#cccccc',material:'soft_tissue',target:false},{id:2,name:'target',color:'#ff0000',material:'tumor',target:true}];
const identity=()=>[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const source=()=>({caseHash:'case',affine:identity(),frame:'RAS+',shape:[4,1,1],mri:Float32Array.from([1,2,3,4]),compartments:[{name:'tissue',color:'#cccccc',mask:Uint8Array.from([1,1,1,0])},{name:'target',color:'#ff0000',mask:Uint8Array.from([0,1,0,0])},{name:'unknown',color:'#777777',mask:Uint8Array.from([0,0,0,1])}]});
const view=(extra={})=>({stateId:'a',scene:{shape:[4,1,1],regions,affine:identity()},removed:[],moved:[],...extra});
test('modeled volume uses authoritative removal, displacement, and later-region ownership without changing MRI/source',()=>{
  const s=source(),v=modeledVolume(s,view({removed:[0],moved:[[1,3]]}));
  assert.equal(v.mri,s.mri);
  assert.deepEqual([...s.compartments[0].mask],[1,1,1,0]);
  assert.deepEqual([...v.compartments[0].mask],[0,0,1,0]);
  assert.deepEqual([...v.compartments[1].mask],[0,0,0,1]);
  assert.equal(v.compartments[2].mask,s.compartments[2].mask);
});
test('state clock/slot changes reuse modeled masks; preview does not mutate committed masks',()=>{
  const s=source(),committed=modeledVolume(s,view()),preview=modeledVolume(s,view({removed:[1]})),next=modeledVolume(s,view({stateId:'later'}));
  assert.equal(next,committed);
  assert.equal(modeledVolume(s,view({stateId:'preview-again',removed:[1]})),preview);
  assert.equal(next.compartments,committed.compartments);
  assert.equal(next.displayIdentity,committed.displayIdentity);
  assert.equal(committed.compartments[1].mask[1],1);
  assert.equal(preview.compartments[1].mask[1],0);
});
test('categorical label data yields independent masks and rejects invalid movement and grid mismatch',()=>{
  const s=source();s.labelSets=[{id:'labels',data:Uint16Array.from([1,2,1,0])}];s.compartments[0]={...s.compartments[0],labelSet:'labels',labelValue:1};
  assert.deepEqual([...modeledVolume(s,view()).compartments[0].mask],[1,0,1,0]);
  assert.throws(()=>modeledVolume(s,view({moved:[[.5,2]]})),/Invalid displacement/);
  assert.throws(()=>modeledVolume(s,view({removed:[8]})),/Invalid removed/);
  assert.throws(()=>modeledVolume(s,view({scene:{shape:[8,1,1],regions}})),/grid mismatch/);
});
test('mapping rejects invalid identities and preserves unknown material choices',()=>{
  assert.equal(validateMapping(regions),null);
  assert.equal(validateMapping([{...regions[0],material:'unknown-new-label'}]),null);
  assert.match(validateMapping([regions[0],regions[0]]),/unique/);
  assert.match(validateMapping([{...regions[0],id:0}]),/positive ID/);
});

test('unchanged source meshes survive rehearsal and changed ownership/material invalidates only affected geometry',()=>{
  const s=source();s.compartments[0].mask[1]=0;s.compartments[0].kind=regions[0].material;s.compartments[1].kind=regions[1].material;
  const surface={positions:new Float32Array(0),source:new Float32Array(0),normals:new Float32Array(0),sourceNormals:new Float32Array(0),indices:new Uint32Array(0)};
  s.compartments[0].surface=surface;s.compartments[1].surface=surface;
  const unchanged=modeledVolume(s,view());assert.equal(unchanged.compartments[0].surface,surface);assert.equal(unchanged.compartments[1].surface,surface);
  assert.equal(unchanged.compartments[0].mask,s.compartments[0].mask);
  const changed=modeledVolume(s,view({removed:[1]}));assert.equal(changed.compartments[0].surface,surface);assert.equal(changed.compartments[1].surface,undefined);
  const mapped=modeledVolume(s,view({scene:{...view().scene,regions:[regions[0],{...regions[1],material:'unknown'}]}}));assert.equal(mapped.compartments[1].surface,undefined);
});
test('all 15 tools have distinct visible research geometry with exact xyzw pose and ghost transparency',()=>{
  const ids=['cranial_drill','cusa','brain_retractor','scalpel','periosteal_elevator','craniotome','dural_scissors','dissector','frazier_suction','bipolar_forceps','micro_scissors','tumor_forceps','irrigation','cortical_probe','subcortical_probe'];
  const signatures=new Set();
  for(const id of ids){const mesh=surgicalToolMesh({id,radiusMm:2,shaftRadiusMm:.5,workingLengthMm:40},{position:[1,2,3],quaternion:[0,0,1,0],parameters:{jawOpeningMm:3}},true);
    assert.deepEqual(mesh.position.toArray(),[1,2,3]);assert.deepEqual(mesh.quaternion.toArray(),[0,0,1,0]);
    assert.ok(mesh.children.length>=3);
    assert.ok(mesh.children.every(m=>m.material.transparent&&m.material.opacity<1));
    signatures.add(JSON.stringify(mesh.children.map(m=>[m.geometry.type,m.geometry.parameters,m.position.toArray(),m.rotation.toArray(),m.scale.toArray()])));
    mesh.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});
  }
  assert.equal(signatures.size,15);
});

test('effect markers share authoritative state, transform source IDs through displacement, and keep fluid spatial',async()=>{
  const {surgicalEffectMarkers,surgicalEffectGroup}=await import('../src/viewer/surgicalGeometry.ts');
  const v=view({stateId:'effects-7',scene:{shape:[4,1,1],regions,resolutionMm:[2,3,4],affine:[[2,0,0,10],[0,3,0,20],[0,0,4,30],[0,0,0,1]]},moved:[[1,3]],contact:[1],thermal:{1:2.5},fluid:{1:4},debris:{2:1},cuts:[{tool:'blade',cells:[0,1],edges:[[0,1]]}],cutEdges:[[0,1]],transectedEdges:[],meshCutEdges:[[500,501]],meshSeams:[{kind:'retained_sheet_seam',points:[[1,2,3],[4,5,6]]},{kind:'retained_strand_break',points:[[7,8,9]]}]});
  const before=JSON.stringify(v),markers=surgicalEffectMarkers(v);
  assert.deepEqual(markers.find(m=>m.kind==='contact').points,[[16,20,30]]);
  assert.deepEqual(markers.find(m=>m.kind==='thermal').points,[[16,20,30]]);
  assert.deepEqual(markers.find(m=>m.kind==='fluid').points,[[12,20,30]]);
  assert.equal(markers.find(m=>m.kind==='thermal').value,2.5);
  assert.equal(markers.filter(m=>m.kind==='cut').length,3,'grid cut deduplicated; displaced sides paired; retained mesh seam separate');
  assert.deepEqual(markers.filter(m=>m.kind==='cut').find(m=>!m.surface).points,[[1,2,3],[4,5,6]]);
  assert.ok(markers.every(m=>m.stateId==='effects-7'));
  assert.equal(JSON.stringify(v),before);
  const group=surgicalEffectGroup(v,true);
  assert.equal(group.userData.stateId,'effects-7');assert.equal(group.userData.preview,true);
  assert.ok(group.children.find(c=>c.name==='effect:cut').isLineSegments);
  assert.ok(group.children.find(c=>c.name==='effect:fluid').isPoints);
  group.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});
});

test('invalid effect cells do not create unrelated geometry; empty effects produce empty group',async()=>{
  const {surgicalEffectMarkers,surgicalEffectGroup}=await import('../src/viewer/surgicalGeometry.ts');
  const v=view({scene:{shape:[4,1,1],regions,resolutionMm:[1,1,1],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},contact:[-1,99,.5],thermal:{99:5},fluid:{1:0,2:-3},cuts:[],meshCutEdges:[[0,1]],meshSeams:[{kind:'retained_strand_break',points:[[NaN,0,0]]}]});
  assert.deepEqual(surgicalEffectMarkers(v),[]);
  assert.equal(surgicalEffectGroup(v).children.length,0);
});

 test('modeled volume validates RAS source alignment before cache reuse, allowing equivalent LPS frames',()=>{
  const s=source();modeledVolume(s,view());
  const shifted=identity();shifted[0][3]=1;
  assert.throws(()=>modeledVolume(s,view({scene:{shape:s.shape,regions,affine:shifted}})),/affine mismatch/);
  const lps=source();lps.frame='LPS+';lps.affine=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]];
  assert.equal(modeledVolume(lps,view()).mri,lps.mri);
  const bad=identity();bad[2][2]=NaN;
  assert.throws(()=>modeledVolume(s,view({scene:{shape:s.shape,regions,affine:bad}})),/finite/);
 });

test('last action summary aggregates effect microsteps and preserves explicit no-effect/contact reasons',async()=>{
  const {summarizeSurgicalEvents}=await import('../src/surgical-types.ts');
  const result=summarizeSurgicalEvents([{kind:'incision',edges:2,meshEdges:1},{kind:'incision',edges:3,meshEdges:2},{kind:'irrigation',volumeMm3:1.25},{kind:'irrigation',volumeMm3:2.5},{kind:'contact_stop',reason:'Unknown material blocks contact'},{kind:'no_material_effect',reason:'No compatible structure intersected by the working tip'}]);
  assert.deepEqual(result[0],{kind:'incision',label:'Incision',count:2,details:['5 grid edges','3 mesh edges']});
  assert.deepEqual(result[1].details,['3.75 mm³']);
  assert.equal(result[2].details[0],'Unknown material blocks contact');
  assert.equal(result[3].details[0],'No compatible structure intersected by the working tip');
  assert.deepEqual(summarizeSurgicalEvents([]),[]);
});

test('effect summary retains latest flap count instead of summing snapshots, and deduplicates controlled sources',async()=>{
  const {summarizeSurgicalEvents}=await import('../src/surgical-types.ts');
  const rows=summarizeSurgicalEvents([{kind:'bone_kerf',cells:3,components:2},{kind:'bone_kerf',cells:1,components:3},{kind:'local_energy',exposure:.5,controlledBleedingSources:['a']},{kind:'local_energy',exposure:.25,controlledBleedingSources:['a','b']},{kind:'mapping',result:'unknown',evidence:[]}]);
  assert.deepEqual(rows[0].details,['4 cells','3 retained components at last sample']);
  assert.deepEqual(rows[1].details,['0.75 exposure units','2 bleeding sources controlled']);
  assert.deepEqual(rows[2].details,['Result: unknown']);
});

test('first obstruction comes from explicit preview outcome without mutating recorded history',async()=>{
  const {surgicalEffectMarkers,surgicalEffectGroup}=await import('../src/viewer/surgicalGeometry.ts');
  const v=view({scene:{shape:[4,1,1],regions,resolutionMm:[1,1,1],affine:identity()},contact:[],thermal:{},fluid:{},cuts:[],history:[{events:[{kind:'contact_stop',position:[1,1,1]}]}]});
  const outcome={status:'rejected',reason:'Occupied by other instrument',completedFraction:0,events:[{kind:'contact_stop',position:[3,4,5]},{kind:'contact_stop',position:[6,7,8]}]};
  const markers=surgicalEffectMarkers(v,outcome).filter(m=>m.kind==='obstruction');
  assert.equal(markers.length,1);assert.deepEqual(markers[0].points,[[3,4,5]]);
  assert.deepEqual(surgicalEffectMarkers(v).find(m=>m.kind==='obstruction').points,[[1,1,1]]);
  assert.equal(surgicalEffectMarkers(v,{...outcome,events:[]}).length,0,'explicit empty outcome does not reuse an old obstruction');
  assert.equal(v.history.length,1);
  const group=surgicalEffectGroup(v,true,outcome);assert.ok(group.children.find(o=>o.name==='effect:obstruction'));
  group.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});
});

test('grid cuts are affine-transformed severed faces with paired displaced sides and retained cells',async()=>{
  const {surgicalEffectMarkers,surgicalEffectGroup}=await import('../src/viewer/surgicalGeometry.ts');
  const v=view({scene:{shape:[3,3,3],regions,resolutionMm:[2,3,4],affine:[[2,.5,0,10],[0,3,0,20],[0,0,4,30],[0,0,0,1]]},contact:[],thermal:{},fluid:{},cuts:[],cutEdges:[[13,22],[22,13]],moved:[],history:[]});
  const before=JSON.stringify(v),markers=surgicalEffectMarkers(v);
  assert.equal(markers.length,1);assert.equal(markers[0].surface,'grid-cut-face');
  assert.deepEqual(markers[0].points,[[13.25,21.5,32],[13.75,24.5,32],[13.75,24.5,36],[13.25,21.5,36],[13.25,21.5,32]]);
  assert.equal(JSON.stringify(v),before);assert.deepEqual(v.removed,[]);
  const paired=surgicalEffectMarkers({...v,moved:[[13,4]]});assert.equal(paired.length,2);
  assert.equal(paired[0].points[0][0],11.25);assert.equal(paired[1].points[0][0],13.25);
  const group=surgicalEffectGroup(v);const surface=group.children.find(o=>o.name==='effect:cut-faces');
  assert.ok(surface.isMesh);assert.equal(surface.geometry.attributes.position.count,6);assert.equal(surface.material.side,2);assert.ok(surface.material.transparent);
  group.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});
});

test('MPR clipping shows cut faces and segments crossing a slab even when every vertex is outside',async()=>{
  const {clipSurgicalMarkersToSlab}=await import('../src/viewer/surgicalGeometry.ts');
  const base={kind:'cut',value:1,color:'#f4d16e',stateId:'same-state'};
  const face={...base,surface:'grid-cut-face',points:[[0,-2,-2],[0,2,-2],[0,2,2],[0,-2,2],[0,-2,-2]]};
  const clipped=clipSurgicalMarkersToSlab([face],2,0,.1);
  assert.equal(clipped.length,1);assert.ok(clipped[0].points.every(p=>p[2]===0));
  assert.ok(clipped[0].points.some(p=>p[1]===-2));assert.ok(clipped[0].points.some(p=>p[1]===2));
  const diagonal={...base,points:[[-2,0,-2],[2,0,2]]};
  assert.deepEqual(clipSurgicalMarkersToSlab([diagonal],2,0,.5)[0].points,[[-.5,0,0],[.5,0,0]]);
  assert.equal(clipSurgicalMarkersToSlab([face],0,4,.5).length,0);
  assert.equal(face.points[0][2],-2,'source marker stays untouched');
});

test('coupled soft-cell raster uses saved precedence independently of label numbers and move order',()=>{
  const s=source(),renamed=regions.map((r,i)=>({...r,id:i?3:98}));
  const v=view({physics:{profile:{schema:'resectionlab.tissue-physics.v2'}},scene:{shape:s.shape,regions:renamed,affine:identity()},moved:[[1,2],[0,2]]});
  const modeled=modeledVolume(s,v);
  assert.deepEqual([...modeled.compartments[0].mask],[0,0,0,0]);
  assert.deepEqual([...modeled.compartments[1].mask],[0,0,1,0]);
  assert.equal(modeled.mri,s.mri);
  assert.deepEqual([...modeledVolume(s,{...v,moved:[...v.moved].reverse()}).compartments[1].mask],[0,0,1,0]);
});

test('drill temperature markers use authoritative physical cells and common slice/3D marker state',async()=>{
  const {surgicalEffectMarkers,clipSurgicalMarkersToSlab}=await import('../src/viewer/surgicalGeometry.ts');
  const v=view({physics:{profile:{bone:{ambientTemperatureC:37}}},contact:[],thermal:{},fluid:{},cuts:[],history:[],drillingV2:{temperatureC:{1:45,2:37,3:NaN}}});
  const markers=surgicalEffectMarkers(v);
  assert.equal(markers.length,1);assert.deepEqual(markers[0].points,[[1,0,0]]);assert.equal(markers[0].value,8);
  assert.equal(clipSurgicalMarkersToSlab(markers,0,1,.5).length,1);
  assert.equal(clipSurgicalMarkersToSlab(markers,0,3,.5).length,0);
});

test('drill event summaries report physical work and removal without summing pending contacts',async()=>{
  const {summarizeSurgicalEvents}=await import('../src/surgical-types.ts');
  const rows=summarizeSurgicalEvents([{kind:'drill_work',inputEnergyJ:2,cuttingWorkJ:1.5,removedVolumeMm3:2,pendingCells:3},{kind:'drill_work',inputEnergyJ:3,cuttingWorkJ:2,removedVolumeMm3:4,pendingCells:2}]);
  assert.equal(rows[0].label,'Bone cutting work');
  assert.deepEqual(rows[0].details,['6 mm³ removed','5 J input','3.5 J cutting work']);
});

test('reference presentation depends on material, preserves camera and never changes source regions',async()=>{
 const {rehearsalPresentation}=await import('../src/rehearsal-presentation.ts');
 const rows=[{id:1,name:'unusual tissue',material:'soft_tissue',color:'#aaaaaa',target:false},{id:2,name:'network',material:'vessel',color:'#bbbbbb',target:false}];
 const before=JSON.stringify(rows),camera={position:[1,2,3],target:[0,0,0],up:[0,1,0]};
 const settings=rehearsalPresentation(rows,{camera});
 assert.equal(settings.layerColor.network,'#e54491');assert.equal(settings.layerOpacity['unusual tissue'],.12);
 assert.equal(settings.volumeRendering.enabled,false);assert.equal(settings.camera,camera);
 assert.equal(JSON.stringify(rows),before);
});
test('initial rehearsal presentation preserves saved appearance, camera, cutaways and source-plane choices',async()=>{
  const {initialRehearsalPresentation}=await import('../src/rehearsal-presentation.ts');
  const rows=[{name:'vessel',material:'vessel',color:'#bb3344'},{name:'target',material:'tumor',color:'#8833bb'}];
  const saved={layerOpacity:{vessel:.1},layerColor:{target:'#abcdef'},showPlane:true,smoothSurfaces:false,layout:'mri-review',camera:{target:[0,0,0],position:[0,50,0],up:[0,0,1]},clipBox:{enabled:true,min:[-2,-2,-2],max:[2,2,2]}};
  const before=structuredClone(saved),settings=initialRehearsalPresentation(rows,saved);
  assert.equal(settings.layerOpacity.vessel,.1);assert.equal(settings.layerOpacity.target,1);assert.equal(settings.layerColor.target,'#abcdef');
  for(const key of ['showPlane','smoothSurfaces','layout','camera','clipBox'])assert.deepEqual(settings[key],saved[key]);
  assert.deepEqual(saved,before);
});
test('fluid footprint shrinks and grows from the same actual ledger volume in 3D and linked MPR without displaying reservoirs',async()=>{
  const {surgicalEffectMarkers,surgicalEffectGroup,clipSurgicalMarkersToSlab,surgicalMarkerRadiusPx,fluidMarkerRadiusMm}=await import('../src/viewer/surgicalGeometry.ts');
  const make=amount=>view({scene:{shape:[4,1,1],regions,resolutionMm:[1,1,1],affine:identity()},contact:[],thermal:{},fluid:amount?{1:amount}:{},sources:[{reservoirMm3:100}],cuts:[]});
  const v=make(8),before=structuredClone(v),markers=surgicalEffectMarkers(v),marker=markers.find(m=>m.kind==='fluid');
  assert.equal(markers.length,1,'a reservoir has no invented spatial marker');assert.equal(marker.value,8);
  assert.ok(Math.abs(fluidMarkerRadiusMm(8)-2*fluidMarkerRadiusMm(1))<1e-10);
  const projected=clipSurgicalMarkersToSlab(markers,0,1,.5)[0];assert.equal(surgicalMarkerRadiusPx(projected,3),3*marker.radiusMm);
  const large=surgicalEffectGroup(v),small=surgicalEffectGroup(make(1));
  const largePoints=large.children.find(c=>c.name==='effect:fluid'),smallPoints=small.children.find(c=>c.name==='effect:fluid');
  assert.ok(Math.abs(largePoints.geometry.getAttribute('fieldDiameterMm').array[0]-2*smallPoints.geometry.getAttribute('fieldDiameterMm').array[0])<1e-6);
  const shader={vertexShader:'void main() { gl_PointSize = size; }',fragmentShader:'#include <clipping_planes_fragment>'};
  largePoints.material.onBeforeCompile(shader,{});assert.match(shader.vertexShader,/gl_PointSize = size \* fieldDiameterMm \* projectionMatrix\[1\]\[1\]/);assert.match(shader.fragmentShader,/gl_PointCoord/);
  assert.deepEqual(surgicalEffectMarkers(make(0)),[]);assert.deepEqual(v,before);
  for(const group of [large,small])group.traverse(o=>{o.geometry?.dispose();o.material?.dispose();});
});
