import assert from 'node:assert/strict';
import {after,before,test} from 'node:test';
import * as React from 'react';
import {createServer} from 'vite';
import {mkdir,mkdtemp,rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

let server,cache,Panel,useNativeApproaches;
before(async()=>{
  const output=fileURLToPath(new URL('../../testcases/outputs/eden-planning-editor/',import.meta.url));
  await mkdir(output,{recursive:true});cache=await mkdtemp(path.join(output,'test-vite-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({NativeApproachPanel:Panel}=await server.ssrLoadModule('/src/NativeApproachPanel.tsx'));
  ({useNativeApproaches}=await server.ssrLoadModule('/src/use-native-approaches.ts'));
});
after(async()=>{await server?.close();if(cache)await rm(cache,{recursive:true,force:true});});
const flush=()=>new Promise(resolve=>setImmediate(resolve));
function* walk(node){if(Array.isArray(node)){for(const item of node)yield*walk(item);}else if(node?.props){yield node;yield*walk(node.props.children);}}
const text=node=>Array.isArray(node)?node.map(text).join(''):node?.props?text(node.props.children):typeof node==='string'||typeof node==='number'?String(node):'';
const input=(tree,label)=>[...walk(tree)].find(node=>node.type==='input'&&node.props['aria-label']===label);
const button=(tree,label)=>[...walk(tree)].find(node=>node.type==='button'&&(node.props['aria-label']===label||text(node)===label));
const point=(tree,label)=>[...walk(tree)].find(node=>node.type?.name==='PointEditor'&&node.props.label===label);
function harness(renderBody){
  const slots=[],effects=[];let index=0,dirty=false;
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,previous=internals.H;
  const same=(a,b)=>a&&b&&a.length===b.length&&a.every((value,i)=>Object.is(value,b[i]));
  const dispatcher={
    useState(value){const i=index++;if(!(i in slots))slots[i]=typeof value==='function'?value():value;return [slots[i],next=>{const result=typeof next==='function'?next(slots[i]):next;if(!Object.is(result,slots[i])){slots[i]=result;dirty=true;}}];},
    useRef(value){const i=index++;return slots[i]??={current:value};},
    useEffect(fn,deps){const i=index++;if(!slots[i]||!same(slots[i].deps,deps)){const old=slots[i],slot={deps,cleanup:old?.cleanup};slots[i]=slot;effects.push(()=>{old?.cleanup?.();slot.cleanup=fn();});}},
  };
  return {render({passive=true}={}){let tree;do{dirty=false;index=0;internals.H=dispatcher;try{tree=renderBody();}finally{internals.H=previous;}if(passive)for(const effect of effects.splice(0))effect();}while(dirty);return tree;},close(){for(const slot of slots)slot?.cleanup?.();internals.H=previous;}};
}
const wholeField={wholeEnvelopeWithinField:true};
function approach(id,patchRadius=15){return {id,name:`Approach ${id}`,sourceSnapshotId:'revision',parameterHash:`initial-${id}`,assessmentStatus:'current',entry:id==='b'?[9,8,60]:[1,2,50],target:id==='b'?[7,6,20]:[1,2,20],radiusMm:id==='b'?4:2,direction:[0,0,-1],entryPatch:{center:id==='b'?[8,8,60]:[0,2,50],normal:[0,0,-1],radiusMm:patchRadius},constraints:{minClearanceMm:id==='b'?3:0,maxLengthMm:id==='b'?80:150,excludedLayerIds:['tract']},measurements:{lengthMm:30,orientation:{azimuthDeg:0,elevationDeg:-90},status:'incomplete',structures:[{layerId:'tract',label:'Supplied CST',kind:'tractography',available:true,clearanceLowerBoundMm:4,confirmedIntersection:false}],coverage:{withinReferenceEnvelope:true},targets:[{available:true,label:'Target',coveredVoxelCenterVolumeMm3:0,coveredVoxelCenterFraction:0,coverage:wholeField,acquisitionCoverage:wholeField}],normalTraversal:{available:true,conservativeVolumeUpperBoundMm3:120},unknowns:['Registration accuracy unmeasured'],assumptions:['Static straight capsule']}};}
function fixture(initial=[approach('a',7),approach('b',22)],{patientFrameReady=true,onEditStateChange,setCursor=()=>{}}={}){
  let saved=structuredClone(initial),revision=0,planning,workspace;
  const calls=[],binding={collectionId:'patient',document:{snapshot_id:'revision',display:{entry_ras_mm:[0,0,90],cursor_ras_mm:[0,0,30]},layers:[]},viewState:{selectedApproachId:'a'}};
  const api={async anatomyPlan(args){calls.push(structuredClone(args));if(args.action==='upsert'){
    const value={...args.approach,id:args.approach.id??'new',sourceSnapshotId:'revision',parameterHash:`edited-${++revision}`,assessmentStatus:'current',measurements:approach('a').measurements};
    value.measurements={...value.measurements,lengthMm:Math.hypot(...value.target.map((v,i)=>v-value.entry[i])),radiusMm:value.radiusMm,constraintFailures:value.entryPatch.radiusMm<1?['entry outside declared entry-centre disk']:[]};
    saved=[...saved.filter(a=>a.id!==value.id),structuredClone(value)];return {approaches:structuredClone(saved),selectedId:value.id};
  }return {approaches:structuredClone(saved)};}};
  function open(){return harness(()=>{
    planning=useNativeApproaches(binding,api,true);
    workspace={planning,binding,patientFrameReady,busy:false,draft:null,state:{cursor:[4,5,60],visible:{},settings:{}},volume:null,selectedStructure:null,setCursor,setSelectedStructure(){},setVisible(){},focusStructure(){},setSettings(){},async save(){calls.push({action:'save',selectedId:planning.selectedId});binding.viewState.selectedApproachId=planning.selectedId;}};
    return Panel({workspace,onEditStateChange});
  });}
  return {binding,api,calls,open,get planning(){return planning;},get workspace(){return workspace;},get saved(){return saved;}};
}
async function ready(f){const h=f.open();h.render();await flush();h.render();return h;}

test('selecting a saved approach initializes every editable field before effects and never leaks another draft',async()=>{
  const f=fixture(),h=await ready(f);
  try{
    let tree=h.render();assert.equal(input(tree,'Entry patch radius').props.value,7);
    input(tree,'Entry patch radius').props.onChange({target:{value:'9'}});h.render();
    f.planning.select('b');tree=h.render({passive:false});
    assert.equal(input(tree,'Entry patch radius').props.value,22);assert.equal(input(tree,'Corridor radius').props.value,4);assert.equal(input(tree,'Minimum clearance').props.value,3);assert.equal(input(tree,'Maximum corridor length').props.value,80);assert.deepEqual(point(tree,'Entry').props.value,[9,8,60]);assert.deepEqual(point(tree,'Target').props.value,[7,6,20]);
    h.render();await new Promise(resolve=>setTimeout(resolve,700));h.render();assert.equal(f.calls.filter(c=>c.action==='upsert').length,0,'a cancelled old timer cannot submit the draft as approach B');
  }finally{h.close();}
});
test('entry-patch edits debounce into the selected saved geometry and preserve its patch plane and constraints',async()=>{
  const f=fixture(),h=await ready(f);
  try{
    const tree=h.render();input(tree,'Entry patch radius').props.onChange({target:{value:'.5'}});h.render();
    await new Promise(resolve=>setTimeout(resolve,700));await flush();const updated=h.render(),call=f.calls.find(c=>c.action==='upsert');
    assert.equal(call.approach.id,'a');assert.equal(call.approach.entryPatch.radiusMm,.5);assert.deepEqual(call.approach.entryPatch.center,[0,2,50]);assert.deepEqual(call.approach.entryPatch.normal,[0,0,-1]);assert.deepEqual(call.approach.constraints.excludedLayerIds,['tract']);
    assert(text(updated).includes('Geometric constraint failures'));assert(text(updated).includes('entry outside declared entry-centre disk'));assert.equal(updated.props['data-approach-dirty'],false);
  }finally{h.close();}
});
test('selecting another card first persists edits to the previous approach, then loads the selected patch',async()=>{
  const f=fixture(),h=await ready(f);
  try{
    input(h.render(),'Entry patch radius').props.onChange({target:{value:'6'}});let tree=h.render();
    const card=[...walk(tree)].find(node=>node.props['data-approach-id']==='b');card.props.onClick();await flush();tree=h.render();
    const call=f.calls.find(c=>c.action==='upsert');assert.equal(call.approach.id,'a');assert.equal(call.approach.entryPatch.radiusMm,6);assert.equal(f.planning.selectedId,'b');assert.equal(input(tree,'Entry patch radius').props.value,22);assert.equal(f.saved.find(a=>a.id==='b').entryPatch.radiusMm,22);
  }finally{h.close();}
});
test('suggestions reuse the selected saved patch centre and plane rather than moving it to the edited entry',async()=>{
  const f=fixture(),h=await ready(f);
  try{
    point(h.render(),'Entry').props.onChange([5,2,50]);
    input(h.render(),'Entry patch radius').props.onChange({target:{value:'6'}});
    button(h.render(),'Suggest approaches').props.onClick();await flush();h.render();
    const actions=f.calls.filter(c=>['upsert','suggest'].includes(c.action));assert.deepEqual(actions.map(c=>c.action),['upsert','suggest']);
    const suggestion=actions[1];assert.deepEqual(suggestion.start,[5,2,50]);assert.deepEqual(suggestion.entryPatch,{center:[0,2,50],normal:[0,0,-1],radiusMm:6});assert.deepEqual(suggestion.constraints.excludedLayerIds,['tract']);
  }finally{h.close();}
});
test('legacy saved geometry without a patch gets its default radius instead of retaining the previous selection',async()=>{
  const a=approach('a',27),b=approach('b');delete b.entryPatch;
  const f=fixture([a,b]),h=await ready(f);
  try{
    assert.equal(input(h.render(),'Entry patch radius').props.value,27);f.planning.select('b');const tree=h.render();assert.equal(input(tree,'Entry patch radius').props.value,15);
  }finally{h.close();}
});
test('Save flushes pending entry, corridor and patch edits before saving; reopening restores all fields',async()=>{
  const f=fixture();let h=await ready(f);
  try{
    point(h.render(),'Entry').props.onChange([4,2,50]);
    input(h.render(),'Corridor radius').props.onChange({target:{value:'5'}});
    input(h.render(),'Entry patch radius').props.onChange({target:{value:'3'}});
    input(h.render(),'Minimum clearance').props.onChange({target:{value:'2'}});
    input(h.render(),'Maximum corridor length').props.onChange({target:{value:'90'}});
    await button(h.render(),'Save study & comparison').props.onClick();await flush();h.render();
    assert.deepEqual(f.calls.filter(c=>['upsert','save'].includes(c.action)).map(c=>c.action),['upsert','save']);
    const saved=f.saved.find(a=>a.id==='a');assert.equal(saved.entryPatch.radiusMm,3);assert.deepEqual(saved.entryPatch.center,[0,2,50]);assert.deepEqual(saved.entry,[4,2,50]);assert.equal(saved.radiusMm,5);
    h.close();h=await ready(f);const reopened=h.render();
    assert.equal(input(reopened,'Entry patch radius').props.value,3);assert.equal(input(reopened,'Corridor radius').props.value,5);assert.equal(input(reopened,'Minimum clearance').props.value,2);assert.equal(input(reopened,'Maximum corridor length').props.value,90);assert.deepEqual(point(reopened,'Entry').props.value,[4,2,50]);assert.equal(f.planning.selectedId,'a');
  }finally{h.close();}
});
test('failed patch edits remain unsaved, block Save completion and can be retried without losing the draft',async()=>{
  const f=fixture(),original=f.api.anatomyPlan;let fail=true;
  f.api.anatomyPlan=async args=>{if(args.action==='upsert'&&fail)throw Error('Source unavailable');return original(args);};
  const h=await ready(f);
  try{
    input(h.render(),'Entry patch radius').props.onChange({target:{value:'4'}});
    await button(h.render(),'Save study & comparison').props.onClick();await flush();let tree=h.render();
    assert.equal(tree.props['data-approach-dirty'],true);assert(text(tree).includes('Edits remain unsaved'));assert.equal(input(tree,'Entry patch radius').props.value,4);assert(!f.calls.some(c=>c.action==='save'));
    fail=false;await button(tree,'Save study & comparison').props.onClick();await flush();tree=h.render();assert.equal(tree.props['data-approach-dirty'],false);assert.equal(f.saved.find(a=>a.id==='a').entryPatch.radiusMm,4);assert(f.calls.some(c=>c.action==='save'));
  }finally{h.close();}
});
test('comparison cards show observed anatomy, incomplete evidence, quantities and constraint failures separately',async()=>{
  const a=approach('a'),b=approach('b');a.measurements.status='known-conflict';a.measurements.constraintFailures=['entry outside declared entry-centre disk'];b.measurements.status='known-conflict';b.measurements.structures[0].confirmedIntersection=true;b.measurements.structures[0].clearanceLowerBoundMm=-1;
  b.measurements.structures.push({layerId:'language',label:'Language bundle',kind:'tractography',available:false});
  const f=fixture([a,b]),h=await ready(f);
  try{
    const tree=h.render(),cardA=[...walk(tree)].find(n=>n.props['data-approach-id']==='a'),cardB=[...walk(tree)].find(n=>n.props['data-approach-id']==='b');
    assert(text(cardA).includes('No detected intersection in assessed anatomy'));assert(text(cardA).includes('Incomplete evidence'));assert(text(cardA).includes('geometric constraint failure'));assert(text(cardA).includes('Target voxel centres reached 0.0 mm³'));assert(text(cardB).includes('Detected anatomical intersection'));assert(text(cardB).includes('Intersects: Supplied CST'));assert(text(cardB).includes('Unavailable: Language bundle'));assert(text(cardB).includes('Incomplete evidence'));
    assert(!text(tree).includes('Known geometric conflict'),'the aggregate backend status does not obscure separate observations and constraints');
  }finally{h.close();}
});
test('switching patch and surface entries binds cards, viewer geometry and consequence columns to the same route',async()=>{
  const target='target',binding='target-binding',slots=['A','B','C'];
  const approaches=['patch','surface'].flatMap(group=>[0,1,2].map(index=>{
    const id=`${group}-${index}`,value=(group==='patch'?10:20)+index;
    const row={...approach(id),origin:group==='patch'?'bounded-native-entry-patch':'native-source-entry-windows',
      entry:[1,2,20+value],targetLayerId:target,measurementHash:`measurements-${id}`};
    row.measurements.lengthMm=value;
    row.measurements.targets[0].coveredVoxelCenterVolumeMm3=value-5;
    row.measurements.normalTraversal.conservativeVolumeUpperBoundMm3=value+10;
    return row;
  }));
  const metric=(label,unit,value)=>({label,unit,value,status:'geometric_estimate'});
  const f=fixture(approaches);
  f.binding.viewState.selectedApproachId='patch-0';
  let finishRequest;
  f.api.anatomyPlan=args=>{
    f.calls.push(structuredClone(args));
    const start=args.reviewApproachId?.startsWith('surface')?3:0;
    const response={approaches,activeTargetLayerId:target,consequenceEvidence:{sourceSnapshotId:'revision',
      targetLayerId:target,targetCandidateBinding:binding,routes:approaches.slice(start,start+3).map((row,index)=>({
        approachId:row.id,name:row.name,comparisonSlot:slots[index],parameterHash:row.parameterHash,measurementHash:row.measurementHash,
        sourceSnapshotId:'revision',targetLayerId:target,targetCandidateBinding:binding,target:{label:'Target',evidenceClass:'supplied_annotation'},
        dimensions:{lengthMm:metric('Corridor length','mm',row.measurements.lengthMm),
          targetVoxelCenterCoverageMm3:metric('Target voxel centres reached','mm3',row.measurements.targets[0].coveredVoxelCenterVolumeMm3),
          nonTargetTraversalUpperBoundMm3:metric('Brain traversal volume upper bound','mm3',row.measurements.normalTraversal.conservativeVolumeUpperBoundMm3)},
        researchDisposition:{reasons:[`Evidence for ${row.id}`]},warnings:{criticalContacts:[],constraintFailures:[],missingEvidence:[]},
        registrations:[],uncertainty:{limitations:[]},sources:[],
      })).reverse()}};
    return f.calls.length===1?Promise.resolve(response):new Promise(resolve=>{finishRequest=()=>resolve(response);});
  };
  const h=await ready(f);
  const card=(tree,id)=>[...walk(tree)].find(node=>node.props['data-approach-id']===id);
  const readout=tree=>{
    const node=[...walk(tree)].find(item=>item.type?.name==='NativeConsequenceReadout');
    return node?.type(node.props);
  };
  function verify(group,index){
    const tree=h.render(),evidence=readout(tree);
    const expected=approaches.filter(row=>row.id.startsWith(group));
    const viewer=f.planning.routes;
    assert.deepEqual(viewer.map(row=>[row.route_id,row.comparisonSlot,row.label,row.entry_mm,row.target_mm,row.tool.working_length_mm]),
      expected.map((row,i)=>[row.id,slots[i],`${slots[i]} · ${row.name}`,row.entry,row.target,row.measurements.lengthMm]));
    assert.equal(viewer.find(route=>route.selected)?.route_id,expected[index].id);
    assert.deepEqual([...walk(evidence)].filter(node=>node.props['data-native-consequence-route'])
      .map(node=>[node.props['data-native-consequence-route'],text(node)]),expected.map((row,i)=>[row.id,slots[i]]));
    assert.deepEqual([...walk(evidence)].filter(node=>node.type==='td').map(text),
      [row=>row.lengthMm,row=>row.targets[0].coveredVoxelCenterVolumeMm3,row=>row.normalTraversal.conservativeVolumeUpperBoundMm3]
        .flatMap(measure=>expected.map(row=>String(measure(row.measurements)))));
    for(const [i,row] of expected.entries()){
      assert(text(card(tree,row.id)).startsWith(`${slots[i]} · ${row.name}${row.measurements.lengthMm}.0 mm`));
    }
    assert(text(evidence).includes(`Evidence for ${expected[index].id}`));
    assert(text(evidence).includes(`Coverage, warnings and registration · ${slots[index]}`));
  }
  async function select(id){
    await card(h.render(),id).props.onClick();
    return h.render();
  }
  try{
    verify('patch',0);
    await select('patch-2');verify('patch',2);
    assert.equal(f.calls.length,1);
    let tree=await select('surface-1');
    assert.equal(f.planning.routes.find(row=>row.selected)?.route_id,'surface-1');
    assert.equal(readout(tree),null,'patch evidence cannot appear under surface slots while loading');
    assert.equal(f.calls.at(-1).reviewApproachId,'surface-1');
    finishRequest();await flush();verify('surface',1);
    await select('surface-2');verify('surface',2);
    assert.equal(f.calls.length,2);
    tree=await select('patch-1');
    assert.equal(readout(tree),null);
    assert.equal(f.calls.at(-1).reviewApproachId,'patch-1');
    finishRequest();await flush();verify('patch',1);
  }finally{h.close()}
});
test('a completed mutation cannot restore an older selection made while it was in flight',async()=>{
  const binding={collectionId:'patient',document:{snapshot_id:'revision'},viewState:{selectedApproachId:'a'}},approaches=[approach('a'),approach('b')];
  let planning,resolve;
  const api={async anatomyPlan(args){return args.action==='list'?{approaches}:new Promise(done=>{resolve=done;});}};
  const h=harness(()=>planning=useNativeApproaches(binding,api,true));
  try{
    h.render();await flush();h.render();const pending=planning.request({action:'upsert',approach:approaches[0]});h.render();planning.select('b');h.render();resolve({approaches,selectedId:'a'});await pending;h.render();assert.equal(planning.selectedId,'b');
  }finally{h.close();}
});

test('switching target restores its saved selection and receipt without remeasuring either target',async()=>{
  const a=approach('a'), b=approach('b'), c=approach('c');
  a.targetLayerId='target-1';b.targetLayerId=c.targetLayerId='target-2';
  const f=fixture([a,b,c]);
  f.binding.viewState={snapshotId:'revision',selectedApproachId:'c',selectedPlanningTargetId:'target-2',selectedApproachIds:{'target-1':'a','target-2':'c'}};
  const targetOptions=[{layerId:'target-1',label:'Target one',target:a.target,entry:a.entry},{layerId:'target-2',label:'Target two',target:b.target,entry:b.entry}];
  const reports={'target-1':{task:{targetLayerId:'target-1'},accepted:1},'target-2':{task:{targetLayerId:'target-2'},accepted:2}};
  f.api.anatomyPlan=async args=>{
    f.calls.push(structuredClone(args));
    return {activeTargetLayerId:args.targetLayerId,approaches:[a,b,c].filter(route=>route.targetLayerId===args.targetLayerId),
      targetOptions,proposalReport:reports[args.targetLayerId]};
  };
  const h=await ready(f);
  const picker=tree=>[...walk(tree)].find(node=>node.type==='select'&&node.props['aria-label']==='Planning target');
  try{
    assert.equal(picker(h.render()).props.value,'target-2');
    assert.equal(f.planning.selectedId,'c');assert.equal(f.planning.report.accepted,2);
    picker(h.render()).props.onChange({target:{value:'target-1'}});await flush();h.render();
    assert.equal(f.planning.selectedId,'a');assert.equal(f.planning.report.accepted,1);
    picker(h.render()).props.onChange({target:{value:'target-2'}});await flush();h.render();
    assert.equal(f.planning.selectedId,'c');assert.equal(f.planning.report.accepted,2);
    assert.deepEqual(f.calls.map(call=>[call.action,call.targetLayerId]),[['list','target-2'],['list','target-1'],['list','target-2']]);
    assert.deepEqual(f.planning.selectedApproachIds,{'target-1':'a','target-2':'c'});
  }finally{h.close();}
});

test('a saved zero-alternative comparison restores its task and can be saved again',async()=>{
  const f=fixture([]);
  f.binding.viewState={snapshotId:'revision',selectedApproachId:null,selectedPlanningTargetId:'target-2',selectedApproachIds:{'target-2':null}};
  f.api.anatomyPlan=async()=>({activeTargetLayerId:'target-2',approaches:[],
    targetOptions:[{layerId:'target-2',label:'Target two',target:[5,8,9],entry:[5,8,40]}],
    proposalReport:{assessmentStatus:'current',accepted:0,task:{targetLayerId:'target-2',entry:[5,8,42],target:[5,8,10],radiusMm:2.5,
      entryPatch:{center:[5,8,41],normal:[0,0,-1],radiusMm:4},sensitivityConfig:{translationMm:2,boundaryMm:1}}}});
  const h=await ready(f);
  try{
    const tree=h.render();
    assert.equal(input(tree,'Entry patch radius').props.value,4);
    assert.equal(input(tree,'Corridor radius').props.value,2.5);
    assert.equal(input(tree,'Registration translation').props.value,2);
    assert.deepEqual(point(tree,'Target').props.value,[5,8,10]);
    assert.equal(button(tree,'Save study & comparison').props.disabled,false);
    assert.equal(f.planning.selectedId,null);
    const changes=[];
    f.workspace.state.visible={'target-2':false,cortex:true,skull:false};
    f.workspace.setVisible=visible=>changes.push(visible);
    f.workspace.setSelectedStructure=layerId=>changes.push(layerId);
    f.workspace.setCursor=endpoint=>changes.push(endpoint);
    button(tree,'Focus selected target').props.onClick();
    assert.deepEqual(changes,[{'target-2':true,cortex:true,skull:false},'target-2',[5,8,10]]);
  }finally{h.close();}
});

test('a prepared lesion requires explicit selection and sends its evidence binding before comparison',async()=>{
  const f=fixture([]);
  const lesion={layerId:'lesion',label:'Enhancing lesion',target:[5,8,9],origin:'Model estimate',
    reviewStatus:'unreviewed',candidateBinding:'a'.repeat(64),sourceSeries:'t1ce',layerRevision:1,labelValue:4};
  f.api.anatomyPlan=async args=>{
    f.calls.push(structuredClone(args));
    return {activeTargetLayerId:args.action==='selectTarget'?args.targetLayerId:null,approaches:[],targetOptions:[lesion]};
  };
  const h=await ready(f);
  try{
    let tree=h.render();
    assert.equal(button(tree,'Suggest approaches').props.disabled,true);
    const picker=[...walk(tree)].find(node=>node.type==='select'&&node.props['aria-label']==='Planning target');
    assert.equal(picker.props.value,'');
    const composition=[];
    f.workspace.state.cranialView='inspection';
    f.workspace.composeCranialView=mode=>composition.push(mode);
    picker.props.onChange({target:{value:'lesion'}});await flush();tree=h.render();
    const selection=f.calls.at(-1);
    assert.equal(selection.action,'selectTarget');
    assert.deepEqual(composition,['custom']);
    assert.equal(selection.targetLayerId,lesion.layerId);
    assert.equal(selection.candidateBinding,lesion.candidateBinding);
    assert.equal(button(tree,'Suggest approaches').props.disabled,false);
    assert(text(tree).includes('unreviewed'));
  }finally{h.close();}
});

test('independent scan cursor stays in its own frame while selecting a saved route',async()=>{
  const cursorWrites=[];
  const f=fixture(undefined,{patientFrameReady:false,setCursor:p=>cursorWrites.push(p)}),h=await ready(f);
  try{
    const tree=h.render();
    for(const label of ['Place patch at cursor','Entry at cursor','Target at cursor','Focus selected target']){
      assert.equal(button(tree,label).props.disabled,true,label);
    }
    await [...walk(tree)].find(n=>n.props['data-approach-id']==='b').props.onClick();
    assert.equal(f.planning.selectedId,'a');
    h.render();
    assert.equal(f.planning.selectedId,'b');
    assert.deepEqual(cursorWrites,[]);
  }finally{h.close();}
});

test('discarding an invalid saved-route edit restores the geometry and clears the navigation guard',async()=>{
  const changes=[];
  const f=fixture(undefined,{onEditStateChange:flag=>changes.push(flag)}),h=await ready(f);
  try{
    const tree=h.render();
    input(tree,'Entry patch radius').props.onChange({target:{value:'-1'}});
    assert.equal(changes.at(-1),true);
    const dirty=h.render();
    assert.equal(button(dirty,'Reassess').props.disabled,true);
    button(dirty,'Discard edits').props.onClick();
    assert.equal(input(h.render(),'Entry patch radius').props.value,7);
    assert.equal(changes.at(-1),false);
    assert.equal(f.calls.some(c=>c.action==='upsert'||c.action==='delete'),false);
  }finally{h.close();}
});
