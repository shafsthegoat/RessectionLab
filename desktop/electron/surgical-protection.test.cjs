'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{createRequire}=require('node:module');
function fixture(){
  const handlers=new Map(),calls=[],dialogs=[],frame={url:'file:///trusted-app.html'},window={webContents:{mainFrame:frame},isDestroyed:()=>false};
  let selection={canceled:false,filePaths:['/workspace/explicit-protection.json']};
  const main=path.join(__dirname,'main.cjs'),localRequire=createRequire(main),engine={request:async(...call)=>{calls.push(call);return {caseHash:'case',changed:true};}};
  const context=vm.createContext({__dirname,process,Buffer,console,__engine:engine,__window:window,__url:frame.url,require:name=>name==='electron'?{app:{requestSingleInstanceLock:()=>false,quit(){},on(){}},dialog:{showOpenDialog:async(...args)=>{dialogs.push(args);return selection;}},ipcMain:{handle:(name,callback)=>handlers.set(name,callback)}}:localRequire(name)});
  vm.runInContext(fs.readFileSync(main,'utf8'),context);vm.runInContext('engine=__engine;window=__window;allowedUrl=__url;currentCaseHash="case";bindOperations();',context);
  return {calls,dialogs,select:value=>{selection=value;},invoke:args=>handlers.get('research:surgicalRehearsal')({sender:window.webContents,senderFrame:frame},args)};
}
const request=()=>({caseHash:'case',command:'attachProtection',expectedStateId:'accepted-state'});
test('protection attachment forwards only the explicit main-process JSON selection',async()=>{
  const f=fixture();await f.invoke(request());
  assert.equal(f.dialogs.length,1);assert.equal(f.dialogs[0][1].filters[0].extensions[0],'json');
  assert.deepEqual(JSON.parse(JSON.stringify(f.calls)),[['surgicalRehearsal',{...request(),protectionPath:'/workspace/explicit-protection.json'},120000]]);
  await assert.rejects(f.invoke({...request(),protectionPath:'/arbitrary/renderer-path.json'}),/Unsupported operation argument/);
  assert.equal(f.calls.length,1);assert.equal(f.dialogs.length,1,'renderer paths are rejected before opening a dialog');
});
test('cancelled selection and foreign or unaccepted workspace requests cannot attach protection',async()=>{
  const f=fixture();f.select({canceled:true,filePaths:[]});assert.equal(await f.invoke(request()),null);assert.equal(f.calls.length,0);
  for(const args of [{...request(),caseHash:'foreign'},{...request(),expectedStateId:undefined},{...request(),source:{collectionId:'external'}}])await assert.rejects(f.invoke(args),/current workspace/);
  assert.equal(f.dialogs.length,1);assert.equal(f.calls.length,0);
});
test('checked consequence reassessment forwards the explicit policy and seal without a file dialog',async()=>{
  const f=fixture(),policyInput={minimumAdditionalTargetMm3:9,maximumCumulativeNonTargetMm3:4940,requiredMetrics:['motorEncounterMm3'],maxBrainStrainSquaredMm3S:null,maxBrainStressSquaredPa2Mm3S:null};
  const args={caseHash:'case',command:'reevaluateConsequences',expectedStateId:'accepted-state',expectedComparisonSeal:'sha256:comparison',policyInput};
  await f.invoke(args);
  assert.deepEqual(JSON.parse(JSON.stringify(f.calls[0][1])),args);assert.equal(f.dialogs.length,0);
  await assert.rejects(f.invoke({...args,analysisPath:'/workspace/renderer-selected.json'}),/Unsupported operation argument/);
  assert.equal(f.calls.length,1);
});
test('interactive approach commands forward bounded editor data and checked candidate identity',async()=>{
  const f=fixture(),constraints={target:{centerRasMm:[-84,118,66],radiusMm:4},entry:{centerRasMm:[-84,118,42],radiusMm:7},tool:{instrument:'cusa',radiusMm:1},requiredEvidence:[]};
  for(const args of [
    {caseHash:'case',command:'generateApproaches',expectedStateId:'accepted-state',constraints,acknowledgeResearchUnknowns:true},
    {caseHash:'case',command:'previewApproach',expectedStateId:'accepted-state',expectedPlanSeal:'sha256:report',candidateId:'approach-1'},
    {caseHash:'case',command:'applyApproach',expectedStateId:'accepted-state',expectedPlanSeal:'sha256:report',candidateId:'approach-1'},
  ]){await f.invoke(args);assert.deepEqual(JSON.parse(JSON.stringify(f.calls.at(-1)[1])),args);}
  assert.equal(f.dialogs.length,0);
  await assert.rejects(f.invoke({caseHash:'case',command:'generateApproaches',reportPath:'/renderer/path'}),/Unsupported operation argument/);
  assert.equal(f.calls.length,3);
});

test('two-tool task forwards explicit controls, phase preview and sealed installation identity',async()=>{
  const f=fixture(),constraints={geometry:{requiredEvidence:['functional']},shortSeconds:.25,longSeconds:1,minimumTargetMm3:1};
  for(const args of [
    {caseHash:'case',command:'generateStrategies',expectedStateId:'state',constraints,acknowledgeResearchUnknowns:true},
    {caseHash:'case',command:'analyzeStrategySensitivity',expectedStateId:'state',expectedPlanSeal:'seal',offsetMm:1,acknowledgeResearchUnknowns:true},
    {caseHash:'case',command:'previewStrategy',expectedStateId:'state',expectedPlanSeal:'seal',candidateId:'long',phase:'reassess'},
    {caseHash:'case',command:'applyStrategy',expectedStateId:'state',expectedPlanSeal:'seal',candidateId:'long'},
  ]){await f.invoke(args);assert.deepEqual(JSON.parse(JSON.stringify(f.calls.at(-1)[1])),args);}
  assert.equal(f.dialogs.length,0);
  await assert.rejects(f.invoke({caseHash:'case',command:'generateStrategies',privateReferencePath:'/private/oracle'}),/Unsupported operation argument/);
});
