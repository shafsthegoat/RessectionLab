/** Generated controls only. No acquired files, frozen engine or model runtime. */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {ALLOWED,PLAN,parseArgs,releaseGate,cleanEnvironment,startWithCleanEnvironment,narrowRequest,stopOwned,sha,jsonSha,canonicalJsonSha,assertPersistedJsonExact,memorySha,verifyAllAssets,descriptors,checkTissues,originalIdentities,checkExpectedArrays} from './first_release_frozen_sidecar.mjs';
import {hydrateWorkspace} from '../desktop/src/workspace-session.ts';
import {checkedRehearsalResponse,rehearsalViewerState} from '../desktop/src/rehearsal-data.ts';
const require=createRequire(import.meta.url),{Sidecar}=require('../desktop/electron/sidecar.cjs');
const h='a'.repeat(64),hash=`sha256:${h}`;
test('persistence fingerprint ignores recursive object-key order while raw JSON fingerprint remains order-sensitive',()=>{
 const before={state:'generated',actions:[{kind:'aspirate',toolId:'aspirator',durationSeconds:2,tipRasMm:[1,2,3]}],nested:{z:{b:2,a:1},a:true}};
 const reordered={nested:{a:true,z:{a:1,b:2}},actions:[{tipRasMm:[1,2,3],durationSeconds:2,toolId:'aspirator',kind:'aspirate'}],state:'generated'};
 assert.notEqual(jsonSha(before),jsonSha(reordered));assert.equal(canonicalJsonSha(before),canonicalJsonSha(reordered));assertPersistedJsonExact(reordered,before,'Generated persisted episode');
});
for(const [name,mutate] of [
 ['scalar value',x=>x.actions[0].durationSeconds++],
 ['numeric type',x=>x.actions[0].durationSeconds='2'],
 ['array order',x=>x.actions[0].tipRasMm.reverse()],
 ['missing property',x=>delete x.actions[0].kind],
 ['extra property',x=>x.actions[0].unexpected=true],
 ['signed zero',x=>x.actions[0].tipRasMm[0]=-0],
 ['null replaced by missing',x=>delete x.unknownRisk],
])test(`exact persistence comparison rejects ${name}`,()=>{
 const expected={actions:[{kind:'aspirate',durationSeconds:2,tipRasMm:[0,2,3]}],unknownRisk:null},actual=structuredClone(expected);mutate(actual);
 assert.throws(()=>assertPersistedJsonExact(actual,expected,'Generated persisted episode'),/values\/types\/array order changed/);
 if(name!=='signed zero')assert.notEqual(canonicalJsonSha(actual),canonicalJsonSha(expected));
});
function gateFixture(){
 const prep={schema:'first-release-frozen-sidecar-preparation-v1',plan:structuredClone(PLAN),auditorSha256:h,controlsSha256:h};
 const actual={preparationSha256:h,auditorSha256:h,controlsSha256:h,executable:{path:'/generated/Contents/Resources/research-engine/ressectionlab-engine'},appAsar:{path:'/generated/Contents/Resources/app.asar'},outputDirectory:'/generated/outputs/audit'};
 const release={schema:'first-release-frozen-sidecar-root-release-v1',authority:'ROOT',status:'RELEASED',buildCompleted:true,unreviewedEngineByteDifferencesAllowed:false,...structuredClone(actual),executable:{...actual.executable,sha256:h},appAsar:{...actual.appAsar,sha256:h},packageManifestSha256:h,engineSourceManifestSha256:h};return {prep,actual,release};
}
test('complete generated release binds plan, auditor, tests, preparation, paths and final hashes',()=>{const f=gateFixture();assert.equal(releaseGate(f.prep,f.release,f.actual),true);});
for(const [name,mutate] of [
 ['withheld release',f=>f.release.status='WITHHELD'],['missing build success',f=>delete f.release.buildCompleted],['wrong authority',f=>f.release.authority='worker'],
 ['auditor changed',f=>f.actual.auditorSha256='b'.repeat(64)],['controls changed',f=>f.actual.controlsSha256='b'.repeat(64)],['preparation changed',f=>f.actual.preparationSha256='b'.repeat(64)],
 ['unbound prepared auditor',f=>f.prep.auditorSha256='b'.repeat(64)],['unbound prepared controls',f=>f.prep.controlsSha256='b'.repeat(64)],
 ['wrong engine path',f=>f.actual.executable.path='/editable/python'],['wrong asar path',f=>f.actual.appAsar.path='/other/app.asar'],['unreleased output',f=>f.actual.outputDirectory='/other'],
 ['missing final engine hash',f=>delete f.release.executable.sha256],['missing final asar hash',f=>delete f.release.appAsar.sha256],
 ['missing source manifest',f=>delete f.release.engineSourceManifestSha256],['missing package manifest',f=>delete f.release.packageManifestSha256],
 ['silent signing exception',f=>f.release.unreviewedEngineByteDifferencesAllowed=true],['changed aspiration plan',f=>f.prep.plan.aspiration.durationSeconds=4],
])test(`release refuses ${name} before launch or payload access`,()=>{const f=gateFixture();mutate(f);assert.throws(()=>releaseGate(f.prep,f.release,f.actual));});
test('CLI requires explicit absolute packaged paths and refuses source/Python fallback',()=>{
 const valid=['--executable','/engine','--package-asar','/app.asar','--preparation','/prep','--release','/release','--output','/outputs'];assert.equal(parseArgs(valid)['--executable'],'/engine');
 for(const input of [[],['--python','/python'],[...valid,'--sourcePath','/src'],[...valid,'--executable','/other'],['--prepare','--output','/x'],valid.map(x=>x==='/engine'?'relative':x)])assert.throws(()=>parseArgs(input));
 assert.deepEqual(parseArgs(['--prepare']),{prepare:true});
});
test('clean environment drops Python/VENV/CONDA/DYLD and restores parent after synchronous spawn',()=>{
 const dirty={PYTHONPATH:'/source',PYTHONHOME:'/venv',VIRTUAL_ENV:'/venv',CONDA_PREFIX:'/conda',DYLD_LIBRARY_PATH:'/inject',LD_PRELOAD:'/inject',QT_PLUGIN_PATH:'/qt',NODE_OPTIONS:'--require /bad',PATH:'/bad',HOME:'/old'};
 const clean=cleanEnvironment(dirty,'/neutral');for(const key of Object.keys(dirty).filter(x=>!['PATH','HOME'].includes(x)))assert.equal(clean[key],undefined);
 assert.equal(clean.HOME,dirty.HOME);assert.equal(cleanEnvironment({},'/neutral').HOME,undefined);assert.equal(clean.PATH,'/usr/bin:/bin:/usr/sbin:/sbin');
 const original=process.env;let captured;
 class Capture{constructor(options){captured={options,env:{...process.env}};}}
 startWithCleanEnvironment(Capture,{executable:'/frozen',cwd:'/neutral',transferDir:'/neutral/transfer',runDir:'/neutral/run'},'/neutral/tmp');
 assert.equal(process.env,original);assert.equal(captured.env.HOME,original.HOME);assert.equal(captured.env.TMPDIR,'/neutral/tmp');assert.equal(captured.options.python,undefined);assert.equal(captured.options.sourcePath,undefined);
 class Throws{constructor(){throw new Error('generated spawn failure');}}assert.throws(()=>startWithCleanEnvironment(Throws,{executable:'/frozen'},'/neutral'));assert.equal(process.env,original);
 for(const extra of [{python:'/python'},{sourcePath:'/src'},{observedSourceRoot:'/src'}])assert.throws(()=>startWithCleanEnvironment(Capture,{executable:'/frozen',...extra},'/neutral'));
});
test('narrow requester denies all generic training/search/model/native paths before process call',async()=>{
 let calls=0;const journal=[],request=narrowRequest({request:async(op)=>{calls++;return {op};}},journal);
 for(const op of ['trainPatient','nativeTraining','generateRoutes','generateNativeRoutes','inspectRefinement','anatomyReconstruct','createSyntheticCase','executeDevelopmentEpisode','importNifti','importDisplaySeries','shutdown'])await assert.rejects(request(op),/not allowed/);
 assert.equal(calls,0);for(const op of ALLOWED)await request(op);assert.equal(calls,7);assert.equal(journal.length,7);
});
function episodeFixture(){
 const affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],volume={caseHash:hash,caseId:'generated',frame:'RAS+',shape:[8,8,8],affine,mri:new Float32Array(512).fill(1),compartments:[{name:'target',mask:new Uint8Array(512).fill(1),color:'#aabbcc'}]};
 const tools=[['aspirator','aspiration',2],['field_manager','field_management',5]].map(([toolId,mode,x])=>({toolId,mode,tipRasMm:[x,3,4],axis:[0,0,1],active:false,workingLengthMm:6,tipLengthMm:1,shaftRadiusMm:.2,tipRadiusMm:.4}));
 const metrics=n=>({targetRemovedMm3:n,nonTargetRemovedMm3:0,residualTargetMm3:400-n,fieldBurden:0,vascularEvents:null,motorRisk:null,languageRisk:null});
 const episode={schema:'resectionlab.source-bound-rehearsal.v2',sourceBinding:{caseHash:hash,shape:volume.shape,affine,frame:'RAS+'},initialStateId:'start',finalStateId:'end',tools,assumedExposureIndicesNative:[[0,0,0]],finalRemovedIndicesNative:[[1,1,1]],history:[{kind:'aspiration'},{kind:'field_management'}],replayFrames:[
  {frameIndex:0,actionIndex:-1,kind:'initial',elapsedSeconds:0,tools,stateBefore:'start',stateAfter:'start',removedIndicesNative:[],contactIndicesNative:[],fieldContactIndicesNative:[],metrics:metrics(0)},
  {frameIndex:1,actionIndex:0,kind:'aspiration',elapsedSeconds:1,tools,stateBefore:'start',stateAfter:'middle',removedIndicesNative:[[1,1,1]],contactIndicesNative:[[1,1,2]],fieldContactIndicesNative:[],metrics:metrics(1)},
  {frameIndex:2,actionIndex:1,kind:'field_management',elapsedSeconds:2,tools,stateBefore:'middle',stateAfter:'end',removedIndicesNative:[],contactIndicesNative:[[1,1,2]],fieldContactIndicesNative:[[1,1,2]],metrics:metrics(1)},
 ]};
 const response={caseHash:hash,stateId:'end',episode,suggestedActions:[],comparison:{snapshotId:'end',alternatives:[{id:'a',label:'A',episode:structuredClone(episode)},{id:'b',label:'B',episode:structuredClone(episode)}]}};return {volume,response};
}
test('generated renderer contract reconstructs distinct aspiration and field masks with two exact prefixes',()=>{
 const {volume,response}=episodeFixture(),before=memorySha(volume.mri),checked=checkedRehearsalResponse(volume,response),view=rehearsalViewerState(volume,checked.episode,2);
 assert.equal(view.removedMask[73],1);assert.equal(view.removedMask[74],0);assert.equal(view.fieldContactMask[74],1);assert.equal(checked.comparison.alternatives.length,2);assert.equal(memorySha(volume.mri),before);
});
for(const [name,mutate] of [
 ['branch prefix mutation',r=>r.comparison.alternatives[1].episode.replayFrames[1].tools[0].tipRasMm[0]++],
 ['stale live state',r=>r.stateId='stale'],['unknown risk invented',r=>r.episode.replayFrames[1].metrics.motorRisk=.1],
 ['duplicate removed cell',r=>r.episode.replayFrames[2].removedIndicesNative=[[1,1,1]]],
 ['source grid changed',r=>r.episode.sourceBinding.shape=[7,8,8]],
])test(`generated renderer refuses ${name}`,()=>{const {volume,response}=episodeFixture();mutate(response);assert.throws(()=>checkedRehearsalResponse(volume,response));});
function tissueFixture(){
 const names=['atropos_csf_like_estimate','atropos_gray_matter_like_estimate','atropos_white_matter_like_estimate','unknown_missing_intensity'];
 const masks=names.map(()=>new Uint8Array(16));masks[0].fill(1,0,3);masks[1].fill(1,3,5);masks[2].fill(1,5,8);masks[3].fill(1,8,16);
 const source={seriesId:'estimated',sourceCaseHash:'generated-estimate',sourceFrameHash:'generated-frame',annotationKind:'estimated'},mri=new Float32Array(16).fill(2);
 const descriptor={...source,planningEligible:false,evaluationEligible:false,registration:{overlayPermitted:false}};
 const workspace={volume:{mri,compartments:[{name:'brain',mask:new Uint8Array(16).fill(1)}]},source:{workspaceSession:{originalFiles:[{name:'generated.nii',role:'imaging',sha256:h,bytes:42}]}},series:[{descriptor,volume:{caseHash:source.sourceCaseHash,mri:mri.slice(),compartments:names.map((name,i)=>({name,mask:masks[i]}))}}]};
 const pin={estimatedSource:source,originals:originalIdentities(workspace.source.workspaceSession.originalFiles),sources:[source],layerCounts:Object.fromEntries(names.map((n,i)=>[n,[3,2,3,8][i]])),unknownCount:8,unknownNativeIndicesSha256:jsonSha([8,9,10,11,12,13,14,15])};return {workspace,pin};
}
test('generated four tissue layers preserve exact eight UNKNOWN cells and brain partition',()=>{const {workspace,pin}=tissueFixture();assert.equal(checkTissues(workspace,pin).exactUnknownNativeCells,true);});
for(const [name,mutate] of [
 ['lost UNKNOWN',f=>f.workspace.series[0].volume.compartments[3].mask[8]=0],
 ['UNKNOWN moved with same count',f=>{const l=f.workspace.series[0].volume.compartments;l[3].mask[8]=0;l[3].mask[0]=1;l[0].mask[0]=0;l[0].mask[8]=1;}],
 ['overlap',f=>f.workspace.series[0].volume.compartments[0].mask[8]=1],
 ['changed source MRI',f=>f.workspace.series[0].volume.mri[0]++],
 ['invented fusion',f=>f.workspace.series[0].descriptor.registration.overlayPermitted=true],
 ['changed original identity',f=>f.workspace.source.workspaceSession.originalFiles[0].sha256='b'.repeat(64)],
 ['missing fourth layer',f=>f.workspace.series[0].volume.compartments.pop()],
])test(`generated tissue payload refuses ${name}`,()=>{const f=tissueFixture();mutate(f);assert.throws(()=>checkTissues(f.workspace,f.pin));});
test('binary verifier traverses working/source/brain masks and refuses missing or mutated bytes',async()=>{
 const bytes=new Uint8Array([1,0,1,0]),d={assetId:'a',dtype:'uint8',byteLength:4,sha256:sha(bytes)},source={brainMask:d,compartments:[{array:d,sourceArray:d}]};
 assert.equal(Object.keys(await verifyAllAssets(source,{readAsset:async()=>bytes})).length,3);
 await assert.rejects(verifyAllAssets(source,{readAsset:async()=>new Uint8Array([0,0,1,0])}),/hash mismatch/);
 await assert.rejects(verifyAllAssets(source,{readAsset:async()=>bytes.slice(1)}));assert.throws(()=>descriptors({mri:{...d,path:'/raw/patient'}}),/path leaked/);
 const expected={mri:{raw_c_order_sha256:sha(bytes)},compartments:{mask:{raw_c_order_sha256:sha(bytes)}},source_compartments:{mask:{raw_c_order_sha256:sha(bytes)}}};
 assert.throws(()=>checkExpectedArrays({compartments:[{name:'mask'}]},{mri:{sha256:sha(bytes)},'compartments.0.array':{sha256:sha(bytes)}},expected));
});
test('actual host Sidecar generated executable verifies clean spawn, binary hydration, four operations and owned shutdown',async()=>{
 const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'frozen-audit-generated-'));let child;
 try{
  const executable=path.join(temporary,'generated-engine'),transferDir=path.join(temporary,'transfer'),cwd=path.join(temporary,'neutral'),runDir=path.join(temporary,'runs');for(const dir of [transferDir,cwd,runDir])await fs.mkdir(dir);
  const {response}=episodeFixture();
  const program=`#!${process.execPath}\nconst fs=require('node:fs'),path=require('node:path'),crypto=require('node:crypto'),rl=require('node:readline');
const transfer=process.argv[process.argv.indexOf('--transfer-dir')+1];const b=Buffer.from(new Float32Array(512).fill(1).buffer);fs.writeFileSync(path.join(transfer,'mri.bin'),b);
const m=Buffer.alloc(512,1);fs.writeFileSync(path.join(transfer,'mask.bin'),m);
const desc=(name,bytes,dtype)=>({path:path.join(transfer,name),dtype,shape:[8,8,8],byteOrder:'little',order:'C',byteLength:bytes.length,sha256:crypto.createHash('sha256').update(bytes).digest('hex')});
const source={caseId:'generated',caseHash:${JSON.stringify(hash)},frame:'RAS+',shape:[8,8,8],spacingMm:[1,1,1],affine:[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],mri:desc('mri.bin',b,'float32'),compartments:[{name:'target',array:desc('mask.bin',m,'uint8'),sourceArray:desc('mask.bin',m,'uint8'),volumeMm3:512}],unknowns:['generated only'],metadata:{}};
const response=${JSON.stringify(response)};
rl.createInterface({input:process.stdin}).on('line',line=>{const q=JSON.parse(line);if(q.op==='shutdown')process.exit(0);let result=q.op==='ping'?{cwd:process.cwd(),env:process.env,argv:process.argv.slice(2)}:q.op==='loadCase'?source:response;process.stdout.write(JSON.stringify({id:q.id,event:'result',result})+'\\n');});\n`;
  await fs.writeFile(executable,program,{mode:0o700});
  child=startWithCleanEnvironment(Sidecar,{executable,cwd,transferDir,runDir},temporary);
  const request=narrowRequest(child,[]),ping=await request('ping');assert.equal(ping.cwd,await fs.realpath(cwd));assert.equal(ping.env.HOME,process.env.HOME);assert.equal(ping.env.PYTHONPATH,undefined);assert.equal(ping.env.VIRTUAL_ENV,undefined);assert.equal(ping.env.DYLD_LIBRARY_PATH,undefined);assert.deepEqual(ping.argv,['--transfer-dir',transferDir,'--run-dir',runDir]);
  const source=await request('loadCase',{path:'/generated-only'}),assets=await verifyAllAssets(source,{readAsset:id=>child.assets.read(id)}),workspace=await hydrateWorkspace(source,{readAsset:id=>child.assets.read(id)});assert.equal(Object.keys(assets).length,3);assert.equal(workspace.volume.mri.length,512);
  for(const op of ['createRehearsal','inspectRehearsal','stepRehearsal','compareRehearsal']){const result=await request(op,{caseHash:hash});assert.equal(checkedRehearsalResponse(workspace.volume,result).comparison.alternatives.length,2);}
  const oldId=source.mri.assetId;await request('loadCase',{path:'/generated-only'});await assert.rejects(child.assets.read(oldId),/Unknown or expired/);
  const result=await stopOwned(child);assert.equal(result.exitCode,0);child=null;
 }finally{if(child)await stopOwned(child);await fs.rm(temporary,{recursive:true,force:true});}
});
