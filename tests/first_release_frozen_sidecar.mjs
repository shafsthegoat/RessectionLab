/** Opt-in packaged-engine audit. --prepare reads source/receipt metadata only.
 * Actual mode needs a separate ROOT release; no Python/source fallback exists.
 * Full protocol records belong under outputs, never in compact Git artifacts.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createRequire} from 'node:module';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {hydrateWorkspace, checkedImagingState} from '../desktop/src/workspace-session.ts';
import {workspaceDisplay} from '../desktop/src/workspace-imaging-data.ts';
import {checkedRehearsalResponse, rehearsalViewerState} from '../desktop/src/rehearsal-data.ts';

export const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
export const PREPARATION='artifacts/first-release-frozen-sidecar-preparation-v3';
export const SELF='tests/first_release_frozen_sidecar.mjs';
export const CONTROLS='tests/test_first_release_frozen_sidecar.mjs';
export const ALLOWED=Object.freeze(['ping','loadCase','createRehearsal','inspectRehearsal','stepRehearsal','compareRehearsal','saveCase']);
export const PLAN=Object.freeze({schema:'frozen-sidecar-audit-plan-v1',
  aspiration:{kind:'aspirate',toolId:'aspirator',durationSeconds:2,advanceAlongInitialAxisMm:2},
  fieldManagement:{kind:'clear_field',toolId:'field_manager',durationSeconds:2,advanceAlongInitialAxisMm:0.5},
  alternatives:2, stop:{kind:'STOP',durationSeconds:0,reason:'frozen_package_protocol_audit_STOP_under_unknown_anatomy'},
  generatedOnlyBeforeRootRelease:true, automaticRetries:0,
  scope:'Packaged host Sidecar, actual bundled engine, binary transfer, source-bound renderer contracts, live rehearsal and fresh-process persistence. No Electron GUI, anatomy/physics accuracy or clinical acceptance.',
  isolation:'Explicit executable from emitted app; packaged host modules; neutral cwd; cleaned Python/VENV/CONDA/DYLD/LD/QT environment; narrow request allowlist. No audit hook or OS sandbox inside the frozen binary: absence of internal model imports, network or subprocess activity is not enforced or claimed.',
});
const RECEIPTS=[
 'artifacts/poc-first-rehearsal-v1/gui-save-audit.json',
 'artifacts/poc-ixi265-tissue-display-package-v1/outcome.json',
 'artifacts/poc-ixi265-tissue-display-package-v1/bundle-inventory.json',
 'artifacts/poc-ixi265-tissue-display-package-v1/package.json',
 'artifacts/ixi265-tissue-display-independent-v1/independent-qa.json',
 'artifacts/poc-synthrad-bc-preparation-v1/receipt.json',
 'artifacts/poc-synthrad-bc-integration-v1/independent-qa.json',
 'artifacts/poc-synthrad-bc-integration-v1/gui-save-independent-qa.json',
];
export const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
export const jsonSha=value=>sha(JSON.stringify(value));
// Saving JSON sorts object keys. Persistence compares values and ordered arrays;
// raw request/result fingerprints deliberately keep their original encoding.
export function canonicalJsonSha(value){
 const ordered=item=>Array.isArray(item)?item.map(ordered):item&&typeof item==='object'?Object.fromEntries(Object.keys(item).sort().map(key=>[key,ordered(item[key])])):item;
 return jsonSha(ordered(value));
}
export function assertPersistedJsonExact(actual,expected,label){
 assert.deepEqual(actual,expected,`${label} values/types/array order changed`);
 assertHash(canonicalJsonSha(actual),canonicalJsonSha(expected),label);
}
export const memorySha=value=>sha(new Uint8Array(value.buffer,value.byteOffset,value.byteLength));
export async function digest(filename){const h=createHash('sha256');for await(const chunk of createReadStream(filename))h.update(chunk);return h.digest('hex');}
const json=async filename=>JSON.parse(await fs.readFile(filename,'utf8'));
const writeJson=async(filename,value)=>fs.writeFile(filename,JSON.stringify(value,null,2)+'\n',{flag:'wx'});
const assertHash=(actual,expected,label)=>assert.equal(actual,expected,`${label} hash mismatch`);
const under=(parent,child)=>{const relative=path.relative(parent,child);return relative!==''&&!relative.startsWith('..'+path.sep)&&relative!=='..'&&!path.isAbsolute(relative);};
export const originalIdentities=rows=>rows.map(({name,role,sha256,bytes})=>({name,role,sha256,bytes})).sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b)));
async function files(directory,prefix=''){
 const result=[];for(const entry of (await fs.readdir(directory,{withFileTypes:true})).sort((a,b)=>a.name.localeCompare(b.name))){
  const rel=path.join(prefix,entry.name);if(entry.isDirectory())result.push(...await files(path.join(directory,entry.name),rel));else if(entry.isFile())result.push(rel);
 }return result;
}
async function hashFiles(root,names){return Object.fromEntries(await Promise.all(names.sort().map(async name=>[name,await digest(path.join(root,name))])));}

/** Preparation uses only existing JSON receipts, source text and hashes. */
export async function prepare(destination=path.join(ROOT,PREPARATION)){
 const receipts=Object.fromEntries(await Promise.all(RECEIPTS.map(async name=>[name,await json(path.join(ROOT,name))])));
 const u=receipts[RECEIPTS[0]].cases.find(x=>x.case==='UCSF-PDGM-0004');assert.ok(u);
 const inventory=receipts[RECEIPTS[2]], iq=receipts[RECEIPTS[4]].result;
 const bc=receipts[RECEIPTS[6]].cases.find(x=>x.subject_id==='1BC077');
 const bg=receipts[RECEIPTS[7]].cases.find(x=>x.subject_id==='1BC077');
 const desktopNames=[...await files(path.join(ROOT,'desktop/src'),'src'),...await files(path.join(ROOT,'desktop/electron'),'electron'),...['index.html','vite.config.ts','tsconfig.json','package.json','pnpm-lock.yaml']];
 const locks=(await fs.readdir(ROOT,{withFileTypes:true})).filter(x=>x.isFile()&&x.name.includes('lock')).map(x=>x.name);
 const pythonNames=[...(await files(path.join(ROOT,'src/resectionlab'),'src/resectionlab')).filter(x=>x.endsWith('.py')),'packaging/sidecar_entry.py','packaging/python_notices.py','scripts/build_electron_sidecar.py','pyproject.toml',...locks];
 const prep={schema:'first-release-frozen-sidecar-preparation-v1',status:'prepared_generated_controls_only_actual_run_withheld',createdUtc:new Date().toISOString(),plan:PLAN,
  auditorSha256:await digest(path.join(ROOT,SELF)),controlsSha256:await digest(path.join(ROOT,CONTROLS)),
  desktopSourceHashes:await hashFiles(path.join(ROOT,'desktop'),desktopNames),pythonSourceHashes:await hashFiles(ROOT,pythonNames),
  receiptHashes:await hashFiles(ROOT,RECEIPTS),
  inputs:{
   ucsf:{path:'outputs/poc-cases/UCSF-PDGM-0004.ressectionlab',sha256:'73cd0792c025e846395bafffe5ec793ef9e1823ceec8e4b0aeb8317783041571',bytes:31495606,caseHash:u.case_hash,
    arrays:u.arrays,auxiliary:u.auxiliary_sources.map(x=>({caseHash:x.case_hash,arrays:x.arrays})),originals:originalIdentities(u.archived_originals.files)},
   ixi:{path:'outputs/poc-ixi265-tissue-display-v1/IXI265-Guys-0845-tissue-estimates.ressectionlab',sha256:'bfa03b853dcb8e6893982e75a6c113e8ba3de1ce88225bb631f8a758ded1350f',bytes:113750604,
    caseHash:inventory.workspace_manifest.referenceCaseHash,originals:originalIdentities(inventory.workspace_manifest.originalFiles),
    sources:inventory.workspace_manifest.images.map(x=>x.source),estimatedSource:iq.added_source,layerCounts:iq.layer_voxel_counts,
    unknownNativeIndicesSha256:jsonSha(iq.unknown_indices_native.map(([x,y,z])=>x*256*150+y*150+z).sort((a,b)=>a-b)),unknownCount:8},
   synthrad:{path:'outputs/poc-synthrad-bc-source-v1/1BC077/SynthRAD2023-1BC077.ressectionlab',sha256:'171101168d1244d509fc64249cd9039a74aa9a35f35535dddc753307e80f905d',caseHash:bc.case_hash,
    arrays:bg.source_CT_arrays_and_metadata,auxiliaryArrays:bg.native_MR_arrays_and_metadata,originals:originalIdentities(bg.originals.files)},
  },
  priorEvidenceQualification:'Input identities derive from the named existing receipts. No acquired array decoding, model execution, frozen engine launch or actual package acceptance occurred during preparation.',
 };
 assert.equal(u.canonical_stamp.sha256,prep.inputs.ucsf.sha256);assert.equal(inventory.bundle_sha256,prep.inputs.ixi.sha256);assert.equal(bc.bundle_sha256,prep.inputs.synthrad.sha256);
 await fs.mkdir(destination,{recursive:true});await writeJson(path.join(destination,'preparation.json'),prep);
 const template={schema:'first-release-frozen-sidecar-root-release-v1',authority:'ROOT',status:'WITHHELD',buildCompleted:false,
  preparationSha256:await digest(path.join(destination,'preparation.json')),auditorSha256:prep.auditorSha256,controlsSha256:prep.controlsSha256,
  executable:{path:null,sha256:null},appAsar:{path:null,sha256:null},packageManifestSha256:null,engineSourceManifestSha256:null,outputDirectory:null,
  scope:PLAN.scope,unreviewedEngineByteDifferencesAllowed:false};
 await writeJson(path.join(destination,'root-release-template.json'),template);return prep;
}

export function parseArgs(argv){
 if(argv.length===1&&argv[0]==='--prepare')return {prepare:true};
 const permitted=new Set(['--executable','--package-asar','--preparation','--release','--output']),out={};
 for(let i=0;i<argv.length;i+=2){assert.ok(permitted.has(argv[i])&&argv[i+1]&&!argv[i+1].startsWith('--'),`Unknown or incomplete argument ${argv[i]}`);assert.equal(out[argv[i]],undefined,'Duplicate argument');out[argv[i]]=argv[i+1];assert.ok(path.isAbsolute(argv[i+1]),'All paths must be absolute');}
 assert.equal(Object.keys(out).length,permitted.size,'Explicit executable, package, preparation, release and output required');return out;
}
export function releaseGate(prep,release,actual){
 assert.equal(prep.schema,'first-release-frozen-sidecar-preparation-v1');assert.deepEqual(prep.plan,PLAN,'Frozen request plan changed');
 assert.equal(release.schema,'first-release-frozen-sidecar-root-release-v1');assert.equal(release.authority,'ROOT');assert.equal(release.status,'RELEASED');assert.equal(release.buildCompleted,true);
 assert.equal(release.unreviewedEngineByteDifferencesAllowed,false);
 for(const key of ['preparationSha256','auditorSha256','controlsSha256'])assertHash(actual[key],release[key],key);
 assertHash(prep.auditorSha256,actual.auditorSha256,'prepared auditor');assertHash(prep.controlsSha256,actual.controlsSha256,'prepared controls');
 for(const key of ['executable','appAsar']){assert.equal(actual[key].path,release[key]?.path,`${key} path mismatch`);assert.match(release[key]?.sha256??'',/^[a-f0-9]{64}$/);}
 assert.equal(actual.outputDirectory,release.outputDirectory,'Output release mismatch');
 for(const key of ['packageManifestSha256','engineSourceManifestSha256'])assert.match(release[key]??'',/^[a-f0-9]{64}$/);
 return true;
}
export function cleanEnvironment(original,temporary){
 const env={PATH:'/usr/bin:/bin:/usr/sbin:/sbin',TMPDIR:temporary,LANG:'en_US.UTF-8',LC_ALL:'en_US.UTF-8',PYTHONNOUSERSITE:'1'};
 if(typeof original.HOME==='string')env.HOME=original.HOME;
 // No inherited Python, virtualenv, dyld, Qt, Node or project override survives.
 return env;
}
export function startWithCleanEnvironment(Sidecar,options,temporary){
 assert.ok(options.executable&&!('python' in options)&&!('sourcePath' in options)&&!('observedSourceRoot' in options),'Source fallback refused');
 const previous=process.env;try{process.env=cleanEnvironment(previous,temporary);return new Sidecar(options);}finally{process.env=previous;}
}
export function narrowRequest(sidecar,journal){return async(op,args={},timeout=120000)=>{
 assert.ok(ALLOWED.includes(op),`Audit request not allowed: ${op}`);
 const row={op,args,startedUtc:new Date().toISOString()};journal.push(row);
 try{const result=await sidecar.request(op,args,timeout);row.resultSha256=jsonSha(result);row.status='result';return result;}
 catch(error){row.status='error';row.error=String(error);throw error;}
 finally{row.elapsedMs=Date.now()-Date.parse(row.startedUtc);}
};}
export async function stopOwned(sidecar){
 const child=sidecar.child,result={pid:child.pid,method:'Sidecar.stop',exitCode:null,signalCode:null};
 await sidecar.stop();
 const exited=()=>child.exitCode!==null||child.signalCode!==null;
 const wait=ms=>new Promise(resolve=>{if(exited())return resolve();const end=()=>{clearTimeout(timer);child.off('exit',end);resolve();};const timer=setTimeout(end,ms);child.once('exit',end);});
 if(!exited()){child.kill('SIGTERM');result.method+='; owned SIGTERM';await wait(2000);}
 if(!exited()){child.kill('SIGKILL');result.method+='; owned SIGKILL';await wait(2000);}
 assert.ok(exited(),'Owned child did not exit');result.exitCode=child.exitCode;result.signalCode=child.signalCode;return result;
}
export function descriptors(value,prefix=''){
 if(!value||typeof value!=='object')return [];
 if('dtype' in value&&'byteLength' in value){assert.equal('path' in value,false,'Binary path leaked');assert.equal(typeof value.assetId,'string');return [[prefix,value]];}
 return Object.entries(value).flatMap(([key,row])=>descriptors(row,prefix?`${prefix}.${key}`:key));
}
export async function verifyAllAssets(source,api){
 const result={};for(const [key,d] of descriptors(source)){const bytes=await api.readAsset(d.assetId);assert.equal(bytes.byteLength,d.byteLength);assertHash(memorySha(bytes),d.sha256.replace(/^sha256:/,''),key);result[key]={sha256:memorySha(bytes),bytes:bytes.byteLength};}return result;
}
export const volumeFingerprint=v=>({caseHash:v.caseHash,shape:v.shape,affineSha256:jsonSha(v.affine),mri:memorySha(v.mri),layers:Object.fromEntries(v.compartments.map(x=>[x.name,memorySha(x.mask)]))});
export const workspaceFingerprint=w=>({primary:volumeFingerprint(w.volume),sources:w.series.map(x=>({seriesId:x.descriptor.seriesId,...volumeFingerprint(x.volume)}))});
export function checkExpectedArrays(source,all,expected,prefix=''){
 const key=name=>prefix?`${prefix}.${name}`:name;
 assertHash(all[key('mri')].sha256,expected.mri.raw_c_order_sha256,'Expected acquired MRI bytes');
 if(expected.brain_mask)assertHash(all[key('brainMask')].sha256,expected.brain_mask.raw_c_order_sha256,'Expected brain bytes');
 assert.deepEqual(source.compartments.map(x=>x.name).sort(),Object.keys(expected.compartments??{}).sort());
 source.compartments.forEach((layer,index)=>{
  assertHash(all[key(`compartments.${index}.array`)].sha256,expected.compartments[layer.name].raw_c_order_sha256,layer.name);
  if(expected.source_compartments?.[layer.name])assertHash(all[key(`compartments.${index}.sourceArray`)].sha256,expected.source_compartments[layer.name].raw_c_order_sha256,`${layer.name} source`);
 });
}
export function checkTissues(workspace,pin){
 const series=workspace.series.find(x=>x.descriptor.seriesId===pin.estimatedSource.seriesId);assert.ok(series,'Estimated tissue source missing');
 assert.equal(series.volume.caseHash,pin.estimatedSource.sourceCaseHash);assert.equal(series.descriptor.sourceFrameHash,pin.estimatedSource.sourceFrameHash);
 assert.equal(series.descriptor.annotationKind,'estimated');assert.equal(series.descriptor.planningEligible,false);assert.equal(series.descriptor.evaluationEligible,false);
 assert.ok(workspace.series.every(x=>x.descriptor.registration.overlayPermitted===false));
 const layers=Object.fromEntries(series.volume.compartments.map(x=>[x.name,x.mask]));assert.deepEqual(Object.keys(layers).sort(),Object.keys(pin.layerCounts).sort());
 const unknown=layers.unknown_missing_intensity,indices=[];const brain=workspace.volume.compartments.find(x=>x.name==='brain')?.mask;assert.ok(brain);assert.equal(brain.length,unknown.length);
 const sums=Object.fromEntries(Object.keys(layers).map(name=>[name,0]));
 for(let index=0;index<unknown.length;index++){
  let total=0;for(const [name,mask] of Object.entries(layers)){assert.ok(mask[index]===0||mask[index]===1);total+=mask[index];sums[name]+=mask[index];}
  assert.equal(total,brain[index],'Tissue and UNKNOWN masks must partition the original brain exactly');if(unknown[index])indices.push(index);
 }
 assert.deepEqual(sums,pin.layerCounts);assert.equal(indices.length,pin.unknownCount);assertHash(jsonSha(indices),pin.unknownNativeIndicesSha256,'Exact UNKNOWN native cells');
 assert.deepEqual(series.volume.mri,workspace.volume.mri,'Estimated display must retain original T1 intensities');
 assert.deepEqual(originalIdentities(workspace.source.workspaceSession.originalFiles),pin.originals);
 for(const expected of pin.sources){const actual=workspace.series.find(x=>x.descriptor.seriesId===expected.seriesId);assert.ok(actual);assert.equal(actual.volume.caseHash,expected.sourceCaseHash);assert.equal(actual.descriptor.sourceFrameHash,expected.sourceFrameHash);assert.equal(actual.descriptor.annotationKind,expected.annotationKind);}
 return {layerCounts:sums,exactUnknownNativeCells:true,originalIdentities:pin.originals.length,brainPartitionExact:true,displayOnly:true};
}

/** Source paths here are metadata bindings, never a fallback execution path. */
async function packageGate(options,prep,release,report){
 const executable=options['--executable'],asarPath=options['--package-asar'];
 assert.equal(await fs.realpath(executable),executable,'Executable must use its exact real path');
 assert.equal(await fs.realpath(asarPath),asarPath,'Package must use its exact real path');
 assert.equal(path.basename(executable),'ressectionlab-engine');assert.equal(path.basename(path.dirname(executable)),'research-engine');
 const resources=path.dirname(path.dirname(executable));assert.equal(path.join(resources,'app.asar'),asarPath);assert.equal(path.basename(resources),'Resources');assert.equal(path.basename(path.dirname(resources)),'Contents');
 assertHash(await digest(executable),release.executable.sha256,'Final executable');assertHash(await digest(asarPath),release.appAsar.sha256,'Final app.asar');
 const fd=await fs.open(executable,'r'),magic=Buffer.alloc(4);try{await fd.read(magic,0,4,0);}finally{await fd.close();}
 assert.ok(['cffaedfe','cefaedfe','feedfacf','feedface','cafebabe','bebafeca'].includes(magic.toString('hex')),'Actual Mach-O bundled engine required');
 const desktopRequire=createRequire(path.join(ROOT,'desktop/package.json'));
 const packagerRequire=createRequire(desktopRequire.resolve('@electron/packager'));
 const asar=await import(pathToFileURL(packagerRequire.resolve('@electron/asar')).href);
 const packageBytes=asar.extractFile(asarPath,'BUILD_INPUT_MANIFEST.json');assertHash(sha(packageBytes),release.packageManifestSha256,'Package build manifest');
 const manifest=JSON.parse(packageBytes);report.packageBinding={executableSha256:release.executable.sha256,appAsarSha256:release.appAsar.sha256,packageManifestSha256:sha(packageBytes),engineByteDifferences:[]};
 const engineRoot=path.dirname(executable),engineNames=await filesAndLinks(engineRoot);assert.deepEqual(Object.keys(manifest.engineHashes).sort(),engineNames.map(x=>x.name).sort(),'Emitted engine inventory differs');
 for(const entry of engineNames){const actual=entry.link===undefined?await digest(path.join(engineRoot,entry.name)):`symlink:${entry.link}`;if(actual!==manifest.engineHashes[entry.name])report.packageBinding.engineByteDifferences.push({path:entry.name,beforeSigningSha256:manifest.engineHashes[entry.name],emittedSha256:actual});}
 assert.equal(report.packageBinding.engineByteDifferences.length,0,'Emitted engine bytes differ from pre-sign build manifest; classification requires separate evidence, not automatic acceptance');
 const engineManifestPath=path.join(engineRoot,'_internal/build_info/BUILD_INPUT_MANIFEST.json');assertHash(await digest(engineManifestPath),release.engineSourceManifestSha256,'Embedded Python build manifest');
 const engineManifest=await json(engineManifestPath);
 assert.deepEqual(engineManifest.source_files,prep.pythonSourceHashes,'Exact frozen Python source inventory differs');
 for(const [name,expected] of Object.entries(prep.pythonSourceHashes)){assertHash(await digest(path.join(ROOT,name)),expected,`Prepared Python ${name}`);if(name.startsWith('src/resectionlab/'))assertHash(await digest(path.join(engineRoot,'_internal',name.slice(4))),expected,`Embedded Python source ${name}`);}
 for(const [name,expected] of Object.entries(prep.desktopSourceHashes)){assertHash(manifest.sourceHashes[name],expected,`Packaged desktop ${name}`);assertHash(await digest(path.join(ROOT,'desktop',name)),expected,`Tested desktop ${name}`);}
 const captured=path.join(options['--output'],'packaged-host');await fs.mkdir(captured);
 for(const [name,expected] of Object.entries(prep.desktopSourceHashes).filter(([name])=>name.startsWith('electron/'))){assert.ok(!name.split('/').includes('..'));const bytes=asar.extractFile(asarPath,name);assertHash(sha(bytes),expected,`Packaged host ${name}`);await fs.mkdir(path.dirname(path.join(captured,name)),{recursive:true});await fs.writeFile(path.join(captured,name),bytes,{flag:'wx'});}
 const hostRequire=createRequire(path.join(captured,'electron/sidecar.cjs'));const host=hostRequire('./sidecar.cjs');ALLOWED.forEach(op=>assert.ok(host.OPERATIONS.has(op)));
 report.packageBinding.packagedHostExecuted=true;report.packageBinding.rendererSourceContractsBound=true;report.packageBinding.compiledRendererGuiExecuted=false;report.packageBinding.embeddedPythonManifestSha256=release.engineSourceManifestSha256;
 return host.Sidecar;
}
async function filesAndLinks(root,prefix=''){
 const result=[];for(const e of await fs.readdir(path.join(root,prefix),{withFileTypes:true})){const name=path.join(prefix,e.name);if(e.isDirectory())result.push(...await filesAndLinks(root,name));else if(e.isSymbolicLink()){const target=await fs.realpath(path.join(root,name));assert.ok(under(root,target),'Engine symlink escapes bundle');result.push({name,link:await fs.readlink(path.join(root,name))});}else if(e.isFile())result.push({name});}return result;
}

export async function run(options){
 const output=options['--output'];assert.ok(under(path.join(ROOT,'outputs'),output),'Actual records must stay under outputs');
 await fs.mkdir(output,{recursive:false}); // Refuse overwrite; preserve every failed attempt.
 const report={schema:'first-release-frozen-sidecar-audit-v1',startedUtc:new Date().toISOString(),status:'running',scope:PLAN.scope,isolation:PLAN.isolation,checks:{},failures:[],sessions:[],journal:[]};
 let current=null,neutral=null,prep=null,release=null;const watched=new Map();
 try{
  prep=await json(options['--preparation']);release=await json(options['--release']);
  releaseGate(prep,release,{preparationSha256:await digest(options['--preparation']),auditorSha256:await digest(path.join(ROOT,SELF)),controlsSha256:await digest(path.join(ROOT,CONTROLS)),executable:{path:options['--executable']},appAsar:{path:options['--package-asar']},outputDirectory:output});
  report.checks.rootRelease=true;
  const Sidecar=await packageGate(options,prep,release,report);
  watched.set(options['--executable'],release.executable.sha256);watched.set(options['--package-asar'],release.appAsar.sha256);
  for(const [name,expected] of Object.entries(prep.receiptHashes))assertHash(await digest(path.join(ROOT,name)),expected,`Input receipt ${name}`);
  for(const pin of Object.values(prep.inputs)){const filename=path.join(ROOT,pin.path),actual=await digest(filename);assertHash(actual,pin.sha256,'Pinned input bundle');if(pin.bytes!==undefined)assert.equal((await fs.stat(filename)).size,pin.bytes);watched.set(filename,actual);}
  neutral=await fs.mkdtemp(path.join(os.tmpdir(),'ressection-frozen-audit-'));assert.ok(!under(ROOT,neutral));
  let sessionCounter=0;
  async function start(){
   const label=`engine-${++sessionCounter}`,base=path.join(neutral,label);await fs.mkdir(base);const cwd=path.join(base,'cwd'),transferDir=path.join(base,'transfer'),runDir=path.join(base,'runs'),scratchTmp=path.join(base,'tmp');for(const dir of [cwd,transferDir,runDir,scratchTmp])await fs.mkdir(dir);
   current=startWithCleanEnvironment(Sidecar,{executable:options['--executable'],cwd,transferDir,runDir},scratchTmp);
   const entry={label,pid:current.child.pid,cwd,environment:cleanEnvironment(process.env,scratchTmp),diagnostics:[],events:[]};report.sessions.push(entry);
   current.on('diagnostic',text=>entry.diagnostics.push(text));current.on('event',event=>entry.events.push({id:event.id,op:event.op,event:event.event,error:event.error??null}));
   const request=narrowRequest(current,report.journal);await request('ping');return request;
  }
  async function stop(){const entry=report.sessions.at(-1);entry.cleanup=await stopOwned(current);current=null;}
  async function load(request,filename,pin){
   const source=await request('loadCase',{path:filename});assert.equal(source.caseHash,pin.caseHash);
   const api={readAsset:id=>current.assets.read(id)},assets=await verifyAllAssets(source,api),workspace=await hydrateWorkspace(source,api);
   assert.deepEqual(originalIdentities(source.workspaceSession.originalFiles),pin.originals);return {source,assets,workspace,fingerprint:workspaceFingerprint(workspace)};
  }
  let request=await start();
  let loaded=await load(request,path.join(ROOT,prep.inputs.ucsf.path),prep.inputs.ucsf);
  const pin=prep.inputs.ucsf,source=loaded.source,volume=loaded.workspace.volume,key=source.caseHash,original=loaded.fingerprint,originalFiles=structuredClone(source.workspaceSession.originalFiles);
  checkExpectedArrays(source,loaded.assets,pin.arrays);
  for(const aux of pin.auxiliary){const index=source.workspaceSession.displaySeries.findIndex(x=>x.volume.caseHash===aux.caseHash);assert.ok(index>=0);checkExpectedArrays(source.workspaceSession.displaySeries[index].volume,loaded.assets,aux.arrays,`workspaceSession.displaySeries.${index}.volume`);}
  const created=await request('createRehearsal',{caseHash:key});checkedRehearsalResponse(volume,created);
  const tool=created.episode.replayFrames[0].tools.find(t=>t.toolId==='aspirator');
  const advanced=await request('stepRehearsal',{caseHash:key,expectedStateId:created.stateId,action:{kind:'aspirate',toolId:'aspirator',durationSeconds:PLAN.aspiration.durationSeconds,tipRasMm:tool.tipRasMm.map((v,i)=>v+PLAN.aspiration.advanceAlongInitialAxisMm*tool.axis[i])}});
  assert.notEqual(advanced.accepted,false);checkedRehearsalResponse(volume,advanced);assert.notEqual(advanced.stateId,created.stateId);assert.ok(advanced.episode.metrics.removedMm3>0);
  await assert.rejects(request('stepRehearsal',{caseHash:key,expectedStateId:created.stateId,action:{kind:'inspect',durationSeconds:1}}),/REHEARSAL_STATE_STALE/);
  const wrongMode=await request('stepRehearsal',{caseHash:key,expectedStateId:advanced.stateId,action:{kind:'aspirate',toolId:'field_manager',durationSeconds:1}});assert.equal(wrongMode.accepted,false);assert.equal(wrongMode.stateId,advanced.stateId);
  const field=created.episode.replayFrames[0].tools.find(t=>t.toolId==='field_manager');
  const managed=await request('stepRehearsal',{caseHash:key,expectedStateId:advanced.stateId,action:{kind:'clear_field',toolId:'field_manager',durationSeconds:PLAN.fieldManagement.durationSeconds,tipRasMm:field.tipRasMm.map((v,i)=>v+PLAN.fieldManagement.advanceAlongInitialAxisMm*field.axis[i])}});
  assert.notEqual(managed.accepted,false);checkedRehearsalResponse(volume,managed);assert.equal(managed.episode.metrics.removedMm3,advanced.episode.metrics.removedMm3);
  assert.deepEqual(managed.episode.finalRemovedIndicesNative,advanced.episode.finalRemovedIndicesNative);assert.ok(managed.episode.metrics.fieldManagementCellSeconds>0);assert.ok(managed.episode.metrics.fieldBurden<advanced.episode.metrics.fieldBurden);
  assert.ok(managed.episode.replayFrames.at(-1).fieldContactIndicesNative.length>0);
  const compared=await request('compareRehearsal',{caseHash:key,expectedStateId:managed.stateId}),checked=checkedRehearsalResponse(volume,compared),prefix=managed.episode.replayFrames;
  assert.equal(compared.stateId,managed.stateId);assert.deepEqual(compared.episode,managed.episode);assert.equal(checked.comparison.alternatives.length,2);assert.equal(new Set(checked.comparison.alternatives.map(x=>x.id)).size,2);
  assert.notEqual(jsonSha(compared.comparison.alternatives[0].episode.replayFrames.slice(prefix.length)),jsonSha(compared.comparison.alternatives[1].episode.replayFrames.slice(prefix.length)),'Two alternatives must differ after their identical prefix');
  for(const branch of checked.comparison.alternatives){const raw=compared.comparison.alternatives.find(x=>x.id===branch.id);assert.deepEqual(raw.episode.replayFrames.slice(0,prefix.length),prefix);const view=rehearsalViewerState(volume,branch.episode,branch.episode.replayFrames.length-1);assert.equal(view.removedMask.reduce((a,b)=>a+b,0),branch.episode.finalRemovedIndicesNative.length);assert.equal(view.tools.length,2);for(const risk of ['motorRisk','languageRisk','vascularEvents'])assert.equal(view.metrics[risk],null);}
  for(const injected of [{research_authority:'unrestricted'},{source_compartments:{fake:true}},{target_names:['reference']},{patient_rehearsal:{state:'forged'}}])await assert.rejects(request('createRehearsal',{caseHash:key,...injected}),/INVALID_ARGUMENT/);
  const invalid=await request('stepRehearsal',{caseHash:key,expectedStateId:managed.stateId,action:{kind:'inspect',durationSeconds:1,annotationReplacement:'forbidden'}});assert.equal(invalid.accepted,false);assert.equal(invalid.stateId,managed.stateId);
  const unchanged=await request('inspectRehearsal',{caseHash:key});assert.deepEqual(unchanged.episode,compared.episode);assert.deepEqual(unchanged.comparison,compared.comparison);
  const stopped=await request('stepRehearsal',{caseHash:key,expectedStateId:managed.stateId,action:PLAN.stop});assert.notEqual(stopped.accepted,false);assert.equal(stopped.episode.metrics.terminated,true);checkedRehearsalResponse(volume,stopped);
  const afterStop=await request('stepRehearsal',{caseHash:key,expectedStateId:stopped.stateId,action:{kind:'inspect',durationSeconds:1}});assert.equal(afterStop.accepted,false);assert.equal(afterStop.stateId,stopped.stateId);
  const view={branchId:compared.comparison.alternatives[0].id,frameIndex:prefix.length-1,selectedTool:'field_manager',visible:true,episodeInitialStateId:created.episode.initialStateId};
  const savedPath=path.join(output,'UCSF-frozen-reviewed.ressectionlab');
  await request('saveCase',{caseHash:key,path:savedPath,workspace:{rehearsalView:view}});
  const forged=path.join(output,'must-not-exist.ressectionlab');await assert.rejects(request('saveCase',{caseHash:key,path:forged,workspace:{patient_rehearsal:{source:'untrusted'}}}),/INVALID_ARGUMENT/);await assert.rejects(fs.stat(forged),/ENOENT/);
  assert.deepEqual(workspaceFingerprint(loaded.workspace),original);
  const episodeHash=canonicalJsonSha(stopped.episode),comparisonHash=canonicalJsonSha(compared.comparison),firstPid=current.child.pid;
  await fs.writeFile(path.join(output,'stopped-rehearsal.json'),JSON.stringify(stopped));await stop();
  request=await start();assert.notEqual(current.child.pid,firstPid);loaded=await load(request,savedPath,pin);assert.deepEqual(loaded.fingerprint,original);assert.deepEqual(loaded.source.workspaceSession.originalFiles,originalFiles);
  const restored=await request('inspectRehearsal',{caseHash:key});checkedRehearsalResponse(loaded.workspace.volume,restored);assertPersistedJsonExact(restored.episode,stopped.episode,'Fresh engine episode');assertPersistedJsonExact(restored.comparison,compared.comparison,'Fresh engine branches');assert.deepEqual(restored.savedView,view);
  report.checks.ucsf={allBinaryAssetsVerified:Object.keys(loaded.assets).length,sourceHashesExact:true,liveAspiration:true,fieldManagementChangesFieldWithoutRemovingTissue:true,allFourRehearsalOperations:true,staleAndPrivilegedStateRefused:true,identicalPrefixAlternatives:2,stopped:true,freshEngineEpisodeSha256:episodeHash,freshEngineComparisonSha256:comparisonHash,sourceFingerprintSha256:jsonSha(original),originalIdentities:originalFiles.length,savedBundleSha256:await digest(savedPath)};
  // Source switching is the actual renderer selection contract plus persisted
  // backend imaging state. It is explicitly not a GUI interaction claim.
  const oldAsset=loaded.source.mri.assetId;loaded=await load(request,path.join(ROOT,prep.inputs.ixi.path),prep.inputs.ixi);await assert.rejects(current.assets.read(oldAsset),/Unknown or expired/);
  report.checks.ixi={...checkTissues(loaded.workspace,prep.inputs.ixi),allBinaryAssetsVerified:Object.keys(loaded.assets).length,sourceFingerprintSha256:jsonSha(loaded.fingerprint)};
  loaded=await load(request,path.join(ROOT,prep.inputs.synthrad.path),prep.inputs.synthrad);checkExpectedArrays(loaded.source,loaded.assets,prep.inputs.synthrad.arrays);
  assert.equal(loaded.workspace.series.length,1);const mr=loaded.workspace.series[0];assert.equal(mr.descriptor.modality,'T1','Native T1 MRI source missing');assert.equal(mr.descriptor.registration.overlayPermitted,false);
  const mrIndex=loaded.source.workspaceSession.displaySeries.findIndex(x=>x.seriesId===mr.descriptor.seriesId);checkExpectedArrays(mr.descriptor.volume,loaded.assets,prep.inputs.synthrad.auxiliaryArrays,`workspaceSession.displaySeries.${mrIndex}.volume`);
  assert.equal(workspaceDisplay(loaded.workspace.volume,mr).caseData,mr.volume);assert.equal(workspaceDisplay(loaded.workspace.volume,mr).primaryOverlaysPermitted,false);
  const bcOriginal=loaded.fingerprint,bcPin=prep.inputs.synthrad;
  let imaging=checkedImagingState(loaded.source,loaded.source.workspaceSession.displaySeries,{...loaded.workspace.imagingState,selectedSeriesId:mr.descriptor.seriesId});
  const bcPath=path.join(output,'SynthRAD-frozen-MR-selected.ressectionlab');await request('saveCase',{caseHash:loaded.source.caseHash,path:bcPath,workspace:{imaging}});loaded=await load(request,bcPath,bcPin);assert.equal(loaded.workspace.imagingState.selectedSeriesId,mr.descriptor.seriesId);assert.deepEqual(loaded.fingerprint,bcOriginal);
  imaging=checkedImagingState(loaded.source,loaded.source.workspaceSession.displaySeries,{...loaded.workspace.imagingState,selectedSeriesId:null});assert.equal(workspaceDisplay(loaded.workspace.volume,null).caseData,loaded.workspace.volume);
  const ctPath=path.join(output,'SynthRAD-frozen-CT-selected.ressectionlab');await request('saveCase',{caseHash:loaded.source.caseHash,path:ctPath,workspace:{imaging}});loaded=await load(request,ctPath,bcPin);assert.equal(loaded.workspace.imagingState.selectedSeriesId,null);assert.deepEqual(loaded.fingerprint,bcOriginal);
  report.checks.synthrad={nativeCtMrSelectionAndPersistence:true,fusionPermitted:false,sourceHashesExact:true,originalIdentities:bcPin.originals.length,sourceFingerprintSha256:jsonSha(bcOriginal)};
  await stop();report.status='passed';
 }catch(error){report.status='failed';report.failures.push({name:error.name,message:error.message,stack:error.stack});}
 finally{
  if(current){try{report.sessions.at(-1).cleanup=await stopOwned(current);}catch(error){report.failures.push({name:error.name,message:error.message});report.status='failed';}}
  for(const [filename,before] of watched){try{assertHash(await digest(filename),before,'Input bundle after audit');}catch(error){report.failures.push({name:error.name,message:error.message});report.status='failed';}}
  report.finishedUtc=new Date().toISOString();report.elapsedSeconds=(Date.parse(report.finishedUtc)-Date.parse(report.startedUtc))/1000;
  report.runtime={node:process.version,auditorSha256:await digest(path.join(ROOT,SELF))};
  await writeJson(path.join(output,'report.json'),report);
  // No raw errors, native coordinates, image values or request arguments enter
  // this compact record; detailed failures remain beside report.json.
  await writeJson(path.join(output,'compact-summary.json'),{schema:report.schema,status:report.status,scope:report.scope,isolation:report.isolation,startedUtc:report.startedUtc,finishedUtc:report.finishedUtc,checks:report.checks,packageBinding:report.packageBinding??null,failures:report.failures.map((x,index)=>({index,name:x.name,details:'report.json'})),requests:report.journal.map(x=>({operation:x.op,status:x.status,elapsedMs:x.elapsedMs})),ownedChildrenExited:report.sessions.every(x=>x.cleanup&&(x.cleanup.exitCode!==null||x.cleanup.signalCode!==null)),generatedControlsAreNotPatientAcceptance:true});
  // Keep the neutral directory and logs on failure; successful transfer dirs
  // are removed by Sidecar.stop, then remove only our owned empty scratch tree.
  if(neutral&&report.status==='passed')await fs.rm(neutral,{recursive:true,force:true});
 }
 return report;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){
 try{const options=parseArgs(process.argv.slice(2));if(options.prepare){await prepare();console.log(JSON.stringify({status:'prepared_only',directory:PREPARATION}));}else{const result=await run(options);console.log(JSON.stringify({status:result.status,report:path.join(options['--output'],'report.json')}));process.exitCode=result.status==='passed'?0:1;}}
 catch(error){console.error(JSON.stringify({status:'refused',name:error.name,message:error.message}));process.exitCode=1;}
}
