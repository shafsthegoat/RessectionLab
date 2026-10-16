/** Opt-in actual host Sidecar -> Python -> assets -> renderer protocol audit.
 * Run with node --experimental-strip-types and explicit --case/--python/output.
 * No GUI, inference, implementation edits, or retained patient copies.
 */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {createReadStream} from 'node:fs';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
import {execFileSync} from 'node:child_process';
import {hydrateWorkspace} from '../desktop/src/workspace-session.ts';
import {checkedRehearsalResponse,rehearsalViewerState} from '../desktop/src/rehearsal-data.ts';

const require=createRequire(import.meta.url);
const {Sidecar,OPERATIONS}=require('../desktop/electron/sidecar.cjs');
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const arg=name=>{const i=process.argv.indexOf(name);return i<0?null:process.argv[i+1]};
const casePath=arg('--case'),python=arg('--python'),output=arg('--output'),artifactsRoot=arg('--artifacts-root');
assert.ok(casePath&&python&&output&&artifactsRoot,'Explicit --case --python --artifacts-root --output required');
assert.ok(path.isAbsolute(casePath)&&path.isAbsolute(python)&&path.isAbsolute(artifactsRoot),'Use absolute source paths');
const operations=['createRehearsal','inspectRehearsal','stepRehearsal','compareRehearsal'];
const recordPaths=[
 'UCSF-PDGM-0004/synthstrip-main/fe693f2aed5cb0bb90aa',
 'UCSF-PDGM-0004/monai-brats-0.5.4/af24e4d8f13e49e37ff8',
 'SynthRAD2023-1BA336/totalsegmentator-skull/32e4c06b003e5b904387',
 'Lausanne-sub-000-ses-20110101/vesselfm-base/0cf5010a6c39c81d7f4b',
];
const digest=async p=>{const h=createHash('sha256');for await(const chunk of createReadStream(p))h.update(chunk);return h.digest('hex')};
const memoryDigest=x=>createHash('sha256').update(new Uint8Array(x.buffer,x.byteOffset,x.byteLength)).digest('hex');
const stamp=async p=>{const s=await fs.stat(p,{bigint:true});return {bytes:String(s.size),mtimeNs:String(s.mtimeNs),sha256:await digest(p)}};
const started=new Date().toISOString();
const report={scope:'Actual Sidecar process/JSON protocol, binary asset authorization, frontend hydration and rehearsal validation; no Electron GUI or physics validation',startedUtc:started,checks:{},failures:[],sessions:[]};
const watched=new Map(),sourceFiles=['desktop/electron/sidecar.cjs','src/resectionlab/desktop_bridge.py','src/resectionlab/rehearsal_bridge.py','src/resectionlab/rehearsal_episode.py','desktop/src/rehearsal-data.ts','desktop/src/viewer/rehearsal.ts','tests/first_release_sidecar.mjs'];
report.testedSourceSha256=Object.fromEntries(await Promise.all(sourceFiles.map(async name=>[name,await digest(path.join(root,name))])));
for(const p of [casePath,...sourceFiles.map(name=>path.join(root,name))])watched.set(p,await stamp(p));
for(const relative of recordPaths){
 const p=path.join(artifactsRoot,'outputs/poc-models',relative,'inference.json');
 const record=JSON.parse(await fs.readFile(p,'utf8'));
 for(const item of [p,...[...(record.inputs??[record.input]),...record.weights,...record.outputs].map(row=>row.path)])if(!watched.has(item))watched.set(item,await stamp(item));
}
const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'ressection-sidecar-independent-'));
const guardLog=path.join(temporary,'python-audit.jsonl');
const launcher=path.join(temporary,'guarded-python');
// Run the same project interpreter/module with an audit hook. This is not a
// replacement engine: Sidecar launches the actual desktop_bridge from src.
// Any external subprocess or model-runtime import is refused and recorded.
await fs.writeFile(launcher,`#!${python}\nimport sys, os, json, runpy, datetime\nlog=${JSON.stringify(guardLog)}\ndef record(value):\n    with open(log,'a') as stream: stream.write(json.dumps(value)+'\\n')\nrecord({'event':'guard_started','pid':os.getpid(),'interpreter':sys.executable,'pythonpath':os.environ.get('PYTHONPATH'),'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})\ndef audit(event,args):\n    denied=event in {'subprocess.Popen','os.system','os.posix_spawn','os.fork','os.forkpty','os.exec'}\n    denied=denied or (event=='import' and str(args[0]).split('.')[0] in {'torch','tensorflow','monai'})\n    if denied:\n        record({'event':'blocked_inference_or_subprocess','audit_event':event,'detail':repr(args)[:1000]})\n        raise RuntimeError('Independent Sidecar audit forbids inference/model imports and subprocesses')\nsys.addaudithook(audit)\nassert sys.argv[1:4]==['-u','-m','resectionlab.desktop_bridge'],sys.argv\nsys.argv=['resectionlab.desktop_bridge',*sys.argv[4:]]\nrunpy.run_module('resectionlab.desktop_bridge',run_name='__main__')\n`,{mode:0o700});
let first=null,second=null;
async function start(label){
 const transferDir=path.join(temporary,label);await fs.mkdir(transferDir);
 const sidecar=new Sidecar({python:launcher,cwd:root,sourcePath:path.join(root,'src'),transferDir});
 const entry={label,pid:sidecar.child.pid,events:[],diagnostics:[]};report.sessions.push(entry);
 sidecar.on('event',event=>entry.events.push({id:event.id,op:event.op,event:event.event,error:event.error??null}));
 sidecar.on('diagnostic',text=>entry.diagnostics.push(text));
 await sidecar.request('ping');return sidecar;
}
const sourceFingerprints=volume=>({mri:memoryDigest(volume.mri),layers:Object.fromEntries(volume.compartments.map(l=>[l.name,memoryDigest(l.mask)]))});
try{
 operations.forEach(op=>assert.ok(OPERATIONS.has(op),`Host whitelist omits ${op}`));
 const prior=execFileSync('git',['show','HEAD:desktop/electron/sidecar.cjs'],{cwd:root,encoding:'utf8'});
 report.priorGuiNegative={source:'Parent reported actual Electron failure before this protocol check',observedMessage:'Unsupported research operation',cause:'The previous host Sidecar whitelist omitted the four rehearsal operations even though the Python bridge accepted them',priorHeadSidecarSha256:createHash('sha256').update(prior).digest('hex'),priorMissingOperations:operations.filter(op=>!prior.includes(`'${op}'`)),currentAllFourAdmitted:true,limitation:'This script does not claim to have independently operated or captured that GUI failure'};
 first=await start('first');
 const source=await first.request('loadCase',{path:casePath});
 assert.equal(source.caseId,'UCSF-PDGM-0004');assert.equal('path' in source.mri,false);assert.equal(typeof source.mri.assetId,'string');
 const hydrated=await hydrateWorkspace(source,{readAsset:id=>first.assets.read(id)});
 const original=sourceFingerprints(hydrated.volume),key=source.caseHash;
 const created=await first.request('createRehearsal',{caseHash:key});
 checkedRehearsalResponse(hydrated.volume,created);
 const tool=created.episode.replayFrames[0].tools.find(t=>t.toolId==='aspirator');
 const advanced=await first.request('stepRehearsal',{caseHash:key,expectedStateId:created.stateId,action:{kind:'aspirate',toolId:'aspirator',durationSeconds:2,tipRasMm:tool.tipRasMm.map((n,i)=>n+2*tool.axis[i])}});
 assert.notEqual(advanced.accepted,false);assert.notEqual(advanced.stateId,created.stateId);assert.ok(advanced.episode.metrics.removedMm3>0);
 checkedRehearsalResponse(hydrated.volume,advanced);
 await assert.rejects(first.request('stepRehearsal',{caseHash:key,expectedStateId:created.stateId,action:{kind:'inspect',durationSeconds:1}}),/REHEARSAL_STATE_STALE/);
 const compared=await first.request('compareRehearsal',{caseHash:key,expectedStateId:advanced.stateId});
 const checked=checkedRehearsalResponse(hydrated.volume,compared);
 assert.equal(compared.stateId,advanced.stateId);assert.equal(compared.comparison.snapshotId,advanced.stateId);
 const prefix=advanced.episode.replayFrames;
 report.branches=[];
 for(const branch of checked.comparison.alternatives){
  const raw=compared.comparison.alternatives.find(row=>row.id===branch.id);
  assert.deepEqual(raw.episode.replayFrames.slice(0,prefix.length),prefix);
  const replay=rehearsalViewerState(hydrated.volume,branch.episode,branch.episode.replayFrames.length-1);
  const count=replay.removedMask.reduce((a,b)=>a+b,0);
  assert.equal(count,branch.episode.finalRemovedIndicesNative.length);
  assert.equal(replay.tools.length,2);assert.ok(replay.tools.every(t=>t.tipRasMm.length===3&&t.axis.length===3));
  assert.equal(replay.metrics.motorRisk,null);assert.equal(replay.metrics.languageRisk,null);assert.equal(replay.metrics.vascularEvents,null);
  report.branches.push({id:branch.id,finalStateId:branch.episode.finalStateId,frames:branch.episode.replayFrames.length,removedNativeCells:count,metrics:branch.episode.metrics,frontendReplayMasksValidated:true});
 }
 // Named protocol authority: callers cannot provide masks, source state, or
 // a replacement authoritative episode through the permitted operations.
 for(const injected of [{research_authority:'unrestricted'},{source_compartments:{fake:true}},{target_names:['reference']}])await assert.rejects(first.request('createRehearsal',{caseHash:key,...injected}),/INVALID_ARGUMENT/);
 const invalid=await first.request('stepRehearsal',{caseHash:key,expectedStateId:advanced.stateId,action:{kind:'inspect',durationSeconds:1,annotationReplacement:'forbidden'}});
 assert.equal(invalid.accepted,false);assert.equal(invalid.stateId,advanced.stateId);
 const unchanged=await first.request('inspectRehearsal',{caseHash:key});
 assert.deepEqual(unchanged.episode,compared.episode);assert.deepEqual(unchanged.comparison,compared.comparison);
 assert.deepEqual(sourceFingerprints(hydrated.volume),original);
 const view={branchId:compared.comparison.alternatives[0].id,frameIndex:prefix.length,selectedTool:'field_manager',visible:true,episodeInitialStateId:created.episode.initialStateId};
 const savedPath=path.join(temporary,'saved.ressectionlab');
 await first.request('saveCase',{caseHash:key,path:savedPath,workspace:{rehearsalView:view}});
 await assert.rejects(first.request('saveCase',{caseHash:key,path:path.join(temporary,'forged.ressectionlab'),workspace:{patient_rehearsal:{source:'untrusted'}}}),/INVALID_ARGUMENT.*Evaluator\/source/);
 await assert.rejects(fs.stat(path.join(temporary,'forged.ressectionlab')),/ENOENT/);
 const firstPid=first.child.pid;await first.stop();assert.ok(first.child.exitCode!==null||first.child.signalCode!==null);first=null;
 second=await start('second');assert.notEqual(second.child.pid,firstPid);
 const reopened=await second.request('loadCase',{path:savedPath});assert.equal(reopened.caseHash,key);
 const rehydrated=await hydrateWorkspace(reopened,{readAsset:id=>second.assets.read(id)});
 assert.deepEqual(sourceFingerprints(rehydrated.volume),original);
 assert.deepEqual(reopened.workspaceSession.originalFiles,source.workspaceSession.originalFiles);
 const restored=await second.request('inspectRehearsal',{caseHash:key});
 checkedRehearsalResponse(rehydrated.volume,restored);
 assert.deepEqual(restored.episode,compared.episode);assert.deepEqual(restored.comparison,compared.comparison);assert.deepEqual(restored.savedView,view);
 const savedStamp=await stamp(savedPath);
 report.checks={allFourHostOperationsReachPython:true,assetPathsReplacedByAuthorizedIds:true,actualFrontendHydration:true,actualFrontendRehearsalValidation:true,nativeTriplesAndModesAccepted:true,nonemptySavedSnapshotBranchPrefixesExact:true,viewerReplayMasksMatchCumulativeRemoval:true,staleActionsRefused:true,authorityAndAnnotationReplacementRefused:true,sourceArraysUnchanged:true,freshProcessLiveAndBothBranchesExact:true,savedViewExact:true,originalFileDescriptorsExact:true};
 report.savedTemporaryBundle=savedStamp;
 await second.stop();assert.ok(second.child.exitCode!==null||second.child.signalCode!==null);second=null;
 const guards=(await fs.readFile(guardLog,'utf8')).trim().split('\n').map(line=>JSON.parse(line));
 assert.equal(guards.filter(row=>row.event==='guard_started').length,2);
 assert.equal(guards.filter(row=>row.event==='blocked_inference_or_subprocess').length,0);
 assert.ok(guards.every(row=>row.pythonpath===path.join(root,'src')));
 report.inferenceGuard={mechanism:'Temporary launcher installs Python audit hook before running the actual project desktop_bridge module; external processes and torch/tensorflow/monai imports refused',events:guards,modelImportsOrExternalSpawnsAttempted:false};
}catch(error){report.failures.push(error.stack??String(error));}
finally{
 if(first)await first.stop();if(second)await second.stop();
 report.watchedFiles=[];
 for(const [p,before] of watched){const after=await stamp(p);report.watchedFiles.push({path:p,before,after,unchanged:JSON.stringify(before)===JSON.stringify(after)});}
 report.sourceCacheAndImplementationUnchanged=report.watchedFiles.every(row=>row.unchanged);
 report.finishedUtc=new Date().toISOString();report.elapsedSeconds=(Date.parse(report.finishedUtc)-Date.parse(started))/1000;
 report.passed=report.failures.length===0&&report.sourceCacheAndImplementationUnchanged;
 await fs.mkdir(path.dirname(path.resolve(output)),{recursive:true});await fs.writeFile(output,JSON.stringify(report,null,2)+'\n');
 await fs.rm(temporary,{recursive:true,force:true});
 console.log(JSON.stringify({passed:report.passed,failures:report.failures,output:path.resolve(output)}));
}
process.exitCode=report.passed?0:1;
