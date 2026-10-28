'use strict';
const fs = require('node:fs/promises');
const {createReadStream} = require('node:fs');
const {createHash} = require('node:crypto');
const path = require('node:path');
const {execFile} = require('node:child_process');
const {promisify,isDeepStrictEqual} = require('node:util');
const execute = promisify(execFile);
const MATCHING_RUNTIME_COMMIT='01e5a365abb7cf9b158a27d49f1e1bfa83e593a2';
const MATCHING_RUNTIME_ARCHIVE_SHA256='bbaeaa5098c918b566523b4c098db96c742454f593a772fa84e597de1d51b3e2';
const MATCHING_RUNTIME_MANIFEST_SHA256='ee0dfb0e4cff704d5f42b2669efb3a24db0da615f01cc4ee631b525cb6d12ccb';
const runtimeRoot=repo=>path.join(repo,'testcases/outputs/integrated-goal-20261026/joint-frozen-runtime-v1/runtime');

function frozenCusaArgs({repo,source,branchId,checkpoint,output,kind='cusa'}) {
  if(typeof branchId!=='string'||!branchId||branchId.length>128||branchId.includes('\0'))throw Error('Choose a saved access approach');
  if(kind==='sector')return[path.join(repo,'scripts/execute_patient_sector_policy.py'),
    '--source',source,'--access-method',branchId,'--source-role','DEVELOPMENT',
    '--checkpoint',checkpoint,'--output',output,'--allow-matching-runtime'];
  if(kind!=='cusa')throw Error('Unknown frozen policy kind');
  return [path.join(repo,'scripts/run_patient_cusa_learning.py'),
    '--source',source,'--branch',branchId,'--source-role','DEVELOPMENT',
    '--frozen-model',checkpoint,'--output',output];
}
function validateFrozenResult(receipt,output,kind='cusa') {
  const episode=receipt?.execution,derived=receipt?.derived;
  if(receipt?.status!=='evaluated'||receipt.optimizerUpdates!==0||
      receipt.task?.role!=='DEVELOPMENT'||!receipt.source?.unchanged||
      receipt.exactReplay?.exactNativeReplay!==true||derived?.reopenedExact!==true||
      !episode||!['success','abstained','incomplete','refused'].includes(episode.status)||
      episode.taskGateMet!==(episode.status==='success')||typeof episode.cleanupComplete!=='boolean'||
      !Number.isFinite(episode.deltaTargetMm3)||episode.deltaTargetMm3<0||
      !Number.isFinite(episode.deltaNonTargetMm3)||episode.deltaNonTargetMm3<0||
      !Number.isSafeInteger(episode.admittedNonStopActions)||episode.admittedNonStopActions<0)throw Error('Frozen execution did not produce a verified episode');
  if(episode.status==='success'&&(!episode.cleanupComplete||episode.deltaTargetMm3<30||
      episode.deltaNonTargetMm3>20||episode.admittedNonStopActions<2))throw Error('Frozen execution has an inconsistent task gate');
  if(kind==='sector'&&episode.status==='success'&&
      (!Number.isFinite(episode.totalNonTargetMm3)||episode.totalNonTargetMm3>200||
       !Number.isFinite(episode.perSectorRemovedMm3?.near)||episode.perSectorRemovedMm3.near<10||
       !Number.isFinite(episode.perSectorRemovedMm3?.far)||episode.perSectorRemovedMm3.far<10))throw Error('Frozen sector execution has an inconsistent task gate');
  if(typeof derived.workspacePath!=='string'||path.extname(derived.workspacePath)!=='.ressectionlab'||
      path.dirname(path.resolve(derived.workspacePath))!==path.resolve(output)||
      !/^[a-f0-9]{64}$/.test(derived.sha256))throw Error('Frozen execution export is outside its experiment folder');
  return {workspacePath:path.resolve(derived.workspacePath),sha256:derived.sha256,
    status:episode.status,taskGateMet:episode.taskGateMet,cleanupComplete:episode.cleanupComplete,
    deltaTargetMm3:episode.deltaTargetMm3,deltaNonTargetMm3:episode.deltaNonTargetMm3};
}
async function fileHash(filename) {
  const hash=createHash('sha256');
  for await(const chunk of createReadStream(filename))hash.update(chunk);
  return hash.digest('hex');
}
function validateMatchingRuntimeAdmission(admission,receipt,manifest,{repo,checkpoint,checkpointSha}) {
  const runtime=runtimeRoot(repo),source=path.join(runtime,'source');
  if(admission?.schema!=='resectionlab.frozen-matching-runtime-admission.v1'||
      admission.reason!=='current_producer_differs_from_frozen_checkpoint'||
      admission.sourceArchiveSha256!==MATCHING_RUNTIME_ARCHIVE_SHA256||
      admission.gitCommit!==MATCHING_RUNTIME_COMMIT||
      admission.manifestSha256!==MATCHING_RUNTIME_MANIFEST_SHA256||
      admission.runtimeSource!==path.resolve(source)||
      admission.canonicalTestcases!==path.resolve(repo,'testcases')||
      admission.checkpointPath!==path.resolve(checkpoint)||
      admission.checkpointFileSha256!==checkpointSha||
      admission.executorSourceSha256!==manifest?.files?.['scripts/execute_patient_sector_policy.py']||
      receipt?.executorSourceSha256!==admission.executorSourceSha256||
      receipt?.model?.path!==path.resolve(checkpoint)||
      receipt?.model?.fileSha256!==checkpointSha||
      receipt?.model?.actorSha256!==admission.checkpointActorSha256||
      !isDeepStrictEqual(admission.matchingProducerIdentity,manifest?.producerIdentity)||
      !isDeepStrictEqual(receipt?.producerIdentity,manifest?.producerIdentity)||
      isDeepStrictEqual(admission.currentProducerIdentity,manifest?.producerIdentity)||
      manifest?.schema!=='resectionlab.frozen-matching-runtime-files.v1'||
      manifest?.gitCommit!==MATCHING_RUNTIME_COMMIT||
      manifest?.sourceArchiveSha256!==MATCHING_RUNTIME_ARCHIVE_SHA256)
    throw Error('Matching frozen runtime admission or producer identity differs');
  return {gitCommit:MATCHING_RUNTIME_COMMIT,sourceArchiveSha256:MATCHING_RUNTIME_ARCHIVE_SHA256};
}
async function matchingRuntimeEvidence(repo,output,checkpoint,receipt) {
  const admissionPath=path.join(path.dirname(output),'matching-runtime-admission.json');
  let admission;
  try {admission=JSON.parse(await fs.readFile(admissionPath,'utf8'));}
  catch(error) {if(error.code==='ENOENT')return null;throw error;}
  const runtime=runtimeRoot(repo),manifestPath=path.join(runtime,'files-manifest.json');
  if(await fileHash(manifestPath)!==MATCHING_RUNTIME_MANIFEST_SHA256)
    throw Error('Pinned matching-runtime file manifest changed');
  const manifest=JSON.parse(await fs.readFile(manifestPath,'utf8'));
  const provenance=validateMatchingRuntimeAdmission(admission,receipt,manifest,
    {repo,checkpoint,checkpointSha:await fileHash(checkpoint)});
  const files=manifest.files;
  if(!files||Object.keys(files).length!==236)throw Error('Matching-runtime file inventory differs');
  for(const [relative,expected] of Object.entries(files)) {
    if(!relative.startsWith('src/resectionlab/')&&
        !['scripts/execute_patient_sector_policy.py','scripts/run_patient_sector_learning.py'].includes(relative))
      throw Error('Matching-runtime file path differs');
    const filename=path.join(runtime,'source',relative);
    if((await fs.lstat(filename)).isSymbolicLink()||await fileHash(filename)!==expected)
      throw Error('Matching-runtime source file changed');
  }
  const linked=path.join(runtime,'source/testcases');
  if(!(await fs.lstat(linked)).isSymbolicLink()||
      await fs.realpath(linked)!==path.resolve(repo,'testcases'))
    throw Error('Matching runtime does not use canonical local testcases');
  return provenance;
}
async function loadFrozenWorkspace(load,source,destination,signal) {
  signal?.throwIfAborted();
  try {
    const result=await load(destination);
    signal?.throwIfAborted();
    return result;
  } catch(error) {
    await load(source);
    throw error;
  }
}
async function runFrozenCusa(options) {
  const {repo,source,output,signal}=options;
  const before=await fileHash(source);
  let logs;
  try {
    logs=await execute(path.join(repo,'.venv/bin/python'),frozenCusaArgs(options),
      {cwd:repo,timeout:650000,maxBuffer:2*1024*1024,signal,windowsHide:true});
  } catch(error) {
    await fs.writeFile(path.join(path.dirname(output),'execution.log'),String(error.stdout??'')+'\n'+String(error.stderr??'')+'\n'+String(error));
    throw Error('Frozen execution stopped; evidence is saved in '+path.dirname(output),{cause:error});
  }
  await fs.writeFile(path.join(path.dirname(output),'execution.log'),logs.stdout+'\n'+logs.stderr);
  signal?.throwIfAborted();
  const receiptPath=path.join(output,'receipt.json');
  if((await fs.stat(receiptPath)).size>4*1024*1024)throw Error('Frozen execution receipt exceeds its limit');
  const receipt=JSON.parse(await fs.readFile(receiptPath,'utf8'));
  const result=validateFrozenResult(receipt,output,options.kind);
  if(before!==await fileHash(source)||receipt.source.sha256!==before||
      await fileHash(result.workspacePath)!==result.sha256)throw Error('Frozen execution source or exported bytes changed');
  const matchingRuntime=await matchingRuntimeEvidence(repo,output,options.checkpoint,receipt);
  if(matchingRuntime&&options.kind!=='sector')throw Error('Matching runtime is limited to frozen sector policies');
  if(options.kind==='sector'&&!matchingRuntime&&
      receipt.executorSourceSha256!==await fileHash(path.join(repo,'scripts/execute_patient_sector_policy.py')))
    throw Error('Frozen sector executor differs without matching-runtime admission');
  return {...result,receiptPath,...(matchingRuntime?{matchingRuntime}:{})};
}
module.exports={frozenCusaArgs,validateFrozenResult,validateMatchingRuntimeAdmission,
  runFrozenCusa,loadFrozenWorkspace};
