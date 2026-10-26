'use strict';
const fs = require('node:fs/promises');
const {createReadStream} = require('node:fs');
const {createHash} = require('node:crypto');
const path = require('node:path');
const {execFile} = require('node:child_process');
const {promisify} = require('node:util');
const execute = promisify(execFile);

function frozenCusaArgs({repo,source,branchId,checkpoint,output}) {
  if(typeof branchId!=='string'||!branchId||branchId.length>128||branchId.includes('\0'))throw Error('Choose a saved access approach');
  return [path.join(repo,'scripts/run_patient_cusa_learning.py'),
    '--source',source,'--branch',branchId,'--source-role','DEVELOPMENT',
    '--frozen-model',checkpoint,'--output',output];
}
function validateFrozenResult(receipt,output) {
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
  const result=validateFrozenResult(receipt,output);
  if(before!==await fileHash(source)||receipt.source.sha256!==before||
      await fileHash(result.workspacePath)!==result.sha256)throw Error('Frozen execution source or exported bytes changed');
  return {...result,receiptPath};
}
module.exports={frozenCusaArgs,validateFrozenResult,runFrozenCusa,loadFrozenWorkspace};
