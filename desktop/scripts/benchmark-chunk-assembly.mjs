#!/usr/bin/env node
/** Bounded exact source assembly comparison; never constructs physics. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,dirname,basename} from 'node:path';
import {createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
import {ChunkedMaskSurface} from '../src/viewer/chunkedSurface.ts';
import {indexSurface} from '../src/viewer/surfaceSmoothing.ts';
const [metadataPath,outputPath]=process.argv.slice(2).map(value=>resolve(value));
if(!metadataPath||!outputPath)throw Error('Supply source mask metadata and output JSON');
const metadata=JSON.parse(await readFile(metadataPath,'utf8'));
const bytes=await readFile(resolve(dirname(metadataPath),basename(metadataPath,'.json')+'.u1'));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
assert.equal(sha(bytes),metadata.maskSha256);
const original=new Uint8Array(bytes),changedIndex=original.findIndex(value=>value===1);
assert(changedIndex>=0,'Source must contain a surface');
const budget=128*1024*1024;
const start=performance.now(),retained=new ChunkedMaskSurface(original.slice(),metadata.shape),classicAdmissionMs=performance.now()-start;
const indexedStart=performance.now(),optimized=new ChunkedMaskSurface(original.slice(),metadata.shape,32,budget),indexedAdmissionMs=performance.now()-indexedStart;
assert(optimized.indexedChunks);assert(optimized.retainedBytes<=budget);
const results=[];
for(let repeat=0;repeat<6;repeat++){
  const value=repeat%2;retained.mask[changedIndex]=value;optimized.mask[changedIndex]=value;
  const updates={};
  for(const [name,chunks] of [['retained',retained],['optimized',optimized]]){
    const t=performance.now();chunks.update([changedIndex]);updates[name]=performance.now()-t;
  }
  let oracle;
  for(const name of repeat%2?['optimized','retained']:['retained','optimized']){
    const t=performance.now(),mesh=name==='optimized'?optimized.indexed():indexSurface(retained.positions(),metadata.shape),ms=performance.now()-t;
    const digest=sha(Buffer.concat([Buffer.from(mesh.positions.buffer),Buffer.from(mesh.indices.buffer)]));
    oracle??=digest;assert.equal(digest,oracle,'Canonical vertex bytes and triangle order must remain exact after each changed cell');
    results.push({repeat,implementation:name,ms,updateMs:updates[name],sha256:digest,vertices:mesh.positions.length/3,triangles:mesh.indices.length/3,retainedBytes:(name==='optimized'?optimized:retained).retainedBytes,remeshedChunks:(name==='optimized'?optimized:retained).lastRemeshedChunks,indexedCorners:(name==='optimized'?optimized:retained).lastIndexedCorners});
  }
}
assert.equal(sha(original),metadata.maskSha256);assert.deepEqual(retained.mask,original);assert.deepEqual(optimized.mask,original);
const median=values=>{const a=[...values].sort((a,b)=>a-b);return (a[(a.length-1)>>1]+a[a.length>>1])/2;};
const summary=Object.fromEntries(['retained','optimized'].map(name=>[name,{medianAssemblyMs:median(results.filter(row=>row.implementation===name&&row.repeat>0).map(row=>row.ms)),medianUpdateMs:median(results.filter(row=>row.implementation===name&&row.repeat>0).map(row=>row.updateMs))}]));
const result={schema:'resectionlab.chunk-assembly-performance.v1',scope:'Exact source bone assembly only after one real source-cell delete/undo; original triangle-soup stitching plus rolling indexing versus retained indexed rows and canonical ID remap. Five balanced samples after first warmup. Excludes normals/fairing/renderer/IPC/physics; update/extraction and initial admission recorded separately.',source:metadata,changedIndex,budget,admission:{classicMs:classicAdmissionMs,indexedMs:indexedAdmissionMs},summary,speedup:summary.retained.medianAssemblyMs/summary.optimized.medianAssemblyMs,results,sourceUnchanged:true};
await writeFile(outputPath,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({outputPath,...summary,speedup:result.speedup,retainedBytes:optimized.retainedBytes}));
