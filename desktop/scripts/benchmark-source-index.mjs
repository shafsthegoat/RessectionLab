#!/usr/bin/env node
/** Exact source-surface indexing CPU benchmark; never constructs physics. */
import assert from 'node:assert/strict';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve,dirname,basename} from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
import {maskSurface} from '../src/viewer/surface.ts';
import {indexSurface,indexSurfaceLookupStatistics} from '../src/viewer/surfaceSmoothing.ts';

const [metadataPath,baselinePath,outputPath]=process.argv.slice(2).map(value=>resolve(value));
if(!metadataPath||!baselinePath||!outputPath)throw Error('Supply source mask metadata, retained indexSurface.ts and output JSON');
const metadata=JSON.parse(await readFile(metadataPath,'utf8'));
const bytes=await readFile(resolve(dirname(metadataPath),basename(metadataPath,'.json')+'.u1'));
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
assert.equal(sha(bytes),metadata.maskSha256);
const mask=new Uint8Array(bytes),soup=maskSurface(mask,metadata.shape);
const {indexSurface:retained}=await import(pathToFileURL(baselinePath));
const results=[];let expected;
for(let repeat=0;repeat<6;repeat++)for(const name of repeat%2?['optimized','retained']:['retained','optimized']){
  const start=performance.now(),indexed=(name==='optimized'?indexSurface:retained)(soup,metadata.shape),ms=performance.now()-start;
  const digest=sha(Buffer.concat([Buffer.from(indexed.positions.buffer),Buffer.from(indexed.indices.buffer)]));expected??=digest;assert.equal(digest,expected,'Exact indexed source positions and triangle order must remain byte-identical');
  results.push({repeat,implementation:name,ms,sha256:digest,vertices:indexed.positions.length/3});
}
const median=values=>{const a=[...values].sort((a,b)=>a-b);return (a[(a.length-1)>>1]+a[a.length>>1])/2;};
const summary=Object.fromEntries(['retained','optimized'].map(name=>[name,{medianMs:median(results.filter(row=>row.implementation===name&&row.repeat>0).map(row=>row.ms))}]));
assert.equal(sha(mask),metadata.maskSha256);
const result={schema:'resectionlab.source-index-performance.v1',scope:'CPU indexing only on exact source bone isosurface; excludes extraction/normals/renderer/IPC/physics. First repeat warmup, five balanced samples.',source:metadata,triangles:soup.length/9,summary,speedup:summary.retained.medianMs/summary.optimized.medianMs,results,exactOutputSha256:expected,sourceUnchanged:true,lookup:indexSurfaceLookupStatistics()};
await writeFile(outputPath,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({outputPath,...summary,speedup:result.speedup}));
