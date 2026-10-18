#!/usr/bin/env node
/** Bounded CPU preparation timing of real viewer functions; no GPU/browser latency. */
import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {dirname,resolve} from 'node:path';
import {performance} from 'node:perf_hooks';
import {inflateSync} from 'node:zlib';
import {modeledVolume} from '../src/use-surgical.ts';
import {surgicalEffectGroup} from '../src/viewer/surgicalGeometry.ts';

const fixturePath=resolve(process.argv[2]??'outputs/integrated-rehearsal-v1/render-fixture.json');
const outputPath=resolve(process.argv[3]??'artifacts/integrated-rehearsal-v1/render-timing.json');
const digest=data=>createHash('sha256').update(data).digest('hex');
const bytes=await readFile(fixturePath),fixture=JSON.parse(bytes);
assert.equal(fixture.schema,'resectionlab.surgical-render-fixture.v1');
function unpack(row){
  assert.equal(row.encoding,'zlib-base64');
  const data=inflateSync(Buffer.from(row.data,'base64'));
  assert.equal(digest(data),row.sha256);
  const buffer=data.buffer.slice(data.byteOffset,data.byteOffset+data.byteLength);
  const result=row.dtype==='<f4'?new Float32Array(buffer):row.dtype==='|u1'?new Uint8Array(buffer):null;
  assert.ok(result,'Supported exact typed-array representation required');assert.equal(result.length,row.length);return result;
}
function source(row){return {...row,mri:unpack(row.mri),compartments:row.compartments.map(layer=>({...layer,mask:unpack(layer.mask)}))};}
function statistics(values){const sorted=[...values].sort((a,b)=>a-b);return {samplesMs:values,medianMs:sorted[2],p95Ms:sorted[Math.ceil(.95*sorted.length)-1],count:values.length,p95Method:'nearest-rank; with five repeats p95 is the maximum'};}
function dispose(group){group.traverse(object=>{object.geometry?.dispose();const materials=Array.isArray(object.material)?object.material:[object.material];for(const material of materials)material?.dispose();});}
const result={schema:'resectionlab.surgical-render-timing.v1',measurement:'CPU modeled-volume and effect-geometry preparation only',
  excludes:['GPU rendering','browser paint','React scheduling','IPC','Python transitions','file parsing/decompression','source-array hashing','geometry disposal'],
  repeats:5,clock:'node:perf_hooks performance.now milliseconds',runtime:process.version,platform:process.platform,architecture:process.arch,
  fixture:{path:fixturePath,sha256:digest(bytes)},sourceFiles:{},cases:[]};
for(const file of ['desktop/src/use-surgical.ts','desktop/src/viewer/surgicalGeometry.ts'])result.sourceFiles[file]=digest(await readFile(new URL('../'+file.replace('desktop/',''),import.meta.url)));
for(const item of fixture.cases){
  const decoded=source(item.source),viewBefore=JSON.stringify([item.intactView,item.updatedView]);
  const before=[digest(Buffer.from(decoded.mri.buffer)),...decoded.compartments.map(layer=>digest(Buffer.from(layer.mask.buffer)))];
  const phases=[];
  for(const phase of ['fresh-source-intact','source-state-update','same-state-cache']){
    const volumeTimes=[],effectTimes=[],combinedTimes=[];let effects=0;
    for(let repeat=0;repeat<5;repeat++){
      // Fresh source identity prevents WeakMap hits across repeats. Immutable
      // source-array references are reused, exactly as resident viewer data.
      const resident={...decoded,compartments:decoded.compartments.map(layer=>({...layer}))};
      let view=item.intactView;
      if(phase!=='fresh-source-intact'){modeledVolume(resident,item.intactView);view=item.updatedView;}
      if(phase==='same-state-cache')modeledVolume(resident,view);
      const begin=performance.now();const modeled=modeledVolume(resident,view);const middle=performance.now();
      const group=surgicalEffectGroup(view);const end=performance.now();
      volumeTimes.push(middle-begin);effectTimes.push(end-middle);combinedTimes.push(end-begin);effects=group.children.length;
      assert.equal(modeled.mri,resident.mri);assert.equal(modeled.shape,resident.shape);dispose(group);
    }
    phases.push({phase,modeledVolume:statistics(volumeTimes),surgicalEffectGroup:statistics(effectTimes),combined:statistics(combinedTimes),effectObjectCount:effects});
  }
  assert.deepEqual(before,[digest(Buffer.from(decoded.mri.buffer)),...decoded.compartments.map(layer=>digest(Buffer.from(layer.mask.buffer)))]);
  assert.equal(JSON.stringify([item.intactView,item.updatedView]),viewBefore);
  result.cases.push({id:item.id,shape:decoded.shape,sourceArrayHashes:before,sourceFile:item.sourceFile,
    intactStateId:item.intactView.stateId,updatedStateId:item.updatedView.stateId,
    removedCells:item.updatedView.removed.length,movedCells:item.updatedView.moved.length,
    actionOutcomes:item.outcomes,sourceArraysUnchanged:true,viewsUnchanged:true,phases});
}
await mkdir(dirname(outputPath),{recursive:true});await writeFile(outputPath,JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify({receipt:outputPath,cases:result.cases.map(c=>({id:c.id,phases:c.phases.map(p=>({phase:p.phase,medianMs:p.combined.medianMs,p95Ms:p.combined.p95Ms}))}))},null,2));
