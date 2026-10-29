// Exact paired worker benchmark from acquired-source exposure/removal masks.
// Frozen worker modules bind the declared baseline; no checkout switching.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {stripTypeScriptTypes} from 'node:module';
import zlib from 'node:zlib';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import os from 'node:os';
import {IncrementalSurfaceEngine} from '../desktop/src/viewer/incrementalSurface.ts';
const [input,baseline,output,repeatArgument='3']=process.argv.slice(2),repetitions=Number(repeatArgument);
const root=path.resolve(import.meta.dirname,'..'),viewer=path.join(root,'desktop/src/viewer');
if(!Number.isInteger(repetitions)||repetitions<2)throw Error('At least two alternating paired repetitions required');
for(const name of [input,baseline,output])if(!path.resolve(name).startsWith(path.join(root,'testcases')+path.sep))throw Error('Use ignored testcases paths');
if(fs.existsSync(output))throw Error('Preserve prior evidence');fs.mkdirSync(output,{recursive:true});
const fixture=JSON.parse(fs.readFileSync(input,'utf8')),frozen=JSON.parse(fs.readFileSync(path.join(baseline,'viewer-manifest.json'),'utf8'));
const sha=value=>crypto.createHash('sha256').update(value).digest('hex');
const retained=['surfaceSafety.ts','surfaceSmoothing.ts','prepareSurface.ts','incrementalSurface.ts',
  ...['chunkedSurface.ts'].filter(name=>fs.existsSync(path.join(baseline,name)))];
const compiled=name=>path.join(output,'retained-'+name.replace('.ts','.mjs'));
const sourceHashes={};
for(const name of retained){
  const frozenPath=path.join(baseline,name),sourcePath=fs.existsSync(frozenPath)?frozenPath:path.join(viewer,name),bytes=fs.readFileSync(sourcePath);
  if(fs.existsSync(frozenPath))assert.equal(sha(bytes),frozen.files['desktop/src/viewer/'+name]);
  sourceHashes[name]={baseline:sha(bytes),current:sha(fs.readFileSync(path.join(viewer,name)))};
  let source=stripTypeScriptTypes(bytes.toString(),{mode:'strip',sourceUrl:pathToFileURL(path.resolve(sourcePath)).href});
  source=source.replace(/from '(\.\/[^']+\.ts)'/g,(_match,specifier)=>{
    const dependency=specifier.slice(2),target=retained.includes(dependency)?compiled(dependency):path.join(viewer,dependency);
    return `from '${pathToFileURL(path.resolve(target)).href}'`;
  });
  fs.writeFileSync(compiled(name),source);
}
const {IncrementalSurfaceEngine:BaselineEngine}=await import(pathToFileURL(path.resolve(compiled('incrementalSurface.ts'))));
for(const name of ['incrementalSurfaceFilter.ts'])if(fs.existsSync(path.join(viewer,name)))
  sourceHashes[name]={baseline:fs.existsSync(path.join(baseline,name))?sha(fs.readFileSync(path.join(baseline,name))):null,current:sha(fs.readFileSync(path.join(viewer,name)))};
const inputs=fixture.layers.map(layer=>{
  const mask=new Uint8Array(zlib.gunzipSync(fs.readFileSync(path.join(path.dirname(input),layer.maskFile))));
  assert.equal(sha(mask),layer.maskSha256);return {...layer,mask};
});
const rows=[],expected=new Map();
const hashes=result=>Object.fromEntries(['positions','normals','indices','smoothPositions','smoothNormals'].map(name=>[name,sha(new Uint8Array(result[name].buffer,result[name].byteOffset,result[name].byteLength))]));
const diagnostics=result=>Object.fromEntries(Object.entries(result.diagnostics).filter(([name])=>name!=='preparationMs'));
function run(mode,repetition){
  const engine=mode==='baseline'?new BaselineEngine():new IncrementalSurfaceEngine();
  for(const layer of inputs){let previous='';
    for(const frame of layer.frames){
      // The application reuses its existing surface when no source cell changed.
      if(previous&&!frame.removedCells.length)continue;
      const identity=layer.kind+':'+frame.stateId;
      const request={jobId:rows.length,layerId:layer.kind,previousIdentity:previous,identity,shape:fixture.shape,affine:fixture.affine,kind:layer.kind};
      if(!previous)request.mask=layer.mask.slice();else{request.indices=new Uint32Array(frame.removedCells);request.values=new Uint8Array(frame.removedCells.length);}
      const started=performance.now(),result=engine.prepare(request),elapsedMs=performance.now()-started;
      const key=layer.kind+':'+frame.position,parity={buffers:hashes(result),diagnostics:diagnostics(result)};
      if(!expected.has(key))expected.set(key,parity);assert.deepEqual(parity,expected.get(key));
      rows.push({mode,repetition,layer:layer.kind,position:frame.position,initial:!previous,elapsedMs,work:result.work,
        removedVolumeMm3:frame.removedVolumeMm3,triangles:result.indices.length/3,vertices:result.positions.length/3,
        exactAllBuffersAndDiagnostics:true,...parity,memory:process.memoryUsage()});
      fs.writeFileSync(path.join(output,'progress.json'),JSON.stringify({mode,repetition,layer:layer.kind,position:frame.position,
        completedPreparations:rows.length,elapsedMs,exactAllBuffersAndDiagnostics:true,updatedAt:new Date().toISOString()}));
      previous=identity;
    }
  }
  for(const layer of inputs)assert.equal(sha(layer.mask),layer.maskSha256);
  fs.writeFileSync(path.join(output,'runs.json'),JSON.stringify(rows,null,2));
  console.log(JSON.stringify({mode,repetition,workerSeconds:rows.filter(row=>row.mode===mode&&row.repetition===repetition).reduce((sum,row)=>sum+row.elapsedMs,0)/1000}));
}
run('baseline','warmup');run('current','warmup');
for(let i=0;i<repetitions;i++)for(const mode of i%2?['current','baseline']:['baseline','current'])run(mode,i);
const quantile=(values,p)=>{const sorted=values.toSorted((a,b)=>a-b),at=(sorted.length-1)*p,low=Math.floor(at);return sorted[low]+(sorted[Math.ceil(at)]-sorted[low])*(at-low);};
const summary={};
for(const mode of ['baseline','current'])for(const layer of ['brain','tumor'])for(const initial of [false,true]){
  const values=rows.filter(row=>row.mode===mode&&row.layer===layer&&row.initial===initial&&typeof row.repetition==='number').map(row=>row.elapsedMs);
  summary[[mode,layer,initial?'initial':'changed'].join(':')]={n:values.length,p50Ms:quantile(values,.5),p95Ms:quantile(values,.95),minimumMs:Math.min(...values),maximumMs:Math.max(...values)};
}
fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({schema:'resectionlab.surface-performance.v1',input,
  fixtureSha256:sha(fs.readFileSync(input)),sourceHashes,summary,exactAllBuffersAndDiagnostics:true,rows,
  measurement:'Serial CPU worker prepare wall time, including extraction, assembly, fairing and normals; excludes hashes, disk writes, warmups, Electron/GPU and screen presentation. Unchanged source masks reuse prior display and are omitted. p95 is descriptive of this small frozen workload.',
  memoryScope:'Per-job post-return process memory samples; maximumRSS is the whole serial paired process, not separate implementation peak estimates.',
  maximumRssBytes:process.resourceUsage().maxRSS*1024,node:process.version,platform:process.platform,arch:process.arch,cpu:os.cpus()[0].model},null,2));
console.log(JSON.stringify(summary));
