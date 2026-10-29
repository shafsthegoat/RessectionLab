// Source-owned CPU worker profile, separate from Electron/GPU presentation.
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';
import crypto from 'node:crypto';
import inspector from 'node:inspector/promises';
import assert from 'node:assert/strict';
import {IncrementalSurfaceEngine} from '../desktop/src/viewer/incrementalSurface.ts';
const [input,output]=process.argv.slice(2);
const root=path.resolve(import.meta.dirname,'..'),fixture=JSON.parse(fs.readFileSync(input,'utf8'));
if(!path.resolve(output).startsWith(path.join(root,'testcases')+path.sep))throw Error('Use ignored testcases output');
if(fs.existsSync(output))throw Error('Preserve prior profiles; choose a new output directory');
fs.mkdirSync(output,{recursive:true});
const sha=value=>crypto.createHash('sha256').update(value).digest('hex');
const session=new inspector.Session();session.connect();
await session.post('Profiler.enable');await session.post('Profiler.start');
const rows=[],engine=new IncrementalSurfaceEngine();
for(const layer of fixture.layers){
  const mask=new Uint8Array(zlib.gunzipSync(fs.readFileSync(path.join(path.dirname(input),layer.maskFile))));
  assert.equal(sha(mask),layer.maskSha256);let previous='';
  for(const frame of layer.frames){
    const identity=layer.kind+':'+frame.stateId;
    const request={jobId:rows.length,layerId:layer.kind,previousIdentity:previous,identity,shape:fixture.shape,affine:fixture.affine,kind:layer.kind};
    if(!previous)request.mask=mask;else {request.indices=new Uint32Array(frame.removedCells);request.values=new Uint8Array(frame.removedCells.length);}
    const result=engine.prepare(request);
    rows.push({layer:layer.kind,position:frame.position,...result.work,diagnostics:result.diagnostics,
      triangles:result.indices.length/3,vertices:result.positions.length/3,
      sourceGeometrySha256:sha(Buffer.concat([Buffer.from(result.positions.buffer),Buffer.from(result.indices.buffer)])),
      displaySha256:sha(Buffer.concat([Buffer.from(result.smoothPositions.buffer),Buffer.from(result.smoothNormals.buffer)])),
      removedVolumeMm3:frame.removedVolumeMm3});
    previous=identity;
    fs.writeFileSync(path.join(output,'rows.json'),JSON.stringify(rows,null,2));
    console.log(JSON.stringify({layer:layer.kind,position:frame.position,ms:result.work.preparationMs,triangles:result.indices.length/3}));
  }
}
const {profile}=await session.post('Profiler.stop');session.disconnect();
fs.writeFileSync(path.join(output,'surface.cpuprofile'),JSON.stringify(profile));
fs.writeFileSync(path.join(output,'report.json'),JSON.stringify({schema:'resectionlab.surface-profile.v1',input,fixtureSha256:sha(fs.readFileSync(input)),
  measurement:'CPU inspector diagnostic prepares every frozen state, including unchanged masks. Includes profiling overhead; not an application job sequence, latency benchmark or screen presentation.',rows,memory:process.memoryUsage()},null,2));
