import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';

let server,cache,describe;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'anatomy-provenance-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',
    server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  describe=(await server.ssrLoadModule('/src/WorkbenchPanels.tsx')).anatomyDescription;
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true,maxRetries:3,retryDelay:50});});

test('virtual geometry never inherits source-annotation or model-estimate attribution',()=>{
  for(const provenance of ['simulated','generated']){
    const label=describe({provenance,kind:'tumor',coverage:'Uniform diffusion; no CSF barriers'},'Source-provided annotation');
    assert.match(label,/Generated research geometry.*Unvalidated/);
    assert.match(label,/no CSF barriers/);
    assert(!label.includes('Source-provided')&&!label.includes('Model estimate'));
  }
});
test('estimated anatomy keeps model and pending review distinct from source labels',()=>{
  assert.equal(describe({provenance:'estimated',model:{name:'SynthStrip'},review_status:'Review pending'},'Source annotation'),
    'SynthStrip estimate · Review pending');
  assert.equal(describe({provenance:'observed',coverage:'Partial MRA field'},'Unknown annotation'),
    'Source-provided annotation. Partial MRA field');
});
test('absent or unfamiliar provenance does not manufacture source attribution',()=>{
  assert.equal(describe(undefined,'Unknown provenance'),'Unknown provenance');
  assert.equal(describe({provenance:'unverified'},'Unknown provenance'),'Unknown provenance');
});
