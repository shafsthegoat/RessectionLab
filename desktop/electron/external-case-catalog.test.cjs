'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const os=require('node:os');
const path=require('node:path');
const crypto=require('node:crypto');
const {ExternalCaseCatalog}=require('./external-case-catalog.cjs');
const sha=data=>crypto.createHash('sha256').update(data).digest('hex');
async function fixture(t){
  const root=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'external cases with spaces ')));t.after(()=>fs.rm(root,{recursive:true,force:true}));
  await fs.mkdir(path.join(root,'catalog'));await fs.mkdir(path.join(root,'cases'));await fs.mkdir(path.join(root,'qc'));
  await fs.writeFile(path.join(root,'.ressectionlab-data-root.json'),JSON.stringify({schema:'ressectionlab-data-root-v1',root_id:'fixture'}));
  await fs.writeFile(path.join(root,'cases','one.ressectionlab'),'native-package');await fs.writeFile(path.join(root,'qc','review.json'),'independent-review');
  const row={case_id:'one',path:'cases/one.ressectionlab',sha256:sha('native-package'),bytes:14,decision:'accepted_with_explicit_limitations',
    review_path:'qc/review.json',review_sha256:sha('independent-review'),capabilities:['source inspection'],missing_anatomy:['skull']};
  const write=async r=>fs.writeFile(path.join(root,'catalog','release.json'),JSON.stringify({schema:'ressectionlab-data-release-v1',cases:r}));
  await write([row]);return {root,row,write,catalog:new ExternalCaseCatalog(root,{checkVolume:async()=>{}})};
}
test('reviewed external package with spaces resolves and changed bytes fail closed',async t=>{
  const {root,catalog}=await fixture(t),rows=await catalog.load();assert.equal(rows.length,1);
  assert.equal(await catalog.resolve(rows[0].id),path.join(root,'cases','one.ressectionlab'));
  await fs.writeFile(path.join(root,'cases','one.ressectionlab'),'altered-volume');
  await assert.rejects(catalog.resolve(rows[0].id),/checksum changed/);
});
test('missing volume explains unavailability; pending cases and escapes are rejected',async t=>{
  const {root,row,write,catalog}=await fixture(t);
  await write([{...row,decision:'awaiting_review'}]);await assert.rejects(catalog.load(),/acceptance/);
  await write([{...row,path:'../outside.ressectionlab'}]);await assert.rejects(catalog.load(),/acceptance/);
  await fs.rename(root,root+'-offline');t.after(()=>fs.rm(root+'-offline',{recursive:true,force:true}));
  await assert.rejects(catalog.load(),/Connect the configured volume/);
});
test('review checksum and symlink dependencies cannot silently change',async t=>{
  const {root,catalog}=await fixture(t),[row]=await catalog.load();
  await fs.writeFile(path.join(root,'qc','review.json'),'different-review');await assert.rejects(catalog.resolve(row.id),/review checksum/);
  await fs.rename(path.join(root,'cases','one.ressectionlab'),path.join(root,'one.ressectionlab'));
  await fs.symlink('../one.ressectionlab',path.join(root,'cases','one.ressectionlab'));await assert.rejects(catalog.load(),/symbolic links/);
});
