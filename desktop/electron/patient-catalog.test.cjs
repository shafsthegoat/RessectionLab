'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const {gunzipSync}=require('node:zlib');
const {createHash}=require('node:crypto');
const {PatientCatalog,RELEASE,validateCatalogMetadata,verifyBundleBytes}=require('./patient-catalog.cjs');
const {CaseLibrary}=require('./case-library.cjs');
const hash=b=>createHash('sha256').update(b).digest('hex');
async function metadata(){
  const releaseBytes=await fs.readFile(path.join(__dirname,'../..',RELEASE.path));
  const catalogBytes=gunzipSync(await fs.readFile(path.join(__dirname,'fixtures/patient-catalog-existing-nine-plus-ixi265-v1.json.gz')));
  assert.equal(hash(releaseBytes),RELEASE.sha256);assert.equal(hash(catalogBytes),RELEASE.catalogSha256);
  return {releaseBytes,catalogBytes,release:JSON.parse(releaseBytes),catalog:JSON.parse(catalogBytes)};
}
async function fixture(t){
  const base=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'rsl-catalog-')));t.after(()=>fs.rm(base,{recursive:true,force:true}));
  const root=path.join(base,'RessectionLab-poc-first'),original=path.join(base,'RessectionLab');await fs.mkdir(root);await fs.mkdir(original);
  const meta=await metadata();
  for(const [name,bytes] of [[RELEASE.path,meta.releaseBytes],[RELEASE.catalogPath,meta.catalogBytes]]){await fs.mkdir(path.dirname(path.join(root,name)),{recursive:true});await fs.writeFile(path.join(root,name),bytes);}
  return {base,root,original,...meta,reader:new PatientCatalog(root)};
}
function watchBundleOpens(t){
  const original=fs.open,opened=[];fs.open=async function(filename,...args){if(String(filename).endsWith('.ressectionlab'))opened.push(filename);return original.call(this,filename,...args);};
  t.after(()=>{fs.open=original;});return opened;
}

test('sealed metadata lists exactly ten own-source cases without any bundle existing or being read',async t=>{
  const f=await fixture(t),opened=watchBundleOpens(t),rows=await f.reader.load();
  assert.equal(rows.length,10);assert.equal(new Set(rows.map(r=>r.id)).size,10);assert(rows.every(r=>r.id.startsWith('catalog-')&&r.qualifiedCatalog));
  assert.equal(rows.filter(r=>r.source==='BTC').length,6);assert.equal(rows.filter(r=>r.source==='IXI').length,1);
  assert.equal(opened.length,0);assert.equal((await fs.readdir(path.join(f.root,'outputs'))).includes('cases'),false);
  const library=new CaseLibrary(path.join(f.base,'user'),path.join(f.root,'outputs/poc-cases'),[],f.root);
  const publicRows=await library.list();assert.equal(publicRows.length,10);
  assert(publicRows.every(r=>!('entry' in r)&&!('aliases' in r)&&!('bundle' in r)&&!('path' in r)));
  await assert.rejects(()=>library.resolve('../../protected.ressectionlab'),/current library/);
  await assert.rejects(()=>library.resolve({id:rows[0].id,path:'/outside'}),/current library/);
  await assert.rejects(()=>library.resolve('a'.repeat(64)),/current library/);assert.equal(opened.length,0);
});

test('an absent optional catalog preserves native, prepared and recent browsing without catalog authority',async t=>{
  const f=await fixture(t),staleId=(await f.reader.load())[0].id;
  const prepared=path.join(f.root,'outputs/poc-cases'),user=path.join(f.base,'user');
  const preparedFile=path.join(prepared,'UCSF-PDGM-0004.ressectionlab'),recentFile=path.join(f.base,'Saved study.rsworkspace');
  const nativeFile=path.join(f.base,'patient-anatomy.json');
  await fs.mkdir(prepared,{recursive:true});
  await fs.writeFile(preparedFile,'tiny synthetic prepared bytes');await fs.writeFile(recentFile,'tiny synthetic saved bytes');
  await fs.writeFile(nativeFile,JSON.stringify({schema:'patient-anatomy-v1',patient_id:'Native study'}));
  const library=new CaseLibrary(user,prepared,[nativeFile],f.root);await library.remember(recentFile);
  const opened=watchBundleOpens(t);
  await fs.unlink(path.join(f.root,RELEASE.catalogPath));
  for(const removeCatalogDirectory of [false,true]) {
    if(removeCatalogDirectory)await fs.rmdir(path.dirname(path.join(f.root,RELEASE.catalogPath)));
    assert.deepEqual(await f.reader.load(),[]);
    const rows=await library.list();assert.equal(rows.length,3);
    assert(rows.every(row=>!row.qualifiedCatalog&&!row.id.startsWith('catalog-')));
    assert.equal(rows.find(row=>row.native)?.title,'Native study');assert.equal(rows.find(row=>row.recent)?.portable,true);
    for(const filename of [nativeFile,preparedFile,recentFile])assert.equal(await library.resolve(hash(filename)),filename);
    await assert.rejects(()=>f.reader.resolve(staleId),/current catalog/);
    await assert.rejects(()=>library.resolve(staleId),/current library/);
    await assert.rejects(()=>library.resolve(preparedFile),/current library/);
  }
  await fs.mkdir(path.dirname(path.join(f.root,RELEASE.catalogPath)));
  await fs.writeFile(path.join(f.root,RELEASE.catalogPath),'{');
  await assert.rejects(()=>library.list(),/catalog checksum/);
  await assert.rejects(()=>library.resolve(hash(preparedFile)),/catalog checksum/);
  assert.equal(opened.length,0);
});

test('optional absence never suppresses an existing changed release or a catalog without its release',async t=>{
  const f=await fixture(t),opened=watchBundleOpens(t),catalogPath=path.join(f.root,RELEASE.catalogPath),releasePath=path.join(f.root,RELEASE.path);
  await fs.unlink(catalogPath);
  await fs.writeFile(releasePath,Buffer.concat([f.releaseBytes,Buffer.from(' ')]));
  await assert.rejects(()=>f.reader.load(),/release checksum/);
  await fs.writeFile(releasePath,'{');await assert.rejects(()=>f.reader.load(),/release checksum/);
  await fs.unlink(releasePath);assert.deepEqual(await f.reader.load(),[]);
  await fs.writeFile(catalogPath,f.catalogBytes);await assert.rejects(()=>f.reader.load(),/frozen release is missing/);
  assert.equal(opened.length,0);
});

test('dangling catalog or release paths remain errors rather than optional absence',async t=>{
  const f=await fixture(t),opened=watchBundleOpens(t),catalogPath=path.join(f.root,RELEASE.catalogPath),releasePath=path.join(f.root,RELEASE.path);
  const absent=path.join(f.base,'absent-target');
  await fs.unlink(catalogPath);await fs.symlink(absent,catalogPath);
  await assert.rejects(()=>f.reader.load(),/metadata path escaped/);
  await fs.unlink(catalogPath);await fs.rmdir(path.dirname(catalogPath));await fs.symlink(absent,path.dirname(catalogPath));
  await assert.rejects(()=>f.reader.load(),/metadata path escaped/);
  await fs.unlink(path.dirname(catalogPath));await fs.unlink(releasePath);await fs.symlink(absent,releasePath);
  await assert.rejects(()=>f.reader.load(),/metadata path escaped/);
  assert.equal(opened.length,0);
});

test('metadata policy refuses protected roles, changed source binding, virtual lineage and escaped paths before payload IO',async()=>{
  const original=await metadata();
  const changes=[
    [f=>f.catalog.entries[0].role='EVAL',/protected/],
    [f=>f.release.base_candidates[0].role='SELECT',/protected/],
    [f=>f.catalog.entries[0].patient_group='UPENN:001',/protected/],
    [f=>f.release.base_candidates[0].patient_group='BTC:sub-PAT26',/protected/],
    [f=>f.release.base_candidates[0].bundle='../protected.ressectionlab',/path/],
    [f=>f.catalog.entries[0].bundle='outputs/cases/../heldout.ressectionlab',/binding/],
    [f=>f.catalog.entries[0].input_binding.candidate.source_files[0].sha256='f'.repeat(64),/binding/],
    [f=>f.catalog.entries[0].imaging[0].source_sha256='f'.repeat(64),/acquired image source/],
    [f=>f.catalog.entries[0].virtual_descendant=true,/virtual/],
    [f=>f.catalog.entries[0].layers[0].provenance='simulated',/generated/],
    [f=>f.catalog.entries[0].layers[0].accepted_anatomical_truth=true,/accepted/],
    [f=>f.catalog.entries[0].tumor_location.lobe='frontal',/location/],
    [f=>f.catalog.plan_hash='sha256:'+'0'.repeat(64),/release identity/],
    [f=>f.catalog.entries[1]=f.catalog.entries[0],/duplicate/],
  ];
  for(const [change,message] of changes){const f=structuredClone(original);change(f);assert.throws(()=>validateCatalogMetadata(f.release,f.catalog),message);}
});

test('corrupt, oversized, redirected or newly changed catalog/release metadata fails closed',async t=>{
  const f=await fixture(t),opened=watchBundleOpens(t),catalogPath=path.join(f.root,RELEASE.catalogPath);
  await fs.writeFile(catalogPath,'{');await assert.rejects(()=>f.reader.load(),/catalog checksum/);
  await fs.writeFile(catalogPath,Buffer.alloc(512*1024+1));await assert.rejects(()=>f.reader.load(),/read bound/);
  await fs.writeFile(catalogPath,f.catalogBytes);
  const library=new CaseLibrary(path.join(f.base,'user'),path.join(f.root,'outputs/poc-cases'),[],f.root),row=(await library.list())[0];
  await fs.writeFile(path.join(f.root,RELEASE.path),Buffer.concat([f.releaseBytes,Buffer.from(' ')]));
  await assert.rejects(()=>library.resolve(row.id),/release checksum/);
  await fs.writeFile(path.join(f.root,RELEASE.path),f.releaseBytes);
  const outside=path.join(f.base,'outside.json');await fs.writeFile(outside,f.catalogBytes);await fs.unlink(catalogPath);await fs.symlink(outside,catalogPath);
  await assert.rejects(()=>f.reader.load(),/metadata path escaped/);assert.equal(opened.length,0);
});

test('released symlink aliases inherit the catalog guard and cannot bypass it as recents or prepared paths',async t=>{
  const f=await fixture(t),entry=f.catalog.entries[0],source=path.join(f.original,entry.bundle);
  await fs.mkdir(path.dirname(source),{recursive:true});await fs.writeFile(source,'synthetic invalid bundle');
  await fs.symlink(path.join(f.original,'outputs/cases'),path.join(f.root,'outputs/cases'));
  const prepared=path.join(f.root,'prepared');await fs.mkdir(prepared);await fs.symlink(source,path.join(prepared,'alias.ressectionlab'));
  const library=new CaseLibrary(path.join(f.base,'user'),prepared,[],f.root);await library.remember(source);
  const rows=await library.list(),selected=rows.find(r=>r.title===entry.case_id);
  assert.equal(rows.length,10);assert.equal(selected.recent,true);assert.equal(selected.qualifiedCatalog,true);
  await assert.rejects(()=>library.resolve(hash(source)),/current library/);
  await assert.rejects(()=>library.resolve(selected.id),/bundle size changed/);
});

test('a catalog-shaped filename redirected outside its exact released root is refused before bundle reads',async t=>{
  const f=await fixture(t),opened=watchBundleOpens(t),row=(await f.reader.load())[0];
  const outside=path.join(f.base,'protected','payload.ressectionlab');await fs.mkdir(path.dirname(outside));await fs.writeFile(outside,'protected sentinel');
  const declared=path.join(f.root,row.entry.bundle);await fs.mkdir(path.dirname(declared),{recursive:true});await fs.symlink(outside,declared);
  await assert.rejects(()=>f.reader.resolve(row.id),/escaped its exact released source/);assert.equal(opened.length,0);
});

test('bundle opening checks all expected bytes and refuses same-size corruption without payload decoding',async t=>{
  const root=await fs.mkdtemp(path.join(os.tmpdir(),'rsl-bundle-hash-'));t.after(()=>fs.rm(root,{recursive:true,force:true}));
  const file=path.join(root,'tiny-synthetic.ressectionlab'),bytes=Buffer.from('small synthetic bytes only');await fs.writeFile(file,bytes);
  await verifyBundleBytes(file,hash(bytes),bytes.length);
  await fs.writeFile(file,Buffer.alloc(bytes.length,7));await assert.rejects(()=>verifyBundleBytes(file,hash(bytes),bytes.length),/bundle checksum changed/);
  await assert.rejects(()=>verifyBundleBytes(file,hash(bytes),bytes.length+1),/bundle size changed/);
  const alias=path.join(root,'alias.ressectionlab');await fs.symlink(file,alias);await assert.rejects(()=>verifyBundleBytes(alias,hash(bytes),bytes.length));
});
