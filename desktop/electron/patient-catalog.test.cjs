'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const {gunzipSync}=require('node:zlib');
const {createHash}=require('node:crypto');
const {PatientCatalog,RELEASE,TWELVE_RELEASE,LOCATION_EVIDENCE,validateCatalogMetadata,validateTwelveCatalogMetadata,validateSourceLocationEvidence,verifyBundleBytes}=require('./patient-catalog.cjs');
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
async function twelveMetadata(){
  const admissionBytes=await fs.readFile(path.join(__dirname,'../..',TWELVE_RELEASE.path));
  const prospectiveBytes=await fs.readFile(path.join(__dirname,'../..',TWELVE_RELEASE.prospectivePath));
  const twelveCatalogBytes=gunzipSync(await fs.readFile(path.join(__dirname,'fixtures/patient-catalog-existing-twelve-execution-v1.json.gz')));
  assert.equal(hash(admissionBytes),TWELVE_RELEASE.sha256);assert.equal(hash(prospectiveBytes),TWELVE_RELEASE.prospectiveSha256);
  assert.equal(hash(twelveCatalogBytes),TWELVE_RELEASE.catalogSha256);
  return {admissionBytes,prospectiveBytes,twelveCatalogBytes,admission:JSON.parse(admissionBytes),prospective:JSON.parse(prospectiveBytes),twelveCatalog:JSON.parse(twelveCatalogBytes)};
}
async function twelveFixture(t){
  const f=await fixture(t),meta=await twelveMetadata();
  for(const [name,bytes] of [[TWELVE_RELEASE.path,meta.admissionBytes],[TWELVE_RELEASE.prospectivePath,meta.prospectiveBytes],[TWELVE_RELEASE.catalogPath,meta.twelveCatalogBytes]]) {
    await fs.mkdir(path.dirname(path.join(f.root,name)),{recursive:true});await fs.writeFile(path.join(f.root,name),bytes);
  }
  return {...f,...meta};
}
async function locationMetadata(){
  const bytes=await fs.readFile(path.join(__dirname,'../..',LOCATION_EVIDENCE.path));assert.equal(hash(bytes),LOCATION_EVIDENCE.sha256);
  return {bytes,evidence:JSON.parse(bytes)};
}
async function installLocationEvidence(f){
  const meta=await locationMetadata(),filename=path.join(f.root,LOCATION_EVIDENCE.path);
  await fs.mkdir(path.dirname(filename),{recursive:true});await fs.writeFile(filename,meta.bytes);return {...meta,filename};
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

test('twelve is preferred over ten without duplicate rows or bundle reads, and IDs bind the chosen release',async t=>{
  const f=await twelveFixture(t),opened=watchBundleOpens(t),rows=await f.reader.load();
  assert.equal(rows.length,12);assert.equal(new Set(rows.map(row=>row.id)).size,12);
  assert.equal(rows.filter(row=>row.source==='BTC').length,6);
  assert.deepEqual(rows.filter(row=>row.source==='SynthRAD2023').map(row=>row.title),['SynthRAD2023-1BA336','SynthRAD2023-1BB033','SynthRAD2023-1BC077']);
  assert.equal(rows.filter(row=>row.anatomy.includes('skull')).length,3);
  assert(rows.every(row=>row.entry.release_id===TWELVE_RELEASE.id&&row.entry.training_admitted===false));
  const library=new CaseLibrary(path.join(f.base,'user'),path.join(f.root,'outputs/poc-cases'),[],f.root);
  const publicRows=await library.list();assert.equal(publicRows.length,12);
  assert(publicRows.every(row=>!('entry' in row)&&!('aliases' in row)&&!('bundle' in row)&&!('path' in row)));
  const shared=rows.find(row=>row.source==='BTC'),oldId='catalog-'+hash(`${RELEASE.sha256}:${RELEASE.catalogSha256}:${shared.entry.patient_group}:${shared.entry.bundle_sha256}`);
  assert.notEqual(shared.id,oldId);await assert.rejects(()=>library.resolve(oldId),/current library/);
  await assert.rejects(()=>library.resolve(shared.entry.bundle),/current library/);
  assert.equal(opened.length,0);
});

test('genuinely absent twelve catalog falls back to ten, then legacy browsing when both catalogs are absent',async t=>{
  const f=await twelveFixture(t),opened=watchBundleOpens(t),twelveRows=await f.reader.load(),catalogPath=path.join(f.root,TWELVE_RELEASE.catalogPath);
  await fs.unlink(catalogPath);
  for(const missing of [null,TWELVE_RELEASE.path,TWELVE_RELEASE.prospectivePath]) {
    if(missing)await fs.unlink(path.join(f.root,missing));
    const rows=await f.reader.load();assert.equal(rows.length,10);assert(rows.every(row=>row.entry.release_id===RELEASE.id));
    await assert.rejects(()=>f.reader.resolve(twelveRows[0].id),/current catalog/);
  }
  await fs.unlink(path.join(f.root,RELEASE.catalogPath));assert.deepEqual(await f.reader.load(),[]);
  const prepared=path.join(f.root,'prepared'),file=path.join(prepared,'Local study.ressectionlab');await fs.mkdir(prepared);await fs.writeFile(file,'tiny synthetic bytes');
  const library=new CaseLibrary(path.join(f.base,'user'),prepared,[],f.root),rows=await library.list();
  assert.equal(rows.length,1);assert.equal(rows[0].qualifiedCatalog,undefined);assert.equal(await library.resolve(rows[0].id),file);
  assert.equal(opened.length,0);
});

test('present twelve metadata is checked even if another part is absent, and an existing catalog requires complete lineage',async t=>{
  const f=await twelveFixture(t),opened=watchBundleOpens(t),catalogPath=path.join(f.root,TWELVE_RELEASE.catalogPath);
  const parts=[[TWELVE_RELEASE.path,f.admissionBytes,/twelve admission checksum/],[TWELVE_RELEASE.prospectivePath,f.prospectiveBytes,/twelve candidate release checksum/]];
  for(const [name,bytes,message] of parts) {
    const filename=path.join(f.root,name);
    await fs.unlink(filename);await assert.rejects(()=>f.reader.load(),/twelve release metadata is missing/);
    await fs.writeFile(filename,Buffer.concat([bytes,Buffer.from(' ')]));await assert.rejects(()=>f.reader.load(),message);
    await fs.unlink(catalogPath);await assert.rejects(()=>f.reader.load(),message);
    await fs.writeFile(filename,bytes);assert.equal((await f.reader.load()).length,10);
    await fs.writeFile(catalogPath,f.twelveCatalogBytes);
  }
  const parentPath=path.join(f.root,RELEASE.path);
  await fs.unlink(parentPath);await assert.rejects(()=>f.reader.load(),/frozen parent release is missing/);
  await fs.writeFile(parentPath,'{}');await assert.rejects(()=>f.reader.load(),/frozen release checksum/);
  await fs.writeFile(parentPath,f.releaseBytes);
  await fs.writeFile(catalogPath,'{');await assert.rejects(()=>f.reader.load(),/twelve catalog checksum/);
  await fs.writeFile(catalogPath,Buffer.alloc(512*1024+1));await assert.rejects(()=>f.reader.load(),/read bound/);
  assert.equal(opened.length,0);
});

test('redirected and dangling twelve metadata never silently falls back to ten',async t=>{
  const f=await twelveFixture(t),opened=watchBundleOpens(t),catalogPath=path.join(f.root,TWELVE_RELEASE.catalogPath);
  for(const [name,bytes] of [[TWELVE_RELEASE.path,f.admissionBytes],[TWELVE_RELEASE.prospectivePath,f.prospectiveBytes],[TWELVE_RELEASE.catalogPath,f.twelveCatalogBytes]]) {
    const filename=path.join(f.root,name),outside=path.join(f.base,'outside.json');await fs.writeFile(outside,bytes);
    await fs.unlink(filename);await fs.symlink(outside,filename);await assert.rejects(()=>f.reader.load(),/metadata path escaped/);
    await fs.unlink(outside);await assert.rejects(()=>f.reader.load(),/metadata path escaped/);
    await fs.unlink(filename);await fs.writeFile(filename,bytes);
  }
  await fs.unlink(catalogPath);await fs.rmdir(path.dirname(catalogPath));await fs.symlink(path.join(f.base,'missing-directory'),path.dirname(catalogPath));
  await assert.rejects(()=>f.reader.load(),/metadata path escaped/);assert.equal(opened.length,0);
});

test('twelve metadata refuses protected roles, changed base-ten membership, source substitutions and altered new bundle identities',async()=>{
  const original={...await metadata(),...await twelveMetadata()};
  const newEntry=f=>f.twelveCatalog.entries.find(entry=>entry.patient_group==='SynthRAD2023:1BB033');
  const changes=[
    [f=>f.admission.prospective_release.sha256='0'.repeat(64),/admission/],
    [f=>f.prospective.parent_release_sha256='0'.repeat(64),/candidate release/],
    [f=>f.prospective.base_candidates[0].role='EVAL',/base candidates/],
    [f=>f.prospective.additional_candidates[0].role='SELECT',/protected/],
    [f=>f.prospective.additional_candidates[0].patient_group='SynthRAD2023:1BB034',/additional candidates/],
    [f=>f.prospective.additional_candidates[0].bundle='../heldout.ressectionlab',/path or identity/],
    [f=>f.prospective.additional_candidates[0].bundle_sha256='0'.repeat(64),/path or identity/],
    [f=>newEntry(f).role='EVAL',/protected/],
    [f=>newEntry(f).input_binding.candidate.source_files[0].sha256='f'.repeat(64),/binding/],
    [f=>newEntry(f).imaging[0].source_sha256='f'.repeat(64),/acquired image source/],
    [f=>newEntry(f).virtual_descendant=true,/virtual/],
    [f=>newEntry(f).training_admitted=true,/unreviewed/],
    [f=>f.twelveCatalog.plan_hash=RELEASE.planHash,/release identity/],
    [f=>f.twelveCatalog.entries[1]=f.twelveCatalog.entries[0],/duplicate/],
  ];
  for(const [change,message] of changes){const f=structuredClone(original);change(f);assert.throws(()=>validateTwelveCatalogMetadata(f.admission,f.prospective,f.release,f.twelveCatalog),message);}
});

test('new SynthRAD aliases remain one catalog row and retain exact bundle and current-release checks before opening',async t=>{
  const f=await twelveFixture(t),opened=watchBundleOpens(t),entry=f.twelveCatalog.entries.find(e=>e.patient_group==='SynthRAD2023:1BB033');
  const source=path.join(f.original,entry.bundle),declared=path.join(f.root,entry.bundle),prepared=path.join(f.root,'prepared');
  await fs.mkdir(path.dirname(source),{recursive:true});await fs.writeFile(source,'tiny synthetic invalid bundle');
  await fs.mkdir(path.dirname(declared),{recursive:true});await fs.symlink(source,declared);
  await fs.mkdir(prepared);await fs.symlink(source,path.join(prepared,'new-case-alias.ressectionlab'));
  const library=new CaseLibrary(path.join(f.base,'user'),prepared,[source],f.root);await library.remember(source);
  const rows=await library.list(),row=rows.find(r=>r.title===entry.case_id);
  assert.equal(rows.length,12);assert.equal(row.qualifiedCatalog,true);assert.equal(row.recent,true);assert.equal(opened.length,0);
  await assert.rejects(()=>library.resolve(hash(source)),/current library/);
  await assert.rejects(()=>library.resolve(row.id),/bundle size changed/);assert.equal(opened.length,1);
  await fs.writeFile(path.join(f.root,TWELVE_RELEASE.path),'{}');await assert.rejects(()=>library.resolve(row.id),/admission checksum/);assert.equal(opened.length,1);
  await fs.writeFile(path.join(f.root,TWELVE_RELEASE.path),f.admissionBytes);
  const outside=path.join(f.base,'protected.ressectionlab');await fs.writeFile(outside,'protected sentinel');await fs.unlink(declared);await fs.symlink(outside,declared);
  await assert.rejects(()=>library.resolve(row.id),/escaped its exact released source/);assert.equal(opened.length,1);
});

test('six source sites decorate only exact twelve-catalog members without changing IDs, canonical anatomy or reading bundles',async t=>{
  const f=await twelveFixture(t),opened=watchBundleOpens(t),before=await f.reader.load();await installLocationEvidence(f);
  const rows=await f.reader.load();assert.deepEqual(rows.map(row=>row.id),before.map(row=>row.id));assert.deepEqual(rows.map(row=>row.entry),before.map(row=>row.entry));
  assert.deepEqual(rows.filter(row=>row.sourceLocation).map(row=>[row.entry.patient_group,row.sourceLocation]),[
    ['BTC:sub-PAT05',{label:'Left Frontal',status:'source-reported-unreviewed'}],['BTC:sub-PAT16',{label:'Right Fronto-temporal',status:'source-reported-unreviewed'}],
    ['BTC:sub-PAT20',{label:'Right Parietal',status:'source-reported-unreviewed'}],['BTC:sub-PAT22',{label:'Right Parietal',status:'source-reported-unreviewed'}],
    ['BTC:sub-PAT25',{label:'Right Temporal',status:'source-reported-unreviewed'}],['BTC:sub-PAT28',{label:'Left Frontal',status:'source-reported-unreviewed'}],
  ]);
  assert(rows.every(row=>['lobe','hemisphere','anatomical_region'].every(key=>row.entry.tumor_location[key]==='UNKNOWN')));
  const prepared=path.join(f.root,'prepared'),variant=path.join(prepared,'BTC-ds001226-sub-PAT05-preop.ressectionlab');
  await fs.mkdir(prepared);await fs.writeFile(variant,'tiny unrelated saved variant');
  const library=new CaseLibrary(path.join(f.base,'user'),prepared,[],f.root);await library.remember(variant);
  const publicRows=await library.list();assert.equal(publicRows.length,13);assert.equal(publicRows.filter(row=>row.sourceLocation).length,6);
  assert.equal(publicRows.find(row=>row.id===hash(variant)).sourceLocation,undefined);
  assert(publicRows.filter(row=>row.sourceLocation).every(row=>row.qualifiedCatalog&&!('entry' in row)&&!('source_files' in row)&&!('hemisphere' in row)));
  assert.equal(opened.length,0);
});

test('missing optional source evidence and missing twelve catalog preserve unlabelled ten and legacy browsing',async t=>{
  const f=await twelveFixture(t),opened=watchBundleOpens(t),meta=await installLocationEvidence(f),before=await f.reader.load();
  await fs.unlink(meta.filename);
  const absent=await f.reader.load();assert.deepEqual(absent.map(row=>row.id),before.map(row=>row.id));assert(absent.every(row=>!row.sourceLocation));
  await fs.rmdir(path.dirname(meta.filename));assert.equal((await f.reader.load()).length,12);
  await installLocationEvidence(f);await fs.unlink(path.join(f.root,TWELVE_RELEASE.catalogPath));
  const ten=await f.reader.load();assert.equal(ten.length,10);assert(ten.every(row=>!row.sourceLocation));
  await fs.unlink(path.join(f.root,RELEASE.catalogPath));assert.deepEqual(await f.reader.load(),[]);
  const prepared=path.join(f.root,'prepared');await fs.mkdir(prepared);await fs.writeFile(path.join(prepared,'Saved study.ressectionlab'),'tiny synthetic bytes');
  const library=new CaseLibrary(path.join(f.base,'user'),prepared,[],f.root),legacy=await library.list();assert.equal(legacy.length,1);assert.equal(legacy[0].sourceLocation,undefined);
  assert.equal(opened.length,0);
});

test('source-location metadata refuses mismatched catalog, source, paper, member, role, review state and reported fields',async()=>{
  const {evidence}=await locationMetadata(),{twelveCatalog}=await twelveMetadata();
  const changes=[
    [f=>f.evidence.catalog.sha256='f'.repeat(64),/catalog binding/],
    [f=>f.evidence.source.sha256='f'.repeat(64),/dataset identity/],
    [f=>f.evidence.source.uri='https://unrelated.invalid/participants.tsv',/dataset identity/],
    [f=>f.evidence.paper_source.sha256='f'.repeat(64),/paper identity/],
    [f=>f.evidence.paper_source.representation='publisher HTML',/paper identity/],
    [f=>f.evidence.status='accepted',/review status/],
    [f=>f.evidence.automatic_seed_placement_allowed=true,/review status/],
    [f=>f.evidence.entries[0].role='EVAL',/protected/],
    [f=>f.evidence.entries[0].patient_group='BTC:sub-PAT26',/out-of-scope/],
    [f=>f.evidence.entries[1]=f.evidence.entries[0],/duplicate/],
    [f=>f.evidence.entries[0].status='reviewed',/unreviewed anatomy/],
    [f=>f.evidence.entries[0].hemisphere='Left',/unreviewed anatomy/],
    [f=>f.evidence.entries[0].anatomical_review_accepted=true,/unreviewed anatomy/],
    [f=>f.evidence.entries[0].source_reported_lateralization='Right',/reported fields/],
    [f=>f.evidence.entries[0].paper_row_evidence['Subject ID']='PAT28',/reported fields/],
    [f=>f.evidence.entries[0].case_id='unrelated saved variant',/row identity/],
    [f=>f.evidence.entries[0].bundle='../protected.ressectionlab',/row identity/],
    [f=>f.evidence.entries[0].bundle_sha256='f'.repeat(64),/frozen member binding/],
    [f=>f.evidence.entries[0].source_files[0].sha256='f'.repeat(64),/frozen member binding/],
    [f=>f.evidence.entries[0].acquisition_record='manifests/unrelated.json',/frozen member binding/],
    [f=>f.entries[0].release_id=RELEASE.id,/selected release/],
  ];
  for(const [change,message] of changes){const f=structuredClone({evidence,entries:twelveCatalog.entries});change(f);assert.throws(()=>validateSourceLocationEvidence(f.evidence,f.entries),message);}
});

test('present changed, oversized or redirected source evidence fails closed before any selected bundle open',async t=>{
  const f=await twelveFixture(t),opened=watchBundleOpens(t),meta=await installLocationEvidence(f),row=(await f.reader.load())[0];
  const library=new CaseLibrary(path.join(f.base,'user'),path.join(f.root,'outputs/poc-cases'),[],f.root);
  await fs.writeFile(meta.filename,'{');await assert.rejects(()=>library.list(),/source-location evidence checksum/);await assert.rejects(()=>library.resolve(row.id),/source-location evidence checksum/);
  await fs.unlink(path.join(f.root,TWELVE_RELEASE.catalogPath));await assert.rejects(()=>library.list(),/source-location evidence checksum/);
  await fs.writeFile(meta.filename,Buffer.alloc(32*1024+1));await assert.rejects(()=>f.reader.load(),/read bound/);
  const outside=path.join(f.base,'outside-evidence.json');await fs.writeFile(outside,meta.bytes);await fs.unlink(meta.filename);await fs.symlink(outside,meta.filename);
  await assert.rejects(()=>f.reader.load(),/metadata path escaped/);await fs.unlink(outside);await assert.rejects(()=>f.reader.load(),/metadata path escaped/);
  assert.equal(opened.length,0);
});
