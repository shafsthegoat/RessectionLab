'use strict';
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const os=require('node:os');
const path=require('node:path');
const crypto=require('node:crypto');
const {ExternalCaseCatalog}=require('./external-case-catalog.cjs');
const {CaseLibrarySelection}=require('./case-library-selection.cjs');
const sha=data=>crypto.createHash('sha256').update(data).digest('hex');
async function fixture(t){
  const root=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'external cases with spaces ')));t.after(()=>fs.rm(root,{recursive:true,force:true}));
  await fs.mkdir(path.join(root,'catalog'));await fs.mkdir(path.join(root,'cases'));await fs.mkdir(path.join(root,'qc'));
  await fs.writeFile(path.join(root,'.ressectionlab-data-root.json'),JSON.stringify({schema:'ressectionlab-data-root-v1',root_id:'fixture'}));
  await fs.writeFile(path.join(root,'cases','one.ressectionlab'),'native-package');await fs.writeFile(path.join(root,'qc','review.json'),'independent-review');
  const row={case_id:'one',person_id:'Fixture:001',role:'TRAIN',path:'cases/one.ressectionlab',sha256:sha('native-package'),bytes:14,decision:'accepted_with_explicit_limitations',
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

async function nativePublication({root,row}, overrides={}, reviewOverrides={}) {
  const bytes='portable-fixture';
  const entry={
    case_id:row.case_id,person_id:row.person_id,patient_id:'fixture-patient',timepoint:'preoperative',
    role:row.role,classification:'real',source:'Fixture',path:'cases/native/one-v1.rsworkspace',
    bytes:Buffer.byteLength(bytes),sha256:sha(bytes),scan_count:2,decision:row.decision,capabilities:['source inspection'],
    limitations:['engineering source review'],missing_anatomy:['skull'],legacy_case_id:row.case_id,legacy_case_sha256:row.sha256,
    ...overrides,
  };
  const evidence=JSON.stringify({...entry,schema:'ressectionlab-native-study-acceptance-v1',independent_review:true,
    catalog_exposure_allowed:true,clinical_validation:false,training_admitted:false,...reviewOverrides});
  const record={...entry,review_path:'qc/native-one.json',review_sha256:sha(evidence)};
  const archive=path.join(root,entry.path),catalogPath=path.join(root,'catalog/native-release.json');
  await fs.mkdir(path.dirname(archive),{recursive:true});
  await fs.writeFile(archive,bytes);
  await fs.writeFile(path.join(root,record.review_path),evidence);
  await fs.writeFile(catalogPath,JSON.stringify({schema:'ressectionlab-native-study-release-v1',root_id:'fixture',cases:[record]}));
  return {archive,catalogPath,record};
}

test('an admitted native publication retains its case ID across versioned archives alongside the legacy case',async t=>{
  const fixtureData=await fixture(t),{root,catalog}=fixtureData;
  const [original]=await catalog.load();
  const first=await nativePublication(fixtureData);
  const [legacy,native]=await catalog.load();
  assert.equal(legacy.id,original.id);
  assert.ok(native.native&&native.portable);
  assert.equal(native.datasetRole,'TRAIN');
  assert.equal(native.timepoint,'preoperative');
  assert.equal(native.scanCount,2);
  assert.equal(await catalog.resolve(legacy.id),path.join(root,'cases','one.ressectionlab'));
  assert.equal(await catalog.resolve(native.id),first.archive);
  const second=await nativePublication(fixtureData,{path:'cases/native/one-v2.rsworkspace'});
  assert.equal((await catalog.load())[1].id,native.id);
  assert.equal(await catalog.resolve(native.id),second.archive);
  await fs.writeFile(second.archive,'altered-fixture!');
  await assert.rejects(catalog.resolve(native.id),/checksum changed/);
});

test('native publication rejects protected roles, mismatched evidence and altered legacy binding',async t=>{
  const fixtureData=await fixture(t),{catalog}=fixtureData;
  for(const [row,review,error] of [
    [{role:'EVAL'},{},/invalid or protected/],
    [{role:'DEVELOPMENT'},{},/frozen role differs/],
    [{legacy_case_sha256:'1'.repeat(64)},{},/legacy case binding differs/],
    [{},{role:'DEVELOPMENT'},/independent admission differs/],
    [{},{catalog_exposure_allowed:false},/independent admission differs/],
    [{},{clinical_validation:true},/independent admission differs/],
  ]) {
    await nativePublication(fixtureData,row,review);
    await assert.rejects(catalog.load(),error);
    const diagnostics=await catalog.discover();
    assert.equal(diagnostics.rows.length,1);
    assert.match(diagnostics.error.message,error);
    assert.match(await catalog.resolve(diagnostics.rows[0].id),/one\.ressectionlab$/);
  }
});

test('source aliases cannot migrate a known person into a different frozen role',async t=>{
  const fixtureData=await fixture(t),{root,catalog}=fixtureData;
  const aliasFile=path.join(root,'catalog','topbrain','person-aliases.json');
  await fs.mkdir(path.dirname(aliasFile));
  await fs.writeFile(aliasFile,JSON.stringify({schema:'ressectionlab-topbrain-person-aliases-v1',people:[
    {canonical_person_id:'Fixture:002',role:'EVAL',aliases:['Other-case-name']},
  ]}));
  for(const person_id of ['Fixture:002','Other-case-name']) {
    await nativePublication(fixtureData,{case_id:'new-case',person_id,legacy_case_id:undefined,legacy_case_sha256:undefined});
    await assert.rejects(catalog.load(),/frozen role differs/);
  }
});

test('native publication dependencies cannot be redirected or treated as optional when dangling',async t=>{
  const fixtureData=await fixture(t),{root,catalog}=fixtureData;
  const {archive,catalogPath}=await nativePublication(fixtureData);
  const admitted=await catalog.load();
  const review=path.join(root,'qc','native-one.json');
  await fs.appendFile(review,' ');
  await assert.rejects(catalog.resolve(admitted[1].id),/review checksum changed/);
  await nativePublication(fixtureData);
  await fs.rename(archive,archive+'.bak');
  await fs.symlink(archive+'.bak',archive);
  await assert.rejects(catalog.load(),/symbolic links/);
  await fs.unlink(catalogPath);
  await fs.symlink('unknown.json',catalogPath);
  await assert.rejects(catalog.load(),/symbolic links/);
});

test('local cases remain open while an external catalog is revoked or its volume is unavailable',async t=>{
  const fixtureData=await fixture(t),{root}=fixtureData;
  const {archive,catalogPath}=await nativePublication(fixtureData);
  const localRoot=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'local cases ')));
  t.after(()=>fs.rm(localRoot,{recursive:true,force:true}));
  const library=new CaseLibrarySelection({userData:path.join(localRoot,'user'),preparedRoot:path.join(localRoot,'prepared'),externalRoot:root});
  library.current.external.checkVolume=async()=>{};
  await fs.mkdir(library.preparedRoot);
  const local=path.join(library.preparedRoot,'portable.rsworkspace');
  await fs.writeFile(local,'local portable');
  await fs.symlink(archive,path.join(library.preparedRoot,'external.rsworkspace'));
  await library.remember(path.join(root,'cases','one.ressectionlab'));
  await library.remember(archive);
  const initial=await library.list();
  assert.equal(initial.length,3);
  const published=initial.find(row=>row.native);
  assert.equal(published.recent,true);
  assert.equal(await library.resolve(published.id),archive);
  assert.ok(library.isManagedNativePath(archive));
  assert.ok(library.isManagedNativePath(archive.replace('/cases/native/','/Cases/Native/')));
  await fs.writeFile(catalogPath,'null');
  assert.equal((await library.list()).length,2);
  assert.match(library.info().error,/native study catalog is invalid/);
  await assert.rejects(library.resolve(published.id),/native study catalog is invalid/);
  await fs.unlink(catalogPath);
  assert.equal((await library.list()).length,2);
  assert.equal(library.info().error,undefined);
  await assert.rejects(library.resolve(published.id),/current release/);
  library.current.external.checkVolume=async()=>{throw new Error('The recorded volume is unavailable.')};
  const [available]=await library.list();
  assert.ok(available.portable);
  assert.equal(await library.resolve(available.id),local);
  assert.match(library.info().error,/volume is unavailable/);
});

test('a reviewed native-only study needs no simulator package',async t=>{
  const fixtureData=await fixture(t),{root,catalog}=fixtureData;
  await fs.unlink(path.join(root,'catalog','release.json'));
  const publication=await nativePublication(fixtureData,{
    case_id:'native-only',role:'DEVELOPMENT',legacy_case_id:undefined,legacy_case_sha256:undefined,
  });
  const [row]=await catalog.load();
  assert.equal(row.title,'native-only');
  assert.equal(row.datasetRole,'DEVELOPMENT');
  assert.equal(await catalog.resolve(row.id),publication.archive);
});
