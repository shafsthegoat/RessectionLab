'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const vm=require('node:vm');
const {gunzipSync}=require('node:zlib');
const {CaseLibrarySelection}=require('./case-library-selection.cjs');
const {RELEASE,TWELVE_RELEASE,LOCATION_EVIDENCE}=require('./patient-catalog.cjs');
const repo=path.join(__dirname,'../..');
async function fixture(t) {
  const base=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'rsl-library-selection-')));t.after(()=>fs.rm(base,{recursive:true,force:true}));
  const root=path.join(base,'RessectionLab-poc-first'),userData=path.join(base,'user');await fs.mkdir(root);
  for(const name of [RELEASE.path,TWELVE_RELEASE.path,TWELVE_RELEASE.prospectivePath,LOCATION_EVIDENCE.path]) {
    const filename=path.join(root,name);await fs.mkdir(path.dirname(filename),{recursive:true});await fs.copyFile(path.join(repo,name),filename);
  }
  const catalog=gunzipSync(await fs.readFile(path.join(__dirname,'fixtures/patient-catalog-existing-twelve-execution-v1.json.gz')));
  await fs.mkdir(path.dirname(path.join(root,TWELVE_RELEASE.catalogPath)),{recursive:true});await fs.writeFile(path.join(root,TWELVE_RELEASE.catalogPath),catalog);
  const options={userData,defaultRoot:null,preparedRoot:path.join(userData,'prepared-cases'),nativePaths:[]};
  return {base,root,userData,options,catalog,selection:new CaseLibrarySelection(options)};
}
const pick=root=>async()=>({canceled:false,filePaths:[root]});
function watchPayloadReads(t) {
  const original=fs.open,opened=[];fs.open=async function(filename,...args){if(/\.(?:ressectionlab|nii|gz)$/.test(String(filename)))opened.push(filename);return original.call(this,filename,...args);};
  t.after(()=>{fs.open=original;});return opened;
}

test('packaged default needs no app.asar catalog; native selection persists canonical root and twelve source-bound rows across fresh instances',async t=>{
  const f=await fixture(t),opened=watchPayloadReads(t),packageFile=path.join(f.base,'app.asar');await fs.writeFile(packageFile,'tiny package sentinel');
  assert.deepEqual(await f.selection.restore(),{folder:null});assert.deepEqual(await f.selection.list(),[]);
  assert.equal(f.selection.current.catalog,null);await assert.rejects(()=>fs.access(f.selection.record),/ENOENT/);
  const alias=path.join(f.base,'chosen-folder-alias');await fs.symlink(f.root,alias);
  assert.deepEqual(await f.selection.choose(pick(alias)),{folder:f.root});
  const rows=await f.selection.list();assert.equal(rows.length,12);assert.equal(rows.filter(row=>row.sourceLocation).length,6);
  const saved=JSON.parse(await fs.readFile(f.selection.record,'utf8'));assert.equal(saved.root,f.root);assert.equal(saved.schema,'case-library-selection-v1');
  assert.deepEqual(saved.catalogIds,rows.map(row=>row.id).sort());
  assert.equal((await fs.stat(f.selection.record)).mode&0o777,0o600);
  const fresh=new CaseLibrarySelection(f.options);assert.deepEqual(await fresh.restore(),{folder:f.root});assert.deepEqual(await fresh.list(),rows);
  await assert.rejects(()=>fresh.resolve(f.root),/current library/);assert.equal(opened.length,0);
  const entry=JSON.parse(f.catalog).entries[0],bundle=path.join(f.root,entry.bundle);await fs.mkdir(path.dirname(bundle),{recursive:true});await fs.writeFile(bundle,'tiny wrong bundle');
  await assert.rejects(()=>fresh.resolve(rows[0].id),/bundle size changed/);assert.equal(opened.length,1);
});

test('development defaults retain the existing checkout without creating a selection record',async t=>{
  const f=await fixture(t),selection=new CaseLibrarySelection({...f.options,defaultRoot:f.root});
  assert.deepEqual(await selection.restore(),{folder:f.root});assert.equal((await selection.list()).length,12);
  await assert.rejects(()=>fs.access(selection.record),/ENOENT/);
});

test('cancelled, empty, malformed and unqualified selections preserve the previous library and exact persisted record',async t=>{
  const f=await fixture(t);await f.selection.choose(pick(f.root));const record=await fs.readFile(f.selection.record),before=await f.selection.list(),current=f.selection.current;
  assert.equal(await f.selection.choose(async()=>({canceled:true,filePaths:[f.base]})),null);
  await assert.rejects(()=>f.selection.choose(async()=>({canceled:false,filePaths:[]})),/one case-library folder/);
  await assert.rejects(()=>f.selection.choose(pick('relative/folder')),/local case-library folder/);
  const empty=path.join(f.base,'empty');await fs.mkdir(empty);await assert.rejects(()=>f.selection.choose(pick(empty)),/no source-checked case catalog/);
  const bad=path.join(f.base,'bad');await fs.cp(f.root,bad,{recursive:true});await fs.writeFile(path.join(bad,TWELVE_RELEASE.catalogPath),'{}');
  await assert.rejects(()=>f.selection.choose(pick(bad)),/catalog checksum/);
  assert.equal(f.selection.current,current);assert.deepEqual(f.selection.info(),{folder:f.root});assert.deepEqual(await f.selection.list(),before);assert.deepEqual(await fs.readFile(f.selection.record),record);
});

test('saved catalog tampering fails closed on restart and at open, but native selections remain independently remembered',async t=>{
  const f=await fixture(t),opened=watchPayloadReads(t);await f.selection.choose(pick(f.root));const row=(await f.selection.list())[0],saved=await fs.readFile(f.selection.record);
  await fs.writeFile(path.join(f.root,TWELVE_RELEASE.catalogPath),'{}');
  await assert.rejects(()=>f.selection.resolve(row.id),/catalog checksum/);
  const fresh=new CaseLibrarySelection(f.options),state=await fresh.restore();assert.equal(state.folder,f.root);assert.match(state.error,/Saved case library is unavailable/);
  await assert.rejects(()=>fresh.list(),/catalog checksum/);await assert.rejects(()=>fresh.resolve(row.id),/catalog checksum/);
  const manual=path.join(f.base,'Manually chosen.ressectionlab');await fs.writeFile(manual,'tiny native-open fixture');await fresh.remember(manual);
  const recent=JSON.parse(await fs.readFile(path.join(f.userData,'anatomy-cases.json'),'utf8'));assert.deepEqual(recent.recent,[manual]);
  assert.equal(await fresh.choose(async()=>({canceled:true,filePaths:[]})),null);assert.equal(fresh.info().error,state.error);assert.deepEqual(await fs.readFile(fresh.record),saved);
  await fs.writeFile(path.join(f.root,TWELVE_RELEASE.catalogPath),f.catalog);await fresh.choose(pick(f.root));assert.equal(fresh.info().error,undefined);assert.equal((await fresh.list()).filter(r=>r.qualifiedCatalog).length,12);assert.equal(opened.length,0);
});

test('settings tamper, missing roots and redirected saved folders do not silently fall back to default browsing',async t=>{
  const f=await fixture(t);await f.selection.choose(pick(f.root));const original=await fs.readFile(f.selection.record),saved=JSON.parse(original);
  for(const bytes of ['{',JSON.stringify({...saved,root:'../outside'}),JSON.stringify({...saved,directory:{device:'0',inode:'0'}}),
    JSON.stringify({...saved,catalogIds:undefined}),JSON.stringify({...saved,catalogIds:Array(12).fill('catalog-'+'0'.repeat(64))}),Buffer.alloc(16*1024+1)]) {
    await fs.writeFile(f.selection.record,bytes);const fresh=new CaseLibrarySelection(f.options);assert.ok((await fresh.restore()).error);await assert.rejects(()=>fresh.list(),/Saved case library/);
  }
  await fs.writeFile(f.selection.record,original);const moved=path.join(f.base,'moved');await fs.rename(f.root,moved);
  const absent=new CaseLibrarySelection(f.options);assert.ok((await absent.restore()).error);await assert.rejects(()=>absent.list(),/Saved case library/);
  await fs.symlink(moved,f.root);const redirected=new CaseLibrarySelection(f.options);assert.match((await redirected.restore()).error,/folder changed/);
  await fs.unlink(f.selection.record);await fs.symlink(path.join(f.base,'missing-record'),f.selection.record);
  const linked=new CaseLibrarySelection(f.options);assert.match((await linked.restore()).error,/settings are invalid/);
});

test('preload folder chooser transmits no renderer-supplied path or configuration',async()=>{
  let api;const calls=[];
  vm.runInNewContext(await fs.readFile(path.join(__dirname,'preload.cjs'),'utf8'),{require(name){assert.equal(name,'electron');return {contextBridge:{exposeInMainWorld(_name,value){api=value;}},ipcRenderer:{invoke(...args){calls.push(args);return Promise.resolve(null);}}};}});
  await api.chooseCaseLibrary({root:'/arbitrary',allowlist:['EVAL']});await api.caseLibraryInfo('/arbitrary');
  assert.deepEqual(calls,[['research:chooseCaseLibrary'],['research:caseLibraryInfo']]);
});

test('live selected catalog metadata loss blocks listing and opaque opens visibly while native remember remains independent',async t=>{
  const f=await fixture(t),opened=watchPayloadReads(t);await f.selection.choose(pick(f.root));
  const row=(await f.selection.list())[0],saved=await fs.readFile(f.selection.record);
  for(const name of [RELEASE.path,TWELVE_RELEASE.path,TWELVE_RELEASE.prospectivePath,TWELVE_RELEASE.catalogPath,LOCATION_EVIDENCE.path])await fs.unlink(path.join(f.root,name));
  await assert.rejects(()=>f.selection.list(),/selected case catalog changed or is missing/);
  assert.match(f.selection.info().error,/Saved case library is unavailable/);
  await assert.rejects(()=>f.selection.resolve(row.id),/selected case catalog/);
  const manual=path.join(f.base,'Native choice.ressectionlab');await fs.writeFile(manual,'tiny native choice');await f.selection.remember(manual);
  assert.deepEqual(JSON.parse(await fs.readFile(path.join(f.userData,'anatomy-cases.json'),'utf8')).recent,[manual]);
  assert.deepEqual(await fs.readFile(f.selection.record),saved);assert.equal(opened.length,0);
});

test('live same-path directory replacement is blocked even when copied catalog bytes and IDs are identical',async t=>{
  const f=await fixture(t),opened=watchPayloadReads(t);await f.selection.choose(pick(f.root));const row=(await f.selection.list())[0];
  const moved=path.join(f.base,'original-directory');await fs.rename(f.root,moved);await fs.cp(moved,f.root,{recursive:true});
  await assert.rejects(()=>f.selection.resolve(row.id),/selected case-library folder changed/);
  assert.match(f.selection.info().error,/folder changed/);await assert.rejects(()=>f.selection.list(),/folder changed/);assert.equal(opened.length,0);
});

test('selected twelve cannot silently downgrade to ten, while unselected development fallback is unchanged',async t=>{
  const f=await fixture(t),opened=watchPayloadReads(t);
  const ten=gunzipSync(await fs.readFile(path.join(__dirname,'fixtures/patient-catalog-existing-nine-plus-ixi265-v1.json.gz')));
  await fs.mkdir(path.dirname(path.join(f.root,RELEASE.catalogPath)),{recursive:true});await fs.writeFile(path.join(f.root,RELEASE.catalogPath),ten);
  await f.selection.choose(pick(f.root));await fs.unlink(path.join(f.root,TWELVE_RELEASE.catalogPath));
  await assert.rejects(()=>f.selection.list(),/selected case catalog changed/);assert.match(f.selection.info().error,/selected case catalog changed/);
  const fresh=new CaseLibrarySelection(f.options);assert.match((await fresh.restore()).error,/selected case catalog changed/);
  const development=new CaseLibrarySelection({...f.options,userData:path.join(f.base,'dev-user'),defaultRoot:f.root});
  assert.equal((await development.list()).length,10);assert.equal(development.info().error,undefined);
  await f.selection.choose(pick(f.root));assert.equal((await f.selection.list()).length,10);assert.equal(f.selection.info().error,undefined);
  assert.equal(JSON.parse(await fs.readFile(f.selection.record,'utf8')).catalogIds.length,10);assert.equal(opened.length,0);
});

test('optional reported-site evidence can disappear without breaking selected admission or its saved IDs',async t=>{
  const f=await fixture(t);await f.selection.choose(pick(f.root));const before=await f.selection.list(),saved=await fs.readFile(f.selection.record);
  await fs.unlink(path.join(f.root,LOCATION_EVIDENCE.path));const after=await f.selection.list();
  assert.equal(after.length,12);assert(after.every(row=>!row.sourceLocation));assert.deepEqual(after.map(row=>row.id),before.map(row=>row.id));
  assert.equal(f.selection.info().error,undefined);assert.deepEqual(await fs.readFile(f.selection.record),saved);
  const fresh=new CaseLibrarySelection(f.options);assert.equal((await fresh.restore()).error,undefined);assert.equal((await fresh.list()).length,12);
});
