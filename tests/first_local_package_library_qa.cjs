'use strict';
// Independent opt-in generated/metadata-only controls. No Electron process or
// patient bundle is launched/read. Temporary bundle bytes below are inert text.
const fs=require('node:fs/promises');
const path=require('node:path');
const os=require('node:os');
const vm=require('node:vm');
const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const {gunzipSync}=require('node:zlib');
const {createRequire}=require('node:module');
const root=path.resolve(__dirname,'..'),electron=path.join(root,'desktop/electron');
const {CaseLibrarySelection}=require(path.join(electron,'case-library-selection.cjs'));
const catalog=require(path.join(electron,'patient-catalog.cjs'));
const {RELEASE,TWELVE_RELEASE,LOCATION_EVIDENCE}=catalog;
const readerSha='2383adf1fc42b82d16dec8b193bda2ba5afc3fa45eb2e1f456ecb836aa913556';
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const canonical=p=>fs.realpath(p);
const pick=p=>async()=>({canceled:false,filePaths:[p]});

async function fixture(base,name='library'){
  const library=path.join(base,name);await fs.mkdir(library);
  for(const relative of [RELEASE.path,TWELVE_RELEASE.path,TWELVE_RELEASE.prospectivePath,LOCATION_EVIDENCE.path]){
    await fs.mkdir(path.dirname(path.join(library,relative)),{recursive:true});
    await fs.copyFile(path.join(root,relative),path.join(library,relative));
  }
  const bytes=gunzipSync(await fs.readFile(path.join(electron,'fixtures/patient-catalog-existing-twelve-execution-v1.json.gz')));
  assert.equal(sha(bytes),TWELVE_RELEASE.catalogSha256);
  await fs.mkdir(path.dirname(path.join(library,TWELVE_RELEASE.catalogPath)),{recursive:true});
  await fs.writeFile(path.join(library,TWELVE_RELEASE.catalogPath),bytes);
  return library;
}

async function main(){
  const output=process.argv[2];assert(output,'Provide a new QA report path');
  const destination=path.resolve(output);assert(destination.startsWith(path.join(root,'artifacts/poc-first-local-package-independent-v1/')));
  assert.equal(sha(await fs.readFile(path.join(electron,'patient-catalog.cjs'))),readerSha);
  const base=await canonical(await fs.mkdtemp(path.join(os.tmpdir(),'rsl-independent-library-')));
  const checks=[],opened=[];let actualPayloadRead=false;
  const originalOpen=fs.open;
  fs.open=async function(filename,...args){
    if(/\.(?:nii(?:\.gz)?|ressectionlab|rsworkspace|tag)$/.test(String(filename))){
      opened.push(path.basename(String(filename)));
      if(!String(filename).startsWith(base+path.sep)){actualPayloadRead=true;throw new Error('Patient payload read forbidden');}
    }
    return originalOpen.call(this,filename,...args);
  };
  const test=async(name,fn)=>{await fn();checks.push(name);};
  try{
    const library=await fixture(base),user=path.join(base,'user'),prepared=path.join(user,'prepared-cases');
    const options={userData:user,defaultRoot:null,preparedRoot:prepared,nativePaths:[]};
    const selection=new CaseLibrarySelection(options);
    await test('packaged empty state has no catalog/app.asar root or persisted implicit selection',async()=>{
      assert.deepEqual(await selection.restore(),{folder:null});assert.deepEqual(await selection.list(),[]);
      assert.equal(selection.current.catalog,null);await assert.rejects(fs.stat(selection.record),{code:'ENOENT'});
    });
    await test('native-only selection canonicalizes, preserves twelve source rows and six reported labels, and persists 0600',async()=>{
      const alias=path.join(base,'alias');await fs.symlink(library,alias);
      assert.deepEqual(await selection.choose(pick(alias)),{folder:library});
      const rows=await selection.list();assert.equal(rows.length,12);assert.equal(rows.filter(r=>r.sourceLocation).length,6);
      assert(rows.every(r=>r.qualifiedCatalog&&!('path' in r)&&!('entry' in r)&&!('aliases' in r)));
      assert.equal((await fs.stat(selection.record)).mode&0o777,0o600);
      const saved=JSON.parse(await fs.readFile(selection.record,'utf8'));assert.equal(saved.root,library);
      assert.equal(saved.directory.inode,String((await fs.stat(library,{bigint:true})).ino));
    });
    const saved=await fs.readFile(selection.record),rows=await selection.list();
    await test('fresh process equivalent restores canonical directory and exact catalog IDs',async()=>{
      const fresh=new CaseLibrarySelection(options);assert.deepEqual(await fresh.restore(),{folder:library});assert.deepEqual(await fresh.list(),rows);
    });
    await test('cancel and invalid choices preserve previous instance, bytes and rows',async()=>{
      const current=selection.current;
      assert.equal(await selection.choose(async()=>({canceled:true,filePaths:[]})),null);
      for(const result of [{canceled:false,filePaths:[]},{canceled:false,filePaths:[library,library]},
        {canceled:false,filePaths:['relative']},{canceled:false,filePaths:[base]}])
        await assert.rejects(selection.choose(async()=>result));
      assert.equal(selection.current,current);assert.deepEqual(await fs.readFile(selection.record),saved);assert.deepEqual(await selection.list(),rows);
    });
    await test('concurrent native selection is refused and cancellation releases lock',async()=>{
      let settle;const waiting=selection.choose(()=>new Promise(resolve=>settle=resolve));
      await assert.rejects(selection.choose(pick(library)),/already open/);
      settle({canceled:true,filePaths:[]});assert.equal(await waiting,null);
      assert.equal(await selection.choose(async()=>({canceled:true,filePaths:[]})),null);
    });
    await test('renderer IDs cannot supply arbitrary path/configuration or fabricated case identity',async()=>{
      for(const id of [library,'../outside',{}, {id:rows[0].id,path:library},'a'.repeat(64)])await assert.rejects(selection.resolve(id));
    });
    await test('catalog ID retains exact bundle hash authority before native open',async()=>{
      const entry=(await selection.current.catalog.load())[0].entry,filename=path.join(library,entry.bundle);
      await fs.mkdir(path.dirname(filename),{recursive:true});await fs.writeFile(filename,'inert wrong bundle fixture');
      await assert.rejects(selection.resolve(rows.find(r=>r.title===entry.case_id).id),/bundle|size|changed|checksum/i);
      await fs.unlink(filename);
    });
    await test('changed sidecar bytes reject before catalog open authority',async()=>{
      const p=path.join(library,LOCATION_EVIDENCE.path),original=await fs.readFile(p);
      await fs.writeFile(p,Buffer.concat([original,Buffer.from(' ')]));
      await assert.rejects(selection.list(),/checksum/);await assert.rejects(selection.resolve(rows[0].id));
      await fs.writeFile(p,original);await selection.choose(pick(library));
    });
    await test('optional report-sidecar absence removes labels without changing qualified source membership',async()=>{
      const p=path.join(library,LOCATION_EVIDENCE.path),original=await fs.readFile(p);await fs.unlink(p);
      const absent=await selection.list();assert.equal(absent.length,12);assert.equal(absent.filter(r=>r.sourceLocation).length,0);
      assert.deepEqual(absent.map(r=>r.id),rows.map(r=>r.id));await fs.writeFile(p,original);
    });
    await test('unchanged reader refuses protected roles, foreign sources, generated/accepted anatomy and promoted location',async()=>{
      const read=async relative=>JSON.parse(await fs.readFile(path.join(library,relative),'utf8'));
      const records=[await read(TWELVE_RELEASE.path),await read(TWELVE_RELEASE.prospectivePath),await read(RELEASE.path),await read(TWELVE_RELEASE.catalogPath)];
      assert.equal(catalog.validateTwelveCatalogMetadata(...records).length,12);
      for(const mutation of [r=>r[3].entries[0].role='EVAL',r=>r[3].entries[0].patient_group='UPENN:001',
        r=>r[3].entries[0].input_binding.candidate.source_files[0].sha256='f'.repeat(64),
        r=>r[3].entries[0].virtual_descendant=true,r=>r[3].entries[0].layers[0].provenance='simulated',
        r=>r[3].entries[0].layers[0].accepted_anatomical_truth=true,r=>r[3].entries[0].tumor_location.lobe='frontal']){
        const changed=structuredClone(records);mutation(changed);assert.throws(()=>catalog.validateTwelveCatalogMetadata(...changed));
      }
    });
    await test('corrupt persisted selection blocks fallback; native remember remains independent',async()=>{
      await fs.writeFile(selection.record,'{');const fresh=new CaseLibrarySelection(options);
      assert((await fresh.restore()).error);await assert.rejects(fresh.list());
      const manual=path.join(base,'native-open.ressectionlab');await fs.writeFile(manual,'inert manually selected bytes');
      await fresh.remember(manual);assert(JSON.parse(await fs.readFile(path.join(user,'anatomy-cases.json'),'utf8')).recent.includes(manual));
      await fs.writeFile(selection.record,saved);await fs.unlink(path.join(user,'anatomy-cases.json'));
    });
    await test('saved settings symlink, oversize and stale directory identity are refused on fresh instance',async()=>{
      for(const value of [Buffer.alloc(16385),JSON.stringify({...JSON.parse(saved),directory:{device:'0',inode:'0'}})]){
        await fs.writeFile(selection.record,value);assert((await new CaseLibrarySelection(options).restore()).error);
      }
      await fs.unlink(selection.record);await fs.symlink(path.join(base,'missing-settings'),selection.record);
      assert((await new CaseLibrarySelection(options).restore()).error);
      await fs.unlink(selection.record);await fs.writeFile(selection.record,saved);
    });
    await test('selected catalog removal in live process refuses instead of empty/ordinary fallback',async()=>{
      const other=await fixture(base,'removable'),s=new CaseLibrarySelection({...options,userData:path.join(base,'remove-user')});
      await s.choose(pick(other));
      for(const relative of [RELEASE.path,TWELVE_RELEASE.path,TWELVE_RELEASE.prospectivePath,TWELVE_RELEASE.catalogPath,LOCATION_EVIDENCE.path])await fs.unlink(path.join(other,relative));
      await assert.rejects(s.list());await assert.rejects(s.resolve(rows[0].id));
      assert(s.info().error);assert.equal(await s.choose(async()=>({canceled:true,filePaths:[]})),null);assert(s.info().error);
    });
    await test('live selected root replacement with same pinned metadata rejects directory-identity change',async()=>{
      const old=await fixture(base,'replaceable'),s=new CaseLibrarySelection({...options,userData:path.join(base,'replace-user')});
      await s.choose(pick(old));await fs.rename(old,path.join(base,'previous-directory'));await fixture(base,'replaceable');
      await assert.rejects(s.list());await assert.rejects(s.resolve(rows[0].id));assert(s.info().error);
    });
    await test('selected twelve-person catalog cannot silently fall back to ten after active catalog loss',async()=>{
      const other=await fixture(base,'fallback'),s=new CaseLibrarySelection({...options,userData:path.join(base,'fallback-user')});
      const oldCatalog=gunzipSync(await fs.readFile(path.join(electron,'fixtures/patient-catalog-existing-nine-plus-ixi265-v1.json.gz')));
      assert.equal(sha(oldCatalog),RELEASE.catalogSha256);await fs.mkdir(path.dirname(path.join(other,RELEASE.catalogPath)),{recursive:true});
      await fs.writeFile(path.join(other,RELEASE.catalogPath),oldCatalog);await s.choose(pick(other));assert.equal((await s.list()).length,12);
      await fs.unlink(path.join(other,TWELVE_RELEASE.catalogPath));await assert.rejects(s.list());assert(s.info().error);
    });
    await test('preload drops supplied library root and policy arguments',async()=>{
      let api;const calls=[];vm.runInNewContext(await fs.readFile(path.join(electron,'preload.cjs'),'utf8'),{
        require(name){assert.equal(name,'electron');return {contextBridge:{exposeInMainWorld(_name,value){api=value;}},
          ipcRenderer:{invoke(...args){calls.push(args);return Promise.resolve(null);}}};}});
      await api.chooseCaseLibrary({root:'/arbitrary',role:'EVAL'});await api.caseLibraryInfo('/arbitrary');
      assert.deepEqual(calls,[['research:chooseCaseLibrary'],['research:caseLibraryInfo']]);
    });
    await test('actual main handlers enforce trusted frame, no renderer root and native directory dialog',async()=>{
      const captures={handlers:new Map(),dialogs:0,engineConfigs:[]},resources=path.join(base,'resources');
      await fs.mkdir(path.join(resources,'research-engine'),{recursive:true});await fs.writeFile(path.join(resources,'research-engine/ressectionlab-engine'),'inert executable sentinel');
      class Sidecar{constructor(config){captures.engineConfigs.push(config);}on(){}request(){throw new Error('Engine must not run');}}
      const hostRequire=createRequire(path.join(electron,'main.cjs'));
      const native={app:{isPackaged:true,getPath:name=>path.join(base,'main-'+name),requestSingleInstanceLock:()=>true,
        on(){},whenReady:()=>new Promise(()=>{}),quit(){}},ipcMain:{handle:(name,fn)=>captures.handlers.set(name,fn)},
        dialog:{async showOpenDialog(_window,opts){captures.dialogs++;captures.options=opts;return {canceled:true,filePaths:[]};}}};
      const context={require:name=>name==='electron'?native:name==='./sidecar.cjs'?{Sidecar}:hostRequire(name),
        process:{...process,argv:[],resourcesPath:resources},__dirname:electron,Buffer,console,URL,structuredClone};
      vm.createContext(context);
      vm.runInContext(await fs.readFile(path.join(electron,'main.cjs'),'utf8')+'\nglobalThis.qaMain={startEngine,bindOperations,setWindow(w,u){window=w;allowedUrl=u;},getLibrary(){return caseLibrary;}};',context);
      await context.qaMain.startEngine();context.qaMain.bindOperations();
      assert.equal(context.qaMain.getLibrary().current.catalog,null);assert.equal(context.qaMain.getLibrary().info().folder,null);
      assert.equal(captures.engineConfigs.length,1);const config=captures.engineConfigs[0];
      assert.equal(config.executable,path.join(resources,'research-engine/ressectionlab-engine'));
      assert(!('python' in config)&&!('sourcePath' in config)&&!('observedSourceRoot' in config));
      const mainFrame={url:'file:///trusted/index.html'},contents={mainFrame},window={webContents:contents,isDestroyed:()=>false};
      context.qaMain.setWindow(window,mainFrame.url);const event={sender:contents,senderFrame:mainFrame};
      const choose=captures.handlers.get('research:chooseCaseLibrary');
      await assert.rejects(choose({...event,sender:{}}),/Untrusted/);
      await assert.rejects(choose(event,{root:library}),/Unsupported/);
      assert.equal(captures.dialogs,0);assert.equal(await choose(event,{}),null);assert.equal(captures.dialogs,1);
      assert.equal(JSON.stringify(captures.options.properties),'["openDirectory"]');
      assert.equal(captures.options.defaultPath,undefined);assert.equal(captures.engineConfigs.length,1);
      await fs.rm(config.transferDir,{recursive:true,force:true});
    });
    assert.equal(actualPayloadRead,false);
    const paths=['desktop/electron/case-library-selection.cjs','desktop/electron/case-library-selection.test.cjs',
      'desktop/electron/main.cjs','desktop/electron/preload.cjs','desktop/electron/case-library.cjs','desktop/electron/patient-catalog.cjs',
      'desktop/src/App.tsx','desktop/src/WorkbenchPanels.tsx','desktop/src/types.ts','desktop/src/workbench.css',
      'desktop/tests/case-library-panel.test.mjs','tests/first_local_package_library_qa.cjs'];
    const pins=Object.fromEntries(await Promise.all(paths.map(async name=>[name,sha(await fs.readFile(path.join(root,name)))])));
    assert.equal(pins['desktop/electron/patient-catalog.cjs'],readerSha);
    const report={schema:'local-package-independent-library-qa-v1',status:'pass',checks_passed:checks.length,checks,pins,
      actual_patient_payloads_read:false,native_engine_or_electron_launched:false,package_build_run:false,
      initial_finding_preserved:'artifacts/poc-first-local-package-independent-v1/initial-library-finding.json',
      scope:'Generated temporary folder and IPC controls, copied pinned catalog metadata only. Final packaged native GUI remains a separate ROOT check.'};
    await fs.writeFile(destination,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({status:'pass',checks_passed:checks.length,report_sha256:sha(await fs.readFile(destination))}));
  }finally{fs.open=originalOpen;await fs.rm(base,{recursive:true,force:true});}
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});
