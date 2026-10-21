'use strict';
// Tiny inert archives and a fully mocked packager/compiler/signer. No build,
// download, archive extraction, Electron launch or patient data access.
const fs=require('node:fs/promises'),sync=require('node:fs'),path=require('node:path'),os=require('node:os');
const vm=require('node:vm'),assert=require('node:assert/strict'),{createHash}=require('node:crypto');
const root=path.resolve(__dirname,'..'),electron=path.join(root,'desktop/electron');
const helper=require(path.join(electron,'local-electron-archive.cjs'));
const {assertLockedElectronVersion}=require(path.join(electron,'electron-version.cjs'));
const sha=b=>createHash('sha256').update(b).digest('hex');
const version='44.5.1',name=`electron-v${version}-darwin-arm64.zip`;
const bytes=Buffer.from('PK\x03\x04independent inert archive bytes; not extracted');
const expected=sha(bytes),checksums={[name]:expected};
const lock="lockfileVersion: '9.0'\nimporters:\n  .:\n    devDependencies:\n      electron:\n        specifier: 44.5.1\n        version: 44.5.1\n";

async function integration(base,mode){
  const repo=path.join(base,mode),desktop=path.join(repo,'desktop'),archive=path.join(repo,'archive');
  const fixtureDir=path.join(desktop,'electron');await fs.mkdir(fixtureDir,{recursive:true});await fs.mkdir(archive);
  for(const directory of ['src','node_modules','sidecar/ressectionlab-engine','sidecar/notices'])await fs.mkdir(path.join(desktop,directory),{recursive:true});
  for(const filename of ['index.html','vite.config.ts','tsconfig.json'])await fs.writeFile(path.join(desktop,filename),'generated');
  await fs.writeFile(path.join(desktop,'src/app.ts'),'generated');await fs.writeFile(path.join(fixtureDir,'fixture.cjs'),'generated');
  await fs.writeFile(path.join(desktop,'pnpm-lock.yaml'),lock);
  await fs.writeFile(path.join(desktop,'package.json'),JSON.stringify({name:'generated',version:'0.0.0',main:'electron/main.cjs'}));
  await fs.writeFile(path.join(desktop,'sidecar/ressectionlab-engine/ressectionlab-engine'),'inert engine');
  await fs.mkdir(path.join(repo,'packaging'));await fs.writeFile(path.join(repo,'packaging/RessectionLab.icns'),'inert icon');
  const filename=path.join(archive,name);await fs.writeFile(filename,bytes);
  if(mode==='missing')await fs.unlink(filename);
  if(mode==='bad-checksum')await fs.writeFile(filename,Buffer.alloc(bytes.length,1));
  if(mode==='symlink'){await fs.rename(filename,path.join(archive,'target'));await fs.symlink(path.join(archive,'target'),filename);}
  const events={compiles:0,packager:0,signs:0,publishes:0,localOptions:[],downloads:0,prechecks:0,postchecks:0};
  const notices={
    async captureRendererNotices(_input,output){await fs.writeFile(path.join(output,'renderer-generated.json'),'{}');return {unresolved:[]};},
    async capturePythonNotices(_input,output){await fs.writeFile(path.join(output,'python-generated.json'),'{}');return {unresolved:[]};},
    async captureElectronNotices(_input,_version,output){await fs.mkdir(path.join(output,'electron'));const files={};
      for(const n of ['LICENSE','LICENSES.chromium.html']){const content=Buffer.from('generated '+n);await fs.writeFile(path.join(output,'electron',n),content);files['electron/'+n]={sha256:sha(content)};}return {files};}
  };
  const modules={
    'node:fs/promises':fs,'node:path':path,'node:crypto':require('node:crypto'),
    './electron-version.cjs':{assertLockedElectronVersion},
    './local-electron-archive.cjs':{
      async localElectronArchive(...args){events.prechecks++;return helper.localElectronArchive(...args);},
      async verifyLocalElectronArchive(...args){events.postchecks++;return helper.verifyLocalElectronArchive(...args);}},
    './dependency-notices.cjs':notices,
    './publish.cjs':{async publishBundle(){events.publishes++;return {}; }},
    'electron/package.json':{version},'electron/checksums.json':checksums,
    'node:child_process':{spawnSync(command,args,options){
      if(command==='/usr/bin/codesign')events.signs++;
      else if(command==='git')return {status:0,stdout:args[0]==='rev-parse'?'generated-revision\n':'',stderr:''};
      else {events.compiles++;if(args[0].endsWith('vite.js')){
        sync.mkdirSync(path.join(options.cwd,'dist'),{recursive:true});sync.writeFileSync(path.join(options.cwd,'dist/index.html'),'generated render');
        if(mode==='changed-before-packager')sync.writeFileSync(filename,Buffer.alloc(bytes.length,2));
      }}return {status:0,stdout:'',stderr:''};}},
    '@electron/packager':{async packager(options){events.packager++;events.localOptions.push(options.electronZipDir??null);
      if(mode==='packager-refusal')throw new Error('generated packager refused');
      const output=path.join(options.out,'generated-darwin-arm64');await fs.mkdir(path.join(output,'RessectionLab.app/Contents/Resources'),{recursive:true});
      for(const n of ['LICENSE','LICENSES.chromium.html'])await fs.writeFile(path.join(output,n),'generated '+n);
      if(mode==='changed-during-packager')await fs.writeFile(filename,Buffer.alloc(bytes.length,3));
      if(mode==='replaced-during-packager'){const replacement=filename+'.new';await fs.writeFile(replacement,bytes);await fs.rename(replacement,filename);}
      if(mode==='removed-during-packager')await fs.unlink(filename);
      return [output];}},
  };
  const fakeRequire=name=>{assert(Object.hasOwn(modules,name),'unexpected module '+name);return modules[name];};
  fakeRequire.resolve=name=>{assert.equal(name,'electron/package.json');return path.join(repo,'installed-electron/package.json');};
  const env=mode==='default'?{}:{RESECTIONLAB_ELECTRON_ZIP_DIR:mode==='empty'?'':archive};
  const context={require:fakeRequire,__dirname:fixtureDir,Buffer,process:{env,execPath:'/inert-node',stdout:{write(){}},stderr:{write(){}}},console};
  const source=await fs.readFile(path.join(electron,'package-macos.cjs'),'utf8');
  const tail="main().catch(error => { process.stderr.write(error.stack+'\\n'); process.exitCode = 1; });";
  assert(source.includes(tail));vm.createContext(context);vm.runInContext(source.replace(tail,'globalThis.qaPackageMain=main;'),context);
  let error=null;try{await context.qaPackageMain();}catch(e){error=e.message;}
  const report=JSON.parse(await fs.readFile(path.join(desktop,'release/build-report.json'),'utf8'));
  if(['valid','default'].includes(mode)){
    assert.equal(error,null);assert.equal(report.status,'built_unverified');assert.equal(events.compiles,2);
    assert.equal(events.packager,1);assert.equal(events.signs,2);assert.equal(events.publishes,1);
    assert.deepEqual(events.localOptions,[mode==='default'?null:archive]);assert.equal(events.postchecks,2);
    const manifest=JSON.parse(await fs.readFile(path.join(report.inputSnapshot,'runtime/BUILD_INPUT_MANIFEST.json'),'utf8'));
    assert.equal(manifest.electronArchiveSource.mode,mode==='default'?'checksum_managed_download_cache':'explicit_local');
    if(mode==='valid'){assert.equal(manifest.electronArchiveSource.sha256,expected);assert(!JSON.stringify(manifest).includes(archive));}
  }else{
    assert(error);assert.equal(report.status,'failed');assert.equal(events.signs,0);assert.equal(events.publishes,0);
    if(['missing','bad-checksum','symlink','empty'].includes(mode)){assert.equal(events.compiles,0);assert.equal(events.packager,0);}
    if(mode==='changed-before-packager'){assert.equal(events.compiles,2);assert.equal(events.packager,0);}
    if(mode.includes('during-packager')||mode==='packager-refusal')assert.equal(events.packager,1);
  }
  assert.equal(events.downloads,0);return {mode,status:error?'refused':'mocked-success',events};
}

async function main(){
  const output=path.resolve(process.argv[2]||'');assert(output.startsWith(path.join(root,'artifacts/poc-first-local-package-independent-v1/')));
  const base=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'rsl-independent-archive-'))),checks=[],results=[];
  const test=async(name,fn)=>{await fn();checks.push(name);};
  try{
    const archive=path.join(base,'helper');await fs.mkdir(archive);const filename=path.join(archive,name);await fs.writeFile(filename,bytes);
    await test('only absent option preserves default mode',async()=>{
      assert.equal(await helper.localElectronArchive(undefined,null,null),null);
      for(const value of ['',null,false,12,'.','../cache'])await assert.rejects(helper.localElectronArchive(value,version,checksums));
    });
    await test('exact checksum binds immutable file identity',async()=>{
      const record=await helper.localElectronArchive(archive,version,checksums);
      assert.equal(record.sha256,expected);assert.equal(record.bytes,bytes.length);assert(Object.isFrozen(record)&&Object.isFrozen(record.identity));
      await helper.verifyLocalElectronArchive(record);
    });
    await test('version, architecture and malformed/upstream checksum mismatches reject',async()=>{
      for(const [v,c] of [['44.5',checksums],[version,{}],[version,{[name]:'f'.repeat(64)}],
        [version,{[name]:expected.toUpperCase()}],[version,{'electron-v44.5.1-darwin-x64.zip':expected}]])
        await assert.rejects(helper.localElectronArchive(archive,v,c));
    });
    await test('directory and ancestor symlinks fail even with exact archive bytes',async()=>{
      const alias=path.join(base,'alias');await fs.symlink(archive,alias);
      await assert.rejects(helper.localElectronArchive(alias,version,checksums));
      const inner=path.join(archive,'inner');await fs.mkdir(inner);await fs.writeFile(path.join(inner,name),bytes);
      await assert.rejects(helper.localElectronArchive(path.join(alias,'inner'),version,checksums));
    });
    await test('file mutation during streaming fails identity recheck',async()=>{
      const originalOpen=fs.open;let changed=false;
      fs.open=async function(p,...args){const handle=await originalOpen.call(this,p,...args);if(p!==filename)return handle;
        const read=handle.read.bind(handle);handle.read=async(...a)=>{const result=await read(...a);if(!changed){changed=true;await fs.writeFile(filename,Buffer.alloc(bytes.length,7));}return result;};return handle;};
      try{await assert.rejects(helper.localElectronArchive(archive,version,checksums),/changed while reading/);}finally{fs.open=originalOpen;await fs.writeFile(filename,bytes);}
    });
    for(const mode of ['valid','default','empty','missing','bad-checksum','symlink','changed-before-packager',
      'changed-during-packager','replaced-during-packager','removed-during-packager','packager-refusal']){
      await test('whole package integration '+mode,async()=>results.push(await integration(base,mode)));
    }
    // Source-basis check of installed packager: explicit local branch throws for
    // absent directory/file and only its else branch calls the downloader.
    const packagerSource=path.join(root,'desktop/node_modules/@electron/packager/dist/packager.js');
    const text=await fs.readFile(packagerSource,'utf8');
    const branch=text.slice(text.indexOf('if (this.opts.electronZipDir)'),text.indexOf('if (this.opts.electronZipDir)')+1000);
    assert(branch.includes('throw new Error')&&branch.includes('return zipPath')&&branch.includes('else {')&&branch.includes('downloadElectronZip(downloadOpts)'));
    checks.push('installed packager explicit local branch does not call downloader fallback');
    const files=['desktop/electron/package-macos.cjs','desktop/electron/local-electron-archive.cjs',
      'desktop/electron/local-electron-archive.test.cjs','desktop/electron/electron-version.cjs','tests/first_local_package_archive_qa.cjs'];
    const pins=Object.fromEntries(await Promise.all(files.map(async p=>[p,sha(await fs.readFile(path.join(root,p)))])));
    const report={schema:'local-package-independent-offline-archive-qa-v1',status:'pass',checks_passed:checks.length,checks,pins,
      installed_packager_source_sha256:sha(await fs.readFile(packagerSource)),generated_integration:results,
      real_archive_read:false,downloads:false,compiler_or_signer_or_engine_executed:false,package_build_run:false,patient_payloads_read:false,
      limitation:'Mocked package integration verifies ordering, options, refusal and publication boundaries; ROOT must separately build and inspect the exact real app.'};
    await fs.writeFile(output,JSON.stringify(report,null,2)+'\n',{flag:'wx'});
    console.log(JSON.stringify({status:'pass',checks_passed:checks.length,report_sha256:sha(await fs.readFile(output))}));
  }finally{await fs.rm(base,{recursive:true,force:true});}
}
main().catch(error=>{console.error(error.stack);process.exitCode=1;});
