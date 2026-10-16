'use strict';
// Opt-in independent host-reader controls. Real reads are metadata-only;
// successful byte/open checks use tiny explicitly generated temporary fixtures.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const os = require('node:os');
const crypto = require('node:crypto');
const vm = require('node:vm');
const {stripTypeScriptTypes} = require('node:module');
const ROOT = path.resolve(__dirname, '..');
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const cli = name => {const i=process.argv.indexOf(name); return i<0 ? null : process.argv[i+1];};
const files = {
  'desktop/electron/patient-catalog.cjs': '--reader-sha256',
  'desktop/electron/case-library.cjs': '--library-sha256',
  'desktop/src/WorkbenchPanels.tsx': '--panel-sha256',
  'desktop/electron/main.cjs': '--main-sha256',
};

async function main() {
  const implementation = {}, source = {};
  for (const [name, flag] of Object.entries(files)) {
    const expected=cli(flag); assert.match(expected??'', /^[a-f0-9]{64}$/, `Externally frozen pin required: ${flag}`);
    const content=await fs.readFile(path.join(ROOT,name));
    assert.equal(hash(content),expected,`${name} changed after freeze`);
    implementation[name]=expected; source[name]=content.toString();
  }
  const {PatientCatalog,RELEASE,validateCatalogMetadata,verifyBundleBytes}=require('../desktop/electron/patient-catalog.cjs');
  const {CaseLibrary}=require('../desktop/electron/case-library.cjs');
  const releaseBytes=await fs.readFile(path.join(ROOT,RELEASE.path));
  const catalogBytes=await fs.readFile(path.join(ROOT,RELEASE.catalogPath));
  assert.equal(hash(releaseBytes),'2c426f82e160be5b11bb19d9d5c4d2ac804ae9d85011cf46a4dc83904adbea84');
  assert.equal(hash(catalogBytes),'2a7ba1c26f101261e3a5fecf84ae998b637a70c5a7f66d43d224153c738ee673');
  const release=JSON.parse(releaseBytes),catalog=JSON.parse(catalogBytes);
  assert.equal(validateCatalogMetadata(release,catalog).length,10);
  const temporary=await fs.mkdtemp(path.join(os.tmpdir(),'independent-ten-person-reader-'));
  const root=await fs.realpath(temporary);
  const originalOpen=fs.open,originalReadFile=fs.readFile;
  let realPayloadReads=0;
  const opened=[];
  function checkRead(filename) {
    if(typeof filename!=='string' && !Buffer.isBuffer(filename))return;
    const resolved=path.resolve(String(filename));
    if(/\.(ressectionlab|nii(?:\.gz)?|npz|pt|safetensors)$/i.test(resolved)) {
      if(!resolved.startsWith(root+path.sep)){realPayloadReads++;throw Error('Independent audit refused real payload access');}
      opened.push(path.relative(root,resolved));
    }
  }
  fs.open=async function(filename,...args){checkRead(filename);return originalOpen.call(fs,filename,...args);};
  fs.readFile=async function(filename,...args){checkRead(filename);return originalReadFile.call(fs,filename,...args);};
  const results={};
  try {
    const write=async(relative,content)=>{const filename=path.join(root,relative);await fs.mkdir(path.dirname(filename),{recursive:true});await fs.writeFile(filename,content);return filename;};
    await write(RELEASE.path,releaseBytes);await write(RELEASE.catalogPath,catalogBytes);
    const reader=new PatientCatalog(root);
    const rows=await reader.load();
    assert.equal(rows.length,10);assert.equal(new Set(rows.map(r=>r.id)).size,10);
    assert(rows.every(row=>/^catalog-[a-f0-9]{64}$/.test(row.id)));
    assert.equal(opened.length,0);
    const selected=rows.find(row=>row.entry.patient_group==='IXI:265');
    for(const id of ['../../outside',selected.entry.bundle,'IXI:265','catalog-'+'0'.repeat(64),{},null]) {
      await assert.rejects(()=>reader.resolve(id),/select a case/);
    }
    assert.equal(opened.length,0);
    results.hard_pinned_metadata_listing={status:'pass',rows:10,unique_opaque_ids:10,payload_reads:0,arbitrary_path_identity_and_unknown_id_rejected:true};

    // Exact byte seals reject altered role, source, acceptance, generated lineage,
    // scope and even harmless whitespace before any bundle can be opened.
    const mutations=[
      ['role',value=>value.entries[0].role='SELECT'],
      ['generated',value=>value.entries[6].virtual_descendant=true],
      ['source_path',value=>value.entries[0].bundle='outputs/protected.ressectionlab'],
      ['fused_registration',value=>value.entries[6].registrations[0].overlayPermitted=true],
      ['location',value=>value.entries[6].tumor_location.lobe='frontal'],
      ['additional_person',value=>value.entries.push(structuredClone(value.entries[6]))],
    ];
    for(const [,mutate] of mutations){const changed=structuredClone(catalog);mutate(changed);await write(RELEASE.catalogPath,JSON.stringify(changed));await assert.rejects(()=>reader.load(),/catalog checksum changed/);}
    await write(RELEASE.catalogPath,Buffer.concat([catalogBytes,Buffer.from('\n')]));await assert.rejects(()=>reader.load(),/catalog checksum changed/);
    await write(RELEASE.catalogPath,catalogBytes);
    await write(RELEASE.path,Buffer.concat([releaseBytes,Buffer.from('\n')]));await assert.rejects(()=>reader.load(),/release checksum changed/);
    await write(RELEASE.path,releaseBytes);
    assert.equal(opened.length,0);
    results.byte_seal_mutations={status:'pass',cases:[...mutations.map(([name])=>name),'catalog_whitespace','release_whitespace'],rejected_before_payload:true};

    // A source alias can be listed without its image; opening still requires
    // that exact release's bounded bundle bytes. No real bundle is used.
    const tiny=Buffer.from('explicitly generated independent byte-check control');
    const filename=await write(selected.entry.bundle,tiny);
    await assert.rejects(()=>reader.resolve(selected.id),/bundle size changed/);
    const beforeSymlink=opened.length;
    const elsewhere=await write('elsewhere/fake.ressectionlab',tiny);
    await fs.unlink(filename);await fs.symlink(elsewhere,filename);
    await assert.rejects(()=>reader.resolve(selected.id),/escaped its exact released/);
    assert.equal(opened.length,beforeSymlink);
    await fs.unlink(filename);await fs.writeFile(filename,tiny);

    // Isolate the resolve/stream component with a generated trusted row. This
    // does not replace or loosen the separately tested real release pin gate.
    const generated=structuredClone(selected);
    generated.bytes=tiny.length;generated.entry.bundle_sha256=hash(tiny);
    generated.entry.input_binding.bundle_bytes=tiny.length;
    const isolated=new PatientCatalog(root);isolated.load=async()=>[generated];
    assert.equal(await isolated.resolve(generated.id),filename);
    await fs.writeFile(filename,Buffer.alloc(tiny.length,1));
    await assert.rejects(()=>isolated.resolve(generated.id),/bundle checksum changed/);
    await fs.writeFile(filename,tiny);
    await verifyBundleBytes(filename,hash(tiny),tiny.length);
    await assert.rejects(()=>verifyBundleBytes(filename,'f'.repeat(64),tiny.length),/bundle checksum changed/);
    await assert.rejects(()=>verifyBundleBytes(filename,hash(tiny),tiny.length+1),/bundle size changed/);
    const symlink=path.join(root,'generated-link.ressectionlab');await fs.symlink(filename,symlink);
    await assert.rejects(()=>verifyBundleBytes(symlink,hash(tiny),tiny.length));
    results.guarded_open={status:'pass',exact_release_with_fake_bundle_rejected:true,foreign_symlink_rejected_before_read:true,
      generated_fixture_roundtrip:true,same_size_hash_tampering_rejected:true,size_change_rejected:true,no_follow_file_open_verified:true,
      successful_open_scope:'Tiny generated file with isolated trusted-row provider; real release listing tested separately; no real bundle opened'};

    const user=path.join(root,'user'),prepared=path.join(root,'outputs/poc-cases');
    await fs.mkdir(prepared,{recursive:true});
    const library=new CaseLibrary(user,prepared,[],root);
    const publicRows=await library.list();
    assert.equal(publicRows.length,10);
    const allowedKeys=new Set(['id','title','recent','bytes','qualifiedCatalog','source','anatomy','missingAnatomy']);
    assert(publicRows.every(row=>Object.keys(row).every(key=>allowedKeys.has(key))));
    assert(!JSON.stringify(publicRows).includes('ressectionlab')&&!JSON.stringify(publicRows).includes(root));
    await library.remember(filename);
    const recentRows=await library.list();assert.equal(recentRows.length,10);
    assert(recentRows.find(row=>row.id===selected.id).recent);
    await assert.rejects(()=>library.resolve(selected.id),/bundle size changed/);
    for(const bad of ['../escape',filename,{id:selected.id},null])await assert.rejects(()=>library.resolve(bad),/current library/);
    results.renderer_boundary={status:'pass',list_contains_only_display_fields:true,recent_catalog_alias_deduplicated:true,open_reenters_catalog_hash_guard:true};

    const panel=source['desktop/src/WorkbenchPanels.tsx'];
    const a=panel.indexOf('export function filterLibraryCases('),b=panel.indexOf('export function CaseLibraryPanel(',a);
    assert(a>=0&&b>a);
    const code=stripTypeScriptTypes(panel.slice(a,b)).replace('export function','function');
    const context={};vm.runInNewContext(code+'\nglobalThis.filter=filterLibraryCases;',context);
    const filter=(query,input=publicRows)=>Array.from(context.filter(input,query),r=>r.title);
    assert.deepEqual(filter('IXI'),['IXI265-Guys-0845']);
    assert.deepEqual(filter('IXI brain vessels'),['IXI265-Guys-0845']);
    assert.equal(filter('tumor').length,7);assert.equal(filter('vessels').length,2);
    assert.equal(filter('IXI tumor').length,0);assert.equal(filter('ventricles').length,0);
    assert.equal(filter('UPENN').length,0);assert.equal(filter('    ').length,10);
    const fixture=[{id:'fake',title:'Caf\u00e9 Study',source:'Generated',anatomy:['brain'],missingAnatomy:['vessels'],receipt:'IXI unrelated shared provenance'}];
    assert.deepEqual(filter('cafe brain',fixture),['Caf\u00e9 Study']);
    assert.equal(filter('vessels',fixture).length,0);assert.equal(filter('IXI',fixture).length,0);
    assert(panel.includes('onClick={()=>onOpen(row.id)}'));
    assert(source['desktop/electron/main.cjs'].includes("openSaved(await caseLibrary.resolve(id))"));
    results.Cases_search={status:'pass',source_and_present_anatomy_only:true,all_query_terms_required:true,
      missing_anatomy_and_foreign_receipt_text_excluded:true,unicode_case_normalization:true,
      button_passes_opaque_id_and_host_resolves_before_open:true,GUI_observation_claimed:false};
    assert.equal(realPayloadReads,0);
  } finally {
    fs.open=originalOpen;fs.readFile=originalReadFile;
    await fs.rm(root,{recursive:true,force:true});
  }
  for(const [name,expected] of Object.entries(implementation))assert.equal(hash(await fs.readFile(path.join(ROOT,name))),expected,'Implementation changed during audit');
  assert.equal(hash(await fs.readFile(path.join(ROOT,RELEASE.catalogPath))),hash(catalogBytes));
  assert.equal(hash(await fs.readFile(path.join(ROOT,RELEASE.path))),hash(releaseBytes));
  const report={schema:'independent-ten-person-reader-audit-v1',status:'pass',created_utc:new Date().toISOString(),
    implementation_sha256:implementation,catalog_sha256:hash(catalogBytes),release_sha256:hash(releaseBytes),
    auditor_sha256:hash(await fs.readFile(__filename)),real_payloads_opened:realPayloadReads,
    controls:results,models_or_solvers_executed:false,scope:'Metadata gate plus tiny generated byte/path controls; no real patient bundle opened or GUI observed'};
  const output=path.join(ROOT,'outputs/poc-first-qa/ten-person-reader-independent-qa.json');
  const text=JSON.stringify(report,null,2)+'\n';assert(!text.includes('/Users/'));
  await fs.writeFile(output,text);
  process.stdout.write(JSON.stringify({status:'pass',artifact:path.relative(ROOT,output),sha256:hash(text)})+'\n');
}
main().catch(error=>{console.error(error);process.exitCode=1;});
