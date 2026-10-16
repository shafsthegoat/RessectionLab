'use strict';
const fs = require('node:fs/promises');
const {constants} = require('node:fs');
const path = require('node:path');
const {createHash} = require('node:crypto');

// A host-owned release, not a renderer-supplied path, role or allowlist.
const RELEASE = Object.freeze({
  id: 'existing-nine-plus-ixi265-v1',
  path: 'manifests/patient-corpus-releases/existing-nine-plus-ixi265-v1.json',
  sha256: '2c426f82e160be5b11bb19d9d5c4d2ac804ae9d85011cf46a4dc83904adbea84',
  catalogPath: 'outputs/patient-corpus-ixi265-v1/catalog.json',
  // Updated only after the qualification run is frozen and independently checked.
  catalogSha256: '2a7ba1c26f101261e3a5fecf84ae998b637a70c5a7f66d43d224153c738ee673',
  catalogHash: 'sha256:554c172ee7061c1ec803a4ea56c8025b5e6969f777bfecfe44987ec8aa2479a3',
  planHash: 'sha256:a1a2623d05270881d4291a1037d303578cb27a4f63d52cce28b5f5db251d8755',
});
const SCOPE = Object.freeze(Object.fromEntries([
  ...['05','16','20','22','25','28'].map(n=>[`BTC:sub-PAT${n}`,{role:'TRAIN',bundle:`outputs/cases/BTC-sub-PAT${n}-structural-evidence.ressectionlab`,source:'BTC'}]),
  ['UCSF-PDGM:004',{role:'DEVELOPMENT',bundle:'outputs/poc-cases/UCSF-PDGM-0004.ressectionlab',source:'UCSF-PDGM'}],
  ['Lausanne:sub-000',{role:'TRAIN',bundle:'outputs/poc-cases/Lausanne-sub-000-ses-20110101.ressectionlab',source:'Lausanne'}],
  ['SynthRAD2023:1BA336',{role:'TRAIN',bundle:'outputs/poc-cases/SynthRAD2023-1BA336.ressectionlab',source:'SynthRAD2023'}],
  ['IXI:265',{role:'TRAIN',bundle:'outputs/poc-ixi265/vesselfm-base-v1/IXI265-vesselfm-estimates.ressectionlab',source:'IXI'}],
]));
const ANATOMY = new Set(['brain','tumor','skull','vessels','ventricles']);
const digest = value => createHash('sha256').update(value).digest('hex');
const stable = value => value && typeof value==='object' ? Array.isArray(value) ? value.map(stable) : Object.fromEntries(Object.keys(value).sort().map(k=>[k,stable(value[k])])) : value;
const canonical = value => JSON.stringify(stable(value));
const equal = (a,b) => canonical(a)===canonical(b);
const need = (condition,message) => {if(!condition)throw new Error(`Patient catalog unavailable: ${message}`);};
const sha = value => typeof value==='string' && /^[a-f0-9]{64}$/.test(value);
function relative(value) {
  return typeof value==='string' && value.length<1024 && !value.includes('\\') && !value.includes('\0') &&
    !path.posix.isAbsolute(value) && value.split('/').every(part=>part && part!=='.' && part!=='..');
}

/** Pure metadata gate. Reading bundle/image bytes is deliberately impossible here. */
function validateCatalogMetadata(release,catalog) {
  need(release?.schema==='patient-corpus-frozen-release-v1'&&release.release_id===RELEASE.id&&
    release.virtual_descendants_admitted===false&&release.protected_roles_unchanged===true&&
    release.training_admission_established===false,'release scope changed');
  need(Array.isArray(release.base_candidates)&&release.base_candidates.length===9&&release.additional_candidate?.patient_group==='IXI:265','release membership changed');
  const candidates=[...release.base_candidates,release.additional_candidate],byGroup=new Map();
  for(const candidate of candidates) {
    const scope=SCOPE[candidate?.patient_group];
    need(scope&&candidate.role===scope.role&&candidate.existing_materialized===true,'protected or out-of-scope source group or role');
    need(relative(candidate.bundle)&&candidate.bundle===scope.bundle&&sha(candidate.bundle_sha256),'bundle path or identity changed');
    need(typeof candidate.case_id==='string'&&candidate.case_id.length<160&&Array.isArray(candidate.source_files)&&candidate.source_files.length>0&&candidate.source_files.length<=16,'source binding missing');
    need(candidate.source_files.every(s=>relative(s.path)&&sha(s.sha256)&&['imaging','annotation'].includes(s.role)),'source path or checksum is invalid');
    need(!byGroup.has(candidate.patient_group),'duplicate source group');byGroup.set(candidate.patient_group,candidate);
  }
  need(catalog?.schema==='patient-corpus-qualification-v1.catalog'&&catalog.release_id===RELEASE.id&&catalog.plan_hash===RELEASE.planHash&&
    catalog.parent_plan_hash===release.base_plan_hash&&catalog.catalog_hash===RELEASE.catalogHash&&catalog.protected_imaging_opened===false,'catalog release identity changed');
  need(Array.isArray(catalog.entries)&&catalog.entries.length===10&&catalog.counts?.qualified_existing_people===10&&
    catalog.counts.newly_acquired_people===0&&catalog.counts.anatomical_truth_accepted_people===0&&catalog.counts.virtual_descendants_counted_as_people===0,'catalog qualification scope changed');
  const seen=new Set();
  for(const entry of catalog.entries) {
    const candidate=byGroup.get(entry?.patient_group),scope=SCOPE[entry?.patient_group];
    need(scope&&entry.role===scope.role&&candidate&&!seen.has(entry.patient_group),'protected, duplicate or out-of-scope catalog entry');seen.add(entry.patient_group);
    need(entry.status==='qualified_with_limitations'&&entry.release_id===RELEASE.id&&entry.existing_materialized===true&&entry.new_patient===false&&
      entry.virtual_descendant===false&&entry.clinical_use_permitted===false&&entry.training_admission_established===false&&entry.training_admitted===false,'entry promotes unreviewed or virtual anatomy');
    need(entry.case_id===candidate.case_id&&entry.bundle===candidate.bundle&&entry.bundle_sha256===candidate.bundle_sha256&&
      entry.input_binding?.plan_hash===RELEASE.planHash&&entry.input_binding.release_id===RELEASE.id&&equal(entry.input_binding.candidate,candidate)&&
      entry.input_binding.bundle_sha256===candidate.bundle_sha256,'entry source or frozen candidate binding changed');
    need(Number.isSafeInteger(entry.input_binding.bundle_bytes)&&entry.input_binding.bundle_bytes>0&&entry.input_binding.bundle_bytes<=512*1024*1024,'bundle size is invalid');
    need(Array.isArray(entry.imaging)&&entry.imaging.length>0&&entry.imaging.length<=32&&entry.imaging.every(image=>
      candidate.source_files.some(source=>source.role==='imaging'&&source.path===image.source_path&&source.sha256===image.source_sha256)&&image.source_voxels_equal===true),'acquired image source changed');
    need(Array.isArray(entry.anatomy_present)&&entry.anatomy_present.length<=5&&new Set(entry.anatomy_present).size===entry.anatomy_present.length&&entry.anatomy_present.every(a=>ANATOMY.has(a))&&
      Array.isArray(entry.missing_anatomy)&&entry.missing_anatomy.every(a=>ANATOMY.has(a)),'anatomy inventory changed');
    need(['lobe','hemisphere','anatomical_region'].every(k=>entry.tumor_location?.[k]==='UNKNOWN'),'anatomical location is not accepted');
    need(Array.isArray(entry.layers)&&entry.layers.every(layer=>layer.accepted_anatomical_truth===false&&!['simulated','generated'].includes(layer.provenance)),
      'generated or accepted anatomy is outside this release');
  }
  return catalog.entries;
}

async function readBoundedMetadata(root,relativePath,maxBytes) {
  const filename=path.join(root,relativePath),resolved=await fs.realpath(filename);
  need(resolved===filename,'metadata path escaped the release root');
  const file=await fs.open(resolved,constants.O_RDONLY|constants.O_NOFOLLOW);
  try {
    const stat=await file.stat();need(stat.isFile()&&stat.size>0&&stat.size<=maxBytes,'metadata exceeds its read bound');
    const buffer=Buffer.alloc(maxBytes+1);const {bytesRead}=await file.read(buffer,0,buffer.length,0);
    need(bytesRead===stat.size&&bytesRead<=maxBytes,'metadata changed during reading');
    return buffer.subarray(0,bytesRead);
  } finally {await file.close();}
}

// An ungenerated catalog is optional. A dangling symlink or redirected parent
// is an existing invalid metadata path, not an absent optional artifact.
async function metadataAbsent(root,relativePath) {
  const parts=relativePath.split('/');let filename=root;
  for(let i=0;i<parts.length;i++) {
    filename=path.join(filename,parts[i]);let stat;
    try {stat=await fs.lstat(filename);}catch(error){if(error.code==='ENOENT')return true;throw error;}
    need(!stat.isSymbolicLink(),'metadata path escaped the release root');
    if(i<parts.length-1)need(stat.isDirectory(),'metadata parent is not a directory');
  }
  return false;
}

/** Streams a bounded expected bundle, without decoding ZIP, JSON or image payloads. */
async function verifyBundleBytes(filename,expectedHash,expectedBytes) {
  need(sha(expectedHash)&&Number.isSafeInteger(expectedBytes)&&expectedBytes>0&&expectedBytes<=512*1024*1024,'invalid bundle checksum request');
  const file=await fs.open(filename,constants.O_RDONLY|constants.O_NOFOLLOW);
  try {
    const before=await file.stat();need(before.isFile()&&before.size===expectedBytes,'bundle size changed; refresh its source qualification');
    const hash=createHash('sha256'),buffer=Buffer.alloc(1024*1024);let offset=0;
    while(offset<expectedBytes){const {bytesRead}=await file.read(buffer,0,Math.min(buffer.length,expectedBytes-offset),offset);need(bytesRead>0,'bundle became incomplete');hash.update(buffer.subarray(0,bytesRead));offset+=bytesRead;}
    const after=await file.stat(),named=await fs.stat(filename);
    need(before.dev===after.dev&&before.ino===after.ino&&before.size===after.size&&before.mtimeMs===after.mtimeMs&&before.ctimeMs===after.ctimeMs&&
      after.dev===named.dev&&after.ino===named.ino&&after.size===named.size&&after.mtimeMs===named.mtimeMs&&after.ctimeMs===named.ctimeMs,'bundle changed during verification');
    need(hash.digest('hex')===expectedHash,'bundle checksum changed; the catalog entry must be requalified');
  } finally {await file.close();}
}

class PatientCatalog {
  constructor(root){this.root=path.resolve(root);}
  // The original checkout is a specifically named same-workspace source, not a
  // caller-controlled escape hatch. Exact source-bound bundle paths still apply.
  roots(){return [this.root,...(path.basename(this.root)==='RessectionLab-poc-first'?[path.join(path.dirname(this.root),'RessectionLab')]:[])];}
  aliases(entry){return this.roots().map(root=>path.join(root,entry.bundle));}
  async load() {
    const root=await fs.realpath(this.root);need(root===this.root,'catalog root must be canonical');
    let releaseBytes,catalogBytes;
    try {releaseBytes=await readBoundedMetadata(root,RELEASE.path,64*1024);}
    catch(error){
      if(error.code==='ENOENT'&&await metadataAbsent(root,RELEASE.path)) {
        if(await metadataAbsent(root,RELEASE.catalogPath))return [];
        throw Error('Patient catalog unavailable: frozen release is missing');
      }
      throw error;
    }
    need(digest(releaseBytes)===RELEASE.sha256,'frozen release checksum changed');
    try {catalogBytes=await readBoundedMetadata(root,RELEASE.catalogPath,512*1024);}
    catch(error){if(error.code==='ENOENT'&&await metadataAbsent(root,RELEASE.catalogPath))return [];throw error;}
    need(digest(catalogBytes)===RELEASE.catalogSha256,'catalog checksum changed; refresh the qualified release');
    let release,catalog;try{release=JSON.parse(releaseBytes);catalog=JSON.parse(catalogBytes);}catch{throw Error('Patient catalog unavailable: invalid metadata JSON');}
    return validateCatalogMetadata(release,catalog).map(entry=>({
      id:'catalog-'+digest(`${RELEASE.sha256}:${RELEASE.catalogSha256}:${entry.patient_group}:${entry.bundle_sha256}`),
      title:entry.case_id,bytes:entry.input_binding.bundle_bytes,qualifiedCatalog:true,source:SCOPE[entry.patient_group].source,
      anatomy:[...entry.anatomy_present],missingAnatomy:[...entry.missing_anatomy],
      entry,aliases:this.aliases(entry),
    }));
  }
  async resolve(id) {
    const row=(await this.load()).find(item=>item.id===id);need(row,'select a case from the current catalog');
    const filename=path.join(this.root,row.entry.bundle),resolved=await fs.realpath(filename);
    need(row.aliases.includes(resolved),'bundle path escaped its exact released source location');
    const selectedRoot=this.roots().find(root=>path.join(root,row.entry.bundle)===resolved);
    need(selectedRoot&&await fs.realpath(selectedRoot)===selectedRoot,'source root is redirected');
    await verifyBundleBytes(resolved,row.entry.bundle_sha256,row.bytes);
    // Recheck release identity and path after streaming, before granting open authority.
    need((await this.load()).some(item=>item.id===id)&&await fs.realpath(filename)===resolved,'release or source path changed while opening');
    return resolved;
  }
}
module.exports={PatientCatalog,RELEASE,validateCatalogMetadata,verifyBundleBytes};
