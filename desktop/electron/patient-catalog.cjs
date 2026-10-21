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
const TWELVE_RELEASE = Object.freeze({
  id: 'existing-twelve-execution-v1',
  path: 'manifests/patient-corpus-releases/existing-twelve-execution-admission-v1.json',
  sha256: '76c3c7e1f3805c136c508f53c4b31eae56b0ea3e9b876b556326a08b1fb681d9',
  prospectiveId: 'existing-ten-plus-synthrad-bc-v1',
  prospectivePath: 'manifests/patient-corpus-releases/existing-ten-plus-synthrad-bc-v1.json',
  prospectiveSha256: 'd1eb5d7642ac01c46ad8d7a2464627408358d438ca0c302bcaa8c71e759bbbd1',
  prospectivePlanHash: 'sha256:0e27cffd8bd556c12460678d33b65b85e2a5f5819a2e09b03b23450b111c1093',
  catalogPath: 'outputs/patient-corpus-twelve-execution-v1/catalog.json',
  catalogSha256: '5f8b833079e0b4673d2b8d01a36fab474574164755ac731a0972fa8071bcd088',
  catalogHash: 'sha256:6354e2e955aa4f4a5e0b892d5e3e8b069d9f8cb046196787b08273e7a94b4083',
  planHash: 'sha256:ecdc8f380437c0f1b1536a12fac0c22a07774c3f97e7d340d25d83f4dc4121d6',
});
const LOCATION_EVIDENCE = Object.freeze({
  path:'artifacts/btc-six-source-location-evidence-v1/evidence.json',
  sha256:'f5d9db8e86f7ecc8343b5c302715f93d65727246dfd94a735d5db94d56c64362',
  sourcePath:'data/diffusion_source/ds001226-v5.0.1/participants.tsv',
  sourceSha256:'2ae814b3e90e5c45ea95342a3658e58ba21d3f2d7bf1caa83ef8f1c74986e71d',
  sourceUri:'https://raw.githubusercontent.com/OpenNeuroDatasets/ds001226/359d372c5e972a161966312128adb365870df949/participants.tsv',
  paperSha256:'38e4890b6dc7d7b3bb84e2abfda005f050e7d93cf3c8a9586f74ce0c1a16589d',
  paperUri:'https://www.eneuro.org/content/5/3/ENEURO.0083-18.2018/tab-figures-data',
  paperMetadataUri:'https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6001263/fullTextXML',
});
const BTC_REPORTED_SITES=Object.freeze(Object.fromEntries([
  ['05','Frontal','Left'],['16','Fronto-temporal','Right'],['20','Parietal','Right'],
  ['22','Parietal','Right'],['25','Temporal','Right'],['28','Frontal','Left'],
].map(([id,location,side])=>[`BTC:sub-PAT${id}`,{participant:`sub-PAT${id}`,location,side}])));
const SCOPE = Object.freeze(Object.fromEntries([
  ...['05','16','20','22','25','28'].map(n=>[`BTC:sub-PAT${n}`,{role:'TRAIN',bundle:`outputs/cases/BTC-sub-PAT${n}-structural-evidence.ressectionlab`,source:'BTC'}]),
  ['UCSF-PDGM:004',{role:'DEVELOPMENT',bundle:'outputs/poc-cases/UCSF-PDGM-0004.ressectionlab',source:'UCSF-PDGM'}],
  ['Lausanne:sub-000',{role:'TRAIN',bundle:'outputs/poc-cases/Lausanne-sub-000-ses-20110101.ressectionlab',source:'Lausanne'}],
  ['SynthRAD2023:1BA336',{role:'TRAIN',bundle:'outputs/poc-cases/SynthRAD2023-1BA336.ressectionlab',source:'SynthRAD2023'}],
  ['IXI:265',{role:'TRAIN',bundle:'outputs/poc-ixi265/vesselfm-base-v1/IXI265-vesselfm-estimates.ressectionlab',source:'IXI'}],
]));
const TWELVE_SCOPE = Object.freeze({...SCOPE,
  'SynthRAD2023:1BB033':{role:'TRAIN',bundle:'outputs/poc-synthrad-bc-source-v1/1BB033/SynthRAD2023-1BB033.ressectionlab',source:'SynthRAD2023',bundleHash:'6da3934b794ad13945ea8d1443a70c46625a75c9b8a7cf2a800d2830c906dd82'},
  'SynthRAD2023:1BC077':{role:'TRAIN',bundle:'outputs/poc-synthrad-bc-source-v1/1BC077/SynthRAD2023-1BC077.ressectionlab',source:'SynthRAD2023',bundleHash:'171101168d1244d509fc64249cd9039a74aa9a35f35535dddc753307e80f905d'},
});
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

function tenCandidates(release) {
  need(release?.schema==='patient-corpus-frozen-release-v1'&&release.release_id===RELEASE.id&&
    release.virtual_descendants_admitted===false&&release.protected_roles_unchanged===true&&
    release.training_admission_established===false,'release scope changed');
  need(Array.isArray(release.base_candidates)&&release.base_candidates.length===9&&release.additional_candidate?.patient_group==='IXI:265','release membership changed');
  return [...release.base_candidates,release.additional_candidate];
}

/** Pure metadata gates. Reading bundle/image bytes is deliberately impossible here. */
function validateCatalogMetadata(release,catalog) {
  return validateEntries(tenCandidates(release),catalog,RELEASE,SCOPE,release.base_plan_hash);
}

function validateTwelveCatalogMetadata(admission,prospective,parentRelease,catalog) {
  need(admission?.schema==='patient-corpus-execution-admission-v1'&&admission.release_id===TWELVE_RELEASE.id&&
    admission.parent_ten_plan_hash===RELEASE.planHash&&admission.prospective_release?.path===TWELVE_RELEASE.prospectivePath&&
    admission.prospective_release.sha256===TWELVE_RELEASE.prospectiveSha256&&admission.prospective_release.plan_hash===TWELVE_RELEASE.prospectivePlanHash&&
    admission.anatomical_truth_accepted===false&&admission.training_admission_established===false&&
    admission.model_execution_allowed===false&&admission.solver_execution_allowed===false&&admission.new_acquisition_allowed===false,'twelve admission changed');
  need(prospective?.schema==='patient-corpus-prospective-release-v1'&&prospective.release_id===TWELVE_RELEASE.prospectiveId&&
    prospective.parent_release_id===RELEASE.id&&prospective.parent_release_path===RELEASE.path&&prospective.parent_release_sha256===RELEASE.sha256&&
    prospective.parent_plan_hash===RELEASE.planHash&&prospective.virtual_descendants_admitted===false&&
    prospective.training_admission_established===false,'twelve candidate release changed');
  need(Array.isArray(prospective.base_candidates)&&prospective.base_candidates.length===10&&
    Array.isArray(prospective.additional_candidates)&&prospective.additional_candidates.length===2,'twelve release membership changed');
  const byGroup=items=>[...items].sort((a,b)=>String(a?.patient_group).localeCompare(String(b?.patient_group)));
  need(equal(byGroup(prospective.base_candidates),byGroup(tenCandidates(parentRelease))),'twelve base candidates changed');
  need(equal(prospective.additional_candidates.map(c=>c?.patient_group).sort(),['SynthRAD2023:1BB033','SynthRAD2023:1BC077']),'twelve additional candidates changed');
  return validateEntries([...prospective.base_candidates,...prospective.additional_candidates],catalog,TWELVE_RELEASE,TWELVE_SCOPE,RELEASE.planHash);
}

function validateEntries(candidates,catalog,pins,allowed,parentPlanHash) {
  const byGroup=new Map();
  for(const candidate of candidates) {
    const scope=allowed[candidate?.patient_group];
    need(scope&&candidate.role===scope.role&&candidate.existing_materialized===true,'protected or out-of-scope source group or role');
    need(relative(candidate.bundle)&&candidate.bundle===scope.bundle&&sha(candidate.bundle_sha256)&&
      (!scope.bundleHash||candidate.bundle_sha256===scope.bundleHash),'bundle path or identity changed');
    need(typeof candidate.case_id==='string'&&candidate.case_id.length<160&&Array.isArray(candidate.source_files)&&candidate.source_files.length>0&&candidate.source_files.length<=16,'source binding missing');
    need(candidate.source_files.every(s=>relative(s.path)&&sha(s.sha256)&&['imaging','annotation'].includes(s.role)),'source path or checksum is invalid');
    need(!byGroup.has(candidate.patient_group),'duplicate source group');byGroup.set(candidate.patient_group,candidate);
  }
  need(catalog?.schema==='patient-corpus-qualification-v1.catalog'&&catalog.release_id===pins.id&&catalog.plan_hash===pins.planHash&&
    catalog.parent_plan_hash===parentPlanHash&&catalog.catalog_hash===pins.catalogHash&&catalog.protected_imaging_opened===false,'catalog release identity changed');
  need(Array.isArray(catalog.entries)&&catalog.entries.length===candidates.length&&catalog.counts?.qualified_existing_people===candidates.length&&
    catalog.counts.newly_acquired_people===0&&catalog.counts.anatomical_truth_accepted_people===0&&catalog.counts.virtual_descendants_counted_as_people===0,'catalog qualification scope changed');
  const seen=new Set();
  for(const entry of catalog.entries) {
    const candidate=byGroup.get(entry?.patient_group),scope=allowed[entry?.patient_group];
    need(scope&&entry.role===scope.role&&candidate&&!seen.has(entry.patient_group),'protected, duplicate or out-of-scope catalog entry');seen.add(entry.patient_group);
    need(entry.status==='qualified_with_limitations'&&entry.release_id===pins.id&&entry.existing_materialized===true&&entry.new_patient===false&&
      entry.virtual_descendant===false&&entry.clinical_use_permitted===false&&entry.training_admission_established===false&&entry.training_admitted===false,'entry promotes unreviewed or virtual anatomy');
    need(entry.case_id===candidate.case_id&&entry.bundle===candidate.bundle&&entry.bundle_sha256===candidate.bundle_sha256&&
      entry.input_binding?.plan_hash===pins.planHash&&entry.input_binding.release_id===pins.id&&equal(entry.input_binding.candidate,candidate)&&
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

/** Patient-level source reports stay separate from reviewed anatomical fields. */
function validateSourceLocationEvidence(evidence,catalogEntries=null) {
  need(evidence?.schema==='btc-six-source-location-evidence-v1'&&evidence.status==='source_reported_unreviewed'&&
    evidence.anatomical_review_accepted===false&&evidence.automatic_seed_placement_allowed===false&&evidence.canonical_catalog_modified===false&&
    evidence.native_mask_localization_established===false&&evidence.training_admission_established===false,'source-location review status changed');
  need(equal(evidence.catalog,{path:TWELVE_RELEASE.catalogPath,sha256:TWELVE_RELEASE.catalogSha256,catalog_hash:TWELVE_RELEASE.catalogHash}),'source-location catalog binding changed');
  const source=evidence.source,paper=evidence.paper_source;
  need(source?.dataset==='BTC_preop'&&source.accession==='ds001226'&&source.version==='5.0.1'&&
    source.commit==='359d372c5e972a161966312128adb365870df949'&&source.path===LOCATION_EVIDENCE.sourcePath&&
    source.sha256===LOCATION_EVIDENCE.sourceSha256&&source.uri===LOCATION_EVIDENCE.sourceUri&&
    source.field==='tumor location'&&source.identity_field==='participant_id','source-location dataset identity changed');
  need(paper?.doi==='10.1523/ENEURO.0083-18.2018'&&paper.pmcid==='PMC6001263'&&paper.table_id==='T1'&&
    paper.sha256===LOCATION_EVIDENCE.paperSha256&&paper.publication_uri===LOCATION_EVIDENCE.paperUri&&paper.metadata_uri===LOCATION_EVIDENCE.paperMetadataUri&&
    paper.representation==='Europe PMC archival full-text JATS XML of the primary article; hash is not of publisher HTML'&&
    equal(paper.selected_fields,['Subject ID','Tumor location','Tumor lateralization']),'source-location paper identity changed');
  need(Array.isArray(evidence.entries)&&evidence.entries.length===6,'source-location membership changed');
  if(catalogEntries!==null)need(Array.isArray(catalogEntries)&&catalogEntries.length===12&&catalogEntries.every(e=>e.release_id===TWELVE_RELEASE.id),'source-location selected release changed');
  const locations=new Map();
  for(const entry of evidence.entries) {
    const expected=BTC_REPORTED_SITES[entry?.patient_group],scope=SCOPE[entry?.patient_group];
    need(expected&&entry.role==='TRAIN'&&!locations.has(entry.patient_group),'protected, duplicate or out-of-scope source-location row');
    need(entry.status==='source_reported_unreviewed'&&entry.hemisphere==='UNKNOWN'&&entry.anatomical_lobe==='UNKNOWN'&&
      entry.anatomical_review_accepted===false&&entry.automatic_seed_placement_allowed===false,'source-location row promotes unreviewed anatomy');
    need(entry.participant_id===expected.participant&&entry.case_id===`BTC-ds001226-${expected.participant}-preop`&&
      entry.bundle===scope.bundle&&sha(entry.bundle_sha256)&&relative(entry.acquisition_record)&&
      Array.isArray(entry.source_files)&&entry.source_files.length>0&&entry.source_files.every(s=>relative(s.path)&&sha(s.sha256)&&['imaging','annotation'].includes(s.role)),
      'source-location row identity changed');
    need(entry.source_reported_location===expected.location&&entry.source_reported_lateralization===expected.side&&
      Number.isSafeInteger(entry.row_number)&&entry.row_number>1&&
      equal(entry.row_evidence,{participant_id:expected.participant,'tumor location':expected.location})&&
      equal(entry.paper_row_evidence,{'Subject ID':expected.participant.slice(4),'Tumor location':expected.location,'Tumor lateralization':expected.side}),
      'source-location reported fields changed');
    if(catalogEntries!==null) {
      const candidate=catalogEntries.find(row=>row.patient_group===entry.patient_group)?.input_binding?.candidate;
      need(candidate&&['case_id','role','bundle','bundle_sha256','source_files','acquisition_record'].every(key=>equal(entry[key],candidate[key]))&&
        candidate.participant?.value===entry.participant_id&&candidate.participant.path===source.path&&candidate.participant.sha256===source.sha256,
        'source-location frozen member binding changed');
    }
    locations.set(entry.patient_group,{label:`${expected.side} ${expected.location}`,status:'source-reported-unreviewed'});
  }
  return locations;
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

async function readPinnedMetadata(root,relativePath,expectedHash,maxBytes,label) {
  let bytes;
  try {bytes=await readBoundedMetadata(root,relativePath,maxBytes);}
  catch(error){if(error.code==='ENOENT'&&await metadataAbsent(root,relativePath))return null;throw error;}
  need(digest(bytes)===expectedHash,`${label} checksum changed`);
  try {return JSON.parse(bytes);}catch{throw Error('Patient catalog unavailable: invalid metadata JSON');}
}

async function loadTwelveMetadata(root) {
  // Check every present part before considering fallback; partial absence must
  // never conceal corrupt or redirected metadata elsewhere in this release.
  const admission=await readPinnedMetadata(root,TWELVE_RELEASE.path,TWELVE_RELEASE.sha256,64*1024,'twelve admission');
  const prospective=await readPinnedMetadata(root,TWELVE_RELEASE.prospectivePath,TWELVE_RELEASE.prospectiveSha256,64*1024,'twelve candidate release');
  const catalog=await readPinnedMetadata(root,TWELVE_RELEASE.catalogPath,TWELVE_RELEASE.catalogSha256,512*1024,'twelve catalog');
  if(!catalog)return null;
  need(admission&&prospective,'twelve release metadata is missing');
  const parent=await readPinnedMetadata(root,RELEASE.path,RELEASE.sha256,64*1024,'frozen release');
  need(parent,'frozen parent release is missing');
  return {entries:validateTwelveCatalogMetadata(admission,prospective,parent,catalog),pins:TWELVE_RELEASE,scope:TWELVE_SCOPE};
}

async function loadTenMetadata(root) {
  const release=await readPinnedMetadata(root,RELEASE.path,RELEASE.sha256,64*1024,'frozen release');
  const catalog=await readPinnedMetadata(root,RELEASE.catalogPath,RELEASE.catalogSha256,512*1024,'catalog');
  if(!catalog)return null;
  need(release,'frozen release is missing');
  return {entries:validateCatalogMetadata(release,catalog),pins:RELEASE,scope:SCOPE};
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
    const active=await loadTwelveMetadata(root)||await loadTenMetadata(root);
    const evidence=await readPinnedMetadata(root,LOCATION_EVIDENCE.path,LOCATION_EVIDENCE.sha256,32*1024,'source-location evidence');
    const locations=evidence?validateSourceLocationEvidence(evidence,active?.pins===TWELVE_RELEASE?active.entries:null):new Map();
    if(!active)return [];
    return active.entries.map(entry=>({
      id:'catalog-'+digest(`${active.pins.sha256}:${active.pins.catalogSha256}:${entry.patient_group}:${entry.bundle_sha256}`),
      title:entry.case_id,bytes:entry.input_binding.bundle_bytes,qualifiedCatalog:true,source:active.scope[entry.patient_group].source,
      anatomy:[...entry.anatomy_present],missingAnatomy:[...entry.missing_anatomy],
      ...(active.pins===TWELVE_RELEASE&&locations.has(entry.patient_group)?{sourceLocation:locations.get(entry.patient_group)}:{}),
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
module.exports={PatientCatalog,RELEASE,TWELVE_RELEASE,LOCATION_EVIDENCE,validateCatalogMetadata,validateTwelveCatalogMetadata,validateSourceLocationEvidence,verifyBundleBytes};
