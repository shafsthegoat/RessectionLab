'use strict';
const fs=require('node:fs/promises');
const path=require('node:path');
const crypto=require('node:crypto');
const {constants}=require('node:fs');
const {execFile}=require('node:child_process');
const {isDeepStrictEqual,promisify}=require('node:util');
const exec=promisify(execFile);
const fail=message=>{throw new Error(`External data library: ${message}`);};
const inside=(root,file)=>file.startsWith(root+path.sep);
// The verified macOS data volume may use a case-insensitive filesystem.
const foldPath=value=>path.resolve(value).normalize('NFC').toLowerCase();
const digest=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
const accepted=new Set(['accepted_for_stated_capabilities','accepted_with_explicit_limitations']);
const visibleRoles=new Set(['TRAIN','DEVELOPMENT']);
const sha=value=>typeof value==='string'&&/^[a-f0-9]{64}$/.test(value);
const label=(value,max=512)=>typeof value==='string'&&value.trim()===value&&value.length>0&&value.length<=max&&!/[\x00-\x1f]/.test(value);
const labels=value=>Array.isArray(value)&&value.length<=256&&value.every(item=>label(item,4096));
const relative=value=>label(value,1024)&&!value.includes('\\')&&value.split('/').every(part=>part&&part!=='.'&&part!=='..');
const fileStamp=stat=>[stat.dev,stat.ino,stat.size,stat.mtimeMs,stat.ctimeMs];
const sameFile=(a,b)=>isDeepStrictEqual(fileStamp(a),fileStamp(b));
const maxReviewBytes=256*1024;
const maxNativeBytes=16*1024**3;
const nativeClaims=['case_id','person_id','patient_id','timepoint','role','classification','source','path','sha256','bytes','scan_count',
  'decision','capabilities','limitations','missing_anatomy','legacy_case_id','legacy_case_sha256'];

async function verifyVolume(root,marker){
  if(process.platform!=='darwin')fail('external volume verification currently requires macOS.');
  const mount=marker.mount_point;
  if(typeof mount!=='string'||!path.isAbsolute(mount)||mount==='/'||!inside(mount,root))fail('invalid volume identity marker.');
  const {stdout:mounts}=await exec('/sbin/mount',[],{timeout:5000,maxBuffer:65536});
  const entry=mounts.split('\n').map(line=>line.split(' on ')).find(parts=>parts.length===2&&parts[1].split(' (')[0]===mount&&parts[0].startsWith('/dev/'));
  if(!entry)fail('the recorded external volume is not mounted.');
  const [rootStat,mountStat]=await Promise.all([fs.stat(root),fs.stat(mount)]);
  if(rootStat.dev!==mountStat.dev)fail('data directory is on a different volume.');
  // The checkout engine supplies a fast native macOS check, including the live
  // filesystem UUID. Packaged engines retain their existing diskutil path.
  const repo=path.resolve(__dirname,'../..'),python=path.join(repo,'.venv/bin/python');
  let checkout=false;try{await fs.access(python);checkout=true;}catch{}
  if(checkout){
    const {stdout}=await exec(python,['-m','resectionlab.data_root','--volume-info',root],{cwd:repo,timeout:5000,maxBuffer:65536});
    const info=JSON.parse(stdout);
    if(info.volume_uuid!==marker.volume_uuid||info.mount_point!==mount||'/dev/'+info.device!==entry[0])fail('connect the recorded external volume.');
    return;
  }
  let stdout;
  for(let attempt=0;attempt<2;attempt++){
    try{({stdout}=await exec('diskutil',['info','-plist',entry[0]],{timeout:15000,maxBuffer:65536}));break;}
    catch(error){if(attempt===1)throw error;}
  }
  const uuid=stdout.match(/<key>VolumeUUID<\/key>\s*<string>([^<]+)<\/string>/)?.[1];
  if(uuid!==marker.volume_uuid||!/<key>Internal<\/key>\s*<false\s*\/>/.test(stdout))fail('connect the recorded external volume.');
}

function libraryError(error,root) {
  return error.message.startsWith('External data library:') ? error :
    new Error(`External data library: unavailable or invalid. Connect the configured volume (${root}). ${error.message}`);
}

// Hash only the selected archive, without decoding or buffering the payload.
async function verifyFile(filename,sha256,expectedBytes) {
  const file=await fs.open(filename,constants.O_RDONLY|constants.O_NOFOLLOW);
  try {
    const before=await file.stat();
    if(!before.isFile()||(expectedBytes!==undefined&&before.size!==expectedBytes))fail('package changed before verification.');
    const hash=crypto.createHash('sha256');
    if(before.size>0)for await(const bytes of file.createReadStream({end:before.size-1,autoClose:false}))hash.update(bytes);
    if(!sameFile(before,await file.stat())||!sameFile(before,await fs.lstat(filename)))fail('package or review changed during verification.');
    if(hash.digest('hex')!==sha256)fail('package or independent review checksum changed.');
  } finally {await file.close();}
}

class ExternalCaseCatalog {
  constructor(root,{checkVolume=verifyVolume}={}) {
    if(typeof root!=='string'||!path.isAbsolute(root))fail('configure an absolute data-root path.');
    this.root=path.resolve(root);
    this.nativeRoot=path.join(this.root,'cases','native');
    this.checkVolume=checkVolume;
  }
  containsPath(filename) {return inside(foldPath(this.root),foldPath(filename));}
  ownsNativePath(filename) {return inside(foldPath(this.nativeRoot),foldPath(filename));}
  async regular(filename,{optional=false}={}) {
    try {
      if(await fs.realpath(filename)!==filename)fail('case dependencies may not escape through symbolic links.');
      const stat=await fs.lstat(filename);
      if(!stat.isFile())fail('expected a regular case dependency.');
      return stat;
    } catch(error) {
      if(!optional||error.code!=='ENOENT')throw error;
      // A dangling alias differs from an absent optional catalog.
      for(let current=filename;inside(this.root,current);current=path.dirname(current)) {
        let stat;
        try {stat=await fs.lstat(current);} catch(error) {if(error.code!=='ENOENT')throw error;}
        if(stat?.isSymbolicLink())fail('case dependencies may not escape through symbolic links.');
      }
      return null;
    }
  }
  async metadata(filename,{limit=8*1024*1024,optional=false,sha256}={}) {
    const stat=await this.regular(filename,{optional});
    if(!stat)return undefined;
    if(stat.size===0||stat.size>limit)fail('invalid catalog metadata size.');
    const file=await fs.open(filename,constants.O_RDONLY|constants.O_NOFOLLOW);
    try {
      const buffer=Buffer.alloc(stat.size+1),{bytesRead}=await file.read(buffer);
      if(bytesRead!==stat.size||!sameFile(stat,await file.stat())||!sameFile(stat,await fs.lstat(filename)))
        fail('catalog metadata changed during reading.');
      const bytes=buffer.subarray(0,bytesRead);
      if(sha256!==undefined&&digest(bytes)!==sha256)fail('independent review checksum changed.');
      return JSON.parse(bytes);
    } finally {await file.close();}
  }
  async verifiedRoot() {
    try {
      if(await fs.realpath(this.root)!==this.root)fail('configured data root must be a real directory.');
      const marker=await this.metadata(path.join(this.root,'.ressectionlab-data-root.json'),{limit:16384});
      if(marker?.schema!=='ressectionlab-data-root-v1'||!label(marker.root_id))fail('identity marker is invalid.');
      await this.checkVolume(this.root,marker);
      return marker;
    } catch(error) {throw libraryError(error,this.root);}
  }
  async legacyRows(marker) {
    const document=await this.metadata(path.join(this.root,'catalog','release.json'),{optional:true});
    if(document===undefined)return [];
    if(document?.schema!=='ressectionlab-data-release-v1'||!Array.isArray(document.cases)||document.cases.length>10000)fail('release catalog is invalid.');
    const ids=new Set(),rows=[];
    for(const row of document.cases) {
      if(!row||!accepted.has(row.decision)||!label(row.case_id)||!relative(row.path)||!row.path.endsWith('.ressectionlab')||
        !sha(row.sha256)||!Number.isSafeInteger(row.bytes)||row.bytes<1||!relative(row.review_path)||!sha(row.review_sha256)||
        !labels(row.capabilities)||!labels(row.missing_anatomy)||(row.source!==undefined&&typeof row.source!=='string')||
        (row.limitations!==undefined&&!labels(row.limitations)))fail('case acceptance record is invalid.');
      const filename=path.join(this.root,row.path),review=path.join(this.root,row.review_path);
      const stat=await this.regular(filename);
      await this.regular(review);
      if(stat.size!==row.bytes)fail(`package changed: ${row.case_id}`);
      const id='external-'+digest(marker.root_id+'\0'+row.case_id);
      if(ids.has(id))fail('duplicate case identity.');
      ids.add(id);
      rows.push({id,title:row.case_id,bytes:row.bytes,aliases:[filename],filename,review,caseId:row.case_id,
        personId:row.person_id,datasetRole:row.role,source:row.source||'External',anatomy:[],capabilities:row.capabilities,
        limitations:row.limitations||[],missingAnatomy:row.missing_anatomy,sha256:row.sha256,reviewSha256:row.review_sha256});
    }
    return rows;
  }
  async personAliases() {
    const document=await this.metadata(path.join(this.root,'catalog','topbrain','person-aliases.json'),{optional:true,limit:256*1024});
    const aliases=new Map();
    if(document===undefined)return aliases;
    if(document?.schema!=='ressectionlab-topbrain-person-aliases-v1'||!Array.isArray(document.people)||document.people.length>1000)
      fail('source person aliases are invalid.');
    for(const person of document.people) {
      if(!label(person?.canonical_person_id)||!label(person.role)||!labels(person.aliases))fail('source person alias is invalid.');
      const identity={personId:person.canonical_person_id,datasetRole:person.role};
      for(const alias of [person.canonical_person_id,...person.aliases]) {
        if(aliases.has(alias)&&!isDeepStrictEqual(aliases.get(alias),identity))fail('source person alias is ambiguous.');
        aliases.set(alias,identity);
      }
    }
    return aliases;
  }
  async nativeRows(marker,legacyRows) {
    const document=await this.metadata(path.join(this.root,'catalog','native-release.json'),{optional:true});
    if(document===undefined)return [];
    if(document?.schema!=='ressectionlab-native-study-release-v1'||document.root_id!==marker.root_id||
      !Array.isArray(document.cases)||document.cases.length>10000)fail('native study catalog is invalid.');
    const aliases=await this.personAliases();
    const legacyByCase=new Map(legacyRows.map(row=>[row.caseId,row]));
    const rolesByPerson=new Map();
    const personId=id=>aliases.get(id)?.personId??id;
    for(const row of legacyRows) {
      if(!row.personId||!row.datasetRole)continue;
      const canonical=personId(row.personId),roles=rolesByPerson.get(canonical)??new Set();
      roles.add(row.datasetRole);
      rolesByPerson.set(canonical,roles);
    }
    const ids=new Set(),paths=new Set(),hashes=new Set(),rows=[];
    for(const row of document.cases) {
      if(!row||!accepted.has(row.decision)||!visibleRoles.has(row.role)||!label(row.case_id)||!label(row.person_id)||
        !label(row.patient_id)||!label(row.timepoint)||!label(row.classification)||!label(row.source)||
        !relative(row.path)||!row.path.startsWith('cases/native/')||!row.path.endsWith('.rsworkspace')||
        !sha(row.sha256)||!Number.isSafeInteger(row.bytes)||row.bytes<1||row.bytes>maxNativeBytes||
        !Number.isInteger(row.scan_count)||row.scan_count<1||row.scan_count>64||
        !relative(row.review_path)||!row.review_path.startsWith('qc/')||!row.review_path.endsWith('.json')||
        !sha(row.review_sha256)||!labels(row.capabilities)||!row.capabilities.length||
        !labels(row.limitations)||!labels(row.missing_anatomy))fail('native study acceptance record is invalid or protected.');
      const family=aliases.get(row.person_id),roles=rolesByPerson.get(row.person_id);
      if((family&&(family.personId!==row.person_id||family.datasetRole!==row.role))||
        (roles&&(roles.size!==1||!roles.has(row.role))))fail('native study person or frozen role differs.');
      const legacy=legacyByCase.get(row.case_id);
      if(legacy ? row.legacy_case_id!==legacy.caseId||row.legacy_case_sha256!==legacy.sha256||
        row.person_id!==personId(legacy.personId)||row.role!==legacy.datasetRole :
        row.legacy_case_id!==undefined||row.legacy_case_sha256!==undefined)fail('native study legacy case binding differs.');
      rolesByPerson.set(row.person_id,new Set([row.role]));
      const id='external-native-'+digest(marker.root_id+'\0'+row.case_id);
      if(ids.has(id)||paths.has(row.path)||hashes.has(row.sha256))fail('duplicate native study identity or package.');
      ids.add(id);paths.add(row.path);hashes.add(row.sha256);
      const filename=path.join(this.root,row.path),review=path.join(this.root,row.review_path);
      const evidence=await this.metadata(review,{limit:maxReviewBytes,sha256:row.review_sha256});
      if(evidence?.schema!=='ressectionlab-native-study-acceptance-v1'||evidence.independent_review!==true||
        evidence.catalog_exposure_allowed!==true||evidence.clinical_validation!==false||evidence.training_admitted!==false||
        !nativeClaims.every(key=>isDeepStrictEqual(row[key],evidence[key])))fail('native study independent admission differs.');
      if((await this.regular(filename)).size!==row.bytes)fail(`native package changed: ${row.case_id}`);
      rows.push({id,title:row.case_id,bytes:row.bytes,aliases:[filename],filename,review,caseId:row.case_id,
        personId:row.person_id,datasetRole:row.role,source:row.source,anatomy:[],capabilities:row.capabilities,
        limitations:row.limitations,missingAnatomy:row.missing_anatomy,sha256:row.sha256,reviewSha256:row.review_sha256,
        native:true,portable:true,timepoint:row.timepoint,scanCount:row.scan_count});
    }
    return rows;
  }
  async discover() {
    try {
      const marker=await this.verifiedRoot();
      const legacy=await this.legacyRows(marker);
      try {return {rows:[...legacy,...await this.nativeRows(marker,legacy)]};}
      catch(error) {return {rows:legacy,error:libraryError(error,this.root)};}
    } catch(error) {throw libraryError(error,this.root);}
  }
  async load() {
    try {
      const {rows,error}=await this.discover();
      if(error)throw error;
      return rows;
    } catch(error) {throw libraryError(error,this.root);}
  }
  async resolve(id) {
    try {
      const {rows,error}=await this.discover();
      const row=rows.find(row=>row.id===id);
      if(!row&&error)throw error;
      if(!row)fail('select a reviewed case from the current release.');
      await verifyFile(row.filename,row.sha256,row.bytes);
      if(!row.native)await verifyFile(row.review,row.reviewSha256);
      const current=(await this.discover()).rows.find(item=>item.id===id);
      if(!isDeepStrictEqual(row,current))fail('release or source path changed while opening.');
      return row.filename;
    } catch(error) {throw libraryError(error,this.root);}
  }
}
module.exports={ExternalCaseCatalog};
