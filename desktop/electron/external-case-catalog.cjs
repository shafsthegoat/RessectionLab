'use strict';
const fs=require('node:fs/promises');
const path=require('node:path');
const crypto=require('node:crypto');
const {createReadStream}=require('node:fs');
const {execFile}=require('node:child_process');
const {promisify}=require('node:util');
const exec=promisify(execFile);
const fail=message=>{throw new Error(`External data library: ${message}`);};
const inside=(root,file)=>file.startsWith(root+path.sep);
const digest=async file=>{const hash=crypto.createHash('sha256');for await(const chunk of createReadStream(file))hash.update(chunk);return hash.digest('hex');};

async function verifyVolume(root,marker){
  if(process.platform!=='darwin')fail('external volume verification currently requires macOS.');
  const mount=marker.mount_point;
  if(typeof mount!=='string'||!path.isAbsolute(mount)||mount==='/'||!inside(mount,root))fail('invalid volume identity marker.');
  const {stdout:mounts}=await exec('/sbin/mount',[],{timeout:5000,maxBuffer:65536});
  const entry=mounts.split('\n').map(line=>line.split(' on ')).find(parts=>parts.length===2&&parts[1].split(' (')[0]===mount&&parts[0].startsWith('/dev/'));
  if(!entry)fail('the recorded external volume is not mounted.');
  let stdout;
  for(let attempt=0;attempt<2;attempt++){
    try{({stdout}=await exec('diskutil',['info','-plist',entry[0]],{timeout:15000,maxBuffer:65536}));break;}
    catch(error){if(attempt===1)throw error;}
  }
  const uuid=stdout.match(/<key>VolumeUUID<\/key>\s*<string>([^<]+)<\/string>/)?.[1];
  if(uuid!==marker.volume_uuid||!/<key>Internal<\/key>\s*<false\s*\/>/.test(stdout))fail('connect the recorded external volume.');
  if((await fs.stat(root)).dev!==(await fs.stat(mount)).dev)fail('data directory is on a different volume.');
}

class ExternalCaseCatalog{
  constructor(root,{checkVolume=verifyVolume}={}){if(typeof root!=='string'||!path.isAbsolute(root))fail('configure an absolute data-root path.');this.root=path.resolve(root);this.checkVolume=checkVolume;}
  async metadata(file,limit=8*1024*1024){
    const stat=await fs.lstat(file);
    if(!stat.isFile()||stat.isSymbolicLink()||stat.size===0||stat.size>limit)fail('invalid catalog metadata.');
    return JSON.parse(await fs.readFile(file,'utf8'));
  }
  async load(){
    try{
      if(await fs.realpath(this.root)!==this.root)fail('configured data root must be a real directory.');
      const marker=await this.metadata(path.join(this.root,'.ressectionlab-data-root.json'),16384);
      if(marker.schema!=='ressectionlab-data-root-v1'||typeof marker.root_id!=='string')fail('identity marker is invalid.');
      await this.checkVolume(this.root,marker);
      let document;
      try{document=await this.metadata(path.join(this.root,'catalog','release.json'));}
      catch(error){if(error.code==='ENOENT')return [];throw error;}
      if(document.schema!=='ressectionlab-data-release-v1'||!Array.isArray(document.cases)||document.cases.length>10000)fail('release catalog is invalid.');
      const ids=new Set(),rows=[];
      for(const row of document.cases){
        if(!['accepted_for_stated_capabilities','accepted_with_explicit_limitations'].includes(row.decision)||
          typeof row.case_id!=='string'||!row.case_id||typeof row.path!=='string'||path.isAbsolute(row.path)||
          !row.path.endsWith('.ressectionlab')||row.path.split(/[\\/]/).includes('..')||
          !/^[a-f0-9]{64}$/.test(row.sha256)||!Number.isSafeInteger(row.bytes)||row.bytes<1||
          typeof row.review_path!=='string'||path.isAbsolute(row.review_path)||row.review_path.split(/[\\/]/).includes('..')||
          !/^[a-f0-9]{64}$/.test(row.review_sha256)||!Array.isArray(row.capabilities)||!Array.isArray(row.missing_anatomy)||
          !row.capabilities.every(value=>typeof value==='string')||!row.missing_anatomy.every(value=>typeof value==='string')||
          (row.source!==undefined&&typeof row.source!=='string')||(row.limitations!==undefined&&(!Array.isArray(row.limitations)||!row.limitations.every(value=>typeof value==='string'))))fail('case acceptance record is invalid.');
        const filename=path.resolve(this.root,row.path),review=path.resolve(this.root,row.review_path);
        if(!inside(this.root,filename)||!inside(this.root,review))fail('case or review path escapes the root.');
        if(await fs.realpath(filename)!==filename||await fs.realpath(review)!==review)fail('case dependencies may not escape through symbolic links.');
        const stat=await fs.stat(filename);
        if(!stat.isFile()||stat.size!==row.bytes)fail(`package changed: ${row.case_id}`);
        const id='external-'+crypto.createHash('sha256').update(marker.root_id+'\0'+row.case_id).digest('hex');
        if(ids.has(id))fail('duplicate case identity.');ids.add(id);
        rows.push({id,title:row.case_id,bytes:row.bytes,aliases:[filename],filename,review,
          source:row.source||'External',anatomy:[],capabilities:row.capabilities,limitations:row.limitations||[],
          missingAnatomy:row.missing_anatomy,sha256:row.sha256,reviewSha256:row.review_sha256});
      }
      return rows;
    }catch(error){if(error.message.startsWith('External data library:'))throw error;
      fail(`unavailable or invalid. Connect the configured volume (${this.root}). ${error.message}`);}
  }
  async resolve(id){
    const row=(await this.load()).find(row=>row.id===id);if(!row)fail('select a reviewed case from the current release.');
    if(await digest(row.filename)!==row.sha256||await digest(row.review)!==row.reviewSha256)fail('package or independent review checksum changed.');
    await this.load();return row.filename;
  }
}
module.exports={ExternalCaseCatalog};
