'use strict';
const fs=require('node:fs/promises');
const {createReadStream,constants}=require('node:fs');
const path=require('node:path');
const os=require('node:os');
const {createHash}=require('node:crypto');
const {AssetRegistry}=require('./assets.cjs');
const MAX_BYTES=512*1024**2;
async function fileHash(filename){const hash=createHash('sha256');for await(const chunk of createReadStream(filename))hash.update(chunk);return hash.digest('hex');}
async function retainPrefixAssets(registry){
 const root=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'ressectionlab-prefix-assets-')));await fs.chmod(root,0o700);
 const snapshot=new AssetRegistry(root);const copied=new Map();let bytes=0,closed=false,active=false;
 async function retain(sourceRegistry){
  if(closed||active)throw Error('Retained prefix asset lease is unavailable');
  active=true;const addedIds=[],addedKeys=[],addedPaths=[];const previousBytes=bytes;
  try{
  const registry=sourceRegistry;
  if(!(registry.records instanceof Map)||registry.records.size>32768)throw Error('Prefix asset registry exceeds its bounded inventory');
  const originalRoot=await fs.realpath(registry.root);
  for(const [id,record] of registry.records){
   if(typeof id!=='string'||!/^[a-f0-9]{64}$/.test(record.sha256)||!Number.isSafeInteger(record.byteLength)||record.byteLength<1)throw Error('Invalid retained prefix asset');
   const prior=snapshot.records.get(id);
   if(prior){if(prior.sha256!==record.sha256||prior.byteLength!==record.byteLength)throw Error('Retained prefix asset ID changed');continue;}
   if(snapshot.records.size>=32768)throw Error('Prefix asset registry exceeds its bounded inventory');
   const source=await fs.realpath(record.path);
   if(source!==record.path||!source.startsWith(originalRoot+path.sep))throw Error('Retained prefix asset escaped its source registry');
   const key=record.sha256+'-'+record.byteLength;
   let destination=copied.get(key);
   if(!destination){
    bytes+=record.byteLength;if(bytes>MAX_BYTES)throw Error('Retained prefix assets exceed 512 MiB');
    destination=path.join(root,key+'.bin');
    await fs.copyFile(source,destination,constants.COPYFILE_EXCL|constants.COPYFILE_FICLONE);
    addedPaths.push(destination);
    const info=await fs.stat(destination);
    if(!info.isFile()||info.size!==record.byteLength||await fileHash(destination)!==record.sha256)throw Error('Retained prefix asset checksum changed');
    await fs.chmod(destination,0o400);copied.set(key,destination);addedKeys.push(key);
   }
   snapshot.records.set(id,{...record,path:destination});addedIds.push(id);
  }
  }catch(error){
   for(const id of addedIds)snapshot.records.delete(id);
   for(const key of addedKeys)copied.delete(key);
   bytes=previousBytes;
   await Promise.all(addedPaths.map(filename=>fs.rm(filename,{force:true})));
   throw error;
  }finally{active=false;}
 }
 try{
  await retain(registry);
  return{registry:snapshot,get bytes(){return bytes;},get files(){return copied.size;},retain,
   close:async()=>{if(active)throw Error('Retained prefix asset copy is still running');closed=true;await fs.rm(root,{recursive:true,force:true});}};
 }catch(error){await fs.rm(root,{recursive:true,force:true});throw error;}
}
module.exports={retainPrefixAssets,fileHash,MAX_BYTES};
