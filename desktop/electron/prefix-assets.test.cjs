'use strict';
const test=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs/promises');const os=require('node:os');const path=require('node:path');const {createHash}=require('node:crypto');
const {AssetRegistry}=require('./assets.cjs');const {retainPrefixAssets,MAX_BYTES}=require('./prefix-assets.cjs');
test('retained source assets survive worker deletion with IDs, bounds and hashes intact',async()=>{
 const root=await fs.mkdtemp(path.join(os.tmpdir(),'prefix-assets-test-'));let held;
 try{
  const bytes=Buffer.from([1,2,3,4]);const file=path.join(root,'source.bin');await fs.writeFile(file,bytes);
  const registry=new AssetRegistry(root);const descriptor={path:file,dtype:'uint8',shape:[4],byteLength:4,byteOrder:'little',order:'C',sha256:createHash('sha256').update(bytes).digest('hex')};
  const first=await registry.register(descriptor);const second=await registry.register(descriptor);
  held=await retainPrefixAssets(registry);assert.equal(held.bytes,4);assert.equal(held.files,1);
  await fs.rm(root,{recursive:true,force:true});assert.deepEqual(Buffer.from(await held.registry.read(first.assetId)),bytes);assert.deepEqual(Buffer.from(await held.registry.read(second.assetId)),bytes);
  const saved=held.registry.records.get(first.assetId).path;await fs.chmod(saved,0o600);await fs.writeFile(saved,Buffer.from([9,2,3,4]));await assert.rejects(held.registry.read(first.assetId),/checksum/);
 }finally{await held?.close();await fs.rm(root,{recursive:true,force:true});}
});
test('retaining assets refuses a changed file, escaped path or aggregate oversized graph',async()=>{
 const root=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'prefix-assets-test-')));
 try{
  const source=path.join(root,'one');await fs.writeFile(source,'x');const registry=new AssetRegistry(root);
  const original={path:source,byteLength:1,sha256:createHash('sha256').update('x').digest('hex')};registry.records.set('known',original);
  await fs.writeFile(source,'y');await assert.rejects(retainPrefixAssets(registry),/checksum changed/);
  registry.records.set('known',{...original,path:path.join(root,'..','outside')});await assert.rejects(retainPrefixAssets(registry));
  registry.records.set('known',{...original,byteLength:MAX_BYTES+1});await assert.rejects(retainPrefixAssets(registry),/512 MiB/);
 }finally{await fs.rm(root,{recursive:true,force:true});}
});
test('same retained registry merges new IDs transactionally within its shared bound',async()=>{
 const root=await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(),'prefix-assets-test-')));let held;
 try{
  const registry=new AssetRegistry(root);const file=path.join(root,'one');await fs.writeFile(file,'a');
  const record={path:file,byteLength:1,sha256:createHash('sha256').update('a').digest('hex')};registry.records.set('original',record);
  held=await retainPrefixAssets(registry);const saved=held.registry;
  registry.records.clear();registry.records.set('duplicate',record);
  await held.retain(registry);assert.equal(held.registry,saved);assert.equal(held.bytes,1);assert.equal(held.files,1);
  const two=path.join(root,'two');await fs.writeFile(two,'b');const second={path:two,byteLength:1,sha256:createHash('sha256').update('b').digest('hex')};
  registry.records.clear();registry.records.set('new',second);registry.records.set('over-limit',{...record,byteLength:MAX_BYTES});
  await assert.rejects(held.retain(registry),/512 MiB/);assert.equal(saved.records.has('new'),false);assert.equal(held.bytes,1);assert.equal(held.files,1);
  registry.records.delete('over-limit');await held.retain(registry);assert.equal(held.bytes,2);assert.equal(held.files,2);assert.deepEqual(Buffer.from(await saved.read('new')),Buffer.from('b'));
  registry.records.set('original',second);await assert.rejects(held.retain(registry),/ID changed/);
  await held.close();await assert.rejects(held.retain(registry),/unavailable/);
 }finally{await held?.close();await fs.rm(root,{recursive:true,force:true});}
});
