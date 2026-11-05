'use strict';
const fs=require('node:fs/promises');
const {constants}=require('node:fs');
const path=require('node:path');
const {randomUUID}=require('node:crypto');
const {CaseLibrary}=require('./case-library.cjs');

const fail=message=>{throw new Error(message);};
const identity=stat=>({device:String(stat.dev),inode:String(stat.ino)});
const sameIdentity=(a,b)=>a?.device===b?.device&&a?.inode===b?.inode;
const catalogIds=rows=>rows.map(row=>row.id).sort();
const sameIds=(a,b)=>a.length===b.length&&a.every((id,index)=>id===b[index]);

// Only the host's native folder dialog may supply a new root. Renderer requests
// carry no paths, release pins, allowlists or engine configuration.
class CaseLibrarySelection {
  constructor({userData,defaultRoot=null,preparedRoot,nativePaths=[],externalRoot=null}) {
    this.userData=userData;this.defaultRoot=defaultRoot;this.preparedRoot=preparedRoot;this.nativePaths=nativePaths;this.externalRoot=externalRoot;
    this.record=path.join(userData,'case-library-settings.json');this.root=defaultRoot;this.blocked=null;this.selecting=false;
    this.current=this.makeLibrary(defaultRoot);this.admission=null;
  }
  makeLibrary(root){return new CaseLibrary(this.userData,root?path.join(root,'outputs/poc-cases'):this.preparedRoot,this.nativePaths,root,this.externalRoot);}
  info(){return {folder:this.root,...(this.blocked?{error:this.blocked}:{})};}
  async readRecord() {
    let stat;try{stat=await fs.lstat(this.record);}catch(error){if(error.code==='ENOENT')return null;throw error;}
    if(!stat.isFile()||stat.isSymbolicLink()||stat.size===0||stat.size>16*1024)fail('Saved case-library settings are invalid. Choose the library folder again.');
    const file=await fs.open(this.record,constants.O_RDONLY|constants.O_NOFOLLOW);
    try {
      const bytes=Buffer.alloc(16*1024+1),{bytesRead}=await file.read(bytes,0,bytes.length,0);
      if(bytesRead!==stat.size||bytesRead>16*1024)fail('Saved case-library settings changed during reading.');
      const value=JSON.parse(bytes.subarray(0,bytesRead).toString('utf8'));
      if(value?.schema!=='case-library-selection-v1'||typeof value.root!=='string'||!path.isAbsolute(value.root)||value.root.length>4096||
        !/^[0-9]+$/.test(value.directory?.device)||!/^[0-9]+$/.test(value.directory?.inode)||
        !Array.isArray(value.catalogIds)||![10,12].includes(value.catalogIds.length)||new Set(value.catalogIds).size!==value.catalogIds.length||
        !value.catalogIds.every(id=>typeof id==='string'&&/^catalog-[a-f0-9]{64}$/.test(id))||!sameIds(value.catalogIds,[...value.catalogIds].sort()))fail('Saved case-library settings are invalid. Choose the library folder again.');
      return value;
    } finally {await file.close();}
  }
  async checkedLibrary(root,expectedIdentity=null,expectedCatalogIds=null) {
    if(typeof root!=='string'||!path.isAbsolute(root))fail('Choose a local case-library folder.');
    const canonical=await fs.realpath(root),stat=await fs.stat(canonical,{bigint:true});
    if(!stat.isDirectory())fail('Choose a folder containing a source-checked case catalog.');
    const directory=identity(stat);
    if(expectedIdentity&&(canonical!==root||!sameIdentity(directory,expectedIdentity)))fail('The saved case-library folder changed. Choose the library folder again.');
    const library=this.makeLibrary(canonical),rows=await library.catalog.load();
    if(!rows.length)fail('This folder has no source-checked case catalog. Choose the folder containing the case library.');
    const ids=catalogIds(rows);
    if(expectedCatalogIds&&!sameIds(ids,expectedCatalogIds))fail('The selected case catalog changed. Choose the library folder again.');
    await library.list();
    if(await fs.realpath(root)!==canonical||!sameIdentity(identity(await fs.stat(canonical,{bigint:true})),directory))fail('The case-library folder changed during selection.');
    return {library,root:canonical,directory,catalogIds:ids};
  }
  async restore() {
    try {
      const saved=await this.readRecord();if(!saved)return this.info();
      this.root=saved.root;
      const selected=await this.checkedLibrary(saved.root,saved.directory,saved.catalogIds);
      this.current=selected.library;this.admission={directory:selected.directory,catalogIds:selected.catalogIds};this.blocked=null;
    } catch(error) {this.blocked=`Saved case library is unavailable. ${error.message}`;}
    return this.info();
  }
  async choose(pickDirectory) {
    if(this.selecting)fail('A case-library folder selection is already open.');
    this.selecting=true;
    try {
      const picked=await pickDirectory();if(picked.canceled)return null;
      if(!Array.isArray(picked.filePaths)||picked.filePaths.length!==1)fail('Choose one case-library folder.');
      const selected=await this.checkedLibrary(picked.filePaths[0]);
      await fs.mkdir(this.userData,{recursive:true});
      const temporary=path.join(this.userData,`.case-library-${randomUUID()}.tmp`);
      try {
        await fs.writeFile(temporary,JSON.stringify({schema:'case-library-selection-v1',root:selected.root,directory:selected.directory,catalogIds:selected.catalogIds},null,2)+'\n',{flag:'wx',mode:0o600});
        await fs.rename(temporary,this.record);
      } finally {await fs.rm(temporary,{force:true});}
      this.current=selected.library;this.root=selected.root;this.admission={directory:selected.directory,catalogIds:selected.catalogIds};this.blocked=null;
      return this.info();
    } finally {this.selecting=false;}
  }
  async checkSelectedAdmission() {
    if(this.blocked)fail(this.blocked);
    if(!this.admission)return;
    try {
      const checkDirectory=async()=>{
        if(await fs.realpath(this.root)!==this.root||!sameIdentity(identity(await fs.stat(this.root,{bigint:true})),this.admission.directory))
          fail('The selected case-library folder changed. Choose the library folder again.');
      };
      await checkDirectory();
      if(!sameIds(catalogIds(await this.current.catalog.load()),this.admission.catalogIds))fail('The selected case catalog changed or is missing. Choose the library folder again.');
      await checkDirectory();
    } catch(error){this.blocked=`Saved case library is unavailable. ${error.message}`;fail(this.blocked);}
  }
  async guarded(operation) {
    await this.checkSelectedAdmission();
    try {const result=await operation();await this.checkSelectedAdmission();return result;}
    catch(error){await this.checkSelectedAdmission();throw error;}
  }
  async list(){return this.guarded(()=>this.current.list());}
  async resolve(id){return this.guarded(()=>this.current.resolve(id));}
  // Native Open remains independent, including while a saved catalog is blocked.
  async remember(filename){return this.current.remember(filename);}
}
module.exports={CaseLibrarySelection};
