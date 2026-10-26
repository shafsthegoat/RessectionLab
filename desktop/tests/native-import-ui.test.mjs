import assert from 'node:assert/strict';
import {test,before,after} from 'node:test';
import * as React from 'react';
import {createServer} from 'vite';
import {mkdtemp,mkdir,rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
let server,cache,NativeImportPanel;
before(async()=>{
  const output=fileURLToPath(new URL('../../testcases/outputs/integrated-goal-20261026/',import.meta.url));
  await mkdir(output,{recursive:true});cache=await mkdtemp(path.join(output,'import-ui-vite-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({NativeImportPanel}=await server.ssrLoadModule('/src/NativeImportPanel.tsx'));
});
after(async()=>{await server?.close();if(cache)await rm(cache,{recursive:true,force:true});});
function* walk(node){if(Array.isArray(node)){for(const child of node)yield*walk(child);}else if(node&&typeof node==='object'&&node.props){yield node;yield*walk(node.props.children);}}
const text=node=>Array.isArray(node)?node.map(text).join(''):node?.props?text(node.props.children):typeof node==='string'||typeof node==='number'?String(node):'';
const find=(tree,type,label)=>[...walk(tree)].find(node=>node.type===type&&(node.props['aria-label']===label||text(node)===label));
function render({sequence='FLAIR',roles=[{value:'1',role:''}],busy=false,draft=null}={}){
  const calls=[],errors=[],internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,previous=internals.H;
  let index=0;const values=[sequence,roles,''];
  internals.H={useState:()=>{const i=index++;return[values[i],value=>{if(i===2)errors.push(value);}];}};
  const workspace={binding:{collectionId:'source',document:{patient_id:'patient'}},api:{anatomyImport:async args=>{calls.push(args);return null;}},args:()=>({collectionId:'source',snapshotId:'snapshot'}),busy,draft,series:'flair',change:async job=>job()};
  try{return{tree:NativeImportPanel({workspace,timepoint:'preoperative',sourceLabel:'FLAIR'}),calls,errors};}finally{internals.H=previous;}
}
test('desktop propagates the acquired sequence for DICOM and NIfTI including T2 and FLAIR',async()=>{
  for(const sequence of ['T1','T1CE','T2','FLAIR','CT','MRA','CTA','TOF']){
    const {tree,calls}=render({sequence});
    await find(tree,'button','Import DICOM series…').props.onClick();
    await find(tree,'button','Import NIfTI…').props.onClick();
    assert.deepEqual(calls.map(c=>[c.kind,c.modality]),[['dicom',sequence],['nifti',sequence]]);
    assert.equal(calls[0].snapshotId,'snapshot');
  }
  assert.equal([...walk(render().tree)].filter(n=>n.type==='option'&&['MR','MRI'].includes(text(n))).length,0);
});
test('segmentation import requires explicit roles and sends every numeric source label',async()=>{
  const {tree,calls}=render({roles:[{value:'1',role:'context'},{value:'2',role:'context'},{value:'4',role:'tumor'}]});
  await find(tree,'button','Import segmentation…').props.onClick();
  assert.deepEqual(calls[0],{collectionId:'source',snapshotId:'snapshot',kind:'labels',seriesId:'flair',label:'Imported segmentation',roleMap:{1:'context',2:'context',4:'tumor'}});
  for(const roles of [[{value:'1',role:''}],[{value:'1',role:'tumor'},{value:'01',role:'brain'}],[{value:'0',role:'brain'}],[]]){
    const invalid=render({roles});await find(invalid.tree,'button','Import segmentation…').props.onClick();
    assert.equal(invalid.calls.length,0);assert.ok(invalid.errors.at(-1));
  }
});
test('an active edit or operation disables scan and segmentation import controls',()=>{
  for(const state of [{busy:true},{draft:{layerId:'draft'}}]){
    const {tree}=render(state);
    assert.equal(find(tree,'button','Import DICOM series…').props.disabled,true);
    assert.equal([...walk(tree)].find(n=>n.type==='fieldset').props.disabled,true);
  }
});
