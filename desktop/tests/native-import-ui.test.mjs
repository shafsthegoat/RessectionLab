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
const scan=(kind,modality,extra={})=>({kind,modality,label:modality,selectionId:'pick-'+modality,source:'',roles:[],...extra});
const annotation=(roles,source='pick-T1CE')=>({kind:'labels',label:'Segmentation',selectionId:'pick-seg',source,roles});
function render({rows=[],confirmed=true,busy=false,draft=null,pending=false,sourcePatientId=''}={}){
  const calls=[],errors=[],messages=[],queueChanges=[],internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,previous=internals.H;
  let index=0;const values=[rows,confirmed,'',sourcePatientId];
  internals.H={useState:()=>{const i=index++;return[values[i],value=>{if(i===0)queueChanges.push(value);if(i===2)errors.push(value);}];},useEffect:()=>{}};
  const workspace={
    binding:{collectionId:'source',document:{snapshot_id:'snapshot',patient_id:'patient',series:[{id:'flair',label:'FLAIR',modality:'FLAIR'}]}},
    args:()=>({collectionId:'source',snapshotId:'snapshot'}),
    api:{
      anatomyPickIntake:async args=>{calls.push(args);return null;},
      anatomyImportBatch:async args=>{calls.push(args);return {document:{snapshot_id:'snapshot'}};},
    },
    busy,draft,series:'flair',planning:{pending},setMessage:message=>messages.push(message),
    run:async job=>job(),change:async job=>job(),
  };
  try{return{tree:NativeImportPanel({workspace,timepoint:'preoperative'}),calls,errors,messages,queueChanges};}finally{internals.H=previous;}
}
test('desktop discovers imaging series and commits reviewed roles in one request',async()=>{
  const {tree,calls,messages,queueChanges}=render({rows:[scan('dicom','CTA'),scan('nifti','FLAIR'),scan('nifti','DWI',{storage:'source-asset',modality:''})]});
  await find(tree,'button','Add DICOM folders…').props.onClick();
  await find(tree,'button','Add NIfTI files…').props.onClick();
  await find(tree,'button','Import selected data').props.onClick();
  assert.deepEqual(calls,[
    {collectionId:'source',snapshotId:'snapshot',kind:'dicom'},
    {collectionId:'source',snapshotId:'snapshot',kind:'nifti'},
    {collectionId:'source',snapshotId:'snapshot',patientId:'patient',timepoint:'preoperative',confirmedAssociation:true,items:[
      {selectionId:'pick-CTA',kind:'dicom',modality:'CTA'},
      {selectionId:'pick-FLAIR',kind:'nifti',modality:'FLAIR'},
      {selectionId:'pick-DWI',kind:'nifti'},
    ]},
  ]);
  assert.deepEqual(queueChanges.at(-1),[]);
  assert.match(messages.at(-1),/already in this study/);
  assert.equal([...walk(tree)].filter(n=>n.type==='option'&&['MR','MRI'].includes(text(n))).length,0);
});
test('segmentation import binds explicit label meanings to the reviewed source',async()=>{
  const roles=[{value:'1',role:'context'},{value:'2',role:'context'},{value:'4',role:'tumor',name:'Enhancing lesion'}];
  const {tree,calls}=render({rows:[annotation(roles),scan('nifti','T1CE')]});
  await find(tree,'button','Import selected data').props.onClick();
  assert.deepEqual(calls[0].items,[
    {selectionId:'pick-seg',kind:'labels',sourceInputIndex:1,label:'Segmentation',roleMap:{1:'context',2:'context',4:{kind:'tumor',label:'Enhancing lesion'}}},
    {selectionId:'pick-T1CE',kind:'nifti',modality:'T1CE'},
  ]);
  for(const roles of [[{value:'1',role:''}],[{value:'1',role:'tumor'},{value:'01',role:'brain'}],[{value:'0',role:'brain'}],[]]){
    const invalid=render({rows:[annotation(roles,'flair')]});
    await find(invalid.tree,'button','Import selected data').props.onClick();
    assert.equal(invalid.calls.length,0);assert.ok(invalid.errors.at(-1));
  }
});
test('study confirmation and resolved source roles are required before import',()=>{
  for(const state of [
    {rows:[scan('nifti','T1CE')],confirmed:false},
    {rows:[scan('nifti','Unknown',{modality:''})]},
    {rows:[annotation([{value:'1',role:'tumor'}],'')]},
  ]){
    assert.equal(find(render(state).tree,'button','Import selected data').props.disabled,true);
  }
});
test('an active edit or operation disables intake controls',()=>{
  for(const state of [{busy:true},{draft:{layerId:'draft'}},{pending:true}]){
    const {tree}=render({...state,rows:[scan('nifti','T1CE')]});
    assert.equal(find(tree,'button','Add DICOM folders…').props.disabled,true);
    assert.equal([...walk(tree)].find(n=>n.type==='fieldset').props.disabled,true);
  }
});

test('metadata aliases must be typed and mixed patient IDs remain blocked',async()=>{
  const identified=scan('dicom','CTA',{patientId:'clinic-id'});
  const unresolved=render({rows:[identified]});
  assert.equal(find(unresolved.tree,'input','Source patient ID').props.value,'');
  assert.equal(find(unresolved.tree,'button','Import selected data').props.disabled,true);

  const verified=render({rows:[identified,scan('nifti','T1')],sourcePatientId:'clinic-id'});
  await find(verified.tree,'button','Import selected data').props.onClick();
  assert.equal(verified.calls[0].sourcePatientId,'clinic-id');
  assert.equal(verified.calls[0].confirmedAssociation,true);

  const mixed=render({rows:[identified,scan('nifti','T1',{patientId:'other-id'})],sourcePatientId:'clinic-id'});
  assert.equal(find(mixed.tree,'button','Import selected data').props.disabled,true);
});
