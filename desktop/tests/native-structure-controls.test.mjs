import assert from 'node:assert/strict';
import {test,before,after} from 'node:test';
import * as React from 'react';
import {createServer} from 'vite';
import {mkdtemp,mkdir,rm} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
let server,cache,NativeStructureControls;
before(async()=>{
  const output=fileURLToPath(new URL('../../testcases/outputs/desktop-verification/',import.meta.url));
  await mkdir(output,{recursive:true});cache=await mkdtemp(path.join(output,'structure-controls-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',optimizeDeps:{noDiscovery:true,include:[]},server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  ({NativeStructureControls}=await server.ssrLoadModule('/src/NativeStructureControls.tsx'));
});
after(async()=>{await server?.close();if(cache)await rm(cache,{recursive:true,force:true});});
function* walk(node){if(Array.isArray(node)){for(const child of node)yield*walk(child);}else if(node&&typeof node==='object'&&node.props){yield node;yield*walk(node.props.children);}}
const text=node=>Array.isArray(node)?node.map(text).join(''):node?.props?text(node.props.children):typeof node==='string'||typeof node==='number'?String(node):'';
const layers=[{id:'tumor',label:'Supplied lesion',kind:'tumor',color:'#ffffff'},{id:'vessel',label:'Vessel estimate',kind:'vessels',color:'#dd0000'},{id:'region',label:'Named brain region',kind:'brain-region',color:'#cccccc'}];
function render(extra={}){
  const visibility=[],selection=[];
  const internals=React.__CLIENT_INTERNALS_DO_NOT_USE_OR_WARN_USERS_THEY_CANNOT_UPGRADE,previous=internals.H;
  internals.H={useState:initial=>[initial,()=>{}],useRef:initial=>({current:initial}),useEffect:effect=>effect()};
  try{
    const tree=NativeStructureControls({volume:{compartments:[]},layers,visibleLayers:{tumor:false,vessel:true,region:false,__fragment_review:true},onVisibilityChange:value=>visibility.push(value),settings:{},onSettingsChange:()=>{},selectedStructure:null,onSelectStructure:value=>selection.push(value),onFocusStructure:()=>{},showSelectionControls:false,...extra});
    return {tree,visibility,selection,button:label=>[...walk(tree)].find(node=>node.type==='button'&&(node.props['aria-label']===label||text(node)===label))};
  }finally{internals.H=previous;}
}
test('the study inventory exposes a hidden lazy label before its display data is loaded',()=>{
  const view=render();
  assert.ok(view.button('Supplied lesion'));
  assert.ok(view.button('Named brain region'));
  assert.equal(view.button('Focus Supplied lesion').props.disabled,true);
  view.button('Supplied lesion').props.onClick();
  assert.deepEqual(view.selection,['tumor']);
  assert.deepEqual(view.visibility,[{tumor:true,vessel:true,region:false,__fragment_review:true}]);
});
test('group display changes retain unrelated anatomy and temporary review visibility',()=>{
  const view=render();let prevented=false;
  view.button('Show all lesion candidates').props.onClick({preventDefault(){prevented=true;}});
  assert.equal(prevented,true);
  assert.deepEqual(view.visibility.at(-1),{tumor:true,vessel:true,region:false,__fragment_review:true});
  view.button('Hide all vascular layers').props.onClick({preventDefault(){}});
  assert.deepEqual(view.visibility.at(-1),{tumor:false,vessel:false,region:false,__fragment_review:true});
});
test('alignment and transfer failures remain actionable without presenting a focusable ready layer',()=>{
  const alignment=render({unavailable:['vessel']});
  assert.match(text(alignment.tree),/Alignment required/);
  assert.equal(alignment.button('Focus Vessel estimate').props.disabled,true);
  const retries=[],failure=render({layerErrors:{vessel:'Source transfer interrupted'},onRetryLayer:id=>retries.push(id)});
  failure.button('Retry').props.onClick();assert.deepEqual(retries,['vessel']);
  const blocked=render({disabled:true});
  for(const button of [...walk(blocked.tree)].filter(node=>node.type==='button'))assert.equal(button.props.disabled,true);
});
