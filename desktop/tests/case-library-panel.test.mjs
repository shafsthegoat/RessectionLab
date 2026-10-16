import assert from 'node:assert/strict';
import {before,after,test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
let server,cache,filter,Panel;
before(async()=>{
  cache=await fs.mkdtemp(path.join(os.tmpdir(),'catalog-panel-'));
  server=await createServer({root:fileURLToPath(new URL('../',import.meta.url)),cacheDir:cache,logLevel:'error',server:{middlewareMode:true,hmr:false,ws:false},appType:'custom'});
  const module=await server.ssrLoadModule('/src/WorkbenchPanels.tsx');filter=module.filterLibraryCases;Panel=module.CaseLibraryPanel;
});
after(async()=>{await server?.close();if(cache)await fs.rm(cache,{recursive:true,force:true});});
const rows=[
  {id:'opaque-btc',title:'BTC-ds001226-sub-PAT05-preop',recent:false,bytes:9392262,qualifiedCatalog:true,source:'BTC',anatomy:['brain','tumor'],missingAnatomy:['vessels'],receipt:{unrelatedEvidence:'IXI265 Lausanne vessel source cohort'}},
  {id:'opaque-ixi',title:'IXI265-Guys-0845',recent:false,bytes:75491573,qualifiedCatalog:true,source:'IXI',anatomy:['brain','vessels'],missingAnatomy:['tumor']},
  {id:'opaque-native',title:'Saved study café',recent:true,bytes:1234,native:true,timepoint:'intraoperative',scanCount:3},
];
test('search matches each case name, own source and present anatomy, never unrelated receipt evidence or missing layers',()=>{
  assert.deepEqual(filter(rows,'   IXI  vessels ').map(r=>r.id),['opaque-ixi']);
  assert.deepEqual(filter(rows,'pat05').map(r=>r.id),['opaque-btc']);
  assert.deepEqual(filter(rows,'BTC tumor').map(r=>r.id),['opaque-btc']);
  assert.deepEqual(filter(rows,'vessel').map(r=>r.id),['opaque-ixi']);
  assert.deepEqual(filter(rows,'Lausanne'),[]);assert.deepEqual(filter(rows,'cafe').map(r=>r.id),['opaque-native']);
  assert.equal(filter(rows,'').length,3);assert.deepEqual(filter(rows,'BTC vessels'),[]);
});
test('Cases displays source-check and unreviewed-anatomy wording while retaining native browse and recent affordances',()=>{
  const html=renderToStaticMarkup(React.createElement(Panel,{cases:rows,onOpen(){},onBrowse(){},busy:false}));
  assert.match(html,/Search cases by name, source, anatomy or reported site/);assert.match(html,/Open a saved case/);
  assert.match(html,/Source checked · anatomy unreviewed/);assert.match(html,/Brain \+ Vessels/);assert.match(html,/Recently opened/);assert.match(html,/3 scans/);
  assert.match(html,/independence across datasets remain unverified/);assert.doesNotMatch(html,/validated patients|independent people|accepted anatomy|unrelatedEvidence/);
});
test('open controls remain disabled while busy and an empty library retains browse guidance',()=>{
  const busy=renderToStaticMarkup(React.createElement(Panel,{cases:rows,onOpen(){},onBrowse(){},busy:true}));
  assert.equal((busy.match(/<button[^>]*disabled=""/g)??[]).length,4);
  const empty=renderToStaticMarkup(React.createElement(Panel,{cases:[],onOpen(){},onBrowse(){},busy:false}));assert.match(empty,/Open or import imaging to start a case/);
});
const sourceRows=[...rows,...[['05','Left Frontal'],['16','Right Fronto-temporal'],['20','Right Parietal'],['22','Right Parietal'],['25','Right Temporal'],['28','Left Frontal']].map(([person,label])=>({
  id:`source-btc-${person}`,title:`BTC-ds001226-sub-PAT${person}-preop`,recent:false,bytes:1,qualifiedCatalog:true,source:'BTC',anatomy:['brain','tumor'],
  sourceLocation:{label,status:'source-reported-unreviewed'},receipt:{unrelatedEvidence:'occipital cerebellum'},
}))];
test('source-reported region and side search only their own reviewed-pending catalog rows',()=>{
  assert.deepEqual(filter(sourceRows,'left frontal').map(row=>row.id),['source-btc-05','source-btc-28']);
  assert.deepEqual(filter(sourceRows,'right parietal').map(row=>row.id),['source-btc-20','source-btc-22']);
  assert.deepEqual(filter(sourceRows,'temporal').map(row=>row.id),['source-btc-16','source-btc-25']);
  assert.deepEqual(filter(sourceRows,'BTC fronto-temporal').map(row=>row.id),['source-btc-16']);
  assert.deepEqual(filter(sourceRows,'IXI frontal'),[]);assert.deepEqual(filter(sourceRows,'occipital'),[]);
  assert.deepEqual(filter([rows[0]],'left frontal'),[]);
  const unrelated={...rows[2],sourceLocation:{label:'Unbound location',status:'source-reported-unreviewed'}};
  assert.deepEqual(filter([unrelated],'unbound'),[]);
});
test('reported sites have explicit source/review context and are not guessed for matching names or accepted anatomy',()=>{
  const html=renderToStaticMarkup(React.createElement(Panel,{cases:sourceRows,onOpen(){},onBrowse(){},busy:false}));
  assert.equal((html.match(/Reported site:/g)??[]).length,6);
  assert.equal((html.match(/Dataset \/ paper report · review pending/g)??[]).length,6);
  assert.match(html,/Reported site: Left Frontal/);assert.match(html,/Reported site: Right Fronto-temporal/);
  const unrelated=renderToStaticMarkup(React.createElement(Panel,{cases:rows,onOpen(){},onBrowse(){},busy:false}));
  assert.doesNotMatch(unrelated,/Reported site:/);assert.doesNotMatch(html,/accepted location|reviewed location|occipital|cerebellum/);
});
test('packaged Cases offers a native library chooser without a renderer path input and retains native Open',()=>{
  const html=renderToStaticMarkup(React.createElement(Panel,{cases:[],onOpen(){},onBrowse(){},onChooseLibrary(){},libraryFolder:{folder:null},busy:false}));
  assert.match(html,/Choose case library/);assert.match(html,/No library folder selected/);assert.match(html,/Open a saved case/);
  assert.match(html,/folder containing your source-checked case catalog/);
  assert.equal((html.match(/<input/g)??[]).length,1);assert.match(html,/<input type="search"/);
});
test('selected folder and recoverable catalog errors are visible while case actions obey the busy state',()=>{
  const html=renderToStaticMarkup(React.createElement(Panel,{cases:rows,onOpen(){},onBrowse(){},onChooseLibrary(){},libraryFolder:{folder:'/Local studies/Research library'},libraryError:'Saved case library is unavailable. Choose its folder again.',busy:true}));
  assert.match(html,/\/Local studies\/Research library/);assert.match(html,/role="alert"/);assert.match(html,/Saved case library is unavailable/);
  assert.equal((html.match(/<button[^>]*disabled=""/g)??[]).length,5);
  const available=renderToStaticMarkup(React.createElement(Panel,{cases:[],onOpen(){},onBrowse(){},onChooseLibrary(){},libraryFolder:{folder:'/Moved folder',error:'Choose the library folder again.'},busy:false}));
  assert.match(available,/role="alert"/);assert.doesNotMatch(available,/<button[^>]*disabled=""/);
});
