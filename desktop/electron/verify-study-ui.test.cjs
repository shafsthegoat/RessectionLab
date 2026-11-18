'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm'),fs=require('node:fs'),path=require('node:path');
const {studyScanEnabledScript,studyScanSelectionScript,studyLayerVisibilityScript,studyVisibilityToggleScript,studyLayerSelectionScript}=require('./verify-study-ui-helpers.cjs');
const run=(script,document,extra={})=>vm.runInNewContext(script,{document,...extra});
function button(extra={}){const calls=[];return {disabled:false,parentElement:{tagName:'DETAILS',open:false},calls,scrollIntoView:()=>calls.push('scroll'),click:()=>calls.push('click'),...extra};}
function documentFor({scans=[],select=null,layers=[],checkboxes=[]}={}){return {querySelector:selector=>selector==='select[aria-label="Source scan"]'?select:null,querySelectorAll:selector=>selector.startsWith('.study-scan-row')?scans:selector.startsWith('input[type=')?checkboxes:selector.startsWith('.study-inspector')?layers:[]};}
function layer(id,label,visible=true){const visibility=button({getAttribute:name=>name==='aria-pressed'?String(visible):null}),name=button();return {dataset:{structureId:id,structureLabel:label},visibility,name,querySelector:selector=>selector==='.native-structure-visibility'?visibility:selector==='.native-structure-name'?name:null};}
test('Scans row interactions use the source ID and refuse missing or disabled scans',()=>{
  const row=button({dataset:{scanId:'source " ] 1'}}),document=documentFor({scans:[row]});
  assert.equal(run(studyScanEnabledScript(row.dataset.scanId),document),true);
  assert.equal(run(studyScanSelectionScript(row.dataset.scanId),document),row.dataset.scanId);
  assert.deepEqual(row.calls,['scroll','click']);assert.equal(row.parentElement.open,true);
  row.disabled=true;assert.equal(run(studyScanEnabledScript(row.dataset.scanId),document),false);
  assert.throws(()=>run(studyScanSelectionScript(row.dataset.scanId),document),/Missing enabled study scan/);
  assert.throws(()=>run(studyScanSelectionScript('absent'),document),/Missing enabled study scan/);
});
test('Anatomy source selector fallback uses native setter and both React events',()=>{
  class Select{set value(value){this.current=value}get value(){return this.current}}
  const events=[],select=Object.assign(new Select(),button(),{options:[{value:'tof',disabled:false}],dispatchEvent:event=>events.push(event.type)}),document=documentFor({select});
  const extra={HTMLSelectElement:Select,Event:class{constructor(type,options){this.type=type;assert.equal(options.bubbles,true)}}};
  assert.equal(run(studyScanSelectionScript('tof'),document,extra),'tof');assert.deepEqual(events,['input','change']);
  assert.throws(()=>run(studyScanSelectionScript('absent'),document,extra),/Missing enabled source option/);
});
test('the inventory retains true source IDs and hidden anatomy without relying on toggle wording',()=>{
  const document=documentFor({layers:[layer('lesion','Lesion candidate',true),layer('region','Named region',false)]});
  const result=JSON.parse(JSON.stringify(run(studyLayerVisibilityScript(),document)));
  assert.deepEqual(result,[{id:'lesion',label:'Show Lesion candidate',checked:true},{id:'region',label:'Show Named region',checked:false}]);
});
test('visibility and named selection open their disclosure and use actual enabled buttons',()=>{
  const row=layer('lesion','Lesion candidate'),document=documentFor({layers:[row]});
  run(studyVisibilityToggleScript('Show Lesion candidate'),document);assert.deepEqual(row.visibility.calls,['scroll','click']);assert(row.visibility.parentElement.open);
  assert.equal(run(studyLayerSelectionScript(['Lesion candidate']),document),'Lesion candidate');assert.deepEqual(row.name.calls,['scroll','click']);assert(row.name.parentElement.open);
  row.visibility.disabled=true;assert.throws(()=>run(studyVisibilityToggleScript('Show Lesion candidate'),document),/Missing enabled visibility/);
  row.name.disabled=true;assert.throws(()=>run(studyLayerSelectionScript(['Lesion candidate']),document),/Missing enabled study layer/);
});
test('fragment-review checkboxes remain separate from anatomy visibility buttons',()=>{
  const checkbox=button({getAttribute:()=> 'Select fragment 4'}),document=documentFor({checkboxes:[checkbox]});
  run(studyVisibilityToggleScript('Select fragment 4'),document);assert.deepEqual(checkbox.calls,['scroll','click']);
});
test('maintained native verifiers cannot show or focus their verification window',()=>{
  for(const filename of ['verify-visual-fidelity.cjs','verify-unified-workspace.cjs','verify-eden-planning.cjs','verify-eden-comparison.cjs','verify-adaptive-patient.cjs','verify-data-rebuild.cjs']){
    const source=fs.readFileSync(path.join(__dirname,filename),'utf8');
    assert.doesNotMatch(source,/\b(?:window|win)\.(?:show|focus)\s*\(/,filename);
    assert.match(source,/--automation-hidden/,filename);
    assert.match(source,/capturePage\(undefined,\{stayHidden:true,stayAwake:true\}\)/,filename);
  }
});
