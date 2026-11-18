'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm');
const {inputScript,enabledControlScript,displayIdentity,pixelSummary,pixelDifference,sourceVoxel,geometry}=require('./verify-eden-planning-helpers.cjs');
function control(tagName,disabled=false){
  class Input{set value(v){this.current=v}get value(){return this.current}}
  class Select extends Input{}
  // Native SELECT owns its value descriptor, as in the browser.
  Object.defineProperty(Select.prototype,'value',Object.getOwnPropertyDescriptor(Input.prototype,'value'));
  const node=new(tagName==='SELECT'?Select:Input)();Object.assign(node,{tagName,disabled,parentElement:null,options:[{value:'tof',disabled:false}],getAttribute:()=> 'Source scan',scrollIntoView:options=>{assert.equal(options.block,'center');events.push('scroll')},dispatchEvent:event=>events.push(event.type)});const events=[];
  const context={document:{querySelectorAll:()=>[node]},HTMLInputElement:Input,HTMLSelectElement:Select,Event:class{constructor(type,options){this.type=type;assert.equal(options.bubbles,true)}}};
  return {node,events,context};
}
test('native SELECT interaction emits React change after input',()=>{const c=control('SELECT');assert.equal(vm.runInNewContext(inputScript('Source scan','tof'),c.context),'tof');assert.deepEqual(c.events,['scroll','input','change']);});
test('native control interaction rejects absent options and disabled controls',()=>{const c=control('SELECT');assert.throws(()=>vm.runInNewContext(inputScript('Source scan','swi'),c.context),/Missing enabled option/);c.node.disabled=true;assert.throws(()=>vm.runInNewContext(inputScript('Source scan','tof'),c.context),/Missing enabled control/);assert.deepEqual(c.events,[]);});
test('enabled control gate requires the real control and rejects disabled or missing selectors',()=>{const c=control('SELECT');assert.equal(vm.runInNewContext(enabledControlScript('Source scan'),c.context),true);c.node.disabled=true;assert.equal(vm.runInNewContext(enabledControlScript('Source scan'),c.context),false);assert.equal(vm.runInNewContext(enabledControlScript('Missing control'),c.context),false);});
test('native INPUT interaction uses native setter and both events',()=>{const c=control('INPUT');vm.runInNewContext(inputScript('Source scan',6.5),c.context);assert.equal(c.node.value,'6.5');assert.deepEqual(c.events,['scroll','input','change']);});
test('source identity requires snapshot, source hash, series and render generation',()=>{assert.equal(displayIdentity(JSON.stringify(['collection','tof','source-sha','snapshot',false,2])).seriesId,'tof');for(const value of ['[]','null',JSON.stringify(['collection','tof','','snapshot',false,2])])assert.throws(()=>displayIdentity(value));});
test('pixel checks reject constant and unchanged signal while detecting source changes',()=>{const bitmap=Buffer.from([35,35,35,255,130,130,130,255,230,230,230,255,0,0,0,255]),a=pixelSummary(bitmap),b=pixelSummary(Buffer.from([65,65,65,255,90,90,90,255,150,150,150,255,0,0,0,255]));assert.equal(a.neutralPixels,3);assert.equal(a.dynamicRange,195);assert.equal(pixelDifference(a,a).changedFraction,0);assert.equal(pixelDifference(a,b).changedFraction,1);assert.equal(pixelSummary(Buffer.alloc(40)).dynamicRange,0);});
test('persistence comparison includes geometry, patch and actual measurements',()=>{const a={id:'a',entry:[1,2,3],entryPatch:{radiusMm:3},measurements:{lengthMm:2},parameterHash:'h',measurementHash:'m',assessmentStatus:'current'};assert.notDeepEqual(geometry(a),geometry({...a,entryPatch:{radiusMm:1}}));assert.notDeepEqual(geometry(a),geometry({...a,measurements:{lengthMm:3}}));});

test('common source cursor maps into the actual registered display grid',()=>{const affine=[[0,-2,0,8],[3,0,0,-5],[0,0,4,7],[0,0,0,1]];assert.deepEqual(sourceVoxel(affine,[0,1,19]),[2,4,3]);assert.throws(()=>sourceVoxel([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]],[0,0,0]),/Singular/);});
