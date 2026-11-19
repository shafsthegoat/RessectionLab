'use strict';
const assert = require('node:assert/strict');
const {inputScript}=require('./verify-eden-planning-helpers.cjs');
const {studyScanSelectionScript}=require('./verify-study-ui-helpers.cjs');

// Compare settled native patient presentations at the same UI dimensions.
// Each patient's own reconstruction determines its fit and available layers.
async function compareAnatomy({win,js,until,ready,click,capture,report,document}){
  const reference=document.reference_series;
  win.setContentSize(1440,900);
  await until('innerWidth===1440 && innerHeight===900');
  if(await js("document.querySelector('select[aria-label=\"Source scan\"]')?.value")!==reference){
    await js(studyScanSelectionScript(reference));
    await ready();
  }
  const setSelect=(label,value)=>js(`(()=>{
    const e=document.querySelector(${JSON.stringify('select[aria-label="'+label+'"]')});
    if(!e||e.disabled)throw Error('Unavailable selector');
    e.value=${JSON.stringify(value)};
    e.dispatchEvent(new Event('change',{bubbles:true}));
  })()`);
  const camera=async preset=>{
    await click('Layout');
    await setSelect('Camera preset',preset);
    await click('Close view controls');
    await ready();
  };
  const compareFrame=async name=>{
    await ready();
    const view=await js(`(()=>{
      const pane=document.querySelector('.rl-viewer-anatomy-image'),v=document.querySelector('.rl-viewer');
      const rect=elem=>{const r=elem.getBoundingClientRect();return {x:r.x,y:r.y,width:r.width,height:r.height}};
      return {frame:window.operativeVerificationFrame,viewport:rect(pane),slices:Array.from(document.querySelectorAll('.rl-viewer-slice-image'),rect),
        layers:JSON.parse(v.dataset.visibleLayerIds),identity:JSON.parse(v.dataset.sourceDisplayIdentity),cameraFit:JSON.parse(pane.dataset.cameraFit||'null'),
        preset:document.querySelector('.anatomy-presets button[aria-pressed="true"]')?.textContent,
        surfaces:document.querySelector('select[aria-label="Native surface representation"]')?.value,
        volume:document.querySelector('button[aria-label="Show structures only"]')?.getAttribute('aria-pressed')==='false',
        alignment:document.querySelector('.patient-model-summary')?.innerText??null};
    })()`);
    assert.equal(view.identity[1],reference);
    assert.equal(view.identity[3],document.snapshot_id);
    assert(view.layers.length && view.frame?.cameraState && view.frame.sourceSignalFingerprint);
    assert.equal(view.surfaces,'conditioned');
    report.comparison[name]=view;
    await capture(name);
  };
  await js("document.querySelector('.anatomy-presets button:nth-child(2)').click()");
  await ready();
  await setSelect('Native surface representation','conditioned');
  await click('Show structures only');
  await ready();
  report.comparison={};
  await camera('left');
  await compareFrame('lateral-structures');
  await camera('anterior');
  await compareFrame('anterior-structures');
  const publisher=document.layers.find(l=>l.source_series===reference && l.id==='publisher_cow_arterial_mesh');
  const arterialMask=document.layers.find(l=>l.source_series===reference && l.id==='supplied_arterial_annotation_union');
  if(publisher && arterialMask){
    const visibility=async (layer,visible)=>{
      const row='[data-structure-id="'+layer.id+'"] .native-structure-visibility';
      await js(`(()=>{const e=document.querySelector(${JSON.stringify(row)});if(!e || e.disabled)throw Error('Unavailable structure visibility');if((e.getAttribute('aria-pressed')==='true')!==${visible})e.click()})()`);
      await ready();
    };
    const initial=report.comparison['anterior-structures'].layers;
    await visibility(arterialMask,true);
    await visibility(publisher,false);
    await compareFrame('anterior-arterial-mask');
    const variant=report.comparison['anterior-arterial-mask'];
    assert(variant.layers.includes(arterialMask.id) && !variant.layers.includes(publisher.id));
    const previous=report.comparison['anterior-structures'].frame.cameraState;
    for(const key of ['position','target','up']){
      assert(variant.frame.cameraState[key].every((v,i)=>Math.abs(v-previous[key][i])<1e-6));
    }
    await visibility(publisher,initial.includes(publisher.id));
    await visibility(arterialMask,initial.includes(arterialMask.id));
  }
  const skull=document.layers.find(l=>['bone','skull'].includes(l.kind));
  if(skull){
    await js(`document.querySelector(${JSON.stringify('[data-structure-id="'+skull.id+'"] .native-structure-name')}).click()`);
    await ready();
    await click('Isolate');
    await js(inputScript(skull.label+' opacity',1));
    await click('Close layer details');
    await compareFrame('anterior-skull');
    assert.deepEqual(report.comparison['anterior-skull'].layers,[skull.id]);
  }
}
module.exports={compareAnatomy};
