'use strict';
const assert=require('node:assert/strict');
const {inputScript}=require('./verify-eden-planning-helpers.cjs');

async function setFieldView(js,choice){
  if(await js('!!document.querySelector(\'[aria-label="Operative field view"]\')')){
    await js(inputScript('Operative field view',choice));
    return;
  }
  // The same physical view can also be captured on the exterior-first build.
  await js(`(()=>{const toggle=[...document.querySelectorAll('.operative-options label')].find(l=>l.textContent.trim()==='Inspection cutaway').querySelector('input');
    if(toggle.checked!==${choice==='section'})toggle.click()})()`);
}

async function verifyOperativeCorridor({win,js,until,capture,match}){
  const current=()=>js(`(()=>{
    const panel=document.querySelector('.operative-playback'),canvas=panel.querySelector('.operative-field-canvas canvas');
    const legend=panel.querySelector('.operative-legend [data-tissue="vessels"]');
    return {vesselLegend:legend?{label:legend.textContent,swatch:getComputedStyle(legend,'::before').backgroundColor}:null,phase:panel.dataset.operativePhase,progress:panel.dataset.operativeProgress,
      removed:Number(panel.querySelector('[data-removed-cells]')?.dataset.removedCells??0),
      camera:JSON.parse(canvas.dataset.camera),field:canvas.getBoundingClientRect().toJSON(),
      sectionSource:canvas.parentElement.dataset.sectionSource??null,view:panel.dataset.fieldView??null,
      surrounding:JSON.parse(canvas.parentElement.dataset.surroundingSources)};
  })()`);
  const end=Number(await js('document.querySelector(\'[aria-label="Operative timeline"]\').max'));
  const views={};
  const seek=async value=>{
    await js(inputScript('Operative timeline',value));
    await until(`Number(document.querySelector('[aria-label="Operative timeline"]').value)===${value}`);
  };
  const snap=async name=>{await capture(name);views[name]=await current();};
  const key=async key=>js(`document.querySelector('.operative-field-canvas canvas').dispatchEvent(new KeyboardEvent('keydown',{key:${JSON.stringify(key)}}))`);
  for(const [width,height] of [[1440,900],[1050,720]]){
    win.setContentSize(width,height);await until(`innerWidth===${width}&&innerHeight===${height}`);
    if(match){
      const rect=match['section-'+width].field;
      // Chrome can change the field height between versions. Standardize the
      // actual WebGL canvas before capture; the source image is never rescaled.
      await js(`(()=>{const canvas=document.querySelector('.operative-field-canvas');
        canvas.style.width=${JSON.stringify(rect.width+'px')};canvas.style.height=${JSON.stringify(rect.height+'px')};
      })()`);
      await until(`Math.abs(document.querySelector('.operative-field-canvas canvas').getBoundingClientRect().height-${rect.height})<.01`);
    }
    await setFieldView(js,'follow');await seek(0);await snap('intact-'+width);
    await seek(2);await snap('exposure-'+width);
    await seek(end-.45);await snap('aspiration-'+width);
    await seek(end);await snap('endpoint-'+width);
    const endpoint=views['endpoint-'+width];
    assert(endpoint.removed>0);
    await key('ArrowLeft');await key('ArrowLeft');await key('ArrowUp');await snap('endpoint-orbit-'+width);
    assert.notDeepEqual(views['endpoint-orbit-'+width].camera,endpoint.camera);
    await key('Home');
    await setFieldView(js,'exterior');await key('Home');await snap('exterior-'+width);
    await setFieldView(js,'section');await key('Home');await snap('section-'+width);
    assert.equal(views['section-'+width].removed,endpoint.removed);
    assert.equal(views['exterior-'+width].removed,endpoint.removed);
    await setFieldView(js,'follow');
    if(endpoint.view==='section'){
      assert(views['section-'+width].sectionSource);
      assert.equal(views['exterior-'+width].view,'exterior');
    }
  }
  await seek(0);await key('Home');
  return views;
}
module.exports={setFieldView,verifyOperativeCorridor};
