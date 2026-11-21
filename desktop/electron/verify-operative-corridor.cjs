'use strict';
const assert=require('node:assert/strict');
const {inputScript}=require('./verify-eden-planning-helpers.cjs');

async function setFieldView(js,choice){
  if(await js('!!document.querySelector(\'[aria-label="Operative field view"]\')')){
    const hasMicroscope=await js(`!!document.querySelector('[aria-label="Operative field view"] option[value="operative"]')`);
    const value=choice==='follow'&&hasMicroscope?'operative':choice==='operative'&&!hasMicroscope?'follow':choice;
    await js(inputScript('Operative field view',value));
    return;
  }
  // The same physical view can also be captured on the exterior-first build.
  await js(`(()=>{const toggle=[...document.querySelectorAll('.operative-options label')].find(l=>l.textContent.trim()==='Inspection cutaway').querySelector('input');
    if(toggle.checked!==${choice==='section'})toggle.click()})()`);
}

async function verifyOperativeCorridor({win,js,until,capture,match,cameras,moviePoses,acceptance,ablations,expectEntry,expectOuterHead,acquiredReview}){
  const current=()=>js(`(()=>{
    const panel=document.querySelector('.operative-playback'),canvas=panel.querySelector('.operative-field-canvas canvas');
    const legend=panel.querySelector('.operative-legend [data-tissue="vessels"]');
    const marker=panel.querySelector('.operative-entry-marker'),box=marker?.getBoundingClientRect();
    return {entryCaption:canvas.parentElement.dataset.entryCaption??null,entry:marker&&!marker.hidden&&getComputedStyle(marker).display!=='none'?{text:marker.textContent,rect:box.toJSON()}:null,submitted:window.__operativeQA?.lastDraw??null,meshes:window.__operativeQA?.layers()??null,vesselLegend:legend?{label:legend.textContent,swatch:getComputedStyle(legend,'::before').backgroundColor}:null,phase:panel.dataset.operativePhase,progress:panel.dataset.operativeProgress,
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
    if(acceptance){
      await setFieldView(js,'section');await key('Home');
      for(const [second,position] of [[9,2.1],[14,3.6],[18,end]]){
        if(position>end)continue;
        await seek(position);await js('new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))');
        const name='matched-'+second+'s-'+width;
        if(cameras){
          const pose=cameras[name]?.submitted;assert(pose,'Missing reference camera '+name);
          await js('window.__operativeQA.paint('+JSON.stringify(pose)+')');
        }
        await snap(name);assert(views[name].submitted,'Missing submitted field matrix');
        const reference=views[name].submitted;
        if(cameras){
          for(const k of ['matrixWorld','projectionMatrix'])assert.deepEqual(reference[k],cameras[name].submitted[k]);
        }
        if(ablations){
          const checkbox=`[...document.querySelectorAll('.operative-options label')].find(l=>l.textContent.trim()==='Field voxel faces').querySelector('input')`;
          const pinned=async suffix=>{
            await js('window.__operativeQA.paint('+JSON.stringify(reference)+')');
            const label=name+'-'+suffix;
            await snap(label);
            assert.equal(views[label].removed,views[name].removed);
            for(const k of ['matrixWorld','projectionMatrix'])assert.deepEqual(views[label].submitted[k],reference[k]);
          };
          await js(`(${checkbox}).click()`);await until(`(${checkbox}).checked`);
          await pinned('raw');
          await js(`(${checkbox}).click()`);await until(`!(${checkbox}).checked`);
          await setFieldView(js,'exterior');await pinned('exterior');
          await setFieldView(js,'section');
        }
        await js('window.__operativeQA.paint(null)');
      }
    }
    await setFieldView(js,'follow');await seek(0);await key('Home');await snap('intact-'+width);
    if(expectEntry){
      const {field,entry,surrounding}=views['intact-'+width];
      assert(entry,'Missing visible entry marker at '+width);
      assert(entry.rect.left>=field.left&&entry.rect.right<=field.right&&entry.rect.top>=field.top&&entry.rect.bottom<=field.bottom,'Entry marker clipped at '+width);
      assert(surrounding.some(layer=>layer.role==='bone'),'Missing source skull at '+width);
      if(expectOuterHead)assert.equal(views['intact-'+width].entryCaption,'Outer head · MRI estimate');
    }
    if(acceptance&&end===4){
      for(const position of [1.99,2,2.01,3.05,4]){
        const started=performance.now();
        await seek(position);
        await js('new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))');
        const elapsed=performance.now()-started;
        views['follow-'+position+'-'+width]={...await current(),seekAndTwoFramesMs:elapsed};
      }
    }
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
    if(acquiredReview&&cameras){
      const optical=await js(`[...document.querySelector('[aria-label="Operative field view"]').options].some(option=>option.value==='operative')`);
      assert(optical,'Expected an explicit opaque operative view');
      for(const [label,position,prior] of [['exposure',2,'exposure'],['endpoint',end,'exterior']]){
        await seek(position);await key('Home');
        await js('new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))');
        const reference=cameras[prior+'-'+width],name='opaque-matched-'+label+'-'+width;
        await js('window.__operativeQA.paint('+JSON.stringify(reference.submitted)+')');
        await snap(name);
        const drawn=views[name];
        assert.equal(drawn.view,'operative');
        assert.equal(drawn.removed,reference.removed);
        for(const k of ['matrixWorld','projectionMatrix'])assert.deepEqual(drawn.submitted[k],reference.submitted[k]);
        for(const k of ['width','height'])assert.equal(drawn.field[k],reference.field[k]);
        await js('window.__operativeQA.paint(null)');
      }
      assert.equal(endpoint.view,'operative');
      assert.equal(views['section-'+width].view,'section');
      assert.equal(views['exterior-'+width].view,'exterior');
      if(width===1440){
        views.operativeOrbitTiming=await js(`(async()=>{
          const probe=window.__operativeQA,gl=probe.last.renderer.getContext();
          const timer=gl.getExtension('EXT_disjoint_timer_query_webgl2');
          const frames=[],canvas=document.querySelector('.operative-field-canvas canvas');
          const next=()=>new Promise(resolve=>requestAnimationFrame(resolve));
          let disjoint=timer&&!!gl.getParameter(timer.GPU_DISJOINT_EXT);
          probe.sampling={timer,frames};
          try{
            for(let i=0;i<80;i++){
              canvas.dispatchEvent(new KeyboardEvent('keydown',{key:i%2?'ArrowLeft':'ArrowRight'}));
              await next();
              if(timer&&gl.getParameter(timer.GPU_DISJOINT_EXT))disjoint=true;
            }
          }finally{probe.sampling=null;}
          const pending=()=>frames.some(f=>f.query&&!gl.getQueryParameter(f.query,gl.QUERY_RESULT_AVAILABLE));
          const start=performance.now();
          while(pending()&&performance.now()-start<1500)await next();
          if(timer&&gl.getParameter(timer.GPU_DISJOINT_EXT))disjoint=true;
          return {method:'Alternating local optical camera arrow-key orbit at the unchanged geometric endpoint',
            cpuMs:frames.map(f=>f.cpuMs),renderIntervalsMs:frames.slice(1).map((f,i)=>f.start-frames[i].start),
            gpuElapsedMs:frames.flatMap(f=>{
              if(!f.query)return [];
              const ms=!disjoint&&gl.getQueryParameter(f.query,gl.QUERY_RESULT_AVAILABLE)?
                gl.getQueryParameter(f.query,gl.QUERY_RESULT)/1e6:null;
              gl.deleteQuery(f.query);return ms===null?[]:[ms];
            }),disjoint};
        })()`);
      }
    }
    if(endpoint.view==='section'){
      assert(views['section-'+width].sectionSource);
      assert.equal(views['exterior-'+width].view,'exterior');
    }
  }
  if(moviePoses){
    win.setContentSize(1920,1080);await until('innerWidth===1920&&innerHeight===1080');
    await setFieldView(js,'follow');await key('Home');
    for(const second of [9,14,18]){
      const state=moviePoses.find(row=>row.frame===second*30),pose=state?.submitted;
      assert(pose?.canvas,'Missing actual film camera and viewport for '+second+' s');
      const r=pose.canvas;
      await js(`(()=>{const field=document.querySelector('.operative-field-canvas');
        field.style.width='${r.width}px';field.style.height='${r.height}px';})()`);
      await seek(state.position);await js('new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))');
      await js('window.__operativeQA.paint('+JSON.stringify(pose)+')');
      const name='film-pose-'+second+'s';await snap(name);
      assert.equal(views[name].removed,state.removed);
      for(const k of ['matrixWorld','projectionMatrix'])assert.deepEqual(views[name].submitted[k],pose[k]);
      for(const k of ['width','height'])assert.equal(views[name].field[k],r[k]);
    }
    await js('window.__operativeQA.paint(null)');
  }
  if(match||moviePoses)await js(`(()=>{const field=document.querySelector('.operative-field-canvas');field.style.width='';field.style.height='';})()`);
  await seek(0);await key('Home');
  return views;
}
module.exports={setFieldView,verifyOperativeCorridor};
