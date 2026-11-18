'use strict';
// electron verify-engine-contours.cjs case output anchor capture-positions [matching-report]
// Open a saved engine case in the ordinary application. Every image is a hidden compositor capture.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const {app, BrowserWindow, ipcMain} = require('electron');
const {inputScript,sourceVoxel} = require('./verify-eden-planning-helpers.cjs');
const [, , casePath, outputPath, anchorArg, positionsArg, matchingReport] = process.argv;
assert(casePath && outputPath && positionsArg, 'Pass case, output directory, anchor, and comma-separated positions');
const source = fs.realpathSync(casePath), output = path.resolve(outputPath);
const anchor = Number(anchorArg), positions = positionsArg.split(',').map(Number);
const main = path.resolve(process.env.SURFACE_VERIFY_MAIN || path.join(__dirname,'main.cjs'));
const hash = data => crypto.createHash('sha256').update(data).digest('hex');
const manifest = JSON.parse(execFileSync('unzip',['-p',source,'manifest.json'],{maxBuffer:256*1024*1024}));
const artifact = manifest.artifacts.surgical_toolbox;
const history = artifact.timeline[artifact.branchId];
const curved = manifest.case_id==='generated-curved-field-v1';
const expected = i => i ? history[i-1].stateAfter : history[0].stateBefore;
for (const i of [anchor,...positions]) assert(Number.isSafeInteger(i) && i>=0 && i<=history.length);
fs.mkdirSync(output,{recursive:false});
const profile = fs.mkdtempSync(path.join(os.tmpdir(),'ressectionlab-contours-'));
const report = {sourceSha256:hash(fs.readFileSync(source)), sourceCaseHash:manifest.case_semantic_hash,
  sourceArrays:manifest.array_sha256, branch:artifact.branchId, anchor, positions, captures:[], seeks:[], hidden:true,
  hardware:{cpu:os.cpus()[0].model, arch:os.arch(), memoryGiB:os.totalmem()/2**30, osRelease:os.release()},
  rendererAssets:{}, ipc:[]};
const directory = path.resolve(path.dirname(main),'../dist/assets');
for(const name of fs.readdirSync(directory).filter(n=>n.endsWith('.js')||n.endsWith('.css'))) {
  report.rendererAssets[name]=hash(fs.readFileSync(path.join(directory,name)));
}
const ledgers = new Map();
const handle = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel,fn)=>handle(channel,async (...args)=>{
  const started=performance.now(), result=await fn(...args);
  if(channel==='research:surgicalRehearsal'){
    report.ipc.push({command:args[1]?.command, state:result.view?.stateId, ms:performance.now()-started});
    if(result.sequenceOutcome)ledgers.set(result.sequenceOutcome.stateId,result.sequenceOutcome);
  }
  return result;
});
app.setPath('userData',profile);
if(process.platform==='darwin')app.setActivationPolicy('accessory');
for(const method of ['show','showInactive','focus'])BrowserWindow.prototype[method]=function(){
  report.hidden=false; throw Error('Hidden capture refused '+method);
};
process.argv.push('--case',source,'--automation-hidden');
delete process.env.RESSECTIONLAB_DEV_URL;
const delay = ms=>new Promise(resolve=>setTimeout(resolve,ms));
app.on('browser-window-created',(_event,win)=>{
  win.once('show',()=>{report.hidden=false;win.hide();});
  win.webContents.once('did-finish-load',async()=>{
    const js = text=>win.webContents.executeJavaScript(text,true);
    const wait = async (expression,limit=150000)=>{
      const started=performance.now();
      do {const value=await js(expression);if(value)return value;await delay(50);}while(performance.now()-started<limit);
      throw Error('Timed out waiting for '+expression+'\n'+await js('document.body.innerText.slice(-5500)'));
    };
    const click = async label=>{
      const button='[...document.querySelectorAll("button")].find(b=>b.textContent.trim()==='+JSON.stringify(label)+'||b.getAttribute("aria-label")==='+JSON.stringify(label)+')';
      await wait('('+button+')?.disabled===false');
      await js('(()=>{const b='+button+';for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==="DETAILS")p.open=true;b.click()})()');
    };
    const set = async (label,value)=>{await js(inputScript(label,value));};
    const ready = async i=>{
      await wait(`document.querySelector('.surgical-panel')?.dataset.recordedPosition==='${i}' &&
        document.querySelector('[data-accepted-tools]')?.dataset.surgicalStateId===${JSON.stringify(expected(i))} &&
        document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0' && !document.querySelector('.viewer-loading')`);
    };
    const seek = async i=>{
      const t=performance.now();
      if(await js(`document.querySelector('.surgical-panel')?.dataset.recordedPosition!=='${i}'`))await set('Surgical timeline',i);
      await ready(i); report.seeks.push({i,state:expected(i),wallMs:performance.now()-t});
    };
    const smooth = async enabled=>{
      const button=enabled?'Conditioned edges':'Source edges';
      if(await js(`[...document.querySelectorAll('[aria-label="Compare surface contours"] button')].some(b=>b.textContent.trim()===${JSON.stringify(button)})`)){
        const pressed=await js(`[...document.querySelectorAll('[aria-label="Compare surface contours"] button')].find(b=>b.textContent.trim()===${JSON.stringify(button)}).getAttribute('aria-pressed')`);
        if(pressed!=='true')await click(button);
        return;
      }
      await js(`(()=>{const e=[...document.querySelectorAll('.surgical-panel input[type=checkbox]')].find(c=>
        /smooth 3d surfaces|conditioned contours/i.test(c.closest('label')?.textContent||c.getAttribute('aria-label')||''));
        if(!e)throw Error('Missing raw/conditioned toggle');if(e.checked!==${enabled})e.click();return e.checked})()`);
    };
    const capture = async (i,appearance,smoothed)=>{
      const started=performance.now();
      await smooth(smoothed); await set('Operative surface material',appearance);
      await ready(i);
      const state=expected(i);
      if(!ledgers.has(state)){
        await click('Calculate sequence breakdown'); await wait("!!document.querySelector('[aria-label=\"Complete sequence quantities\"]')");
      }
      const ledger=ledgers.get(state); assert(ledger,'Need ledger at '+i);
      // Input setters scroll controls into view. Restore a fixed page and panel before measuring pixels.
      await js(`(()=>{window.scrollTo(0,0); document.querySelector('.surgical-panel')?.closest('aside')?.scrollTo(0,0);
        return new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))})()`);
      await delay(180);
      const snap=await wait(`(()=>{const p=document.querySelector('[data-accepted-tools]'), v=document.querySelector('.rl-viewer'),
        frames=window.contourQA.frames, f=frames.findLast(f=>!f.interactive&&f.surgicalStateId===${JSON.stringify(state)}),
        gpu=f&&window.contourQA.gpu.findLast(g=>g.serial===f.serial&&g.completed);if(!gpu||v.dataset.surfacesPending!=='0')return null;
        const r=p.getBoundingClientRect(), checks=[...document.querySelectorAll('.surgeon-view input[type=checkbox]')];
        return {state:p.dataset.surgicalStateId, camera:JSON.parse(p.dataset.opticalCamera), actualCamera:f.cameraState,
          tools:JSON.parse(p.dataset.acceptedTools),removedCells:Number(p.dataset.removedCells),
          cutawayOff:checks.filter(c=>/inspection-only/i.test(c.parentElement.textContent)).length===2&&checks.filter(c=>/inspection-only/i.test(c.parentElement.textContent)).every(c=>!c.checked),
          view:p.dataset.operativeView, render:f, fence:gpu, rect:{x:r.x,y:r.y,width:r.width,height:r.height}}})()`);
      assert.equal(snap.view,'surgeon');assert(snap.cutawayOff);assert.equal(snap.state,state);
      assert(!win.isVisible());
      const file='boundary-'+i+'-'+appearance+'-'+(smoothed?'conditioned':'raw');
      const shot=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(output,file+'.png'),shot.resize({width:1440,height:900}).toPNG());
      const crop=await win.webContents.capturePage(Object.fromEntries(Object.entries(snap.rect).map(([key,v])=>[key,Math.round(v)])),{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(output,file+'-field.png'),crop.toPNG());
      const {width,height}=crop.getSize();
      const center=crop.crop({x:Math.floor(width*.23),y:Math.floor(height*.34),
        width:Math.floor(width*.53),height:Math.floor(height*.40)});
      report.captures.push({file,i,appearance,smoothed,...snap,ledger,displayToCaptureMs:performance.now()-started,
        centerFieldPixelsSha256:hash(center.toBitmap()),fieldPixelsSha256:hash(crop.toBitmap())});
      console.log('Captured',file);
    };
    try{
      win.setContentSize(1440,900); win.webContents.setBackgroundThrottling(false);
      await js(`window.contourQA={frames:[],gpu:[],surfaces:[]};for(const [ev,arr] of [['anatomy-frame','frames'],['anatomy-gpu','gpu'],['anatomy-surface','surfaces']])document.addEventListener(ev,e=>{const a=window.contourQA[arr];a.push(e.detail);if(a.length>1024)a.shift()})`);
      await wait("!!document.querySelector('.rl-viewer canvas') && !document.querySelector('.viewer-loading')");
      await click('Rehearse');
      await wait("!!document.querySelector('[aria-label=\"Surgical timeline\"]')");
      await seek(anchor);
      if(!await js("!!document.querySelector('[data-operative-view=surgeon][data-optical-camera]')"))await click('Surgeon / Microscope');
      await wait("!!document.querySelector('[data-operative-view=surgeon][data-optical-camera]')");
      await set('Viewing angle',curved?34:18);
      await set('Field / magnification',curved?52:42);
      for(const i of positions){
        await seek(i);
        for(const [appearance,conditioned] of [['labels',false],['labels',true],['operative',true]])await capture(i,appearance,conditioned);
      }
      if(positions.length>1){await seek(positions[0]);}
      report.surfaces=await js('window.contourQA.surfaces');
      report.performance=await js(`(()=>{const frames=window.contourQA.frames.filter(f=>f.surgicalStateId);
        return {count:frames.length,cpuSubmissionMs:frames.map(f=>f.cpuMs),fencePollUpperMs:window.contourQA.gpu.filter(g=>g.completed&&g.surgicalStateId).map(g=>g.completionPollMs)}})()`);
      if(matchingReport){
        const conditioned=report.surfaces.filter(s=>s.diagnostics?.policy==='engine-shared-implicit-v1');
        assert(conditioned.length,'Missing live engine shared-field diagnostics');
        for(const layer of conditioned)assert(layer.diagnostics.maxDisplacementMm<=layer.diagnostics.displacementLimitMm);
        for(const position of positions){
          const labelViews=report.captures.filter(c=>c.i===position && c.appearance==='labels');
          assert.equal(labelViews.length,2);
          assert.notEqual(labelViews[0].centerFieldPixelsSha256,labelViews[1].centerFieldPixelsSha256,
            'Raw and conditioned views must differ inside the operative field');
        }
        const legend='[aria-label="Operative anatomy legend"]';
        await wait(`!!document.querySelector('${legend}')`);
        report.legend=await js(`(()=>{const el=document.querySelector('${legend}'),r=el.getBoundingClientRect();
          return {box:{width:r.width,height:r.height},rows:[...el.querySelectorAll('[data-region]')].map(e=>({name:e.dataset.region,
            title:e.textContent,sourceColor:e.dataset.sourceColor,fieldColor:e.dataset.fieldColor}))}})()`);
        const field=report.captures[0].rect;
        assert(report.legend.box.width*report.legend.box.height<field.width*field.height*.15,'Legend obscures the field');
      }
      if(matchingReport && manifest.case_semantic_hash==='sha256:f7985ce9761aeba770ac752156f0b874c78c8dd8078b25455f773006aab557ee'){
        const native=manifest.metadata.surgical_adapter;
        const legend='[aria-label="Operative anatomy legend"]';
        const target=report.legend.rows.find(r=>r.name==='tumor'),vessel=report.legend.rows.find(r=>r.name==='vessels');
        assert.match(target.title,/supplied.*enhancing/i);assert.match(vessel.title,/vessel.*estimate/i);
        assert.notEqual(target.sourceColor,vessel.sourceColor);
        assert.notEqual(vessel.fieldColor,vessel.sourceColor);
        await js(`document.querySelector('${legend} [data-region="tumor"]').click()`);
        const targetLayer=native.sourceLayers.tumor;
        const inspect=`document.querySelector('${legend} [role="status"]')?.textContent`;
        report.targetInspection=await wait(inspect);
        for(const fact of [targetLayer.layerId,targetLayer.provenance.method,targetLayer.reviewStatus])assert(report.targetInspection.includes(fact));
        await js(`document.querySelector('${legend} [data-region="tumor"]').click()`);
        // These two source patches are exposed at the fixed historical camera.
        const bounds=report.captures[0].rect;
        const pick=async (name,points)=>{
          const layer=native.sourceLayers[name];
          for(const [u,v] of points){
            const x=bounds.x+bounds.width*u,y=bounds.y+bounds.height*v;
            await js(`document.querySelector('.rl-viewer-anatomy-image').dispatchEvent(new MouseEvent('dblclick',{bubbles:true,clientX:${x},clientY:${y}}))`);
            await delay(150);
            const detail=await js(inspect);
            if(!detail?.includes(layer.layerId))continue;
            const ras=await js('window.contourQA.frames.at(-1).cursor');
            const voxel=sourceVoxel(layer.registeredAffine,ras).map(Math.round);
            assert(detail.toLowerCase().includes('source voxel '+voxel.join(', ')),detail);
            for(const fact of [layer.provenance.model||layer.provenance.method,layer.reviewStatus,layer.sampling])assert(detail.includes(fact));
            const evidence={relativeField:[u,v],ras,voxel,detail};
            const nativeRoot=process.env.CONTOUR_NATIVE_STUDY;
            if(nativeRoot){
              const file=path.join(nativeRoot,layer.sourceArray.path);
              assert.equal(hash(fs.readFileSync(file)),layer.sourceArray.sha256);
              const sample=JSON.parse(execFileSync(path.resolve(__dirname,'../../.venv/bin/python'),['-c',[
                'import sys,json,numpy as np',
                'a=np.load(sys.argv[1],mmap_mode="r",allow_pickle=False)',
                'affine,ras,ijk=json.loads(sys.argv[2])',
                'voxels=np.argwhere(a>0)',
                'world=voxels @ np.array(affine)[:3,:3].T+np.array(affine)[:3,3]',
                'distance=np.linalg.norm(world-np.array(ras),axis=1)',
                'i=int(distance.argmin())',
                'print(json.dumps(dict(sourceValue=int(a[tuple(ijk)]),nearestPositiveVoxel=voxels[i].tolist(),distanceToPositiveCenterMm=float(distance[i]))))',
              ].join('\n'),file,JSON.stringify([layer.registeredAffine,ras,voxel])],{encoding:'utf8'}));
              // A nearest rounded surface point can land on either side of a
              // binary label. Record the actual value and true positive-center distance.
              Object.assign(evidence,{sourceArraySha256:layer.sourceArray.sha256,...sample});
              assert(sample.distanceToPositiveCenterMm<Math.sqrt(3),'Pick lies more than a 1 mm source voxel diagonal from this label');
            }
            if(name==='vessels')assert.match(detail,/unassessed/i);
            assert.equal(await js("document.querySelector('[data-accepted-tools]').dataset.surgicalStateId"),expected(positions[0]));
            const image=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
            fs.writeFileSync(path.join(output,name==='tumor'?'picked-supplied-target.png':'picked-estimated-vessel.png'),image.resize({width:1440,height:900}).toPNG());
            return evidence;
          }
          throw Error('No positive registered pick for '+name+' at the fixed microscope camera');
        };
        report.targetPick=await pick('tumor',[[.5352,.524],[.535,.53],[.539,.523],[.4948,.5528],[.74,.598],[.531,.532],[.537,.513]]);
        report.vesselPick=await pick('vessels',[[.914,.347],[.89,.36],[.747,.448],[.934,.37],[.704,.51]]);
      }
      report.hardware.gpu=app.getGPUFeatureStatus();
      assert.equal(hash(fs.readFileSync(source)),report.sourceSha256);
      if(matchingReport){
        const ref=JSON.parse(fs.readFileSync(matchingReport));
        for(const key of ['sourceSha256','sourceCaseHash','sourceArrays','branch','anchor','positions'])assert.deepEqual(report[key],ref[key],key);
        assert.equal(report.captures.length,ref.captures.length);
        report.captures.forEach((c,i)=>{for(const key of ['file','state','camera','tools','removedCells','cutawayOff','ledger','rect']){
          assert.deepEqual(c[key],ref.captures[i][key],key+' '+c.file);
        }
          // Three's orbit matrix recomposition can vary in the last few floating-point bits.
          for(const key of ['position','target','up'])for(let j=0;j<3;j++){
            assert(Math.abs(c.actualCamera[key][j]-ref.captures[i].actualCamera[key][j])<1e-8,'Rendered camera '+key);
          }
        });
        report.matchedBaseline=true;
      }
      report.passed=true;
    }catch(error){report.error=String(error.stack||error);report.body=await js('document.body.innerText').catch(()=>null);console.error(report.error);}
    finally{fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2)+'\n');
      fs.rmSync(profile,{recursive:true,force:true});app.exit(report.passed&&report.hidden?0:1);}
  });
});
require(main);
