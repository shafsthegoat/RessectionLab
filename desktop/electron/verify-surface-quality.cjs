'use strict';
// Hidden Electron capture of the live viewer. Native studies are tested on disposable copies.
const {app,ipcMain}=require('electron');
const {execFile}=require('node:child_process');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const {inputScript}=require('./verify-eden-planning-helpers.cjs');
const main=process.env.SURFACE_VERIFY_MAIN||path.join(__dirname,'main.cjs');
const output=path.resolve(process.env.SURFACE_VERIFY_OUTPUT);
fs.mkdirSync(output,{recursive:false});
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-surface-quality-'));
app.setPath('userData',profile);
process.argv.push('--automation-hidden');
const nativeIndex=process.argv.indexOf('--anatomy'),caseIndex=process.argv.indexOf('--case');
assert((nativeIndex>=0)!==(caseIndex>=0),'Choose --anatomy or --case');
const inputIndex=(nativeIndex>=0?nativeIndex:caseIndex)+1,source=fs.realpathSync(process.argv[inputIndex]);
const sha=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const sourceHash=sha(source);
const document=nativeIndex>=0&&path.basename(source)==='patient-anatomy.json' ? JSON.parse(fs.readFileSync(source,'utf8')) : null;
const cacheSource=process.env.SURFACE_VERIFY_CACHE_FROM;
const retention=process.env.SURFACE_VERIFY_RETAIN_CACHE==='1';
let studyDirectory;
const engine=require(path.join(path.dirname(main),'sidecar.cjs')),Sidecar=engine.Sidecar;
let sidecarPid;
engine.Sidecar=class extends Sidecar{
  constructor(options){super(options);sidecarPid=this.child.pid;}
};
function clone(from,to){
  const target=fs.realpathSync(from);
  if(fs.statSync(target).isDirectory()){
    fs.mkdirSync(to); for(const name of fs.readdirSync(target))clone(path.join(target,name),path.join(to,name));
  }else fs.copyFileSync(target,to,fs.constants.COPYFILE_FICLONE);
}
if(document){
  studyDirectory=path.join(profile,'study');fs.mkdirSync(studyDirectory);
  for(const member of ['patient-anatomy.json','arrays','originals','revisions']){
    const from=path.join(path.dirname(source),member),destination=path.join(studyDirectory,member);
    if(fs.existsSync(from))clone(from,destination);
  }
  const availableCache=cacheSource??(process.env.SURFACE_VERIFY_CACHE_MODE==='empty'?null:path.join(path.dirname(source),'meshes'));
  if(availableCache&&fs.existsSync(availableCache))clone(availableCache,path.join(studyDirectory,'meshes'));
  process.argv[inputIndex]=path.join(studyDirectory,'patient-anatomy.json');
}
const report={source,sourceHash,windowEverVisible:false,captures:{},requests:[],errors:[],memory:{peakElectronKiB:0,peakSidecarKiB:0,peakCombinedKiB:0},startedAt:new Date().toISOString()};
if(document){
  report.snapshotId=document.snapshot_id;
  report.grids=[...document.series,...document.layers].map(x=>({id:x.id,shape:x.data.shape,
    gridSpacingMm:x.grid_spacing_mm??[0,1,2].map(axis=>Math.hypot(...x.affine.slice(0,3).map(row=>row[axis]))),
    affine:x.affine,acquisitionResolution:x.acquisition_resolution}));
}
const dist=path.resolve(path.dirname(main),'../dist');
report.build=Object.fromEntries(fs.readdirSync(path.join(dist,'assets')).filter(n=>!n.endsWith('.map')).map(n=>[n,sha(path.join(dist,'assets',n))]));
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms)),started=performance.now();
let active=0;
const handle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(name,handler)=>handle(name,async(event,args)=>{
  const at=performance.now()-started;active++;
  try{
    const result=await handler(event,args);
    if(['research:startupWorkspace','research:anatomyView','research:anatomyRegister'].includes(name)){
      const layers=result?.compartments?.map(({name,kind,qa,surfaceDiagnostics,volumeMm3,mask,surface})=>({name,kind,qa,surfaceDiagnostics,volumeMm3,maskSha256:mask?.sha256,triangles:surface?.indices?.shape?.reduce((a,b)=>a*b,1)/3,points:surface?.positions?.shape?.reduce((a,b)=>a*b,1)/3,indicesSha256:surface?.indices?.sha256,sourceMeshSha256:surface?.source?.sha256,displayMeshSha256:surface?.positions?.sha256}));
      report.requests.push({name,action:args?.action,at,ms:performance.now()-started-at,sourceSignalSha256:result?.mri?.sha256,layers});
    }
    return result;
  }catch(error){report.requests.push({name,error:String(error),at});throw error;}
  finally{active--;}
});
function sampleMemory(){
  const electron=app.getAppMetrics().reduce((sum,row)=>sum+(row.memory.workingSetSize||0),0);
  report.memory.peakElectronKiB=Math.max(report.memory.peakElectronKiB,electron);
  if(sidecarPid)execFile('/bin/ps',['-o','rss=','-p',String(sidecarPid)],{timeout:1500},(error,stdout)=>{
    if(error)return;
    const sidecar=Number(stdout.trim());
    report.memory.peakSidecarKiB=Math.max(report.memory.peakSidecarKiB,sidecar);
    report.memory.peakCombinedKiB=Math.max(report.memory.peakCombinedKiB,electron+sidecar);
  });
}
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.on('console-message',(_event,detail)=>{if(detail.level==='error')report.errors.push(detail.message);});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setContentSize(1440,960);
    const memory=setInterval(sampleMemory,300),js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(check,label,seconds=250)=>{
      const deadline=performance.now()+seconds*1000;
      while(performance.now()<deadline){const value=await check();if(value)return value;await delay(125);}
      throw Error('Timed out waiting for '+label);
    };
    const click=async label=>{
      await js(`(()=>{const buttons=[...document.querySelectorAll('button')];const b=buttons.find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)})??buttons.find(e=>e.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});b.click()})()`);
    };
    const set=async(selector,value)=>{
      await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});if(!e||e.disabled)throw Error('Missing enabled control');Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(e,${JSON.stringify(value)});e.dispatchEvent(new Event('change',{bubbles:true}))})()`);
    };
    const ready=async(minimum=-1)=>{
      await delay(125);
      return wait(async()=>{
        if(report.requests.some(r=>r.error&&r.name.startsWith('research:anatomy')))throw Error(report.requests.find(r=>r.error).error);
        if(active)return null;
        const state=await js(`(()=>{const v=document.querySelector('.rl-viewer'),p=window.surfaceVerification,f=p.frames.at(-1);return {frame:f,gpu:f&&p.gpu.findLast(g=>g.serial===f.serial&&g.completed),visible:v?.dataset.rendererReady==='true',pending:!!document.querySelector('.viewer-loading,[data-layer-pending=true]'),layers:v?.dataset.visibleLayerIds,modality:v?.dataset.sourceModality}})()`);
        return state.gpu&&state.visible&&!state.pending&&state.frame.serial>minimum&&!state.frame.interactive?state:null;
      },'GPU-confirmed resting anatomy');
    };
    const capture=async(name,minimum=-1)=>{
      const view=await ready(minimum);
      assert(!window.isVisible());
      const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      assert(!image.isEmpty(),'No hidden-window screenshot');
      fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());
      const paneBounds=await js(`[...document.querySelectorAll('.rl-viewer-anatomy-image,.rl-viewer-slice')].map(e=>{
        const r=e.getBoundingClientRect(); return {className:e.className,x:r.x,y:r.y,width:r.width,height:r.height,windowWidth:innerWidth};
      })`);
      const imageScale=image.getSize().width/paneBounds[0].windowWidth;
      const pixels=Object.fromEntries(paneBounds.map(pane=>{
        const id=['anatomy-image','axial','coronal','sagittal'].find(name=>pane.className.includes(name));
        const rect={x:Math.ceil((pane.x+8)*imageScale),y:Math.ceil((pane.y+40)*imageScale),
          width:Math.floor((pane.width-16)*imageScale),height:Math.floor((pane.height-58)*imageScale)};
        const data=image.crop(rect).toBitmap();
        return [id,{rect,sha256:crypto.createHash('sha256').update(data).digest('hex')}];
      }));
      report.captures[name]={...view,imageSize:image.getSize(),atMs:performance.now()-started,pixels};
      console.log('Captured '+name);
      return view.frame;
    };
    const serial=()=>js('window.surfaceVerification.frames.at(-1)?.serial??-1');
    const smooth=async(enabled)=>{
      await click('Appearance');
      await js(`(()=>{const c=document.querySelector('input[aria-label="Smooth 3D surfaces"]');if(c.checked!==${enabled})c.click()})()`);
      await click('Close view controls');
    };
    const camera=async(preset)=>{
      await click('Layout');await set('[aria-label="Camera preset"]',preset);await click('Close view controls');
    };
    const sidePanel=async label=>{await js(`(()=>{const x=[...document.querySelectorAll('summary')].find(e=>e.textContent.includes(${JSON.stringify(label)}));if(x)x.parentElement.open=true;})()`);};
    try{
      await js(`window.surfaceVerification={frames:[],gpu:[],surface:[],refinement:[]};for(const [name,key] of [['anatomy-frame','frames'],['anatomy-gpu','gpu'],['anatomy-surface','surface'],['anatomy-refined','refinement']])document.addEventListener(name,e=>{window.surfaceVerification[key].push(e.detail);});`);
      await ready();report.freshProcessToReadyMs=performance.now()-started;
      if(process.env.SURFACE_VERIFY_NATIVE_PRESENTATION==='1'){
        await js(`(()=>{const button=[...document.querySelectorAll('.anatomy-presets button')].find(b=>b.textContent.trim()==='Structures');
          if(!button||button.disabled)throw Error('Native Structures presentation unavailable');
          button.click();
        })()`);
      }
      await click('Show structures only');
      const hasBone=report.requests.some(request=>request.layers?.some(layer=>['skull','bone'].includes(layer.kind)&&layer.points>0));
      const skull=hasBone&&await js(`!!document.querySelector('select[aria-label="Skull view"]')`);
      if(skull){await sidePanel('Display & fragments');await set('[aria-label="Skull view"]','transparent');}
      if(process.env.SURFACE_VERIFY_NATIVE_PRESENTATION==='1'){
        const brain=document.layers.find(layer=>layer.kind==='brain');
        if(brain){
          await click(brain.label);
          await js(inputScript(`${brain.label} opacity`,.16));
          await click('Close layer details');
        }
      }
      const pair=async(name,frameView=async()=>{})=>{
        await smooth(false);await frameView();
        const raw=await capture('source-'+name);
        const warmStarted=performance.now();
        await smooth(true);await capture('smooth-'+name,raw.serial);
        report.captures['smooth-'+name].toggleToCaptureMs=performance.now()-warmStarted;
      };
      await pair('oblique',()=>click('Fit 3D view'));
      await pair('anterior',()=>camera('anterior'));
      if(skull){
        await set('[aria-label="Skull view"]','exterior');
        await pair('skull-anterior');
        await set('[aria-label="Skull view"]','transparent');
      }
      if(process.env.SURFACE_VERIFY_FOCUS==='1'){
        const labels=await js(`[...document.querySelectorAll('.native-structure-name')].map(e=>e.textContent.trim())`);
        for(const [kind,re] of [['brain',/brain/i],['tumor',/tumor|lesion/i],['vessels',/vessel|arter|vein|branch/i]]){
          const label=labels.find(name=>re.test(name));if(!label)continue;
          await click(label);await click('Isolate');
          const selector='input[aria-label="'+label+' opacity"]';
          const opacity=await js(`document.querySelector(${JSON.stringify(selector)}).value`);
          const setOpacity=async value=>{await js(`(()=>{const e=document.querySelector(${JSON.stringify(selector)});Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${JSON.stringify(String(value))});e.dispatchEvent(new Event('input',{bubbles:true}));})()`);};
          if(opacity!=='1')await setOpacity(1);
          await pair(kind,()=>click('Focus '+label));
          if(opacity!=='1')await setOpacity(opacity);
          await click('Restore others');
        }
      }
      await click('Fit 3D view');
      const resting=await ready();
      const center=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5)}})()`);
      window.webContents.sendInputEvent({type:'mouseDown',...center,button:'left',clickCount:1});
      for(let k=1;k<=90;k++){
        window.webContents.sendInputEvent({type:'mouseMove',x:center.x+k,y:center.y+Math.round(10*Math.sin(k/9)),modifiers:['leftButtonDown']});
        await delay(16);
      }
      window.webContents.sendInputEvent({type:'mouseUp',x:center.x+90,y:center.y+Math.round(10*Math.sin(10)),button:'left',clickCount:1});
      await capture('post-orbit-refinement',resting.frame.serial);
      const data=await js('window.surfaceVerification');
      const percentile=(values,p)=>values.length?[...values].sort((a,b)=>a-b)[Math.floor((values.length-1)*p)]:null;
      const metrics=frames=>{
        const fences=frames.flatMap(f=>data.gpu.find(g=>g.serial===f.serial&&g.completed)?.milliseconds??[]);
        return {frames:frames.length,cpuP50Ms:percentile(frames.map(f=>f.cpuMs),.5),cpuP95Ms:percentile(frames.map(f=>f.cpuMs),.95),fenceP50Ms:percentile(fences,.5),fenceP95Ms:percentile(fences,.95),submissionGapP50Ms:percentile(frames.slice(1).map((f,i)=>f.at-frames[i].at),.5)};
      };
      report.resting=metrics(data.frames.filter(f=>!f.interactive));report.orbit=metrics(data.frames.filter(f=>f.interactive));
      assert(report.orbit.frames>0,'No actual interactive frame submitted');
      assert(!report.windowEverVisible,'Verification window appeared');
      function sameFrame(before,after,label){
        for(const point of ['target','position','up'])assert(before.frame.cameraState[point].every((value,i)=>Math.abs(value-after.frame.cameraState[point][i])<1e-5),label+' camera differs');
        assert.equal(before.frame.sourceSignalFingerprint,after.frame.sourceSignalFingerprint,label+' source signal differs');
        assert.deepEqual(before.frame.cursor,after.frame.cursor,label+' slice cursor differs');
      }
      for(const [name,view] of Object.entries(report.captures)){
        const raw=report.captures[name.replace(/^smooth-/,'source-')];if(!name.startsWith('smooth-')||!raw)continue;
        sameFrame(raw,view,name);
        for(const plane of ['axial','coronal','sagittal'])assert.equal(raw.pixels[plane].sha256,view.pixels[plane].sha256,name+' changed '+plane+' source pixels');
      }
      if(process.env.SURFACE_VERIFY_NATIVE_PRESENTATION==='1'){
        assert.notEqual(report.captures['source-oblique'].pixels['anatomy-image'].sha256,
          report.captures['smooth-oblique'].pixels['anatomy-image'].sha256,
          'Conditioned contours should change the actual 3D pixels at the same source and camera');
      }
      if(process.env.SURFACE_VERIFY_EXPECTED){
        const before=JSON.parse(fs.readFileSync(process.env.SURFACE_VERIFY_EXPECTED,'utf8'));
        assert(before.passed&&before.sourceHash===report.sourceHash,'Expected successful baseline with exact same source');
        for(const [name,view] of Object.entries(report.captures)){
          if(name==='post-orbit-refinement')continue;
          sameFrame(before.captures[name],view,name+' baseline');
        }
        const identities=log=>log.requests.flatMap(request=>request.layers??[]).map(layer=>[
          layer.name,layer.indicesSha256,layer.sourceMeshSha256,layer.maskSha256,layer.volumeMm3,layer.qa?.componentCount
        ]);
        assert.deepEqual(identities(before),identities(report),'Source masks, topology, geometry or source volume changed');
        report.baselineMatched=true;
      }
      if(retention&&studyDirectory&&fs.existsSync(path.join(studyDirectory,'meshes')))clone(path.join(studyDirectory,'meshes'),path.join(output,'meshes'));
      report.passed=true;
    }catch(error){
      report.failure=String(error.stack||error);
      report.body=await js('document.body.innerText').catch(()=>null);
      const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true}).catch(()=>null);
      if(image&&!image.isEmpty())fs.writeFileSync(path.join(output,'failure.png'),image.toPNG());
    }finally{
      clearInterval(memory);sampleMemory();
      Object.assign(report,await js('window.surfaceVerification').catch(()=>({})));
      report.sourceUnchanged=sha(source)===sourceHash;
      report.passed=Boolean(report.passed&&report.sourceUnchanged&&!report.windowEverVisible&&!report.errors.length);
      fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));
      console.log(JSON.stringify({passed:report.passed,failure:report.failure,errors:report.errors,orbit:report.orbit}));
      if(!report.passed)process.exitCode=1;
      app.quit();
    }
  });
});
app.on('quit',()=>fs.rmSync(profile,{recursive:true,force:true}));
require(main);
