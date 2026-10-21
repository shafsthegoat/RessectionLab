'use strict';
// Read-only viewer verification. Preferences and patient sessions are not saved.
const {app}=require('electron'),fs=require('node:fs'),os=require('node:os'),path=require('node:path'),crypto=require('node:crypto');
const output=path.resolve(process.env.SURFACE_VERIFY_OUTPUT||'../outputs/surface-quality-verification/render');
fs.mkdirSync(output,{recursive:true});app.setPath('userData',fs.mkdtempSync(path.join(os.tmpdir(),'rslab-surface-quality-')));
const nativeIndex=process.argv.indexOf('--anatomy'),caseIndex=process.argv.indexOf('--case');
const originalSource=path.resolve(process.argv[(nativeIndex>=0?nativeIndex:caseIndex)+1]);
const sourceHash=()=>crypto.createHash('sha256').update(fs.readFileSync(originalSource)).digest('hex');
const sourceBefore=sourceHash();let disposable=null;
// Opening a native study runs the existing automatic alignment workflow.
// Verification therefore uses an APFS-cloned disposable study, never the source.
if(nativeIndex>=0){
  disposable=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-surface-study-'));const directory=path.join(disposable,'study');
  fs.mkdirSync(directory);for(const member of ['patient-anatomy.json','arrays','originals','meshes','revisions','workspace-view.json']){
    const from=path.join(path.dirname(originalSource),member);if(fs.existsSync(from))fs.cpSync(from,path.join(directory,member),{recursive:true,mode:fs.constants.COPYFILE_FICLONE});
  }
  process.argv[nativeIndex+1]=path.join(directory,'patient-anatomy.json');
}
app.on('will-quit',()=>{if(disposable)fs.rmSync(disposable,{recursive:true,force:true});});
const report={startedAt:Date.now(),errors:[],views:[],surface:[],frames:[],gpu:[],refinement:[],memory:[]},pause=ms=>new Promise(r=>setTimeout(r,ms));
app.on('browser-window-created',(_event,window)=>{
  window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message)});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1440,960);window.show();
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expr,seconds=300)=>{for(let i=0;i<seconds*4;i++){if(await js(expr))return;await pause(250)}throw Error('Timed out '+expr)};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing '+${JSON.stringify(label)});b.click()})()`);await pause(200)};
    const ready=async()=>{await wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&document.querySelector('.rl-viewer').dataset.surfacesPending==='0'&&document.querySelector('.native-workspace select[aria-label="Source scan"]')?.disabled!==true`);await pause(500)};
    const capture=async name=>{await ready();fs.writeFileSync(path.join(output,name+'.png'),(await window.webContents.capturePage()).toPNG());report.views.push({name,at:Date.now()});console.log('Captured '+name);};
    try{
      await js(`window.surfaceVerification={surface:[],frames:[],gpu:[],refinement:[]};for(const [event,key]of [['anatomy-surface','surface'],['anatomy-frame','frames'],['anatomy-gpu','gpu'],['anatomy-refined','refinement']])document.addEventListener(event,e=>window.surfaceVerification[key].push(e.detail));`);
      await pause(1000);await ready();
      if(await js(`!!document.querySelector('select[aria-label="Skull view"]')`))await js(`(()=>{const s=document.querySelector('select[aria-label="Skull view"]');s.value='transparent';s.dispatchEvent(new Event('change',{bubbles:true}))})()`);
      await click('Structures');await capture('smooth-oblique');
      await click('Appearance');await js(`(()=>{const c=document.querySelector('input[aria-label="Smooth 3D surfaces"]');if(c.checked)c.click()})()`);await pause(300);await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);await capture('source-oblique');
      await click('Appearance');await js(`document.querySelector('input[aria-label="Smooth 3D surfaces"]').click()`);
      await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);
      await click('Layout');await js(`(()=>{const s=document.querySelector('select[aria-label="Camera preset"]');s.value='anterior';s.dispatchEvent(new Event('change',{bubbles:true}))})()`);await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);await capture('smooth-anterior');
      await click('Volume');await pause(1200);await capture('volume-anterior');
      await click('Appearance');await js(`(()=>{const s=document.querySelector('select[aria-label="Volume rendering preset"]');window.surfaceVerification.presets=[...s.options].map(o=>({value:o.value,disabled:o.disabled}));})()`);await js(`document.querySelector('button[aria-label="Close view controls"]').click()`);
      const nativeCt=await js(`(()=>{const s=document.querySelector('select[aria-label="Source scan"]');return s?[...s.options].find(o=>/ct-brain/i.test(o.value))?.value:null})()`);
      if(nativeCt){
        await click('Anatomy');
        await js(`(()=>{const s=document.querySelector('select[aria-label="Skull view"]');if(s){s.value='transparent';s.dispatchEvent(new Event('change',{bubbles:true}))}})()`);
        await js(`(()=>{const s=document.querySelector('select[aria-label="Source scan"]');s.value=${JSON.stringify(nativeCt)};s.dispatchEvent(new Event('change',{bubbles:true}))})()`);
        await ready();await pause(1200);await capture('ct-volume-anterior');report.ctScan=nativeCt;
      }
      const presented=[];window.webContents.beginFrameSubscription(true,()=>presented.push(performance.now()));
      const center=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5)}})()`);
      window.webContents.sendInputEvent({type:'mouseDown',...center,button:'left',clickCount:1});
      const framesDir=path.join(output,'orbit');fs.mkdirSync(framesDir,{recursive:true});
      for(let i=1;i<=60;i++){
        window.webContents.sendInputEvent({type:'mouseMove',x:center.x+i*2,y:center.y+Math.round(12*Math.sin(i/10)),modifiers:['leftButtonDown']});await pause(16);
        if(i%5===0)fs.writeFileSync(path.join(framesDir,String(i/5).padStart(3,'0')+'.png'),(await window.webContents.capturePage()).toPNG());
      }
      window.webContents.sendInputEvent({type:'mouseUp',x:center.x+120,y:center.y,button:'left',clickCount:1});window.webContents.endFrameSubscription();await pause(2000);
      Object.assign(report,await js('window.surfaceVerification'));
      const gaps=presented.slice(1).map((v,i)=>v-presented[i]).sort((a,b)=>a-b);
      report.rotation={deliveredFrames:presented.length,medianFps:gaps.length?1000/gaps[Math.floor(gaps.length/2)]:null,p95Ms:gaps.length?gaps[Math.floor(gaps.length*.95)]:null,captureOverhead:true};
      const pure=[];window.webContents.beginFrameSubscription(true,()=>pure.push(performance.now()));
      window.webContents.sendInputEvent({type:'mouseDown',...center,button:'left',clickCount:1});
      for(let i=1;i<=90;i++){window.webContents.sendInputEvent({type:'mouseMove',x:center.x-i,y:center.y,modifiers:['leftButtonDown']});await pause(16)}
      window.webContents.sendInputEvent({type:'mouseUp',x:center.x-90,y:center.y,button:'left',clickCount:1});window.webContents.endFrameSubscription();await pause(2000);
      const intervals=pure.slice(1).map((v,i)=>v-pure[i]).sort((a,b)=>a-b);
      report.orbitWithoutScreenshots={deliveredFrames:pure.length,medianFps:intervals.length?1000/intervals[Math.floor(intervals.length/2)]:null,p95Ms:intervals.length?intervals[Math.floor(intervals.length*.95)]:null};
      Object.assign(report,await js('window.surfaceVerification'));
      if(!report.surface.some(s=>!s.empty)&&!process.env.SURFACE_VERIFY_BASELINE)throw Error('No prepared anatomy was observed');
      if(process.env.SURFACE_VERIFY_REHEARSAL){
        await js(`document.querySelector('button[aria-label="Rehearse"]').click()`);await pause(500);
        const prepare=await js(`!![...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Prepare instrument scene')`);
        if(prepare){await click('Prepare instrument scene');await pause(500);}
        await wait(`!!document.querySelector('.rehearsal-anatomy')`);
        await js(`document.querySelector('.rehearsal-anatomy').open=true`);await click('Reference appearance');await ready();await capture('rehearsal-reference');
        report.rehearsalChecked=true;
      }

      report.memory=await app.getAppMetrics();report.passed=report.errors.length===0;
    }catch(error){report.passed=false;report.failure=error.stack;fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG());}
    report.originalSource=originalSource;report.sourceBefore=sourceBefore;report.sourceAfter=sourceHash();report.sourceUnchanged=report.sourceBefore===report.sourceAfter;report.disposableStudy=!!disposable;
    if(!report.sourceUnchanged){report.passed=false;report.failure='Original source file changed during verification';}
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:report.passed,failure:report.failure,errors:report.errors,rotation:report.rotation}));if(disposable){fs.rmSync(disposable,{recursive:true,force:true});disposable=null;}app.exit(report.passed?0:1);
  });
});
require('./main.cjs');
