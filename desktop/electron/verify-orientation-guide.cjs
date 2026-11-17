'use strict';
// Actual application, shared GPU canvas, hidden captures and normal UI actions.
const {app,dialog}=require('electron');
if(process.platform==='darwin')app.setActivationPolicy('accessory');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {verificationDirectory}=require('./verification-output.cjs');
const root=path.resolve(__dirname,'../..'),input=path.resolve(process.env.ORIENTATION_CASE||path.join(root,'testcases/cases/UCSF-PDGM-0007-operative-demo.ressectionlab'));
const output=verificationDirectory(root),reopen=process.env.ORIENTATION_REOPEN==='1';
assert(input.startsWith(path.join(root,'testcases')+path.sep),'Use a disposable or existing local testcases input');
assert(output.startsWith(path.join(root,'testcases')+path.sep)&&!fs.existsSync(output),'Use a fresh testcases output directory');fs.mkdirSync(output,{recursive:true});
const fatal=error=>{console.error(error?.stack||String(error));app.exit(1);};
dialog.showErrorBox=(title,message)=>fatal(Error(title+': '+message));process.on('uncaughtException',fatal);process.on('unhandledRejection',fatal);
const sha=p=>crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex'),original=sha(input);
const native=path.extname(input)==='.rsworkspace',saved=path.join(output,native?'orientation-saved.rsworkspace':'orientation-saved.ressectionlab');
// Native header Save updates the opened portable archive in place. Only open
// a disposable copy; its original bytes/provenance stay at the supplied input.
const workingInput=native?saved:input;if(native)fs.copyFileSync(input,workingInput);
app.setPath('userData',path.join(output,'hidden-profile'));process.argv.push('--case',workingInput,'--automation-hidden');
dialog.showSaveDialog=async()=>({canceled:false,filePath:saved});dialog.showOpenDialog=async()=>({canceled:false,filePaths:[saved]});
const report={input,inputSha256:original,workingInput,output,windowEverVisible:false,captures:[],checks:[],states:[],errors:[]};
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const near=(a,b)=>assert(a.length===b.length&&a.every((n,i)=>Math.abs(n-b[i])<1e-7),`${a} != ${b}`);
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true;window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    window.webContents.setBackgroundThrottling(false);window.setSize(1460,980);
    const js=code=>window.webContents.executeJavaScript(code,true);
    const wait=async(expression,seconds=180)=>{const start=Date.now();while(Date.now()-start<seconds*1000){if(await js(expression))return;await delay(100);}throw Error('Timeout '+expression+'\n'+await js('document.body.innerText'));};
    const click=async label=>{await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(label)}||b.getAttribute('aria-label')===${JSON.stringify(label)});if(!b||b.disabled)throw Error('Missing enabled '+${JSON.stringify(label)});b.click();})()`);await delay(120);};
    const ready=()=>wait(`document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('.viewer-loading')&&!document.querySelector('.rl-viewer-preparing')&&window.orientationFrames?.length>0&&![...document.querySelectorAll('.surgical-panel [role="status"]')].some(n=>/^(Computing instrument action|Preparing an exact recorded state|Preparing accepted display|Verifying saved source-bound rehearsal)/.test(n.textContent.trim()))`);
    const frame=()=>js('window.orientationFrames.at(-1)');
    const capture=async name=>{
      await ready();await delay(250);assert(!window.isVisible());
      const image=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());report.captures.push(name+'.png');
      const state=await js(`(()=>{const e=document.querySelector('.rl-body-guide-image'),r=e.getBoundingClientRect();return {frame:window.orientationFrames.at(-1),bounds:{x:r.x,y:r.y,width:r.width,height:r.height},caption:document.querySelector('[data-guide-caption]').textContent,windowWidth:innerWidth};})()`);
      assert(await js(`(()=>{const a=document.querySelector('.rl-body-guide').getBoundingClientRect(),b=document.querySelector('.operative-readout')?.getBoundingClientRect();return !b||!a.width||a.right<=b.left||a.left>=b.right||a.bottom<=b.top||a.top>=b.bottom;})()`),'Operative readout covers the guide');
      if(state.bounds.width>0){
        const ratio=image.getSize().width/state.windowWidth,b=state.bounds;
        const crop=image.crop({x:Math.round(b.x*ratio),y:Math.round(b.y*ratio),width:Math.round(b.width*ratio),height:Math.round(b.height*ratio)});
        fs.writeFileSync(path.join(output,name+'-guide.png'),crop.toPNG());
        const pixels=crop.toBitmap();let mint=0,neutral=0;
        for(let i=0;i<pixels.length;i+=4){const b=pixels[i],g=pixels[i+1],r=pixels[i+2];if(g>90&&g>r*1.1&&b>r*1.05)mint++;if(r>65&&g>65&&b>65&&Math.max(r,g,b)-Math.min(r,g,b)<45)neutral++;}
        state.guidePixels={mint,neutral};assert(mint>15&&neutral>80,'Guide has no visible mint head and neutral body');
      }
      report.states.push({name,...state});return state;
    };
    const option=async(label,value)=>{await js(`(()=>{const e=[...document.querySelectorAll('select')].find(e=>e.getAttribute('aria-label')===${JSON.stringify(label)});if(!e)throw Error('Missing select');e.value=${JSON.stringify(value)};e.dispatchEvent(new Event('change',{bubbles:true}));})()`);await delay(250);await ready();};
    const checkbox=async value=>{await click('Appearance');await js(`(()=>{const e=document.querySelector('[aria-label="Body orientation guide"]');if(e.checked!==${value})e.click();})()`);await click('Close view controls');await delay(250);await ready();};
    try{
      await js(`window.orientationFrames=[];document.addEventListener('anatomy-frame',e=>{window.orientationFrames.push(e.detail);if(window.orientationFrames.length>200)window.orientationFrames.shift();});`);
      await ready();
      if(reopen){
        assert.equal(await js('document.querySelector(".rl-body-guide").hidden'),true);assert.equal((await frame()).orientationGuide,null);
        report.checks.push('Cold reopen preserves hidden guide');
        await checkbox(true);await capture('01-cold-reopened-enabled');
      }else{
        await click('Anatomy');await ready();await checkbox(true);
        for(const preset of ['anterior','left','superior','oblique']){
          await click('Layout');await option('Camera preset',preset);await click('Close view controls');
          const state=await capture('01-'+preset);assert(state.frame.orientationGuide);
          if(preset==='anterior'){
            const sides=await js(`[...document.querySelectorAll('[data-guide-side]')].map(e=>({side:e.dataset.guideSide,left:parseFloat(e.style.left)}))`);
            assert(sides.find(s=>s.side==='right').left<sides.find(s=>s.side==='left').left,'Front view must show patient right on viewer left');
          }
        }
        report.checks.push('Anterior, lateral, superior and oblique guide rendered with correct patient sidedness');
        const point=await js(`(()=>{const e=document.querySelector('.rl-body-guide-image'),r=e.getBoundingClientRect(),x=Math.round(r.x+r.width/2),y=Math.round(r.y+r.height/2);return {x,y,passes:!!document.elementFromPoint(x,y)?.closest('.rl-viewer-anatomy-image')&&!document.elementFromPoint(x,y)?.closest('.rl-body-guide')};})()`);assert(point.passes,'Guide intercepts mouse input');
        const before=await frame();window.webContents.sendInputEvent({type:'mouseDown',x:point.x,y:point.y,button:'left',clickCount:1});
        window.webContents.sendInputEvent({type:'mouseMove',x:point.x+30,y:point.y+15,button:'left'});window.webContents.sendInputEvent({type:'mouseUp',x:point.x+30,y:point.y+15,button:'left',clickCount:1});await delay(250);await ready();
        const orbited=await frame();assert.notDeepEqual(orbited.orientationGuide.cameraQuaternion,before.orientationGuide.cameraQuaternion);
        window.webContents.sendInputEvent({type:'mouseWheel',x:point.x,y:point.y,deltaY:80,canScroll:true});await delay(350);await ready();
        const zoomed=await frame();near(zoomed.orientationGuide.cameraQuaternion,orbited.orientationGuide.cameraQuaternion);assert.notDeepEqual(zoomed.cameraState.position,orbited.cameraState.position);
        report.checks.push('Drag through guide rotates brain and guide; wheel changes brain zoom without changing guide orientation');
        await click('Layout');await click('Four views');await capture('02-four-views');
        await js(`document.querySelector('.rl-viewer-axial .rl-viewer-expand').click()`);await delay(250);await ready();assert.equal((await frame()).orientationGuide,null);report.checks.push('Expanded source slice submits no guide draw');await click('Restore linked views');
        await click('Layout');await click('3D + slices');window.setSize(1050,720);await delay(300);const compact=await capture('03-compact');assert(compact.bounds.width>=104&&compact.bounds.height>0);window.setSize(1460,980);await delay(250);
        if(!native){
          await click('Rehearse');await wait(`!!document.querySelector('[aria-label="Surgical branch"]')`,300);await ready();
          const branches=await js(`[...document.querySelector('[aria-label="Surgical branch"]').options].map(o=>o.value)`);
          let postures=0;
          for(const [branch,label] of [['position-supine','Supine'],['position-semisitting','Semisitting']]){
            if(!branches.includes(branch))continue;
            await option('Surgical branch',branch);await ready();
            const end=await js(`Number(document.querySelector('[aria-label="Surgical timeline"]').max)`);
            await js(`(()=>{const e=document.querySelector('[aria-label="Surgical timeline"]');Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(e,${end});e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));})()`);await delay(400);await ready();
            const state=await capture('04-'+branch);assert.equal(state.frame.orientationGuide.posture,label);assert(state.frame.orientationGuide.down);assert.equal(state.caption,label);
            postures++;
          }
          if(postures)report.checks.push('Recorded surgical posture and gravity shown from displayed replay state');
          await click('Surgeon / Microscope');await ready();window.setSize(1050,720);await delay(300);await capture('04-compact-microscope');
          assert(await js(`!!document.querySelector('.operative-readout')`));report.checks.push('Compact microscope guide stays clear of the operative readout');window.setSize(1460,980);await delay(250);
        }
        if(process.env.ORIENTATION_RECOVER==='1'){
          await js(`document.querySelector('.rl-viewer canvas').getContext('webgl2').getExtension('WEBGL_lose_context').loseContext()`);
          await wait(`[...document.querySelectorAll('button')].some(b=>b.textContent==='Retry graphics')`);await click('Retry graphics');
          await wait(`![...document.querySelectorAll('button')].some(b=>b.textContent==='Retry graphics')`);await ready();await capture('05-recovered-graphics');assert((await frame()).orientationGuide);
          report.checks.push('Graphics context recovery recreates the guide');
        }
        await checkbox(false);assert.equal((await frame()).orientationGuide,null);assert.equal(await js('document.querySelector(".rl-body-guide").hidden'),true);
        await click('Save');await wait(`document.body.innerText.includes(${JSON.stringify(native?'Workspace saved · scans, layers and views preserved':'Workspace saved locally')})`);assert(fs.existsSync(saved));report.saved=saved;
        report.checks.push('Appearance hides guide and normal Save writes a disposable workspace');
      }
      report.inputPreserved=sha(input)===original;assert(report.inputPreserved);assert(!report.windowEverVisible);
      report.passed=true;fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify({passed:true,output,checks:report.checks}));app.exit(0);
    }catch(error){report.errors.push(error.stack||String(error));report.body=await js('document.body.innerText').catch(()=>null);await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true}).then(image=>fs.writeFileSync(path.join(output,'failure.png'),image.toPNG())).catch(()=>{});fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));fatal(error);}
  });
});
if(process.env.ORIENTATION_APP_BUNDLE){
  const bundle=path.resolve(process.env.ORIENTATION_APP_BUNDLE),resources=path.join(bundle,'Contents/Resources');
  assert(fs.existsSync(path.join(resources,'app.asar'))&&fs.existsSync(path.join(resources,'research-engine/ressectionlab-engine')));
  Object.defineProperty(process,'resourcesPath',{value:resources});Object.defineProperty(app,'isPackaged',{value:true});app.setAppPath(path.join(resources,'app.asar'));
  report.bundle=bundle;require(path.join(resources,'app.asar/electron/main.cjs'));
}else require('./main.cjs');
