'use strict';
// Real Chromium/GPU pixels, isolated preferences, no simulated renderer.
const {app}=require('electron'),fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const output=path.resolve(process.env.ALIGNMENT_VERIFY_OUTPUT||'../outputs/alignment-verification/render');
fs.mkdirSync(output,{recursive:true});
app.setPath('userData',fs.mkdtempSync(path.join(os.tmpdir(),'rslab-alignment-pixels-')));
const pause=ms=>new Promise(r=>setTimeout(r,ms)),report={errors:[],views:[]};
app.on('browser-window-created',(_event,window)=>{
 window.webContents.on('console-message',(_event,d)=>{if(d.level==='error')report.errors.push(d.message)});
 window.webContents.once('did-finish-load',async()=>{
  const js=code=>window.webContents.executeJavaScript(code,true);
  const wait=async(expr)=>{for(let i=0;i<600;i++){if(await js(expr))return;await pause(200)}throw Error('Timed out: '+expr)};
  const ready=()=>wait(`!!document.querySelector('.rl-viewer canvas')&&!document.querySelector('.viewer-loading')&&!document.querySelector('select[aria-label="Source scan"]')?.disabled`);
  const select=async(label,value)=>{await js(`(()=>{const s=document.querySelector('select[aria-label="${label}"]');if(!s)throw Error('Missing ${label}');s.value=${JSON.stringify(value)};s.dispatchEvent(new Event('change',{bubbles:true}))})()`);await pause(200);await ready();await pause(800)};
  const capture=async name=>{
   const rect=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.width),height:Math.floor(r.height)}})()`);
   const image=await window.webContents.capturePage(rect);fs.writeFileSync(path.join(output,name+'.png'),image.toPNG());
   const pixels=image.toBitmap();let pink=0;
   for(let i=0;i<pixels.length;i+=4){const b=pixels[i],g=pixels[i+1],r=pixels[i+2];if(r>70&&r>g*1.3&&r>b*1.1&&b>g*1.15)pink++}
   const record={name,pinkPixels:pink};report.views.push(record);console.log(JSON.stringify(record));return pink;
  };
  try{
   window.webContents.setBackgroundThrottling(false);window.setSize(1440,960);window.show();
   await ready();
   await select('Source scan','ct-bone');
   await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()==='Anatomy');if(!b)throw Error('Missing Anatomy tab');b.click()})()`);
   await wait(`!!document.querySelector('select[aria-label="Skull view"]')`);
   await select('Vessel display','intracranial');
   // Test both mesh and scalar-volume composition from two camera directions.
   for(const representation of ['Structures','Volume']){
    await js(`(()=>{const b=[...document.querySelectorAll('button')].find(b=>b.textContent.trim()===${JSON.stringify(representation)});b.click()})()`);await pause(900);
    for(let direction=0;direction<2;direction++){
     await select('Skull view','exterior');const exterior=await capture(`${representation}-${direction}-exterior`);
     await select('Skull view','transparent');const transparent=await capture(`${representation}-${direction}-transparent`);
     await select('Skull view','cutaway');const cutaway=await capture(`${representation}-${direction}-cutaway`);
     if(transparent<20||cutaway<20||exterior>Math.max(30,transparent*.02))throw Error(`Occlusion failed: ${JSON.stringify({representation,direction,exterior,transparent,cutaway})}`);
     await select('Skull view','exterior');
     const r=await js(`(()=>{const r=document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect();return {x:Math.round(r.x+r.width*.5),y:Math.round(r.y+r.height*.5)}})()`);
     window.webContents.sendInputEvent({type:'mouseDown',...r,button:'left',clickCount:1});
     for(let i=1;i<=30;i++){window.webContents.sendInputEvent({type:'mouseMove',x:r.x+i*5,y:r.y,modifiers:['leftButtonDown']});await pause(20)}
     window.webContents.sendInputEvent({type:'mouseUp',x:r.x+150,y:r.y,button:'left',clickCount:1});await pause(800);
    }
   }
   report.quality=await js(`document.querySelector('.workspace-alignment-banner')?.textContent`);
   report.passed=report.errors.length===0;
  }catch(e){report.passed=false;report.failure=e.stack;fs.writeFileSync(path.join(output,'failure.png'),(await window.webContents.capturePage()).toPNG())}
  fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(report.passed?0:1);
 });
});
require('./main.cjs');
