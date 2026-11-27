'use strict';

// Open the ordinary practice workspace. All edits below use Chromium input
// events; the IPC observer records native requests and responses without changing them.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const {app, dialog, ipcMain} = require('electron');
const {setTimeout:delay} = require('node:timers/promises');
const source = fs.realpathSync(process.argv[2]), output = path.resolve(process.argv[3]);
const editPath=process.argv.includes('--path');
fs.mkdirSync(output,{recursive:true});
const reportPath=path.join(output,'guided-ui.json'), edited=path.join(output,'guided-edit.ressectionlab');
assert(!fs.existsSync(reportPath)&&!fs.existsSync(edited),'Select a fresh output directory');
const scratch=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-guided-'));
const record={source,windowEverVisible:false,actions:[],previews:[],screens:[]};
let latest,win;
const nativeHandle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(channel,handler)=>nativeHandle(channel,async(...args)=>{
  const result=await handler(...args);
  if(channel==='research:surgicalRehearsal'&&result?.view){
    if(args[1].command==='preview'){
      record.preview={status:result.status,stateId:result.view.stateId,request:args[1].action};
      record.previews.push(record.preview);
    }
    else if(!result.playback){
      latest=result;
      if(args[1].command==='step')record.actions.push({request:args[1].action,
        status:result.outcome?.status,stateId:result.view.stateId,cue:result.walkthroughCue});
    }
  }
  return result;
});
app.setPath('userData',path.join(scratch,'profile'));
if(process.platform==='darwin')app.setActivationPolicy('accessory');
process.argv.push('--case',source,'--automation-hidden');
delete process.env.RESSECTIONLAB_DEV_URL;
dialog.showSaveDialog=async()=>({filePath:edited,canceled:false});
const write=()=>fs.writeFileSync(reportPath,JSON.stringify(record,null,2)+'\n');
app.on('browser-window-created',(_,window)=>{
  win=window;
  win.on('show',()=>{record.windowEverVisible=true;win.hide();});
  win.webContents.once('did-finish-load',async()=>{
    const wc=win.webContents, js=code=>wc.executeJavaScript(code,true);
    const byText=label=>`[...document.querySelectorAll('button,summary')].find(e=>e.textContent.trim()===${JSON.stringify(label)})`;
    const field=label=>`document.querySelector('[aria-label="${label}"]')`;
    async function wait(check,description,ms=55000){
      const start=performance.now();
      while(performance.now()-start<ms){if(await check())return;await delay(55);}
      throw Error('Timed out waiting for '+description);
    }
    async function clickExpr(expr){
      await wait(()=>js(`!!(${expr})&&!(${expr}).matches(':disabled')`),expr);
      let parent=true;
      while(parent){
        const point=await js(`(()=>{const e=${expr};let target=e;for(let p=e.parentElement;p;p=p.parentElement)if(p.matches('details:not([open])')&&p.querySelector(':scope>summary')!==e)target=p.querySelector(':scope>summary');target.scrollIntoView({block:'center',behavior:'instant'});const r=target.getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2,parent:target!==e}})()`);
        await wc.debugger.sendCommand('Input.dispatchMouseEvent',{type:'mousePressed',x:point.x,y:point.y,button:'left',buttons:1,clickCount:1});
        await wc.debugger.sendCommand('Input.dispatchMouseEvent',{type:'mouseReleased',x:point.x,y:point.y,button:'left',buttons:0,clickCount:1});
        parent=point.parent;await delay(55);
      }
    }
    async function capture(name){
      await js('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))');
      assert(!win.isVisible()); const image=await wc.capturePage(undefined,{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(output,name+'.png'),image.resize({width:1080}).toPNG());
      record.screens.push(name);
    }
    async function fillNumber(label,n){
      await clickExpr(field(label));
      await wc.debugger.sendCommand('Input.dispatchKeyEvent',{type:'keyDown',key:'a',code:'KeyA',
        modifiers:process.platform==='darwin'?4:2,commands:['selectAll']});
      await wc.debugger.sendCommand('Input.dispatchKeyEvent',{type:'keyUp',key:'a',code:'KeyA'});
      await wc.debugger.sendCommand('Input.insertText',{text:String(n)});
      assert.equal(await js(`Number((${field(label)}).value)`),n);
    }
    async function displayed(id){
      await wait(()=>js(`document.querySelector('.surgical-panel')?.dataset.stateId===${JSON.stringify(id)}
        && document.querySelector('[data-accepted-tools]')?.dataset.surgicalStateId===${JSON.stringify(id)}
        && document.querySelector('.rl-viewer')?.dataset.rendererReady==='true'
        && window.guidedPracticeGpuState===${JSON.stringify(id)}`),'rendered native boundary');
    }
    const control=()=>js("document.querySelector('[aria-label=\"Instrument practice\"]')?.dataset??null");
    const ready=()=>js("document.querySelector('[aria-label=\"Instrument practice\"]')?.getAttribute('aria-busy')==='false' && document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'");
    try{
      wc.debugger.attach('1.3');await wc.debugger.sendCommand('Emulation.setFocusEmulationEnabled',{enabled:true});
      await js("document.addEventListener('anatomy-gpu',({detail})=>{if(detail.completed)window.guidedPracticeGpuState=detail.surgicalStateId})");
      win.setSize(1380,880);assert(!win.isVisible());
      await clickExpr(byText('Rehearse'));
      await wait(()=>js("!!document.querySelector('[aria-label=\"Surgical timeline\"]')"),'surgical workspace',150000);
      for(const name of ['Verify saved replay','Verify replay for planning','Rehearse/edit stage']){
        if(await js(`Boolean((${byText(name)})&&!((${byText(name)}).disabled))`))await clickExpr(byText(name));
      }
      await wait(()=>Boolean(latest?.walkthroughCue?.status==='ready'),'valid native guide');
      await wait(ready,'accepted operative view');const initial=latest.view.stateId,guide=latest.walkthroughCue;
      record.initial={stateId:initial,cue:guide};
      await wait(()=>js("!!document.querySelector('[aria-label=\"Guided rehearsal\"]')"),'guide controls');
      const actual=await js("({label:document.querySelector('.walkthrough-cue h3').textContent,duration:Number(document.querySelector('[aria-label=\"Guided maneuver duration\"]').value)})");
      assert.equal(actual.label,guide.label);assert.equal(actual.duration,guide.action.duration??1);
      await clickExpr(`[...document.querySelectorAll('button')].find(e=>e.textContent.trim().startsWith('Grip ·'))`);
      const draft=await control();assert.notEqual(draft.practiceAcceptedPosition,draft.practiceProposedPosition);
      assert.equal(draft.practiceStateId,initial);assert.equal(record.actions.length,0);
      record.landmark={accepted:draft.practiceAcceptedPosition,proposed:draft.practiceProposedPosition};
      await clickExpr(byText('Discard'));await clickExpr(byText('Preview maneuver'));
      await wait(()=>!!record.preview,'native preview');assert.equal(latest.view.stateId,initial);
      assert.deepEqual(record.preview.request,guide.action);assert.equal(record.preview.status,'accepted');
      await wait(()=>js(`!!(${byText('Dismiss preview')})`),'visible preview');
      record.performEnabledDuringPreview=await js(`!(${byText('Perform maneuver')}).disabled`);
      assert.equal(record.performEnabledDuringPreview,false);
      await displayed(record.preview.stateId);await capture('01-guided-preview');
      await clickExpr(byText('Dismiss preview'));await wait(ready,'dismissed preview');
      await clickExpr(byText('Perform maneuver'));await wait(()=>record.actions.length===1,'recorded guided action');
      await wait(ready,'committed geometry');assert.deepEqual(record.actions[0].request,guide.action);
      assert.equal(record.actions[0].stateId,record.preview.stateId);assert.equal(latest.walkthroughCue.position,guide.position+1);
      await displayed(record.actions[0].stateId);await capture('02-guided-next');
      let cue=latest.walkthroughCue;
      while(cue.status==='ready'&&(editPath?!(cue.action.path?.length>1):!cue.action.pose&&!cue.action.path?.length)){
        assert(cue.position<12,'Select an operative guide with an early editable maneuver');
        const count=record.actions.length;
        await clickExpr(byText('Perform maneuver'));await wait(()=>record.actions.length===count+1,'next exact cue');
        await wait(ready,'next accepted surface');
        assert.deepEqual(record.actions.at(-1).request,cue.action);cue=latest.walkthroughCue;
      }
      assert.equal(cue.status,'ready');
      const tip=cue.action.path?.at(-1)??cue.action.pose;
      record.originalEditedCue=cue;
      const altered=Number(cue.action.duration??1)+.06, changedR=Number((tip.position[0]+.2).toFixed(4));
      const component=field('Guided tip R');
      assert.equal(await js(`Number((${component}).value)`),tip.position[0]);
      await fillNumber('Guided maneuver duration',altered);
      await fillNumber('Guided tip R',changedR);
      const prior=latest.view.stateId, actionCount=record.actions.length;
      await clickExpr(byText('Preview maneuver'));await wait(()=>record.preview?.request.duration===altered,'edited native preview');
      const revised=record.preview.request.path?.at(-1)??record.preview.request.pose;
      assert.deepEqual(revised.position,[changedR,...tip.position.slice(1)]);
      assert.deepEqual(revised.quaternion,tip.quaternion);
      assert.deepEqual(record.preview.request.path?.slice(0,-1),cue.action.path?.slice(0,-1));
      assert.equal(record.preview.status,'accepted');assert.equal(latest.view.stateId,prior);
      assert.equal(actionCount,record.actions.length);
      await displayed(record.preview.stateId);
      await clickExpr(byText('Dismiss preview'));await wait(ready,'dismiss edited preview');
      await clickExpr(byText('Perform maneuver'));await wait(()=>record.actions.length===actionCount+1,'edited native action');
      assert.equal(record.actions.at(-1).status,'accepted');assert.deepEqual(record.actions.at(-1).request,record.preview.request);
      assert.equal(record.actions.at(-1).stateId,record.preview.stateId);
      assert.equal(record.actions.at(-1).cue.status,'modified');await wait(ready,'edited geometry');
      assert(await js("document.body.innerText.includes('Your episode follows a different path')"));
      win.setSize(1050,720);await displayed(record.actions.at(-1).stateId);await capture('03-edited-compact');
      await clickExpr(byText('Save'));await wait(()=>fs.existsSync(edited),'edited practice save');
      record.saved=edited;record.passed=true;assert(!record.windowEverVisible);write();app.exit(0);
    }catch(e){
      record.error=String(e.stack||e);record.visibleText=(await js('document.body.innerText').catch(()=>'' )).slice(-6500);
      console.error(record.error);
      await capture('failure').catch(()=>{});write();app.exit(1);
    }
  });
});
app.on('quit',()=>fs.rmSync(scratch,{recursive:true,force:true}));
require('./main.cjs');
