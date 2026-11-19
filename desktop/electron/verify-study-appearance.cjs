'use strict';
// Electron SCRIPT disposable-native-manifest target-layer output-dir [--cold]
const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const {execFileSync}=require('node:child_process');
const fs=require('node:fs'),os=require('node:os'),path=require('node:path');
const {app,dialog}=require('electron');
const {inputScript,displayIdentity}=require('./verify-eden-planning-helpers.cjs');
const [nativeArg,target,outputArg]=process.argv.slice(2);
const native=fs.realpathSync(nativeArg),output=path.resolve(outputArg);
const testing=fs.realpathSync(path.resolve(__dirname,'../../testcases/outputs'))+path.sep;
assert(native.startsWith(testing)&&output.startsWith(testing),'Use a disposable native study under testcases/outputs');
const cold=process.argv.includes('--cold'),stage=cold?'restored':'saved';
const saved=path.join(output,'appearance.rsworkspace'),receipt=path.join(output,stage+'.json');
fs.mkdirSync(output,{recursive:true});
assert(!fs.existsSync(receipt)&&(cold||!fs.existsSync(saved)),'Choose a fresh output directory');
const manifest=JSON.parse(fs.readFileSync(native)),targetInfo=manifest.layers.find(l=>l.id===target);
assert(targetInfo,'Missing target layer');
const expected=cold?JSON.parse(fs.readFileSync(path.join(output,'saved.json'))):null;
const digest=file=>createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const scratch=fs.mkdtempSync(path.join(os.tmpdir(),'study-appearance-'));
const report={pid:process.pid,stage,windowEverVisible:false,manifestSha256:digest(native)};
app.setPath('userData',scratch);
if(process.platform==='darwin')app.setActivationPolicy('accessory');
process.argv.push('--automation-hidden',cold?'--case':'--anatomy',cold?saved:native);
delete process.env.RESSECTIONLAB_DEV_URL;
dialog.showSaveDialog=async(...args)=>{
  assert.equal(args.at(-1).title,'Save portable patient study');
  assert(!fs.existsSync(saved));return {canceled:false,filePath:saved};
};
const pause=ms=>new Promise(resolve=>setTimeout(resolve,ms));
function nearCamera(actual,reference){
  const error=Math.max(...['position','target','up'].flatMap(key=>actual[key].map((x,i)=>Math.abs(x-reference[key][i]))));
  assert(error<1e-7,'Saved camera differs from submitted camera by '+error);return error;
}
app.on('browser-window-created',(_,win)=>{
  win.on('show',()=>{report.windowEverVisible=true;win.hide();});
  win.webContents.once('did-finish-load',async()=>{
    const js=expr=>win.webContents.executeJavaScript(expr,true);
    const wait=async expr=>{
      const start=Date.now();
      while(Date.now()-start<120000){const value=await js(expr);if(value)return value;await pause(125);}
      throw Error('Timed out waiting for '+expr+'\n'+await js('document.body.innerText.slice(-2200)'));
    };
    const click=async label=>{
      const expr='[...document.querySelectorAll("button")].find(b=>!b.disabled&&(b.textContent.trim()==='+JSON.stringify(label)+'||b.getAttribute("aria-label")==='+JSON.stringify(label)+'))';
      await wait(expr+'!==undefined');console.log('Selecting',label);
      await js('(()=>{const b='+expr+';for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==="DETAILS")p.open=true;b.click()})()');
    };
    const snap=async(width,height)=>{
      win.setContentSize(width,height);await wait('innerWidth==='+width+'&&innerHeight==='+height);
      await state();await pause(500);
      assert(!win.isVisible());const shot=await win.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(output,stage+'-'+width+'.png'),shot.resize({width,height}).toPNG());
      const panes=await js(`[...document.querySelectorAll('.rl-viewer-anatomy-image,.rl-viewer-slice-image')].map(e=>{const r=e.getBoundingClientRect();return {x:Math.ceil(r.x),y:Math.ceil(r.y),width:Math.floor(r.right)-Math.ceil(r.x),height:Math.floor(r.bottom)-Math.ceil(r.y)}})`);
      (report.layout??={})[width]=panes;
    };
    const state=async()=>{
      const value=await wait(`(()=>{const v=document.querySelector('.rl-viewer'),raw=document.querySelector('.native-workspace')?.dataset.nativeViewState;
        if(!raw||v?.dataset.rendererReady!=='true'||v.dataset.surfacesPending!=='0'||document.querySelector('.viewer-loading,.rl-viewer-preparing,[data-layer-pending=true]'))return false;
        const pair=window.__appearanceFrames.findLast(p=>p.gpu?.completed&&p.frame.displayIdentity===v.dataset.sourceDisplayIdentity&&!p.frame.interactive&&p.frame.selectedStructure===${JSON.stringify(target)});
        return pair&&{identity:v.dataset.sourceDisplayIdentity,view:JSON.parse(raw),visible:JSON.parse(v.dataset.visibleLayerIds),frame:pair.frame,gpu:pair.gpu};})()`);
      const {frame,gpu,view}=value,source=displayIdentity(value.identity),scan=view.states[view.seriesId],row=manifest.series.find(s=>s.id===view.seriesId);
      assert.equal(source.snapshotId,manifest.snapshot_id);assert.equal(source.seriesId,row.id);
      const asset=[row.data,...(row.preview_pyramid??[]).map(level=>level.data)].find(d=>d.shape.join()===frame.sourceShape.join());
      assert(asset,'Rendered grid is missing from the source study');
      const file=path.join(path.dirname(native),asset.path);
      assert.equal(digest(file),asset.sha256);
      const python=path.resolve(__dirname,'../../.venv/bin/python');
      const rawSha=execFileSync(python,['-c','import hashlib,numpy as np,sys; a=np.load(sys.argv[1],mmap_mode="r"); print(hashlib.sha256(np.ascontiguousarray(a,dtype="<f4").tobytes()).hexdigest())',file],{encoding:'utf8'}).trim();
      assert.equal(source.sha256,rawSha);
      assert.equal(view.selectedStructure,target);
      assert.equal(scan.settings.layerColor[target],'#82c7ba');assert.equal(scan.settings.layerOpacity[target],.81);
      assert.equal(scan.visible[report.hiddenLayer],false);
      assert.deepEqual([...value.visible].sort(),Object.keys(scan.visible).filter(id=>scan.visible[id]).sort());
      assert.equal(frame.sourceSignalFingerprint,gpu.sourceSignalFingerprint);
      return {source,visible:value.visible.sort(),view,frame};
    };
    try {
      win.webContents.setBackgroundThrottling(false);win.setContentSize(1440,900);
      await js(`window.__appearanceFrames=[];
        document.addEventListener('anatomy-frame',({detail:f})=>{const records=window.__appearanceFrames;
          records.push({frame:{serial:f.serial,displayIdentity:f.displayIdentity,sourceShape:f.sourceShape,sourceSignalFingerprint:f.sourceSignalFingerprint,cameraState:f.cameraState,cursor:f.cursor,selectedStructure:f.selectedStructure,interactive:f.interactive,drawCalls:f.drawCalls}});
          if(records.length>100)records.shift();});
        document.addEventListener('anatomy-gpu',({detail:g})=>{const p=window.__appearanceFrames.findLast(x=>x.frame.serial===g.serial&&x.frame.displayIdentity===g.displayIdentity);if(p)p.gpu=g;});`);
      await click('Anatomy');
      await wait('!!document.querySelector('+JSON.stringify('.native-structure-row[data-structure-id='+JSON.stringify(target)+']')+')');
      if(cold){
        assert(expected.passed&&expected.pid!==process.pid);report.hiddenLayer=expected.hiddenLayer;
        assert.equal(digest(saved),expected.archiveSha256);
      }else{
        await click('Focus '+targetInfo.label);
        await js(inputScript(targetInfo.label+' opacity',.81));await js(inputScript(targetInfo.label+' color','#82c7ba'));
        const other=await wait(`(()=>{const r=[...document.querySelectorAll('.native-structure-row:not(.is-hidden)')].find(e=>e.dataset.structureId!==${JSON.stringify(target)}&&e.dataset.structureId!=='brain'&&e.querySelector('.native-structure-visibility:not(:disabled)'));return r&&{label:r.dataset.structureLabel,id:r.dataset.structureId}})()`);
        report.hiddenLayer=other.id;await click('Hide '+other.label);
        await state();await click('Scans');await click('Save patient study file…');
        await wait("document.querySelector('footer')?.innerText.includes('Patient study file saved')");await click('Anatomy');
      }
      await snap(1440,900);report.display=await state();
      const archived=JSON.parse(execFileSync('unzip',['-p',saved,'workspace-view.json'])),active=archived.states[archived.seriesId];
      assert.equal(archived.selectedStructure,target);assert.equal(archived.snapshotId,manifest.snapshot_id);
      assert.equal(archived.selectedApproachId,report.display.view.selectedApproachId);
      assert.deepEqual(active.visible,report.display.view.states[archived.seriesId].visible);
      assert.deepEqual(active.settings.layerOpacity,report.display.view.states[archived.seriesId].settings.layerOpacity);
      assert.deepEqual(active.settings.layerColor,report.display.view.states[archived.seriesId].settings.layerColor);
      report.archiveCameraErrorMm=nearCamera(report.display.frame.cameraState,active.settings.camera);report.archiveSha256=digest(saved);
      if(cold){
        assert.deepEqual(report.display.visible,expected.display.visible);
        report.restoredCameraErrorMm=nearCamera(report.display.frame.cameraState,expected.display.frame.cameraState);
        assert.equal(report.display.frame.sourceSignalFingerprint,expected.display.frame.sourceSignalFingerprint);
        assert.deepEqual(report.display.frame.cursor,expected.display.frame.cursor);assert.equal(report.manifestSha256,expected.manifestSha256);
      }
      await snap(1050,720);assert(!report.windowEverVisible);report.passed=true;
    }catch(error){report.passed=false;report.error=String(error.stack??error);console.error(report.error);}
    fs.writeFileSync(receipt,JSON.stringify(report,null,2)+'\n');app.exit(report.passed?0:1);
  });
});
app.on('will-quit',()=>fs.rmSync(scratch,{recursive:true,force:true}));
require('./main.cjs');
