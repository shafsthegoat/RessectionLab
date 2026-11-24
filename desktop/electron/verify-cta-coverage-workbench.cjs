'use strict';

// Exercise the ordinary intake, coverage controls and portable workspace in a hidden window.
const {app,dialog,ipcMain}=require('electron');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {inputScript}=require(path.join(process.env.CTA_COVERAGE_DESKTOP??__dirname+'/..','electron/verify-eden-planning-helpers.cjs'));
const config=JSON.parse(fs.readFileSync(process.env.CTA_COVERAGE_CONFIG,'utf8'));
const phase=config.phase;
assert(['create','reopen'].includes(phase));
assert(!fs.existsSync(config.output)&&!fs.existsSync(config.profile));
if(phase==='create')assert(!fs.existsSync(config.workspace)&&!fs.existsSync(config.package));
fs.mkdirSync(config.output,{recursive:true});
fs.mkdirSync(config.profile,{recursive:true});
if(phase==='create')fs.mkdirSync(config.workspace,{recursive:true});
app.setPath('userData',config.profile);
if(process.platform==='darwin')app.setActivationPolicy('accessory');
process.argv.push('--automation-hidden');
delete process.env.RESSECTIONLAB_DEV_URL;

const report={phase,passed:false,windowEverVisible:false,captures:{},operations:[],signedRuntime:{isPackaged:app.isPackaged,executable:process.execPath}};
let selected;
dialog.showOpenDialog=async()=>{
  const result=selected;
  selected=undefined;
  assert(result?.length,'Unexpected open dialog');
  return {canceled:false,filePaths:result};
};
dialog.showSaveDialog=async()=>({canceled:false,filePath:config.package});
const originalHandle=ipcMain.handle.bind(ipcMain);
ipcMain.handle=(channel,handler)=>originalHandle(channel,async(event,...args)=>{
  try{
    const result=await handler(event,...args);
    if(['research:anatomyCreate','research:anatomyImportBatch','research:anatomyCtaArterialCoverage','research:anatomySavePackage'].includes(channel))
      report.operations.push({channel,snapshotId:result?.document?.snapshot_id,layerId:result?.layerId,counts:result?.counts});
    return result;
  }catch(error){report.operations.push({channel,error:String(error)});throw error;}
});

const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const stem=filename=>path.basename(filename).replace(/\.nii(?:\.gz)?$/i,'');
app.on('browser-window-created',(_event,window)=>{
  window.on('show',()=>{report.windowEverVisible=true; window.hide();});
  window.webContents.once('did-finish-load',async()=>{
    const js=code=>window.webContents.executeJavaScript(code,true);
    const select=label=>`[...document.querySelectorAll('select')].find(element=>element.getAttribute('aria-label')===${JSON.stringify(label)})`;
    const button=label=>`[...document.querySelectorAll('button')].find(element=>element.getAttribute('aria-label')===${JSON.stringify(label)}||element.textContent.trim()===${JSON.stringify(label)}||element.querySelector('span')?.textContent.trim()===${JSON.stringify(label)})`;
    const wait=async(test,name)=>{
      const deadline=Date.now()+300000;
      while(Date.now()<deadline){if(await test())return;await delay(120);}
      throw Error(`Timed out waiting for ${name}`);
    };
    const click=async label=>{
      const control=button(label);
      await wait(()=>js(`!!(${control})&&!(${control}).matches(':disabled')`),label);
      await js(`(${control}).click()`);
    };
    const state=()=>js(`JSON.parse(document.querySelector('.native-workspace')?.dataset.nativeViewState||'null')`);
    const frame=()=>js('window.ctaWorkbenchFrame');
    const ready=async()=>{
      await wait(()=>js(`document.querySelector('.rl-viewer')?.dataset.rendererReady==='true'&&
        document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&
        !document.querySelector('.viewer-loading,[data-layer-pending="true"]')&&window.ctaWorkbenchFrame?.cameraState`),'settled CTA viewer');
      await delay(450);
    };
    const pressed=async label=>wait(()=>js(`(${button(label)})?.getAttribute('aria-pressed')==='true'`),label+' selection');
    const details=()=>js(`document.querySelector('[data-cta-coverage-panel]').open=true`);
    const capture=async name=>{
      await ready();
      assert(!window.isVisible());
      const shot=await window.webContents.capturePage(undefined,{stayHidden:true,stayAwake:true});
      fs.writeFileSync(path.join(config.output,name+'.png'),shot.toPNG());
      const current=await state();
      const currentFrame=await frame();
      report.captures[name]={state:current,signal:currentFrame.sourceSignalFingerprint,cursor:currentFrame.cursor,camera:currentFrame.cameraState};
      return report.captures[name];
    };
    const sameGeometry=(a,b)=>{
      assert.equal(a.signal,b.signal);
      assert.deepEqual(a.cursor,b.cursor);
      assert.deepEqual(a.state.patientCutaway,b.state.patientCutaway);
      assert.deepEqual(a.state.states[a.state.seriesId].settings.clipping,b.state.states[b.state.seriesId].settings.clipping);
      for(const key of ['position','target','up'])assert(a.camera[key].every((value,i)=>Math.abs(value-b.camera[key][i])<1e-8));
    };
    try{
      report.signedRuntime.loadedUrl=window.webContents.getURL();
      await js(`document.addEventListener('anatomy-frame',event=>window.ctaWorkbenchFrame=event.detail)`);
      if(phase==='create'){
        await click('Cases');
        await click('New patient study');
        await js(inputScript('New patient ID',config.patient));
        selected=[config.workspace];
        await click('Create study…');
        await wait(async()=> (await state())?.patientId===config.patient,'new native workspace');
        await click('Scans');
        selected=[config.cta];
        await click('Add NIfTI files…');
        const sequence=`Sequence for ${stem(config.cta)}`;
        await wait(()=>js(`!!(${select(sequence)})`),'CTA intake role');
        await js(inputScript(sequence,'CTA'));
        selected=config.annotations.map(item=>item.path);
        await click('Add segmentations…');
        for(const annotation of config.annotations){
          const source=`Source scan for ${stem(annotation.path)}`;
          await wait(()=>js(`!!(${select(source)})`),source);
          const sourceId=await js(`(${select(source)}).options[1].value`);
          await js(inputScript(source,sourceId));
          const item=`(${select(source)}).closest('.intake-item')`;
          for(let i=0;i<annotation.labels.length;i++){
            if(i)await js(`Array.from((${item}).querySelectorAll('button')).find(b=>b.textContent==='Add label').click()`);
            for(const [control,value] of [['label',annotation.labels[i].value],['role',annotation.labels[i].role],['name',annotation.labels[i].name]])
              await js(`(()=>{const section=${item};${inputScript(`Segmentation ${control} ${i+1}`,value).replace('document.querySelectorAll','section.querySelectorAll')}})()`);
          }
        }
        await js(`document.querySelector('input[aria-label="Confirm patient and timepoint"]').click()`);
        await click('Import selected data');
        await wait(()=>report.operations.some(row=>row.channel==='research:anatomyImportBatch'&&!row.error),'committed original CTA and both annotation files');
        await click('3D workspace');
        await ready();
        await click('Clip');
        await js(`Array.from(document.querySelectorAll('label')).find(e=>e.textContent.trim()==='Cutaway').querySelector('input').click()`);
        await js(inputScript('Cutaway position',config.cutMm));
        await click('Flip side');
        await click('Close view controls');
        await ready();
        const manifest=JSON.parse(fs.readFileSync(path.join(config.workspace,'patient-anatomy.json'),'utf8'));
        const roles=new Map(config.annotations.flatMap(annotation=>annotation.labels.map(label=>[label.name,label.coverage])));
        await details();
        for(const layer of manifest.layers){
          if(!roles.get(layer.label))continue;
          await js(inputScript(`Arterial coverage role for ${layer.label}`,roles.get(layer.label)));
        }
        const before={state:await state(),signal:(await frame()).sourceSignalFingerprint,cursor:(await frame()).cursor,camera:(await frame()).cameraState};
        await click('Build combined arteries');
        await pressed('Combined arteries');
        const combined=await capture('combined');
        sameGeometry(before,combined);
        const result=report.operations.find(row=>row.channel==='research:anatomyCtaArterialCoverage'&&!row.error);
        assert.equal(result.counts.focus_voxels,config.focusVoxels);
        assert.equal(result.counts.wider_voxels,config.widerVoxels);
        assert.equal(result.counts.wider_only_voxels,config.additionalVoxels);
        await click('Focused branches');
        await pressed('Focused branches');
        sameGeometry(combined,await capture('focused'));
        await details();
        await click('Update combined arteries');
        await pressed('Combined arteries');
        assert.equal((await state()).snapshotId,result.snapshotId);
        await click('Focused branches');
        await pressed('Focused branches');
        await click('Scans');
        await click('Save patient study file…');
        await wait(()=>report.operations.some(row=>row.channel==='research:anatomySavePackage'&&!row.error),'portable CTA package');
      }else{
        await wait(async()=>(await state())?.patientId===config.patient,'cold portable CTA workspace');
        await click('3D workspace');
        await pressed('Focused branches');
        const focused=await capture('reopened-focused');
        await click('Combined arteries');
        await pressed('Combined arteries');
        sameGeometry(focused,await capture('reopened-combined'));
      }
      report.passed=!report.windowEverVisible;
    }catch(error){
      report.error=error.stack??String(error);
      report.intake=await js(`document.querySelector('.import-data')?.innerText??null`).catch(()=>null);
      try{const image=await window.webContents.capturePage(undefined,{stayHidden:true});fs.writeFileSync(path.join(config.output,'failure.png'),image.toPNG());}catch{}
    }finally{
      fs.writeFileSync(path.join(config.output,'report.json'),JSON.stringify(report,null,2)+'\n');
      if(!report.passed)console.error(report.error);
      app.exit(report.passed?0:1);
    }
  });
});
if(!app.isPackaged)require('./main.cjs');
