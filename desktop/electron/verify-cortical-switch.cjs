'use strict';
const assert=require('node:assert/strict');
const {studyScanSelectionScript}=require('./verify-study-ui-helpers.cjs');

// Observe the texture and uniforms actually bound at indexed cortical draws.
// No test uniforms, material substitutions, or writes to application state.
function observeCorticalDraws(){
  const proto=WebGL2RenderingContext.prototype,uploads=new WeakMap(),programs=new WeakMap();
  for(const name of ['texImage3D','texSubImage3D']){
    const original=proto[name];
    proto[name]=function(...args){
      const data=args.at(-1);
      if(args[0]===this.TEXTURE_3D&&data instanceof Float32Array){
        const texture=this.getParameter(this.TEXTURE_BINDING_3D);
        if(texture)uploads.set(texture,data);
      }
      return original.apply(this,args);
    };
  }
  const draw=proto.drawElements;
  proto.drawElements=function(...args){
    const p=this.getParameter(this.CURRENT_PROGRAM);
    if(!programs.has(p))programs.set(p,this.getUniformLocation(p,'uCorticalScan'));
    const location=programs.get(p);
    if(location&&this.canvas.isConnected){
      const active=this.getParameter(this.ACTIVE_TEXTURE);
      this.activeTexture(this.TEXTURE0+this.getUniform(p,location));
      const data=uploads.get(this.getParameter(this.TEXTURE_BINDING_3D));
      this.activeTexture(active);
      if(data){
        const uniform=name=>Array.from(this.getUniform(p,this.getUniformLocation(p,name)));
        window.readCorticalDraw=async()=>({
          sha256:Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256',data)),v=>v.toString(16).padStart(2,'0')).join(''),
          voxelCount:data.length,worldToVoxel:uniform('uCorticalWorldToVoxel'),
          window:uniform('uCorticalWindow'),modelMatrix:uniform('modelMatrix')
        });
      }
    }
    return draw.apply(this,args);
  };
  document.addEventListener('anatomy-frame',e=>{
    const {cameraState,sourceSignalFingerprint,cursor,displayIdentity}=e.detail;
    window.sourceSwitchFrame={cameraState,sourceSignalFingerprint,cursor,displayIdentity};
  });
}

function visibleCortexPixels(image){
  const data=image.toBitmap(); // Electron stores BGRA bytes.
  let pixels=0;
  for(let i=0;i<data.length;i+=4)
    if(data[i+2]>60&&data[i+2]>data[i+1]+22&&data[i]>data[i+1]+4)pixels++;
  return pixels;
}

function pixelDifference(a,b){
  assert.deepEqual(a.getSize(),b.getSize());
  const left=a.toBitmap(),right=b.toBitmap();
  let change=0,sum=0;
  for(let i=0;i<left.length;i+=4){
    const deltas=[0,1,2].map(c=>Math.abs(left[i+c]-right[i+c]));
    sum+=deltas.reduce((a,b)=>a+b,0);
    if(Math.max(...deltas)>=8)change++;
  }
  return {pixels:left.length/4,pixelsChangedBy8:change,meanAbsoluteChannelDelta:sum/(left.length/4*3)};
}

async function verifyCorticalSwitch({js,until,click,ready,capture,report,document}){
  const initial=document.reference_series;
  const secondary=document.series.find(r=>r.id!==initial&&/^CT/.test(r.modality))??document.series.find(r=>r.id!==initial);
  assert(secondary,'Choose a paired study with cortical labels');
  await js("document.querySelector('.anatomy-presets button:nth-child(2)').click()");
  await ready();
  await click('Show structures only');
  await click('Layout');
  await js("const preset=document.querySelector('select[aria-label=\"Camera preset\"]');preset.value='left';preset.dispatchEvent(new Event('change',{bubbles:true}))");
  await click('Close view controls');
  const views=[],images=[];
  for(const [index,id] of [initial,secondary.id,initial].entries()){
    if(index){
      await js('window.readCorticalDraw=null');
      await js(studyScanSelectionScript(id));
    }
    await until(`JSON.parse(document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity||'[]')[1]===${JSON.stringify(id)} && !!window.readCorticalDraw`);
    await ready();
    const frame=await js('window.sourceSwitchFrame');
    assert.equal(JSON.parse(frame.displayIdentity)[1],id);
    const state=await js(`(async()=>{
      const region=selector=>{const {x,y,width,height}=document.querySelector(selector).getBoundingClientRect();
        return {x:Math.round(x+width*.16),y:Math.round(y+height*.16),width:Math.floor(width*.68),height:Math.floor(height*.68)}};
      return {gpu:await window.readCorticalDraw(),anatomyRegion:region('.rl-viewer-anatomy-image'),
        sliceRegion:region('.rl-viewer-axial .rl-viewer-slice-image'),
        visible:JSON.parse(document.querySelector('.rl-viewer').dataset.visibleLayerIds).sort()};
    })()`);
    const name='switch-'+(index+1)+'-'+id;
    const image=await capture(name);
    images.push({anatomy:image.crop(state.anatomyRegion),slice:image.crop(state.sliceRegion)});
    views.push({id,frame,...state,name,corticalColorPixels:visibleCortexPixels(images.at(-1).anatomy)});
  }
  report.sourceSwitch={views,ctOrSecondary:secondary.modality,
    firstToSecond:{anatomy:pixelDifference(images[0].anatomy,images[1].anatomy),slice:pixelDifference(images[0].slice,images[1].slice)},
    returned:{anatomy:pixelDifference(images[0].anatomy,images[2].anatomy),slice:pixelDifference(images[0].slice,images[2].slice)}};
  const pair=report.sourceSwitch.firstToSecond;
  const primary=report.patientViews[initial],other=report.patientViews[secondary.id];
  assert.equal(primary.contextHash,other.contextHash);
  assert.notEqual(primary.sourceHash,other.sourceHash);
  for(const view of views){
    assert(view.corticalColorPixels>1000,'An opaque foreground surface hides the cortical material');
    assert.equal(view.gpu.sha256,primary.sourceHash);
    assert.deepEqual(view.gpu.worldToVoxel,views[0].gpu.worldToVoxel);
    assert.deepEqual(view.gpu.window,views[0].gpu.window);
    assert.deepEqual(view.visible,views[0].visible);
    for(const key of ['position','up','target'])assert(view.frame.cameraState[key].every((n,i)=>Math.abs(n-views[0].frame.cameraState[key][i])<.001));
  }
  assert.notEqual(views[0].frame.sourceSignalFingerprint,views[1].frame.sourceSignalFingerprint);
  assert(pair.slice.pixelsChangedBy8>100,'Selected slice must visibly change');
  assert(pair.anatomy.meanAbsoluteChannelDelta<2,'3D anatomy changed under source switching');
  assert.equal(report.sourceSwitch.returned.anatomy.meanAbsoluteChannelDelta,0);
  assert.equal(report.sourceSwitch.returned.slice.meanAbsoluteChannelDelta,0);
}
module.exports={observeCorticalDraws,verifyCorticalSwitch};
