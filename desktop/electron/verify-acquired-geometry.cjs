'use strict';

const assert = require('node:assert/strict');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');

// Installed only inside hidden acceptance runs. Fragment derivatives use the
// rendered triangle positions after deformation. Material colours, texture,
// vertex normals, lighting and shadows contribute nothing to this comparator.
function observeGeometry(){
  const probe=window.acquiredGeometry={active:false,draws:0,scenes:new Map()};
  const proto=WebGL2RenderingContext.prototype,source=proto.shaderSource,locations=new WeakMap();
  proto.shaderSource=function(shader,code){
    if(this.getShaderParameter(shader,this.SHADER_TYPE)===this.FRAGMENT_SHADER &&
       code.includes('varying vec3 vViewPosition;') && code.includes('void main()') && code.includes('gl_FragColor')){
      const at=code.lastIndexOf('}');
      code=code.slice(0,at)+`
        if(uQaGeometricFaces){
          vec3 face=normalize(cross(dFdx(vViewPosition),dFdy(vViewPosition)));
          if(face.z<0.0)face=-face;
          float value=0.20+0.75*max(0.0,dot(face,normalize(vec3(-0.49,0.58,0.65))));
          gl_FragColor=vec4(vec3(value),1.0);
        }
      `+code.slice(at);
      code=code.replace('void main()','uniform bool uQaGeometricFaces;\nvoid main()');
    }
    return source.call(this,shader,code);
  };
  for(const key of ['drawArrays','drawElements']){
    const draw=proto[key];
    proto[key]=function(...args){
      const target=this.canvas.closest('.operative-field-canvas') ||
        this.canvas.closest('.rl-viewer')?.querySelector('[data-accepted-tools][data-operative-view="surgeon"]');
      if(!probe.active || !target)return draw.apply(this,args);
      const program=this.getParameter(this.CURRENT_PROGRAM);
      if(!locations.has(program))locations.set(program,this.getUniformLocation(program,'uQaGeometricFaces'));
      const flag=locations.get(program);
      if(!flag)return draw.apply(this,args);
      this.uniform1i(flag,1);probe.draws++;
      try{return draw.apply(this,args);}finally{this.uniform1i(flag,0);}
    };
  }
  const events=window.__THREE_DEVTOOLS__??=new EventTarget();
  events.addEventListener('observe',({detail:renderer})=>{
    if(!renderer.isWebGLRenderer)return;
    const render=renderer.render.bind(renderer);
    renderer.render=(scene,camera)=>{
      if(probe.active && renderer.domElement.isConnected &&
         (renderer.domElement.closest('.operative-field-canvas')||
          renderer.domElement.closest('.rl-viewer')?.querySelector('[data-accepted-tools]'))){
        probe.scenes.set(scene.uuid,{scene,camera});
      }
      return render(scene,camera);
    };
  });
  probe.buffers=async()=>{
    const rows=[];
    async function digest(array){
      if(!array)return null;
      const bytes=new Uint8Array(array.buffer,array.byteOffset,array.byteLength);
      const hash=await crypto.subtle.digest('SHA-256',bytes);
      return Array.from(new Uint8Array(hash),v=>v.toString(16).padStart(2,'0')).join('');
    }
    for(const {scene,camera} of probe.scenes.values()){
      const wanted=[];
      scene.traverseVisible(m=>{if(m.isMesh&&(m.userData.compartment||m.userData.sourceLayer))wanted.push(m);});
      for(const mesh of wanted){
        const g=mesh.geometry, pos=g.getAttribute('position');
        if(!pos?.count)continue;
        rows.push({name:mesh.userData.compartment??mesh.name,source:mesh.userData.sourceLayer?.directId,
          vertices:pos.count,triangles:(g.index?.count??pos.count)/3,
          positionSHA256:await digest(pos.array),indexSHA256:await digest(g.index?.array),
          world:mesh.matrixWorld.toArray(),projection:camera.projectionMatrix.toArray()});
      }
    }
    return rows;
  };
}

async function captureGeometry({win,js,until,capture,name,width,height}){
  await js('window.acquiredGeometry.scenes.clear();window.acquiredGeometry.draws=0;window.acquiredGeometry.active=true');
  // Resize back to the exact saved dimensions to request the ordinary complete
  // render, including the engine's normal viewport/scissor and compositing.
  async function redraw(){
    win.setContentSize(width+1,height);await until('innerWidth==='+String(width+1));
    win.setContentSize(width,height);await until('innerWidth==='+width);
    await js('new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))');
    await js('if(window.__operativeQA?.pinned)window.__operativeQA.paint(window.__operativeQA.pinned)');
  }
  try{
    await redraw();
    await until('window.acquiredGeometry.draws>0',15000);
    await capture(name+'-faces');
    const meshes=await js('window.acquiredGeometry.buffers()');
    assert(meshes.length,'No owned surface vertices submitted for the geometry pass');
    return {draws:await js('window.acquiredGeometry.draws'),meshes};
  }finally{
    await js('window.acquiredGeometry.active=false;window.acquiredGeometry.scenes.clear()');
    await redraw();
  }
}

async function captureOpening({win,js,until,capture,report,reference}){
  report.opening={};
  for(const [width,height] of [[1440,900],[1050,720]]){
    win.setContentSize(width,height);await until('innerWidth==='+width+' && innerHeight==='+height);
    if(reference){
      const rect=reference.opening['open-'+width].viewport;
      await js(`(()=>{const field=document.querySelector('.operative-field-canvas'),rect=${JSON.stringify(rect)};
        field.style.transform='';field.style.width=rect.width+'px';field.style.height=rect.height+'px';
        const b=field.getBoundingClientRect();
        field.style.transform='translate('+(rect.left-b.left)+'px,'+(rect.top-b.top)+'px)';
      })()`);
      await js('new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))');
    }
    for(const [label,time] of [['open',2],['retracted',3],['endpoint',4]]){
      await js(inputScript('Operative timeline',time));
      await until('Number(document.querySelector(\'[aria-label="Operative timeline"]\').value)==='+time);
      const name=label+'-'+width;
      await js('new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))');
      const prior=reference?.opening[name];
      if(prior)await js('window.__operativeQA.paint('+JSON.stringify(prior.submitted)+')');
      await capture(name);
      const field=await js(`(()=>{const f=document.querySelector('.operative-field-canvas'),panel=f.closest('.operative-playback');
        return {camera:JSON.parse(f.querySelector('canvas').dataset.camera),viewport:f.getBoundingClientRect().toJSON(),
          sources:JSON.parse(f.dataset.surroundingSources),removed:Number(panel.querySelector('[data-removed-cells]')?.dataset.removedCells??0),
          mode:panel.dataset.fieldView,phase:panel.dataset.operativePhase,progress:panel.dataset.operativeProgress};})()`);
      assert.equal(field.mode,'operative');
      const submitted=await js('window.__operativeQA.lastDraw');
      if(prior){
        assert.deepEqual(field.viewport,prior.viewport);
        assert.equal(field.removed,prior.removed);
        for(const key of ['matrixWorld','projectionMatrix'])assert.deepEqual(submitted[key],prior.submitted[key]);
      }
      report.opening[name]={...field,submitted,diagnostic:await captureGeometry({win,js,until,capture,name,width,height})};
      if(prior)await js('window.__operativeQA.paint(null)');
    }
    if(reference){
      report.retraction??={};
      const frozen=reference.opening['open-'+width].submitted;
      // Replay to blade contact, full hold and the identical contact state.
      // The final seek releases the displayed displacement without removing cells.
      for(const [label,time] of [['contact',2.16],['held',3],['released',2.16]]){
        await js(inputScript('Operative timeline',time));
        await until('Number(document.querySelector(\'[aria-label="Operative timeline"]\').value)==='+time);
        await js('new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))');
        await js('window.__operativeQA.paint('+JSON.stringify(frozen)+')');
        const name='retraction-'+label+'-'+width;
        await capture(name);
        const record=await js(`(()=>{const p=window.__operativeQA,f=document.querySelector('.operative-field-canvas');
          let cortex;
          p.last.scene.traverseVisible(mesh=>{if(mesh.userData.sourceLayer?.cortex)cortex=mesh;});
          const panel=f.closest('.operative-playback');
          return {registeredSource:cortex?.userData.sourceLayer?.directId,
            influence:cortex?.morphTargetInfluences?.[0],camera:p.lastDraw,viewport:f.getBoundingClientRect().toJSON(),
            removed:Number(panel.querySelector('[data-removed-cells]')?.dataset.removedCells??0),
            phase:panel.dataset.operativePhase,progress:panel.dataset.operativeProgress};})()`);
        assert(record.registeredSource,'Missing registered gray matter');
        assert.equal(record.removed,0);
        assert(Math.abs(record.influence-(label==='held'?1:0))<1e-9);
        for(const key of ['matrixWorld','projectionMatrix'])assert.deepEqual(record.camera[key],frozen[key]);
        report.retraction[name]=record;
      }
      await js('window.__operativeQA.paint(null)');
      await js("(()=>{const f=document.querySelector('.operative-field-canvas');f.style.width='';f.style.height='';f.style.transform=''})()");
    }
  }
}

module.exports={observeGeometry,captureGeometry,captureOpening};
