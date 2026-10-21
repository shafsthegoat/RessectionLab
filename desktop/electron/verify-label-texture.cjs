'use strict';
// Separate real-WebGL correctness check, deliberately outside primary timings.
// Run with Electron. All generated bundles and evidence stay under testcases/.
const {app,BrowserWindow}=require('electron');
const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const root=path.resolve(__dirname,'../..');
const output=path.resolve(process.env.LABEL_GPU_OUTPUT||path.join(root,'testcases/performance/rehearsal-2026-10-19/label-gpu'));
if(!output.startsWith(path.join(root,'testcases')+path.sep))throw Error('Keep generated GPU evidence under testcases/');
fs.mkdirSync(output,{recursive:true});
app.setPath('userData',path.join(output,'user-data'));
const source=`
import * as THREE from ${JSON.stringify(path.join(root,'desktop/node_modules/three/build/three.module.js'))};
import {LabelTextureUpdates} from ${JSON.stringify(path.join(root,'desktop/src/viewer/labelTextureUpdates.ts'))};
export function verify(){
  const renderer=new THREE.WebGLRenderer({antialias:false});
  const shape=[65,3,5],width=shape[2],height=shape[1]*shape[0];
  const data=Uint8Array.from({length:width*height*4},(_,i)=>(i*17+3)%256);
  const texture=new THREE.Data3DTexture(data,width,shape[1],shape[0]);
  texture.format=THREE.RGBAFormat;texture.type=THREE.UnsignedByteType;texture.unpackAlignment=1;texture.needsUpdate=true;
  const target=new THREE.WebGLRenderTarget(width,height,{type:THREE.UnsignedByteType,format:THREE.RGBAFormat,depthBuffer:false});
  const material=new THREE.RawShaderMaterial({glslVersion:THREE.GLSL3,uniforms:{uData:{value:texture},uHeight:{value:shape[1]}},
    vertexShader:'precision highp float;in vec3 position;void main(){gl_Position=vec4(position,1.);}',
    fragmentShader:'precision highp float;precision highp sampler3D;uniform sampler3D uData;uniform int uHeight;out vec4 color;void main(){int row=int(gl_FragCoord.y);color=texelFetch(uData,ivec3(int(gl_FragCoord.x),row%uHeight,row/uHeight),0);}' });
  const geometry=new THREE.PlaneGeometry(2,2),scene=new THREE.Scene(),camera=new THREE.Camera();scene.add(new THREE.Mesh(geometry,material));
  const updates=new LabelTextureUpdates(),checks=[],snapshots=[];
  const check=name=>{
    renderer.setRenderTarget(target);renderer.render(scene,camera);const actual=new Uint8Array(data.length);
    renderer.readRenderTargetPixels(target,0,0,width,height,actual);
    const mismatch=actual.findIndex((value,i)=>value!==data[i]);
    if(mismatch!==-1)throw Error(name+' GPU mismatch at '+mismatch+': '+actual[mismatch]+' != '+data[mismatch]);
    const error=renderer.getContext().getError();if(error)throw Error(name+' GL error '+error);
    checks.push(name);snapshots.push({...updates.work});
  };
  const patch=indices=>{const values=Uint8Array.from({length:indices.length*4},(_,i)=>(255-i*7)%256);indices.forEach((index,i)=>data.set(values.subarray(i*4,i*4+4),index*4));return {indices:Uint32Array.from(indices),values};};
  try{
    updates.sync(renderer,texture);check('initial full upload');
    for(const [name,indices] of [['first cell',[0]],['interior RGBA cell',[(2*shape[1]+1)*width+2]],['last cell',[width*height-1]],['separate dirty regions',[1,width*height-2]]]){
      if(updates.update(renderer,texture,shape,patch(indices),false)!=='partial')throw Error('Expected partial upload for '+name);
      updates.sync(renderer,texture);check(name);
    }
    updates.update({initTexture:renderer.initTexture.bind(renderer)},texture,shape,patch([7]),false);updates.sync(renderer,texture);check('unavailable copy API full fallback');
    let copies=0;
    const failing={initTexture:renderer.initTexture.bind(renderer),copyTextureToTexture(...args){if(++copies===2)throw Error('Injected copy failure');renderer.copyTextureToTexture(...args);}};
    updates.update(failing,texture,shape,patch([0,width*height-1]),false);updates.sync(renderer,texture);check('mid-batch failure complete upload');
    return {passed:true,checks,shape,rgbaBytes:data.length,work:{...updates.work},snapshots,three:THREE.REVISION,renderer:renderer.getContext().getParameter(renderer.getContext().RENDERER),meaning:'Actual WebGL2 texelFetch/readPixels comparison of every RGBA byte against CPU full image; separate correctness run, not performance timing'};
  }finally{target.dispose();geometry.dispose();material.dispose();texture.dispose();renderer.dispose();renderer.forceContextLoss();}
}`;
const fixture=path.join(output,'fixture.ts');fs.writeFileSync(fixture,source);
app.whenReady().then(async()=>{
  let window;
  try{
    const {build}=await import('vite');
    const result=await build({configFile:false,logLevel:'error',build:{write:false,minify:false,lib:{entry:fixture,name:'LabelGpuCheck',formats:['iife']}}});
    const bundle=(Array.isArray(result)?result:[result]).flatMap(item=>item.output).find(item=>item.type==='chunk').code;
    window=new BrowserWindow({width:500,height:300,show:false,webPreferences:{nodeIntegration:false,contextIsolation:true,sandbox:true}});
    await window.loadURL('data:text/html,<title>Label GPU correctness check</title>');
    const report=await window.webContents.executeJavaScript(bundle+';LabelGpuCheck.verify()',true);
    report.recordedAt=new Date().toISOString();
    report.helperSha256=crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'desktop/src/viewer/labelTextureUpdates.ts'))).digest('hex');
    fs.writeFileSync(path.join(output,'report.json'),JSON.stringify(report,null,2));console.log(JSON.stringify(report));app.exit(0);
  }catch(error){fs.writeFileSync(path.join(output,'failure.txt'),String(error.stack));console.error(error);app.exit(1);}
});
