import test from 'node:test';
import assert from 'node:assert/strict';
import {DataUtils} from 'three';
import {isCtModality,volumeSettings,volumeTransfer,volumeRayInterval,opticalAlpha,mriBoundaryExtinction} from './volumeTransfer.ts';
import {buildOpticalDensity,volumeOpticalAtlas} from './volumeOpticalDensity.ts';
import {inverseAffine} from './coordinates.ts';
import {checkedViewerSettings} from '../workspace-view-settings.ts';
const identity=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
test('volume rays use millimetres on mirrored, anisotropic and oblique grids, including camera inside',()=>{
  assert.deepEqual(volumeRayInterval([-10,1,1],[1,0,0],[3,3,3],identity),[9.5,12.5]);
  assert.deepEqual(volumeRayInterval([1,1,1],[1,0,0],[3,3,3],identity),[0,1.5]);
  assert.equal(volumeRayInterval([-10,5,1],[1,0,0],[3,3,3],identity),null);
  const affine=[[0,-3,0,10],[-2,0,0,20],[0,0,4,30],[0,0,0,1]];
  const interval=volumeRayInterval([0,18,34],[1,0,0],[3,3,3],affine);
  assert.ok(Math.abs(interval[0]-2.5)<1e-12&&Math.abs(interval[1]-11.5)<1e-12);
  assert.deepEqual(affine,[[0,-3,0,10],[-2,0,0,20],[0,0,4,30],[0,0,0,1]]);
});
test('cutaways retain the declared RAS side and work when the ray parallels the plane',()=>{
  assert.deepEqual(volumeRayInterval([-10,1,1],[1,0,0],[3,3,3],identity,[1,0,0,-1]),[11,12.5]);
  assert.deepEqual(volumeRayInterval([-10,1,1],[1,0,0],[3,3,3],identity,[-1,0,0,1]),[9.5,11]);
  assert.equal(volumeRayInterval([-10,1,1],[1,0,0],[3,3,3],identity,[0,1,0,-2]),null);
});
test('opacity is invariant to ray sample spacing and presets retain transparent air/background',()=>{
  const once=opticalAlpha(.3,.7,12),small=opticalAlpha(.3,.7,.5);
  assert.ok(Math.abs(once-(1-(1-small)**24))<1e-12);
  assert.equal(opticalAlpha(.3,0,12),0);
  for(const preset of ['ct-bone','ct-tissue','mri']){
    const lut=volumeTransfer(preset,[0,900]);
    assert.equal(lut.data.length,4096);assert.equal(lut.data[3],0);
    assert.ok(lut.data[4095]>0);
    assert.ok(lut.range[1]>lut.range[0]);
  }
  assert.equal(volumeSettings({},'CT').preset,'ct-bone');
  assert.equal(volumeSettings({},'T1CE').preset,'mri');
  assert.equal(volumeSettings({density:99}).density,2);
});
test('saved volume choices round-trip detached from source arrays and reject malformed settings',()=>{
  const source={compartments:[],brainMask:null,metadata:{}},input={volumeRendering:{enabled:true,preset:'ct-bone',density:.45}};
  const saved=checkedViewerSettings(source,input);assert.deepEqual(saved,input);
  saved.volumeRendering.density=.1;assert.equal(input.volumeRendering.density,.45);
  for(const volumeRendering of [null,[],{enabled:1},{density:NaN},{density:-1},{density:3},{preset:'CTA-vessels'},{arbitrary:1}])
    assert.throws(()=>checkedViewerSettings(source,{volumeRendering}));
});

test('explicit skull modes survive settings validation',()=>{
 const source={compartments:[],brainMask:null,metadata:{}};
 for(const skullMode of ['exterior','transparent','cutaway'])assert.equal(checkedViewerSettings(source,{skullMode}).skullMode,skullMode);
 assert.throws(()=>checkedViewerSettings(source,{skullMode:'guess'}));
});

test('presentation and slice annotations reopen in legacy saves and reject malformed choices',()=>{
  const source={compartments:[],brainMask:null,metadata:{}};
  for(const presentation of ['scan','segmentation','vascular'])for(const sliceOverlay of ['contour','fill']){
    const stored={presentation,sliceOverlay,volumeRendering:{enabled:true,preset:'auto',density:.11}};
    const reopened=checkedViewerSettings(source,JSON.parse(JSON.stringify(stored)));
    assert.deepEqual(reopened,stored);
    reopened.volumeRendering.density=.9;
    assert.equal(stored.volumeRendering.density,.11);
  }
  for(const key of ['presentation','sliceOverlay'])for(const invalid of ['unknown',null,1,[],{}])
    assert.throws(()=>checkedViewerSettings(source,{[key]:invalid}));
});

test('angiography is automatic only for declared CTA, with MRI sequence and segmentation-only behavior',()=>{
  assert.equal(volumeSettings({},'CT').preset,'ct-bone');assert.equal(volumeSettings({},'CTA').preset,'ct-angiography');
  assert.equal(volumeSettings({preset:'ct-angiography'},'CT').preset,'ct-angiography');
  assert.equal(volumeSettings({preset:'ct-angiography'},'T1CE').preset,'mri');
  assert.equal(volumeSettings({},'segmentation only').enabled,false);assert.equal(volumeSettings({},'analytic signal').enabled,false);
  assert.equal(volumeSettings({enabled:true},'generated image signal').enabled,false);
  assert.notDeepEqual(volumeTransfer('mri',[0,900],{},'T2').data,volumeTransfer('mri',[0,900],{},'T1').data);
  const custom={points:[[0,0,0,0,0],[1,.8,.5,.2,.4]]};
  assert.deepEqual(volumeTransfer('mri',[0,900],custom,'T1').data,volumeTransfer('mri',[0,900],custom,'T2').data);
  const source={compartments:[],brainMask:null,metadata:{}},settings={smoothSurfaces:false,volumeRendering:{preset:'ct-angiography',density:.3}};
  assert.deepEqual(checkedViewerSettings(source,settings),settings);assert.throws(()=>checkedViewerSettings(source,{smoothSurfaces:1}));
});

test('CT thresholds require an acquired CT modality even when a CT preset was saved',()=>{
  assert.equal(isCtModality('  CTA'),true);assert.equal(isCtModality('CTAX'),false);assert.equal(isCtModality('T1CE'),false);
  for(const preset of ['ct-bone','ct-tissue','ct-angiography']){
    for(const modality of ['MRI','T1CE','T2 FLAIR','3D TOF MRA'])
      assert.equal(volumeSettings({preset},modality).preset,'mri');
    for(const modality of ['CT','CTA','CT angiography'])
      assert.equal(volumeSettings({preset},modality).preset,preset);
  }
  assert.equal(volumeSettings({},'CT angiography').preset,'ct-angiography');
  assert.equal(volumeSettings({},'T1CE post-contrast').preset,'mri');
  const input={preset:'ct-bone',density:.3},saved=structuredClone(input);
  volumeSettings(input,'T1CE');assert.deepEqual(input,saved,'presentation must not overwrite persisted choices');
});

test('declared bright-blood MRI suppresses background and emphasizes acquired high signal without coloring vessels',()=>{
  const structural=volumeTransfer('mri',[0,900],{},'T1CE');
  const alpha=(lut,x)=>lut.data[Math.round(x*1023)*4+3];
  for(const modality of ['MRA','3D TOF','MRI angiography']){
    assert.equal(volumeSettings({},modality).preset,'mri');
    const lut=volumeTransfer('mri',[0,900],{},modality);
    assert.deepEqual(lut.range,structural.range);
    assert.equal(alpha(lut,0),0);
    assert.ok(alpha(lut,.4)<alpha(structural,.4),'background tissue must recede');
    assert.ok(alpha(lut,.9)>alpha(structural,.9),'bright acquired signal must remain detailed');
    const top=[...lut.data.slice(-4,-1)];
    assert.ok(Math.max(...top)-Math.min(...top)<.15,'signal remains neutral ivory, not an arterial classification');
  }
  assert.deepEqual(volumeTransfer('mri',[0,900],{},'T1CE post-contrast').data,structural.data);
  const points=[[0,0,0,0,0],[1,.8,.5,.2,.4]],custom=structuredClone(points);
  assert.deepEqual(volumeTransfer('mri',[0,900],{points},'MRA').data,volumeTransfer('mri',[0,900],{points},'T1CE').data);
  assert.deepEqual(points,custom);
});

test('transparent MRI context retains precise extinction and emphasizes physical boundaries only for automatic transfers',()=>{
  const automatic=volumeTransfer('mri',[0,900],undefined,'T1CE');
  assert.ok(automatic.data instanceof Float32Array);
  // This low extinction rounded to zero in the byte LUT. It must contribute
  // proportionally at low density instead of appearing as a transfer band.
  const faint=automatic.data[Math.round(.1*1023)*4+3];
  assert.ok(faint>0&&faint<1/510);
  assert.equal(automatic.gradientOpacity,true);
  assert.equal(volumeTransfer('ct-bone',[-1000,1000]).gradientOpacity,false);
  assert.equal(volumeTransfer('mri',[0,900],{points:[[0,0,0,0,0],[1,1,1,1,.4]]}).gradientOpacity,false);
  assert.equal(mriBoundaryExtinction(0),.06);
  assert.equal(mriBoundaryExtinction(.01),.06);
  assert.equal(mriBoundaryExtinction(.075),1);
  assert.ok(mriBoundaryExtinction(.025)<.2);
  // Equivalent physical boundaries on 1mm and 2mm grids have the same
  // derivative after central difference and the affine covector transform.
  assert.equal(mriBoundaryExtinction((.1-0)/2),mriBoundaryExtinction((.2-0)/4));
  const coarse=opticalAlpha(.07*.025*mriBoundaryExtinction(.025),1,200);
  const full=opticalAlpha(.07*.025,1,200);
  assert.ok(coarse<full*.25,'uniform interior context must recede visibly');
});

test('CTA tapers lower HU haze, gives high HU a cooler tint and honors custom transfer points',()=>{
  const lut=volumeTransfer('ct-angiography',[-1000,2000],undefined,'CTA');
  const sample=hu=>lut.data.slice(Math.round((hu+1000)/4000*1023)*4,Math.round((hu+1000)/4000*1023)*4+4);
  assert.deepEqual(lut.range,[-1000,3000]);
  assert.equal(sample(85)[3],0);
  assert.ok(sample(220)[3]<.07);
  const [midR,midG,midB,midExtinction]=sample(360);
  assert.ok(midExtinction>.30&&midExtinction<.35);
  assert.ok(midR>midG&&midG>midB&&midB>.32,'mid HU remains warm beige');
  const [highR,highG,highB,highExtinction]=sample(1800);
  assert.ok(highExtinction>.8&&highB>highG&&highG>highR);
  assert.ok(highR>.90&&highB-highR<.08,'mineral HU remains close to neutral');
  const points=[[0,0,0,0,0],[.5,.8,.2,.1,.2],[1,1,.8,.4,.9]],saved=structuredClone(points);
  assert.deepEqual(volumeTransfer('ct-angiography',[-1000,2000],{points},'CTA').data,
    volumeTransfer('ct-bone',[-1000,2000],{points,window:[-1000,3000]},'CTA').data,'explicit saved colors/extinction take precedence over the automatic preset');
  assert.deepEqual(points,saved);
});

test('coarse shadow density integrates extinction after the nonlinear transfer with source support',()=>{
  const lut=volumeTransfer('ct-angiography',[-1000,3000],undefined,'CTA');
  const signal=new Float32Array([-1000,1000]);
  const source={signal,shape:[2,1,1],affine:identity,inverse:identity};
  const sampled=buildOpticalDensity(source,lut);
  assert.deepEqual(sampled.grid,[1,1,1]);
  // A transfer of the mean HU would produce transparent air/background.
  assert.ok(sampled.fine[0]>.30&&sampled.fine[0]<.32);
  assert.equal(sampled.fine[0],sampled.fine[1]);
  assert.equal(buildOpticalDensity({...source,support:new Uint8Array([0,1])},lut).fine[0],sampled.fine[0]);
  assert.equal(buildOpticalDensity({...source,support:new Uint8Array([1,0])},lut).fine[0],0);
  assert.deepEqual(signal,new Float32Array([-1000,1000]));
});

test('secondary MRI shell density uses the physical derivative from the acquisition affine',()=>{
  const lut=volumeTransfer('mri',[0,1],undefined,'T1CE');
  const source=(spacing, signal)=>{
    const affine=[[0,-2,0,0],[-spacing,0,0,0],[0,0,6,0],[0,0,0,1]];
    return {signal:new Float32Array(signal),shape:[3,1,1],affine,inverse:inverseAffine(affine)};
  };
  const a=buildOpticalDensity(source(4,[.4,.5,.6]),lut);
  const b=buildOpticalDensity(source(8,[.3,.5,.7]),lut);
  assert.deepEqual(a.grid,[3,1,1]);
  assert.ok(Math.abs(a.fine[3]/a.fine[2]-mriBoundaryExtinction(.025))<1e-7);
  assert.ok(Math.abs(a.fine[3]-b.fine[3])<1e-7);
});

test('CT shadow cache includes the filtered skull fraction without changing primary extinction',()=>{
  const lut=volumeTransfer('ct-angiography',[-1000,3000],undefined,'CTA');
  const affine=[[4,0,0,0],[0,4,0,0],[0,0,4,0],[0,0,0,1]];
  const signal=Float32Array.of(1000,1000,1000);
  const source={signal,shape:[3,1,1],affine,inverse:inverseAffine(affine)};
  const mask=Uint8Array.of(0,1,0),weights=[.25,.5,.25];
  const plain=buildOpticalDensity(source,lut);
  const masked=buildOpticalDensity(source,lut,{mask,radius:[.5,.5,.5]});
  for(let cell=0;cell<3;cell++){
    const index=2*cell;
    assert.equal(plain.fine[index],masked.fine[index]);
    assert.equal(plain.coarse[index],masked.coarse[index]);
    assert.ok(Math.abs(masked.fine[index+1]/masked.fine[index]-weights[cell])<1e-6);
  }
  const edge=buildOpticalDensity(source,lut,{mask:Uint8Array.of(1,0,0),radius:[.5,.5,.5]});
  assert.ok(Math.abs(edge.fine[1]/edge.fine[0]-.75)<1e-6,'coverage clamps at the acquisition face');
  assert.deepEqual(mask,Uint8Array.of(0,1,0));
  assert.deepEqual(signal,Float32Array.of(1000,1000,1000));
});

test('transfer row and both optical scales survive atlas packing in source XYZ order',()=>{
  const lut=volumeTransfer('mri',[0,1],undefined,'TOF');
  const affine=[[4,0,0,0],[0,4,0,0],[0,0,4,0],[0,0,0,1]];
  const shape=[56,5,56];
  const signal=new Float32Array(shape.reduce((n,v)=>n*v,1));
  const x=19,y=2,z=27;
  signal[(x*shape[1]+y)*shape[2]+z]=1;
  const source={signal,shape,affine,inverse:inverseAffine(affine)};
  const atlas=volumeOpticalAtlas(lut,source);
  assert.deepEqual(atlas.grid,shape);
  const read=(col,row,channel)=>DataUtils.fromHalfFloat(atlas.data[4*(row*atlas.width+col)+channel]);
  const row=1+Math.floor(x/atlas.columns)*shape[1]+y;
  const col=(x%atlas.columns)*shape[2]+z;
  assert.ok(Math.abs(read(col,row,0)-lut.data.at(-1))<.0005);
  assert.ok(Math.abs(read(col,row,1)-lut.data.at(-1)/8)<.0001);
  assert.equal(read(col+1,row,0),0);
  assert.ok(read(col+1,row,1)>0);
  // Preserve an extinction value below the old 8-bit quantization threshold.
  const faint=lut.data.findIndex((v,i)=>i%4===3&&v>.0007&&v<.001);
  assert.ok(Math.abs(read((faint-3)/4,0,3)-lut.data[faint])<.000001);
  assert.equal(volumeOpticalAtlas(lut).data.byteLength,8192);
});
