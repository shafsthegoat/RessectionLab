import assert from 'node:assert/strict';
import {test} from 'node:test';
import {drawsOnSlices,isSliceTargetLayer,sliceFocusFlags,sliceLayerBits,sliceOutlineColor,sliceOutlineWidth,
  sliceOverlayStyle,sliceStructureScope} from './sliceOverlay.ts';

const layers=[
  {name:'brain',kind:'brain',color:'#b8a8b6'},
  {name:'tumor',kind:'tumor',color:'#65bfd3'},
  {name:'skull',kind:'bone',color:'#ece2ca'},
  {name:'arteries',kind:'artery',color:'#aa352b'},
  {name:'edema',kind:'edema',color:'#cb9e60'},
  {name:'cortex',kind:'brain-region',displayKind:'gray-matter',color:'#c69891'},
  {name:'enhancing_tumor',color:'#e5a269'},
];

test('slices default to the target and the selected structure; saved views keep their explicit choice',()=>{
  assert.equal(sliceStructureScope(undefined),'focus');
  assert.equal(sliceStructureScope({}),'focus');
  assert.equal(sliceStructureScope({sliceStructures:'focus'}),'focus');
  assert.equal(sliceStructureScope({sliceStructures:'all'}),'all');
  assert.equal(sliceOverlayStyle({}),'contour','views without a saved style use outlines');
  assert.equal(sliceOverlayStyle({sliceOverlay:'fill'}),'fill');
  assert.equal(sliceOverlayStyle({sliceOverlay:'contour'}),'contour');
});

test('declared lesion kinds and undeclared supplied annotations are slice targets; context is not',()=>{
  assert.deepEqual(layers.map(isSliceTargetLayer),[false,true,false,false,true,false,true]);
  for(const kind of ['lesion','tumor-core','tumor-whole','enhancing-tumor','nonenhancing-tumor','lesion-context','synthetic-target'])
    assert.equal(isSliceTargetLayer({name:kind,kind}),true,kind);
  for(const kind of ['vessels','vein','sinus','dura','csf','ventricles','head-envelope','outer-head-tissue'])
    assert.equal(isSliceTargetLayer({name:kind,kind}),false,kind);
  // A layer's name or colour is not evidence of a lesion.
  assert.equal(isSliceTargetLayer({name:'tumor_mask',kind:'brain'}),false);
});

test('only target and selected layers draw on slices unless all structures are requested',()=>{
  assert.deepEqual(sliceFocusFlags(layers,null),[false,true,false,false,true,false,true]);
  assert.deepEqual(sliceFocusFlags(layers,'arteries'),[false,true,false,true,true,false,true]);
  assert.deepEqual(layers.map(layer=>drawsOnSlices(layer,'skull',{})),[false,true,true,false,true,false,true]);
  assert.deepEqual(layers.map(layer=>drawsOnSlices(layer,null,{sliceStructures:'all'})),layers.map(()=>true));
});

test('slice focus packs into four 32-bit words for all 128 layer slots',()=>{
  const flags=Array.from({length:130},(_,i)=>[0,1,31,32,63,64,127,128,129].includes(i));
  const bits=sliceLayerBits(flags);
  assert.deepEqual([...bits],[(1|2|(1<<31))|0,(1|(1<<31))|0,1,(1<<31)|0]);
  const reuse=new Int32Array([7,7,7,7]);
  assert.equal(sliceLayerBits([false,true],reuse),reuse);
  assert.deepEqual([...reuse],[2,0,0,0]);
  const unpack=(words,i)=>((words[i>>5]>>(i&31))&1)===1;
  flags.slice(0,128).forEach((on,i)=>assert.equal(unpack(bits,i),on,`layer ${i}`));
});

test('slice outlines use the colour the 3D surface shows, including user overrides',()=>{
  const cta={displayIdentity:'native/revision',modality:'CTA'};
  const vascular={presentation:'vascular',vesselShading:'cta-contrast'};
  assert.equal(sliceOutlineColor(layers[1],{},cta),'#65bfd3');
  assert.equal(sliceOutlineColor(layers[1],{layerColor:{tumor:'#112233'}},cta),'#112233');
  assert.equal(sliceOutlineColor(layers[3],vascular,cta),'#a32e22','CTA contrast shading colours arteries by type in 3D and 2D');
  assert.equal(sliceOutlineColor(layers[3],{...vascular,vesselShading:'source',layerColor:{arteries:'#445566'}},cta),'#445566');
  assert.equal(sliceOutlineColor(layers[3],{...vascular,presentation:'scan'},cta),'#aa352b','Scan keeps the layer palette in 3D and 2D');
  assert.equal(sliceOutlineColor(layers[3],vascular,{modality:'CTA'}),'#aa352b','legacy cases have no CTA vessel finish');
  assert.equal(sliceOutlineColor(layers[3],vascular,cta,true),'#aa352b','operative vessels keep their label colour');
});

test('outline width is about 1.5 CSS pixels in whole device pixels',()=>{
  assert.equal(sliceOutlineWidth(2),1.5);
  assert.equal(sliceOutlineWidth(1),2);
  assert.equal(sliceOutlineWidth(1.5)*1.5,2);
  assert.equal(sliceOutlineWidth(3),5/3);
  for(const invalid of [0,-1,Number.NaN,Infinity])assert.equal(sliceOutlineWidth(invalid),2);
});
