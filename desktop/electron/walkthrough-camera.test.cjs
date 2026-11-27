'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');

test('a parked shaft on opposite sides of the same opening changes the optical azimuth',async()=>{
  const {clearWorkingView}=await import('./walkthrough-camera.mjs');
  const az=-80*Math.PI/180,tilt=35*Math.PI/180,d=160;
  const direction=[Math.sin(tilt)*Math.sin(az),-Math.sin(tilt)*Math.cos(az),Math.cos(tilt)];
  const target=[32,32,61],position=target.map((v,i)=>v+d*direction[i]);
  const options={camera:{target,position},azimuthDeg:-80,fieldMm:64,aspect:16/9,outward:[0,0,1],
    targets:[[32,37,88]]};
  const slot=x=>({position:[x,32,85],quaternion:[1,0,0,0],inserted:false,contactGeometry:[
    {a:[0,0,0],b:[0,0,-40],radiusMm:2,part:'shaft'},
    {a:[0,0,-40],b:[0,0,-120],radiusMm:7,part:'handle'}]});
  const left=clearWorkingView({...options,slots:{left:slot(0)}});
  const right=clearWorkingView({...options,slots:{right:slot(84)}});
  assert.equal(left.azimuthDeg,80);assert.equal(right.azimuthDeg,-80);
  assert(left.frameMargin>0&&right.frameMargin>0);
  assert(left.parkedClearanceMm>10&&right.parkedClearanceMm>10);
});

test('a clear field retains the authored camera and both support sites',async()=>{
  const {clearWorkingView}=await import('./walkthrough-camera.mjs');
  const result=clearWorkingView({camera:{target:[0,0,0],position:[0,-60,120]},azimuthDeg:0,
    fieldMm:65,aspect:16/9,outward:[0,0,1],targets:[[-13,0,0],[13,0,0]],slots:{},
    supportPair:[[-13,0,0],[13,0,0]]});
  assert.equal(result.azimuthDeg,0);assert.equal(result.parkedClearanceMm,null);
  assert(Math.abs(result.supportSeparationMm-26)<1e-9);assert(result.frameMargin>.5);
});
