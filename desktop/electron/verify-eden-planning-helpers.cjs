'use strict';
const assert=require('node:assert/strict');
const crypto=require('node:crypto');

// Use native setters to bypass React's value tracker, then dispatch the event
// appropriate to the control. Both events make the browser interaction explicit.
function inputScript(label,value){return `(()=>{const e=[...document.querySelectorAll('[aria-label]')].find(n=>n.getAttribute('aria-label')===${JSON.stringify(label)});if(!e||e.disabled)throw Error('Missing enabled control '+${JSON.stringify(label)});if(e.tagName!=='SELECT'&&e.tagName!=='INPUT')throw Error('Unsupported control');for(let p=e.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;const value=${JSON.stringify(String(value))};if(e.tagName==='SELECT'&&![...e.options].some(o=>o.value===value&&!o.disabled))throw Error('Missing enabled option '+value);e.scrollIntoView({block:'center'});Object.getOwnPropertyDescriptor(e.tagName==='SELECT'?HTMLSelectElement.prototype:HTMLInputElement.prototype,'value').set.call(e,value);e.dispatchEvent(new Event('input',{bubbles:true}));e.dispatchEvent(new Event('change',{bubbles:true}));return e.value})()`;}
function enabledControlScript(label){return `(()=>{const e=[...document.querySelectorAll('[aria-label]')].find(n=>n.getAttribute('aria-label')===${JSON.stringify(label)});return !!e&&!e.disabled})()`;}
function displayIdentity(value){
  const identity=JSON.parse(value);assert(Array.isArray(identity)&&identity.length===6,'Malformed native source display identity');
  assert(identity.slice(0,4).every(v=>typeof v==='string'&&v.length>0),'Missing source identity members');
  assert.equal(typeof identity[4],'boolean');assert(Number.isInteger(identity[5]));
  return {collectionId:identity[0],seriesId:identity[1],sha256:identity[2],snapshotId:identity[3],original:identity[4],generation:identity[5]};
}
function pixelSummary(bitmap){
  assert(bitmap.length>0&&bitmap.length%4===0);let neutral=0,min=255,max=0;const signal=[];
  for(let i=0;i<bitmap.length;i+=4){const low=Math.min(bitmap[i],bitmap[i+1],bitmap[i+2]),high=Math.max(bitmap[i],bitmap[i+1],bitmap[i+2]);const gray=high-low<=18&&low>=24&&high<=246;
    const value=gray?Math.round((bitmap[i]+bitmap[i+1]+bitmap[i+2])/3):0;signal.push(value);if(gray){neutral++;min=Math.min(min,value);max=Math.max(max,value);}}
  return {sha256:crypto.createHash('sha256').update(bitmap).digest('hex'),neutralPixels:neutral,dynamicRange:neutral?max-min:0,signal};
}
function pixelDifference(a,b){
  assert.equal(a.signal.length,b.signal.length,'Source crop dimensions changed');let comparable=0,changed=0,totalChange=0;
  for(let i=0;i<a.signal.length;i++){if(a.signal[i]||b.signal[i]){comparable++;const delta=Math.abs(a.signal[i]-b.signal[i]);totalChange+=delta;if(delta>=8)changed++;}}
  return {comparable,changed,changedFraction:comparable?changed/comparable:0,meanAbsoluteDifference:comparable?totalChange/comparable:0};
}
function sourceVoxel(affine,point){
  assert(Array.isArray(affine)&&affine.length===4&&point.length===3,'Missing source affine');
  const [[a,b,c],[d,e,f],[g,h,i]]=affine,det=a*(e*i-f*h)-b*(d*i-f*g)+c*(d*h-e*g);
  assert(Number.isFinite(det)&&Math.abs(det)>1e-12,'Singular source affine');
  const inverse=[[e*i-f*h,c*h-b*i,b*f-c*e],[f*g-d*i,a*i-c*g,c*d-a*f],[d*h-e*g,b*g-a*h,a*e-b*d]],delta=point.map((v,j)=>v-affine[j][3]);
  return inverse.map(row=>row.reduce((sum,v,j)=>sum+v*delta[j],0)/det);
}
function geometry(a){return {id:a.id,entry:a.entry,target:a.target,radiusMm:a.radiusMm,entryPatch:a.entryPatch,parameterHash:a.parameterHash,measurementHash:a.measurementHash,measurements:a.measurements,assessmentStatus:a.assessmentStatus};}
module.exports={inputScript,enabledControlScript,displayIdentity,pixelSummary,pixelDifference,sourceVoxel,geometry};
