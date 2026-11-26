import assert from 'node:assert/strict';
import {test} from 'node:test';
import {acceptedToolTip,createCursorFollower,sourceTargetMeasure} from '../src/tool-follow.ts';
import {formatLabelDistance} from '../src/viewer/toolNavigation.ts';

function cursorStore(initial){
  let cursor=initial;const writes=[];
  return {read:()=>cursor,write:p=>{cursor=p;writes.push(p);},move:p=>{cursor=p;},writes,get cursor(){return cursor;}};
}

test('Follow tool moves the linked cursor to each new tip and restores the user place when it stops',()=>{
  const store=cursorStore([1,2,3]),follower=createCursorFollower(store.read,store.write);
  follower.follow([10,20,30],'study');
  assert.deepEqual(store.cursor,[10,20,30]);
  follower.follow([10,20,30],'study');
  assert.equal(store.writes.length,1,'an unchanged tip does not rewrite the cursor');
  follower.follow([11,20,30],'study');
  assert.deepEqual(store.cursor,[11,20,30]);
  follower.stop('study');
  assert.deepEqual(store.cursor,[1,2,3],'stopping returns to the pre-follow cursor');
  follower.follow([5,5,5],'study');follower.follow([NaN,0,0],'study');
  assert.deepEqual(store.cursor,[5,5,5],'non-finite tips are ignored');
});

test('A cursor the user moved, or one from another study, is never overwritten on stop',()=>{
  const store=cursorStore([1,2,3]),follower=createCursorFollower(store.read,store.write);
  follower.follow([10,20,30],'study');
  store.move([4,4,4]);
  follower.stop('study');
  assert.deepEqual(store.cursor,[4,4,4],'the manual cursor stays');
  follower.follow([10,20,30],'study');
  follower.stop('other study');
  assert.deepEqual(store.cursor,[10,20,30],'a different scope keeps the current cursor');
  follower.follow([7,7,7],'second');
  assert.deepEqual(store.cursor,[7,7,7]);
  follower.stop('second');
  assert.deepEqual(store.cursor,[10,20,30],'a new scope remembers the cursor at its own start');
});

test('The followed tip is the selected inserted hand, or the only inserted instrument',()=>{
  const slot=(position,inserted)=>({position,inserted,quaternion:[0,0,0,1],instrument:'cusa',active:false,grip:[],parameters:{}});
  const view={slots:{left:slot([1,2,3],true),right:slot([4,5,6],false)}};
  assert.deepEqual(acceptedToolTip(view,'left'),{hand:'left',position:[1,2,3]});
  assert.deepEqual(acceptedToolTip(view,'right'),{hand:'left',position:[1,2,3]});
  view.slots.right.inserted=true;
  assert.deepEqual(acceptedToolTip(view,'right'),{hand:'right',position:[4,5,6]});
  view.slots.left.inserted=false;view.slots.right.inserted=false;
  assert.equal(acceptedToolTip(view,'left'),null,'withdrawn instruments have no tip in the field');
  assert.equal(acceptedToolTip(null,'left'),null);
  view.slots.left.inserted=true;view.slots.right.inserted=true;
  assert.equal(acceptedToolTip(view,'missing'),null,'two inserted tools with no selected hand stay ambiguous');
});

test('Tip-to-original-target distance uses the declared target cells of an axis-aligned source',()=>{
  const shape=[12,12,12],count=12**3,mask=new Uint8Array(count),other=new Uint8Array(count);
  for(let i=4;i<=7;i++)for(let j=4;j<=7;j++)for(let k=4;k<=7;k++)mask[(i*12+j)*12+k]=1;
  other[0]=1;
  // LPS source, 2 mm along the first voxel axis.
  const volume={affine:[[-2,0,0,30],[0,-1,0,40],[0,0,1,-5],[0,0,0,1]],frame:'LPS+',shape,
    compartments:[{name:'tumor',mask},{name:'brain',mask:other}]};
  const measure=sourceTargetMeasure(volume,['tumor']);
  assert.ok(measure);
  // RAS = (2 i - 30, j - 40, k - 5). Voxel (5.5, 5.5, 5.5) is the label centre.
  const centre=measure([2*5.5-30,5.5-40,5.5-5]);
  assert.deepEqual(centre,{inside:true,distanceMm:2,exact:true},'nearest face lies 2 voxels away on the 1 mm axes');
  const outside=measure([2*5.5-30,5.5-40,0.5-5]);
  assert.deepEqual(outside,{inside:false,distanceMm:3,exact:true});
  assert.equal(formatLabelDistance(outside),'3.0 mm');
  assert.equal(sourceTargetMeasure(volume,[]),null);
  assert.equal(sourceTargetMeasure(volume,['missing']),null);
  const oblique={...volume,frame:'RAS+',affine:[[Math.cos(.3),-Math.sin(.3),0,0],[Math.sin(.3),Math.cos(.3),0,0],[0,0,1,0],[0,0,0,1]]};
  assert.equal(sourceTargetMeasure(oblique,['tumor']),null,'oblique source cells are not measured');
  const empty=sourceTargetMeasure({...volume,compartments:[{name:'tumor',mask:new Uint8Array(count)}]},['tumor']);
  assert.equal(empty([0,0,0]),null,'an empty target has no boundary');
});
