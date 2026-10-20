import test from 'node:test';
import assert from 'node:assert/strict';
import {surgicalPlaybackDelay} from '../src/surgical-playback.ts';
test('recorded playback uses authoritative elapsed effects, including partial and zero-time commands',()=>{
  const history=[{elapsedSeconds:.025},{elapsedSeconds:0},{elapsedSeconds:4.5}];
  assert.deepEqual(history.map((_,i)=>surgicalPlaybackDelay(history,i)),[{milliseconds:25,recorded:true},{milliseconds:0,recorded:true},{milliseconds:4500,recorded:true}]);
});
test('missing historical timing explicitly selects review cadence',()=>{
  for(const elapsedSeconds of [undefined,-1,NaN,Infinity])assert.deepEqual(surgicalPlaybackDelay([{elapsedSeconds}],0),{milliseconds:600,recorded:false});
});
