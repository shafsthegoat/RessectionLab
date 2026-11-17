import test from 'node:test';
import assert from 'node:assert/strict';
import {approachRoutes,approachStatus,lowerBoundText} from '../src/native-approaches.ts';
const approach=(id,entry=[1,2,30])=>({id,name:id,entry,target:[1,2,10],radiusMm:3,direction:[0,0,-1],constraints:{},sourceSnapshotId:'revision',parameterHash:'parameters',assessmentStatus:'current',measurements:{status:'incomplete'}});
test('anatomical approach overlay preserves physical source coordinates and radius without execution admission',()=>{
 const a=approach('source');const routes=approachRoutes([a],'source');assert.deepEqual(routes[0].entry_mm,a.entry);assert.deepEqual(routes[0].target_mm,a.target);assert.equal(routes[0].tool.shaft_radius_mm,3);assert.equal(routes[0].tool.working_length_mm,20);assert.equal(routes[0].scope,'geometric-approach');assert.equal(routes[0].selected,true);assert.equal(approachStatus(a),'incomplete');assert.equal(approachStatus({...a,assessmentStatus:'stale'}),'stale');
});
test('selected saved approach remains displayed when more than three paths were saved',()=>{
 const a=['a','b','c','d'].map(id=>approach(id));const routes=approachRoutes(a,'d');assert.equal(routes.length,3);assert(routes.some(r=>r.route_id==='d'&&r.selected));
});
test('selecting among three comparisons retains each corridor color slot',()=>{
 const a=['a','b','c'].map(id=>approach(id));const first=approachRoutes(a,'a'),last=approachRoutes(a,'c');assert.deepEqual(first.map(r=>[r.route_id,r.comparisonSlot]),last.map(r=>[r.route_id,r.comparisonSlot]));assert(last[2].selected);
});

test('displayed clearance lower bounds round down for positive and negative values',()=>{assert.equal(lowerBoundText(.75898),'0.7');assert.equal(lowerBoundText(-.121),'-0.2');assert.equal(lowerBoundText(null),'—');assert.equal(lowerBoundText(Infinity),'—')});
