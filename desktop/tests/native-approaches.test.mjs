import test from 'node:test';
import assert from 'node:assert/strict';
import {approachAssessment,approachRoutes,approachStatus,lowerBoundText} from '../src/native-approaches.ts';
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

const measured=(structures,extra={})=>({...approach('a'),measurements:{status:'incomplete',structures,unknowns:['Registration accuracy unmeasured'],targets:[],normalTraversal:{available:false},coverage:{withinReferenceEnvelope:true},...extra}});
test('observed intersections, evidence completeness and geometric constraints are independent',()=>{
 const clear={layerId:'artery',label:'Artery estimate',available:true,clearanceLowerBoundMm:2,confirmedIntersection:false};
 const a=measured([clear],{status:'known-conflict',constraintFailures:['entry outside declared entry-centre disk']});
 let s=approachAssessment(a);assert.equal(s.observed,'no-intersection');assert.equal(s.observedLabel,'No detected intersection in assessed anatomy');assert.equal(s.completeness,'incomplete');assert.equal(s.minimumBoundMm,2);assert.deepEqual(s.constraintFailures,['entry outside declared entry-centre disk']);
 s=approachAssessment(measured([{...clear,confirmedIntersection:true,clearanceLowerBoundMm:-2}],{status:'known-conflict'}));assert.equal(s.observed,'intersection');assert.equal(s.completeness,'incomplete');assert.equal(s.intersections[0].label,'Artery estimate');
});
test('missing, empty, stale and unresolved measurements never become anatomical absence',()=>{
 const absent={layerId:'language',label:'Language bundle',available:false,confirmedIntersection:false};
 let s=approachAssessment(measured([absent]));assert.equal(s.observed,'unavailable');assert.equal(s.minimumBoundMm,null);assert.equal(s.targetReachedMm3,null);assert.equal(s.traversalMm3,null);assert(s.limitations.some(text=>text.includes('Language bundle')));
 s=approachAssessment(measured([],{status:'no-detected-conflict',unknowns:[]}));assert.equal(s.observed,'unavailable');assert.equal(s.completeness,'incomplete');
 const unresolved={...absent,available:true,intersectsOrUnresolved:true,clearanceLowerBoundMm:-.1};
 s=approachAssessment(measured([unresolved]));assert.equal(s.observed,'no-intersection');assert.equal(s.unresolved.length,1);assert(s.limitations.some(text=>text.includes('unresolved')));
 s=approachAssessment({...measured([{...unresolved,confirmedIntersection:true}]),assessmentStatus:'stale'});assert.equal(s.observed,'unavailable');assert.equal(s.completeness,'stale');assert.equal(s.minimumBoundMm,null);assert.equal(s.intersections.length,0);
});
test('comparison quantities preserve measured zero and hide unavailable target and traversal quantities',()=>{
 const a=measured([{layerId:'tract',label:'Tract',available:true,clearanceLowerBoundMm:0}],{targets:[{available:true,coveredVoxelCenterVolumeMm3:0},{available:false,coveredVoxelCenterVolumeMm3:99}],normalTraversal:{available:true,conservativeVolumeUpperBoundMm3:0}});
 const s=approachAssessment(a);assert.equal(s.minimumBoundMm,0);assert.equal(s.targetReachedMm3,0);assert.equal(s.traversalMm3,0);
});
