import assert from 'node:assert/strict';
import test from 'node:test';
import {scenarioReadout,mappingText,latestNavigation,bleedingSummary} from '../operative-scenarios.ts';
import {mappingSites,woundEmitters,injuryMarkGrid} from './operativeScenarioScene.ts';
import {bloodShare,fieldLiquidAppearance} from './surgicalFieldOptics.ts';
import {nearestLayers,tipProximity,workingTip} from './tipProximity.ts';

const shape=[10,10,10],affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
const cell=(x,y,z)=>(x*shape[1]+y)*shape[2]+z;
const base=()=>({scene:{shape,affine,regions:[]},observations:[],metrics:{},slots:{},removed:[]});
const mapping=(result,families,extra={})=>({kind:'model-estimate',category:'mapping',probe:'subcortical',position:[4,4,4],result,parameters:{amplitudeMa:8},
  effectiveCurrentMa:8,estimatedReachMm:7.9,uncertaintyMm:1.37,families,unknownFamilies:[],
  tracts:[{id:'cst_left',regionName:'Corticospinal tract (tract)',family:'motor',estimatedDistanceMm:6.2}],...extra});

test('mapping text says model estimate, names the tract and never calls no response safe',()=>{
  const response=mappingText(mapping('model_estimate_response',{motor:{result:'model_estimate_response',nearestTract:'cst_left',estimatedDistanceMm:6.2}}));
  assert.match(response,/^Model estimate at 8\.0 mA: motor response likely, CST within ~6\.2 mm \(±1\.4 mm\)$/);
  const silent=mappingText(mapping('model_estimate_no_response',{motor:{result:'model_estimate_no_response',nearestTract:'cst_left',estimatedDistanceMm:12}}));
  assert.match(silent,/no motor response/);assert.match(silent,/not evidence of safety/);assert.doesNotMatch(silent,/safe\b(?! ty)/);
  const unknown=mappingText(mapping('unknown',{motor:{result:'unknown',reason:'The scene declares no tract evidence for this function family.'}},{tracts:[]}));
  assert.equal(unknown,'UNKNOWN: no tract evidence');
  const outside=mappingText(mapping('unknown',{motor:{result:'unknown',reason:'The tip is outside declared tract coverage: no retained motor tract within 20 mm.'}}));
  assert.equal(outside,'UNKNOWN: tip outside the declared tract coverage');
});

test('navigation reads the latest screen reading and separates rehearsal truth',()=>{
  const view={...base(),observations:[{kind:'navigation',reportedImageRasMm:[1,2,3],imageRegion:{regionId:3,name:'Cortex',material:'soft_tissue'},distanceToPlannedTumourMm:14.25}],
    navigationV1:{privilegedTruth:{readings:[{tipRasMm:[1,2,1],registrationErrorMm:[0,0,2],navigationErrorMm:6.4,nearestTumourEdge:{discrepancyMm:4.1},trueDistanceToRemainingTumourMm:9.8}]}}};
  assert.equal(latestNavigation(view).index,1);
  const lines=scenarioReadout(view);
  assert.deepEqual(lines.map(l=>l.tone),['navigation','truth']);
  assert.equal(lines[0].text,'Navigation · image shows Cortex · 14.3 mm to planned tumour');
  assert.equal(lines[1].text,'Rehearsal truth · navigation off 6.4 mm · tumour actually 9.8 mm away · edge moved 4.1 mm');
});

test('brain shift, bleeding with kind UNKNOWN and injury lines',()=>{
  const view={...base(),
    brainShiftV1:{started:true,currentPeakMm:2.5,declaration:{peakSagMm:5,settlingSteps:6},state:{level:3,settlingSteps:6},plannedTargetRasMm:[0,0,0],shiftedTargetRasMm:[0,0,1.2]},
    vesselBleedingV1:{wounds:{a:{kind:'uncertain',sealFraction:0,lossMm3:400,rateMm3PerSecond:20,cells:[1]},b:{kind:'vein',sealFraction:0,lossMm3:100,rateMm3PerSecond:5,cells:[2]}},
      bloodLossMm3:500,lossByKindMm3:{},tumourOozeMm3:0,rateMm3PerSecond:25,depositedMm3:500},
    surgicalInjuryV1:{contused:[1,2],lacerated:[3],vesselInjured:[4],totalsMm3:{contused:2,lacerated:1,vessel:1},avulsions:0,eventCount:2,events:[{structures:['cst']}]}};
  const text=scenarioReadout(view).map(l=>l.text);
  assert.equal(text[0],'Brain shift scenario · peak 2.5 mm (settling to 5.0 mm) · target moved 1.2 mm');
  assert.equal(text[1],'Bleeding · 2 open wounds (1 venous, 1 kind UNKNOWN) · loss 0.50 mL · 1.5 mL/min · UNKNOWN kind bled at the arterial law (0.40 mL)');
  assert.match(text[2],/^Injury recorded · contused 2 mm³ · lacerated 1 mm³ · 1 vessel cells · cst$/);
  assert.equal(bleedingSummary(view.vesselBleedingV1).unknown,1);
  assert.deepEqual(scenarioReadout({...base(),brainShiftV1:{...view.brainShiftV1,started:false}}).map(l=>l.text),['Brain shift scenario · begins at dural opening']);
});

test('mapping sites and wound emitters come from the view',()=>{
  const view={...base(),observations:[mapping('unknown',{}),mapping('model_estimate_response',{})],
    vesselBleedingV1:{wounds:{a:{id:'a',kind:'artery',sealFraction:.2,cells:[cell(1,1,1),cell(3,1,1)],rateMm3PerSecond:3,calibreMm:1.4},
      s:{id:'s',kind:'vein',sealFraction:1,cells:[cell(5,5,5)]}}}};
  const sites=mappingSites(view);
  assert.deepEqual(sites.map(s=>[s.result,s.latest,s.reachMm]),[['unknown',false,null],['model_estimate_response',true,7.9]]);
  const wounds=woundEmitters(view,p=>[p[0],p[1],p[2]-1]);
  assert.equal(wounds.length,1);assert.deepEqual(wounds[0].position,[2,1,0]);assert.equal(wounds[0].arterial,true);
});

test('pooled blood occludes once it dominates the retained fluid',()=>{
  const view={...base(),metrics:{fluidAddedMm3:100},vesselBleedingV1:{depositedMm3:80}};
  assert.equal(bloodShare(view),.8);
  assert.equal(fieldLiquidAppearance(.8).opaque,true);assert.equal(fieldLiquidAppearance(.8).opacity,1);
  assert.equal(fieldLiquidAppearance(.2).opaque,false);
  assert.equal(bloodShare({...view,vesselBleedingV1:undefined}),0);
});

test('injury marks cover contused and lacerated cells on the source grid',()=>{
  const grid=injuryMarkGrid({contused:[cell(4,4,4)],lacerated:[cell(5,4,4)]},shape);
  const at=(x,y,z,c)=>grid.data[(((x-grid.origin[0])*grid.size[1]+(y-grid.origin[1]))*grid.size[2]+(z-grid.origin[2]))*4+c];
  assert.equal(at(4,4,4,0),255);assert.equal(at(5,4,4,1),255);assert.equal(at(4,4,4,1),0);
  assert.equal(injuryMarkGrid({contused:[],lacerated:[]},shape),null);
});

test('tip proximity uses the working tip, skips removed cells and carries the tip into the source frame',()=>{
  const tract=new Uint8Array(1000),vessel=new Uint8Array(1000);tract[cell(5,5,8)]=1;vessel[cell(5,5,2)]=1;vessel[cell(5,5,4)]=1;
  const layers=[{name:'Corticospinal tract (tract)',mask:tract,group:'tract'},{name:'Artery',mask:vessel,group:'vessel'}];
  const slots={left:{instrument:'cusa',position:[5,5,5],inserted:false},right:{instrument:'bipolar_forceps',position:[5,5,5],inserted:true}};
  assert.equal(workingTip(slots,'left').slot,'right');
  const near=tipProximity(slots,'left',layers,shape,affine,{removed:new Set([cell(5,5,4)])});
  assert.deepEqual([near.tract,near.vessel],[{name:'Corticospinal tract',distanceMm:3},{name:'Artery',distanceMm:3}]);
  const shifted=tipProximity(slots,'right',layers,shape,affine,{toSource:p=>[p[0],p[1],p[2]+2]});
  assert.equal(shifted.tract.distanceMm,1);
  assert.deepEqual(nearestLayers([5,5,5],layers,shape,affine,.5),[]);
});

test('target contour follows the target seen along the opening axis',async()=>{
  const {targetContour}=await import('./operativeScenarioScene.ts');
  const points=[];
  for(let i=0;i<4000;i++){const t=Math.acos(1-2*((i+.5)/4000)),f=Math.PI*(1+Math.sqrt(5))*i;
    points.push(5+3*Math.sin(t)*Math.cos(f),5+3*Math.sin(t)*Math.sin(f),5+3*Math.cos(t));}
  const contour=targetContour([new Float32Array(points)],affine,[0,0,1]);
  for(const p of contour.loop){assert.ok(Math.abs(Math.hypot(p[0]-5,p[1]-5)-3)<.15);assert.ok(Math.abs(p[2]-2)<.01);}
  assert.equal(contour.extremes.length,72);
});
