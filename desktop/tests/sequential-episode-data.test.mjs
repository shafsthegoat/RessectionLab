import {declaredStructureColor,planningSliceMemberships} from '../src/interactive-planning-data.ts';
import test from 'node:test';
import assert from 'node:assert/strict';
import {SurgicalComparisonContext} from '../src/surgical-comparison-context.ts';
import {sequentialEpisodePacket,sequentialEpisodeOverlay} from '../src/sequential-episode-data.ts';
const h=n=>'sha256:'+String(n).repeat(64),f=n=>String(n).repeat(64);
function fixture(){
 const episodes=[['learned',17],['learned',43],['greedy',0],['search',0],['fixed-narrow-greedy',0]].map(([method,seed],index)=>({branchId:`b${index}`,label:method,method,seed,checkpoint:method==='learned'?{sha256:f(1),selectedUpdates:1}:null,originalEpisode:{sha256:f(2)},nativeActions:1,acceptedStateIds:[h(0),h(1)],finalStateId:h(1),snapshotHash:h(2),actionHistoryHash:h(3),attempts:[{attemptIndex:0,nativePositionBefore:0,nativePositionAfter:0,action:{operation:'apply'},status:'rejected',stateBefore:h(0),stateAfter:h(0),toolPositionBefore:[0,0,0],toolPositionAfter:[0,0,0],effectiveParametersBefore:{radiusMm:1.5},effectiveParametersAfter:{radiusMm:1.5}},{attemptIndex:1,nativePositionBefore:0,nativePositionAfter:1,action:{operation:'apply'},status:'accepted',stateBefore:h(0),stateAfter:h(1),toolPositionBefore:[0,0,0],toolPositionAfter:[1,0,0],effectiveParametersBefore:{radiusMm:.4},effectiveParametersAfter:{radiusMm:.4}}],endpoint:{configHash:h(4)},progress:{originalTargetTreatmentMm3:27,originalTargetRemovedMm3:27,originalTargetCells:2,residualOriginalTargetCells:1},neuralRemovedMm3:0,modeledSeconds:1,cleanup:{debrisMm3:0,fluidMm3:0,debrisCollectedMm3:0},structureRemovedMm3:{'motor-cortex':0},structureCutSourceMm3:{'motor-cortex':0},computation:{tier:{name:'low',max_physical_calls:56,max_wall_seconds:25},physicalCalls:2,planningPhysicalCalls:0,executionPhysicalCalls:2,wallSeconds:1,previewSkillAttempts:0,previewRefusals:0,canceledRequests:0}}));
 const packet={schema:'resectionlab.recorded-sequential-episodes.v1',caseHash:h(7),sourceSceneId:h(6),sourceStateId:h(0),policyHash:h(5),configHash:h(4),parent:200,tier:'low',scope:'generated',contentHash:h(8),geometry:{structures:[],targetSourceCells:[1,2],targetPointsRasMm:[[1,0,0],[2,0,0]]},episodes};
 const response={caseHash:h(7),planningAdmission:{policyHash:h(5)},branchId:'b0',position:1,branches:episodes.map(e=>({id:e.branchId,steps:e.nativeActions})),view:{scene:{sceneId:h(6)},stateId:h(1),removed:[1]}};
 return {artifact:{sequentialEpisodes:packet},response};
}
test('five real method identities join source policy and native branches',()=>{const {artifact,response}=fixture();assert.equal(sequentialEpisodePacket(artifact,response).episodes.length,5)});
for(const [label,mutate] of [
 ['source',x=>x.response.view.scene.sceneId=h(9)],['case',x=>x.response.caseHash=h(9)],['policy',x=>x.response.planningAdmission.policyHash=h(9)],
 ['edited branch',x=>x.response.branches[0].steps++],['checkpoint',x=>x.artifact.sequentialEpisodes.episodes[0].checkpoint=null],
 ['missing control',x=>x.artifact.sequentialEpisodes.episodes.pop()],['seed identity',x=>x.artifact.sequentialEpisodes.episodes[1].seed=71],
 ['refused state effect',x=>x.artifact.sequentialEpisodes.episodes[0].attempts[0].stateAfter=h(1)],
 ['omitted accepted action',x=>x.artifact.sequentialEpisodes.episodes[0].attempts.pop()],
 ['bad measured total',x=>x.artifact.sequentialEpisodes.episodes[0].progress.originalTargetTreatmentMm3=NaN],
])test(`rejects mismatched ${label}`,()=>{const x=fixture();mutate(x);assert.equal(sequentialEpisodePacket(x.artifact,x.response),null)});
test('overlay uses only accepted paths and current residual; edited state clears it',()=>{const {artifact,response}=fixture(),p=sequentialEpisodePacket(artifact,response);let o=sequentialEpisodeOverlay(p,response,response.view);assert.equal(o.routes.length,1);assert.deepEqual(o.recorded.residualPointsRasMm,[[2,0,0]]);response.view.stateId=h(9);assert.equal(sequentialEpisodeOverlay(p,response,response.view),null)});

test('interior frames retain admitted policy identity without planning authority',()=>{
 const {artifact,response}=fixture(),context=new SurgicalComparisonContext(),owner={};
 context.accept(response,owner,false);const wire={...response,playback:{readOnly:true,branchId:'b0'},view:{...response.view,stateId:h(9)}};delete wire.planningAdmission;
 const bound=context.accept(wire,owner,true);assert.equal(bound.planningAdmission.available,false);
 const packet=sequentialEpisodePacket(artifact,bound);assert(packet);assert(sequentialEpisodeOverlay(packet,bound,bound.view));
 assert.equal(sequentialEpisodePacket(artifact,context.accept(wire,{},true)),null);
});

test('original corticospinal identity keeps the existing violet smaller overlap marker',()=>{
 assert.equal(declaredStructureColor('corticospinal'),declaredStructureColor('corticospinal-tract'));
 const dots=planningSliceMemberships({recorded:{structures:[{id:'motor-cortex',label:'motor',pointsRasMm:[[0,0,0]]},{id:'corticospinal',label:'CST',pointsRasMm:[[0,0,0]]}]}},0,0,.5);
 assert.equal(dots[0].radiusPx,7);assert.equal(dots[1].radiusPx,4);assert.equal(dots[1].color,'#c6a2ff');
});
