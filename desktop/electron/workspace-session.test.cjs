'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {validateWorkspaceResult}=require('./workspace-session.cjs');
const {Sidecar}=require('./sidecar.cjs');
function fixture(){const f=JSON.parse(fs.readFileSync(path.join(__dirname,'../tests/fixtures/development-scripted.json'),'utf8')).result;
 f.case.workspaceSession={schema:'integrated-workspace-session-v1',referenceCaseHash:f.case.caseHash,sessionHash:'sha256:'+'b'.repeat(64),displaySeries:[],imagingState:{selectedSeriesId:null},
  episodeReplay:{episode:f.episode,episodeCanonicalJson:f.episodeCanonicalJson,frameIndex:8,visible:true,validation:'authoritative_generated_replay',accountingQualification:'historical_timings_not_remeasured'}};return f.case;}
test('old source and valid revalidated recorded workspace accepted by host envelope',()=>{const source=fixture();assert.equal(validateWorkspaceResult(source),source);delete source.workspaceSession;assert.equal(validateWorkspaceResult(source),source)});
for(const [name,mutate] of [['source',s=>s.referenceCaseHash='bad'],['admission',s=>s.episodeReplay.validation='hash-only'],['timing',s=>delete s.episodeReplay.accountingQualification],['geometry',s=>s.episodeReplay.episode.tools[0].working_length_mm++],['selection',s=>s.episodeReplay.frameIndex=-1],['aux replay',s=>s.imagingState.selectedSeriesId='aux']])test(`host refuses stale ${name} before clearing/exposing assets or sending result`,async()=>{
 const source=fixture();mutate(source.workspaceSession);const pending={op:'loadCase',resolve:()=>assert.fail('resolved'),reject:()=>{},timeout:null};
 const fake={pending:new Map([['load',pending]]),assets:{clear:()=>assert.fail('cleared'),expose:()=>assert.fail('exposed')},emit:()=>assert.fail('forwarded')};
 await assert.rejects(Sidecar.prototype.handle.call(fake,{id:'load',event:'result',result:source}));
});
test('valid loaded workspace is exposed before its renderer result',async()=>{
 const source=fixture(),order=[],pending={op:'loadCase',resolve:value=>{assert.equal(value,source);order.push('resolved')},timeout:null};
 const fake={pending:new Map([['load',pending]]),assets:{clear:()=>order.push('clear'),expose:async v=>{order.push('expose');return v}},emit:()=>order.push('event')};
 await Sidecar.prototype.handle.call(fake,{id:'load',event:'result',result:source});assert.deepEqual(order,['clear','expose','event','resolved']);assert.equal(fake.pending.size,0);
});
