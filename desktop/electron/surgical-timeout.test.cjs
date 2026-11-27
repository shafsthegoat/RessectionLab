'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {surgicalTimeoutMs,surgicalTransportLimitMs,progressRenewalCeilingMs,ADMISSION_STALL_MS,ADMISSION_CEILING_MS}=require('./surgical-timeout.cjs');
const {Sidecar}=require('./sidecar.cjs');
test('cold admission budget is a progress-renewed stall window restricted to named surgical replay operations',()=>{
  assert.equal(ADMISSION_STALL_MS,3600000);assert.equal(ADMISSION_CEILING_MS,86400000);
  for(const command of ['inspect','seek','seekFrame','selectBranch','undo','redo','selectScene']){
    assert.equal(surgicalTimeoutMs(command),ADMISSION_STALL_MS,command);
    assert.equal(surgicalTransportLimitMs('surgicalRehearsal',{command}),3600000,command);
    assert.equal(progressRenewalCeilingMs('surgicalRehearsal',{command}),ADMISSION_CEILING_MS,command);
  }
  for(const [operation,args] of [['surgicalRehearsal',{command:'step'}],['surgicalRehearsal',{command:'fullCase'}],['loadCase',{command:'inspect'}],['anatomyPrepare',{action:'run'}]])
    assert.equal(progressRenewalCeilingMs(operation,args),null,operation);
  assert.equal(surgicalTimeoutMs('fullCase'),300000);
  assert.equal(surgicalTimeoutMs('plan'),300000);
  assert.equal(surgicalTimeoutMs('generatePreparations'),600000);
  assert.equal(surgicalTransportLimitMs('surgicalRehearsal',{command:'generatePreparations'}),600000);
  for(const command of ['step','preview','create','branch','reset'])assert.equal(surgicalTimeoutMs(command),120000,command);
  for(const operation of ['trainPatient','inspectRehearsal','loadCase','saveCase','cancel'])assert.equal(surgicalTransportLimitMs(operation,{command:'inspect'}),300000);
  for(const command of ['fullCase','step','preview','not-an-operation'])assert.equal(surgicalTransportLimitMs('surgicalRehearsal',{command}),300000);
});
test('real sidecar admission accepts the scoped long budget and retains its normal cancellation path',async t=>{
  const engine=Object.create(Sidecar.prototype),messages=[];
  Object.assign(engine,{closed:false,pending:new Map(),child:{stdin:{write:message=>messages.push(JSON.parse(message))}}});
  t.after(()=>{for(const pending of engine.pending.values())clearTimeout(pending.timeout);});
  const request=engine.request('surgicalRehearsal',{command:'inspect'},3600000);
  assert.equal(messages[0].timeoutMs,3600000);assert.equal(messages[0].op,'surgicalRehearsal');
  const cancel=engine.request('cancel',{requestId:messages[0].id});
  assert.equal(messages[1].args.requestId,messages[0].id);
  const finish=(message,result)=>{const pending=engine.pending.get(message.id);clearTimeout(pending.timeout);engine.pending.delete(message.id);pending.resolve(result);};
  finish(messages[1],{cancelled:true});finish(messages[0],null);
  assert.deepEqual(await cancel,{cancelled:true});assert.equal(await request,null);
  for(const [op,args] of [['surgicalRehearsal',{command:'fullCase'}],['surgicalRehearsal',{command:'step'}],['loadCase',{command:'inspect'}]])await assert.rejects(engine.request(op,args,3600000),/Invalid operation time budget/);
  await assert.rejects(engine.request('surgicalRehearsal',{command:'inspect'},3600001),/Invalid operation time budget/);
});
test('anatomy reconstruction accepts its bounded model job budget through the real transport',async t=>{
  const engine=Object.create(Sidecar.prototype),messages=[];
  Object.assign(engine,{closed:false,pending:new Map(),child:{stdin:{write:message=>messages.push(JSON.parse(message))}}});
  t.after(()=>{for(const pending of engine.pending.values())clearTimeout(pending.timeout);});
  assert.equal(surgicalTransportLimitMs('anatomyReconstruct'),1800000);
  const request=engine.request('anatomyReconstruct',{model:'synthstrip'},1800000);
  assert.equal(messages.length,1);assert.equal(messages[0].timeoutMs,1800000);
  const pending=engine.pending.get(messages[0].id);clearTimeout(pending.timeout);engine.pending.delete(messages[0].id);pending.resolve({document:{}});
  assert.deepEqual(await request,{document:{}});
  await assert.rejects(engine.request('anatomyReconstruct',{},1800001),/Invalid operation time budget/);
  await assert.rejects(engine.request('anatomyImport',{},1800000),/Invalid operation time budget/);
});
test('owned surgical request identifiers bind cancellation and never enter action arguments',async t=>{
  const engine=Object.create(Sidecar.prototype),messages=[];
  Object.assign(engine,{closed:false,pending:new Map(),child:{stdin:{write:message=>messages.push(JSON.parse(message))}}});
  t.after(()=>{for(const pending of engine.pending.values())clearTimeout(pending.timeout);});
  const requestId='surgical-6d97b9f1-fd14-4fe2-96e2-a9dc43bb8b83';
  const request=engine.request('surgicalRehearsal',{requestId,command:'plan',caseHash:'case'});
  assert.equal(messages[0].id,requestId);assert.equal(Object.hasOwn(messages[0].args,'requestId'),false);
  await assert.rejects(engine.request('surgicalRehearsal',{requestId,command:'plan'}),/Invalid surgical request/);
  await assert.rejects(engine.request('surgicalRehearsal',{requestId:'another-operation',command:'plan'}),/Invalid surgical request/);
  const cancellation=engine.request('cancel',{requestId});assert.equal(messages[1].args.requestId,requestId);
  for(const [message,result] of [[messages[1],{cancelled:true}],[messages[0],null]]){const pending=engine.pending.get(message.id);clearTimeout(pending.timeout);engine.pending.delete(message.id);pending.resolve(result);}
  assert.deepEqual(await cancellation,{cancelled:true});assert.equal(await request,null);
});

test('preparation owns its cancellable transport identity, with short review and bounded run',async t=>{
  const engine=Object.create(Sidecar.prototype),messages=[];
  Object.assign(engine,{closed:false,pending:new Map(),child:{stdin:{write:message=>messages.push(JSON.parse(message))}}});
  t.after(()=>{for(const pending of engine.pending.values())clearTimeout(pending.timeout);});
  const requestId='prepare-6d97b9f1-fd14-4fe2-96e2-a9dc43bb8b83';
  const request=engine.request('anatomyPrepare',{action:'run',collectionId:'patient',requestId},3600000);
  assert.deepEqual(messages[0],{id:requestId,op:'anatomyPrepare',args:{action:'run',collectionId:'patient'},timeoutMs:3600000});
  await assert.rejects(engine.request('anatomyPrepare',{action:'run',requestId},3600000),/Invalid prepare request/);
  await assert.rejects(engine.request('anatomyPrepare',{action:'plan'},3600000),/Invalid operation time budget/);
  const cancel=engine.request('cancel',{requestId});
  assert.equal(messages[1].args.requestId,requestId);
  for(const [message,result] of [[messages[1],{cancelled:true}],[messages[0],null]]){
    const pending=engine.pending.get(message.id);
    clearTimeout(pending.timeout);engine.pending.delete(message.id);pending.resolve(result);
  }
  assert.deepEqual(await cancel,{cancelled:true});
  assert.equal(await request,null);
});

test('replay admission progress renews its stall window up to the ceiling; fixed budgets are not renewed',async t=>{
  t.mock.timers.enable({apis:['setTimeout','Date']});
  const engine=Object.create(Sidecar.prototype),messages=[];
  Object.assign(engine,{closed:false,pending:new Map(),child:{stdin:{write:message=>messages.push(JSON.parse(message))}},assets:{expose:async value=>value}});
  t.after(()=>{for(const pending of engine.pending.values())clearTimeout(pending.timeout);});
  const settled=promise=>promise.then(value=>({value}),error=>({error:error.message}));
  const replay=settled(engine.request('surgicalRehearsal',{command:'inspect'},1000));
  const id=messages[0].id;
  for(let index=0;index<10;index++){t.mock.timers.tick(900);await engine.handle({id,event:'progress',progress:{fraction:index/10,message:'Verifying'}});}
  assert.equal(engine.pending.has(id),true,'advancing verification is still pending after 9 s on a 6 s budget');
  t.mock.timers.tick(6100);
  assert.deepEqual(await replay,{error:'Verification stopped reporting progress within its local time budget'});
  assert.equal(messages.at(-1).op,'cancel');assert.equal(messages.at(-1).args.requestId,id);
  const step=settled(engine.request('surgicalRehearsal',{command:'step'},1000));
  const stepId=messages.at(-1).id;
  for(let index=0;index<10;index++){t.mock.timers.tick(900);if(engine.pending.has(stepId))await engine.handle({id:stepId,event:'progress',progress:{fraction:0,message:'step'}});}
  assert.deepEqual(await step,{error:'Operation exceeded its local time budget'});
});
test('a renewed replay never outlives its absolute ceiling',async t=>{
  t.mock.timers.enable({apis:['setTimeout','Date']});
  const engine=Object.create(Sidecar.prototype),messages=[];
  Object.assign(engine,{closed:false,pending:new Map(),child:{stdin:{write:message=>messages.push(JSON.parse(message))}}});
  t.after(()=>{for(const pending of engine.pending.values())clearTimeout(pending.timeout);});
  const replay=engine.request('surgicalRehearsal',{command:'seek'},ADMISSION_STALL_MS).catch(error=>error.message);
  const id=messages[0].id;
  let elapsed=0;
  while(engine.pending.has(id)&&elapsed<ADMISSION_CEILING_MS+ADMISSION_STALL_MS){t.mock.timers.tick(600000);elapsed+=600000;if(engine.pending.has(id))await engine.handle({id,event:'progress',progress:{fraction:.5,message:'Verifying'}});}
  assert.equal(await replay,'Verification exceeded its absolute local time ceiling');
  assert.ok(elapsed>=ADMISSION_CEILING_MS&&elapsed<=ADMISSION_CEILING_MS+600000,String(elapsed));
});
