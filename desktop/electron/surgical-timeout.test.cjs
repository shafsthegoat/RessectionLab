'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {surgicalTimeoutMs,surgicalTransportLimitMs}=require('./surgical-timeout.cjs');
const {Sidecar}=require('./sidecar.cjs');
test('one-hour cold admission budget is restricted to named surgical replay operations',()=>{
  for(const command of ['inspect','seek','seekFrame','selectBranch','undo','redo','selectScene']){
    assert.equal(surgicalTimeoutMs(command),3600000,command);
    assert.equal(surgicalTransportLimitMs('surgicalRehearsal',{command}),3600000,command);
  }
  assert.equal(surgicalTimeoutMs('fullCase'),300000);
  assert.equal(surgicalTimeoutMs('plan'),300000);
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
