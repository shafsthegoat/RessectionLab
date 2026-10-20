'use strict';
const test=require('node:test');
const assert=require('node:assert/strict');
const {Sidecar,MAX_RESPONSE_LINE_BYTES}=require('./sidecar.cjs');
function fixture(){
  const engine=Object.create(Sidecar.prototype),received=[],failures=[],kills=[];
  Object.assign(engine,{closed:false,buffer:'',processing:Promise.resolve(),pending:new Map(),child:{kill:signal=>kills.push(signal)}});
  engine.handle=async message=>received.push({id:message.id,length:message.result.payload.length});
  engine.emit=(_event,value)=>failures.push(value);
  return {engine,received,failures,kills};
}
const line=(id,size)=>JSON.stringify({id,event:'result',result:{payload:'x'.repeat(size)}})+'\n';
test('actual consume admits a split long surgical response larger than the former 8 MiB buffer',async()=>{
  const f=fixture(),message=line('long-surgical',9*1024*1024);
  for(let i=0;i<message.length;i+=64*1024)f.engine.consume(message.slice(i,i+64*1024));
  await f.engine.processing;
  assert.deepEqual(f.received,[{id:'long-surgical',length:9*1024*1024}]);
  assert.equal(f.engine.buffer,'');assert.equal(f.engine.closed,false);assert.deepEqual(f.failures,[]);assert.deepEqual(f.kills,[]);
});
test('the 32 MiB limit applies per response line rather than to a batch of valid stdout messages',async()=>{
  const f=fixture();f.engine.consume(line('first',17*1024*1024)+line('second',17*1024*1024));await f.engine.processing;
  assert.deepEqual(f.received,[{id:'first',length:17*1024*1024},{id:'second',length:17*1024*1024}]);assert.equal(f.engine.closed,false);
});
test('over-limit complete, partial and UTF-8 responses fail before JSON parsing and clear retained buffers',async()=>{
  for(const message of ['x'.repeat(MAX_RESPONSE_LINE_BYTES+1),'x'.repeat(MAX_RESPONSE_LINE_BYTES+1)+'\n','é'.repeat(MAX_RESPONSE_LINE_BYTES/2+1)]){
    const f=fixture();f.engine.consume(message);await f.engine.processing;
    assert.equal(f.engine.closed,true);assert.equal(f.engine.buffer,'');assert.deepEqual(f.received,[]);assert.deepEqual(f.kills,['SIGTERM']);
    assert.match(f.failures[0].error.message,/32 MiB protocol limit/);
    f.engine.consume(line('ignored after failure',1));await f.engine.processing;assert.deepEqual(f.received,[]);
  }
});
