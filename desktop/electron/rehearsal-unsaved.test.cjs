const test = require('node:test');
const assert = require('node:assert/strict');
const {RehearsalUnsaved} = require('./rehearsal-unsaved.cjs');
test('unsaved actions guard navigation until successful install/save; failed and no-op actions stay clean',async()=>{
 let asks=0, discard=false;
 const guard=new RehearsalUnsaved(async()=>{asks++;return discard;});
 guard.changed({command:'preview'},{});guard.changed({command:'step'},{changed:false});guard.changed({command:'step'},{outcome:{status:'rejected'}});
 assert.equal(await guard.allowNavigation(),true);assert.equal(asks,0);
 guard.changed({command:'step'},{outcome:{status:'accepted'}});
 assert.equal(await guard.allowNavigation(),false);assert.equal(guard.dirty,true);
 discard=true;assert.equal(await guard.allowNavigation(),true);
 assert.equal(guard.dirty,true,'failed replacement does not clear dirty state');
 guard.installed();assert.equal(guard.dirty,false);
 guard.changed({command:'create'},{});guard.saved();assert.equal(await guard.allowNavigation(),true);
 guard.changed({source:{collectionId:'native'},command:'step'},{});assert.equal(guard.dirty,false,'native histories are persisted atomically');
});
