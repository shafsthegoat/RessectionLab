'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),{createRequire}=require('node:module');

test('actual main IPC whitelist forwards prefix frame acknowledgments and still rejects undeclared fields',async()=>{
  const handlers=new Map(),calls=[],frame={url:'file:///trusted-app.html'},window={webContents:{mainFrame:frame},isDestroyed:()=>false};
  const main=path.join(__dirname,'main.cjs'),localRequire=createRequire(main),engine={request:async(...call)=>{calls.push(call);return {changed:false};}};
  const context=vm.createContext({__dirname,process,Buffer,console,__engine:engine,__window:window,__url:frame.url,require:name=>name==='electron'?{app:{requestSingleInstanceLock:()=>false,quit(){},on(){}},ipcMain:{handle:(name,callback)=>handlers.set(name,callback)}}:localRequire(name)});
  vm.runInContext(fs.readFileSync(main,'utf8'),context);vm.runInContext('engine=__engine;window=__window;allowedUrl=__url;bindOperations();',context);
  const invoke=args=>handlers.get('research:surgicalRehearsal')({sender:window.webContents,senderFrame:frame},args);
  const args={caseHash:'case',command:'seekPrefixFrame',branchId:'main',position:2,frameIndex:1,prefixToken:'prefix-replay.v1/owner/3',prefixFieldTokens:{flapComponents:'a'.repeat(64)}};
  assert.deepEqual(await invoke(args),{changed:false});assert.deepEqual(calls,[['surgicalRehearsal',args,120000]]);
  await assert.rejects(invoke({...args,prefixPrivateState:{}}),/Unsupported operation argument/);assert.equal(calls.length,1);
});
