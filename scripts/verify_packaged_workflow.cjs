'use strict';
// Install the existing test harness at the signed app's first main-module line.
// The archive, executable, production options and application flags stay real.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path');
const {spawn}=require('node:child_process');
const app=fs.realpathSync(process.env.WORKFLOW_APP),output=path.resolve(process.env.WORKFLOW_OUTPUT);
const harness=path.resolve(__dirname,'../desktop/electron/verify-surgical-workflow.cjs');
const executable=path.join(app,'Contents/MacOS/RessectionLab');
assert(!fs.existsSync(output),'Preserve previous evidence');fs.mkdirSync(path.dirname(output),{recursive:true});
const record={schema:'resectionlab-actual-packaged-workflow-v1',passed:false,executable,harness,output};
const log=fs.createWriteStream(output+'.launcher.log');let diagnostic='',socket;
const child=spawn(executable,['--automation-hidden','--inspect-brk=127.0.0.1:0'],
 {cwd:path.dirname(output),env:{...process.env,WORKFLOW_ACTUAL_APP:'1'},stdio:['ignore','pipe','pipe'],detached:true});
let closed=false;child.once('close',()=>{closed=true;});
const exited=()=>child.exitCode!==null||child.signalCode!==null;
for(const stream of [child.stdout,child.stderr])stream.on('data',data=>{diagnostic+=data;log.write(data);});
const delay=ms=>new Promise(resolve=>setTimeout(resolve,ms));
const wait=async(test,label,seconds=90)=>{const started=Date.now();while(Date.now()-started<seconds*1000){if(await test())return;if(exited())throw Error('Application exited during '+label);await delay(50);}throw Error('Timeout '+label);};
const waitClosed=milliseconds=>new Promise(resolve=>{if(closed)return resolve(true);const timer=setTimeout(()=>resolve(false),milliseconds);child.once('close',()=>{clearTimeout(timer);resolve(true);});});
const terminate=signal=>{try{process.kill(-child.pid,signal);}catch(error){if(error.code!=='ESRCH')throw error;}};
(async()=>{
 try{
  let address;await wait(()=>{address=diagnostic.match(/Debugger listening on (ws:\/\/127\.0\.0\.1:\d+\/[^\s]+)/)?.[1];return address;},'main inspector startup');
  socket=new WebSocket(address);
  await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
  let serial=0;const pending=new Map(),scripts=new Map(),pauses=[];
  socket.addEventListener('message',event=>{const value=JSON.parse(String(event.data));
   if(value.method==='Debugger.scriptParsed')scripts.set(value.params.scriptId,value.params.url);
   if(value.method==='Debugger.paused')pauses.push(value.params);
   const task=pending.get(value.id);if(task){pending.delete(value.id);value.error?task.reject(Error(JSON.stringify(value.error))):task.resolve(value.result);}});
  const command=(method,params={})=>new Promise((resolve,reject)=>{const id=++serial,timer=setTimeout(()=>{pending.delete(id);reject(Error('Timeout '+method));},30000);pending.set(id,{resolve:value=>{clearTimeout(timer);resolve(value);},reject:error=>{clearTimeout(timer);reject(error);}});socket.send(JSON.stringify({id,method,params}));});
  await command('Debugger.enable');
  await command('Debugger.setBreakpointByUrl',{urlRegex:'[/]app[.]asar[/]electron[/]main[.]cjs$',lineNumber:0});
  await command('Runtime.runIfWaitingForDebugger');
  let frame;
  for(let attempts=0;attempts<5;attempts++){
   await wait(()=>pauses.length,'initial main pause');const pause=pauses.shift();
   const current=pause.callFrames[0],url=scripts.get(current.location.scriptId)||current.url;
   if(url.endsWith('/app.asar/electron/main.cjs')){frame=current;record.mainUrl=url;break;}
   await command('Debugger.resume');
  }
  assert(frame,'Pause in the actual packaged main before installing hooks');
  const injection="process.getBuiltinModule('module').createRequire("+JSON.stringify(harness)+")("+JSON.stringify(harness)+")";
  const result=await command('Debugger.evaluateOnCallFrame',{callFrameId:frame.callFrameId,expression:injection});
  assert(!result.exceptionDetails,JSON.stringify(result.exceptionDetails));
  record.harnessInstalledBeforeMain=true;
  await command('Debugger.resume');await command('Debugger.disable');socket.close();socket=null;
  await wait(exited,'complete packaged workflow',900);
  record.exitCode=child.exitCode;assert.equal(child.exitCode,0);
  const workflow=JSON.parse(fs.readFileSync(path.join(output,'report.json'),'utf8'));
  assert(workflow.passed);assert(workflow.launcher.isPackaged);assert.equal(workflow.launcher.executable,executable);
  assert(workflow.launcher.originalPackagedEngineOptions);record.passed=true;
 }catch(error){record.error=String(error);}
 finally{
  socket?.close();if(!record.passed||!exited())terminate('SIGTERM');
  if(!await waitClosed(5000)){terminate('SIGKILL');await waitClosed(5000);}
  record.signalCode=child.signalCode;
  if(!closed){record.passed=false;record.error='Application did not terminate after bounded cleanup';child.stdout.destroy();child.stderr.destroy();}
  log.end();
  fs.writeFileSync(output+'.launcher.json',JSON.stringify(record,null,2)+'\n');
  console.log(JSON.stringify({passed:record.passed,error:record.error,output}));process.exitCode=record.passed?0:1;
 }
})();
