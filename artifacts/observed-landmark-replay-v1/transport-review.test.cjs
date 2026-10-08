'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
const {createRequire}=require('node:module');
const root=path.resolve(__dirname,'../..'), electron=path.join(root,'desktop/electron');
const actual=JSON.parse(fs.readFileSync(path.join(__dirname,'authentic-before.json')));
function handler(result){
 const h=new Map(), mainFile=path.join(electron,'main.cjs'),req=createRequire(mainFile);
 const frame={url:'file:///trusted-local-app.html'},window={isDestroyed:()=>false,webContents:{mainFrame:frame}};
 const context=vm.createContext({__dirname:electron,process,Buffer,console,__engine:{request:async()=>result},__window:window,__url:frame.url,
 require:n=>n==='electron'?{app:{requestSingleInstanceLock:()=>false,quit(){},on(){}},ipcMain:{handle:(n,f)=>h.set(n,f)}}:req(n)});
 vm.runInContext(fs.readFileSync(mainFile,'utf8'),context);
 vm.runInContext('engine=__engine;window=__window;allowedUrl=__url;bindOperations();',context);
 return ()=>h.get('research:inspectObservedLandmarkUpdate')({sender:window.webContents,senderFrame:frame},{action:'open',studyId:'resect-case4-sparse-update-v1'});
}
const mutations={
 'extra excluded payload':r=>r.validationLandmarks='FORBIDDEN_SENTINEL',
 'altered source coordinate unchanged hash':r=>r.state.observations[0].sourceRasMm[0]+=1,
 'wrong B membership unchanged hash':r=>r.state.observations[0].landmarkId=2,
 'during observation leaked before phase':r=>r.state.observations[0].observedRasMm=r.state.observations[0].sourceRasMm,
 'finite clock fabricated':r=>r.state.elapsedSeconds=1,
 'nonfinite coordinate':r=>r.state.observations[0].sourceRasMm[0]=Infinity,
 'invented tool pose':r=>r.state.toolPose={status:'FORBIDDEN_SENTINEL'},
 'source binding changed':r=>r.binding.sourceImageSha256='sha256:'+'0'.repeat(64),
 'unbound snapshot selection':r=>r.snapshot.landmarkId=14,
};
for(const [name,mutate]of Object.entries(mutations))test('main refuses '+name,async()=>{const r=structuredClone(actual);mutate(r);await assert.rejects(handler(r)());});
test('authentic result passes',async()=>assert.deepEqual(await handler(actual)(),actual));
test('sidecar does not emit excluded result before validation',async()=>{
 const {Sidecar}=require(path.join(electron,'sidecar.cjs'));
 const x=Object.create(Sidecar.prototype);x.pending=new Map();const events=[];
 x.emit=(name,msg)=>events.push([name,msg]);x.assets={expose:async x=>x};
 const r=structuredClone(actual);r.validationLandmarks='FORBIDDEN_SENTINEL';
 x.pending.set('review',{op:'inspectObservedLandmarkUpdate',resolve(){},reject(){},timeout:undefined});
 try{await x.handle({id:'review',event:'result',result:r});}catch(e){}
 assert.equal(events.some(([name,msg])=>name==='event'&&msg.event==='result'),false);
});
