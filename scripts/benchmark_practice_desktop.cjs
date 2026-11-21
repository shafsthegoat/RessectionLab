'use strict';

// Run the same trusted manual controls against two production-built renderers.
// Each invocation gets a fresh Electron profile and a separate evidence folder.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { execFileSync, spawn } = require('node:child_process');
const { parseArgs } = require('node:util');
const { summarize } = require('./summarize_practice_desktop.cjs');

const root = path.resolve(__dirname, '..');
const mount = '/Volumes/shafrir';
const dataRoot = path.join(mount, 'old coding projects/RessectionLabData');
const evidenceRoot = path.join(dataRoot, 'qc/rehearsal-performance-20261121');
const historical = path.join(dataRoot, 'qc/interactive-operative-practice-20261121');
const defaultBaseline = 'a07df30938636547a3e6622fdc12a475a2ac9461';
const controls = {
  tissue:{sourceSha256:'a2ce46b66912199e891c761fff64114444ed925cf5b2a9ffbecbf93db2b8ce2a',
    caseHash:'sha256:c298887cb67b1034d782d682b01ee03cdaf86ee13237602eb3239ec114d5a710',
    actionSha256:'64ed965204b26c53d636a3118c5d548ecbfb987d02d9006fc7965dca594342c4',branch:'configured-start',startPosition:1},
  cranial:{sourceSha256:'0f8a8800558250fa7de1dcdc0f398ec53525b2370308d5eea9cfd0e3d445b269',
    caseHash:'sha256:9818506d2f779b8fece10d048e0f21e2357f86851936ad0f5b86ee6f6cfaa1e0',startPosition:105},
};
const sha = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const write = (filename, data) => fs.writeFileSync(filename, JSON.stringify(data, null, 2) + '\n');
const command = (program, args, cwd = root) => execFileSync(program, args, { cwd, encoding: 'utf8' }).trim();

function verifyStorage() {
  const info = command('diskutil', ['info', '-plist', mount]);
  const disk = JSON.parse(execFileSync('plutil', ['-convert', 'json', '-o', '-', '-'], { input: info }));
  assert.equal(disk.MountPoint, mount);
  assert.equal(disk.VolumeUUID, '9CC3623C-E38B-3EEE-B6BB-B9A06C193700');
  const marker = read(path.join(dataRoot,'.ressectionlab-data-root.json'));
  assert.equal(marker.schema, 'ressectionlab-data-root-v1');
  assert.equal(marker.mount_point, mount);
  assert.equal(marker.volume_uuid, disk.VolumeUUID);
  assert.equal(marker.root_id, '756c4d3b-aabf-4837-a354-52aa500af3f9');
  return marker;
}

function workloads(manifestFile) {
  const fixtureDirectory=manifestFile?path.dirname(path.resolve(manifestFile)):historical;
  const files=manifestFile?read(manifestFile):{
    tissue:'ready/bipolar-thermal-source.ressectionlab',
    cranial:'../continuous-rehearsal-20261118/generated-cleanup/final/complete.ressectionlab',
    tissueActions:'mechanics/paired-actions.json',
  };
  const actions=fs.realpathSync(path.resolve(fixtureDirectory,files.tissueActions));
  assert.equal(sha(actions),controls.tissue.actionSha256,'Use the already archived paired tissue action stream');
  return Object.entries(controls).map(([id,identity])=>{
    const source=fs.realpathSync(path.resolve(fixtureDirectory,files[id]));
    assert.equal(sha(source),identity.sourceSha256);
    const manifest=JSON.parse(execFileSync('unzip',['-p',source,'manifest.json'],{maxBuffer:32*1024*1024}));
    assert.equal(manifest.case_semantic_hash,identity.caseHash);
    return {id,...identity,source,caseId:manifest.case_id,arraySha256:manifest.array_sha256,
      physicalModel:manifest.metadata.surgical_properties,researchSetup:manifest.artifacts.surgical_toolbox.researchSetup,
      ...(id==='tissue'?{actions,args:['--paired-actions',actions]}:{args:[]})};
  });
}

function build(workspace, revision) {
  const main=path.join(workspace,'desktop/electron/main.cjs');
  const electron=path.join(workspace,'desktop/node_modules/.bin/electron');
  assert(fs.existsSync(electron) && fs.existsSync(path.join(workspace,'desktop/dist/index.html')));
  const binaryDirectory=path.join(workspace,'src/resectionlab');
  const nativeBinaries=Object.fromEntries(fs.readdirSync(binaryDirectory).filter(name=>name.endsWith('.so')).map(name=>
    [name,sha(path.join(binaryDirectory,name))]));
  assert(Object.keys(nativeBinaries).some(name=>name.startsWith('_geometry_cpp.')), 'Build the ordinary native extension first');
  return {workspace, main, electron, revision:revision ?? command('git',['rev-parse','HEAD'],workspace),nativeBinaries};
}

function loadSnapshot() {
  return {loadAverage:os.loadavg(),freeBytes:os.freemem(),
    competingCpu:command('ps',['-A','-o','pid=,pcpu=,rss=,comm=']).split('\n').flatMap(line=>{
      const record=line.trim().match(/^(\d+)\s+([\d.]+)\s+(\d+)\s+(.+)$/);
      return record&&Number(record[2])>10&&Number(record[1])!==process.pid?
        [{pid:Number(record[1]),cpuPercent:Number(record[2]),residentKiB:Number(record[3]),process:path.basename(record[4])}]:[];
    })};
}

async function invoke(which, workload, physicalDirectory, logicalDirectory, phase, attempt, extra=[]) {
  fs.mkdirSync(physicalDirectory,{recursive:true});
  const log=fs.openSync(path.join(physicalDirectory,phase+'.log'),'wx');
  const cache=fs.mkdtempSync(path.join(os.tmpdir(),'rslab-practice-benchmark-'));
  const env = {...process.env, REHEARSAL_PERF_MAIN:which.main, REHEARSAL_PERF_BUILD_REVISION:which.revision,
    REHEARSAL_PERF_NATIVE_BINARIES:JSON.stringify(which.nativeBinaries),
    XDG_CACHE_HOME:cache, PYTHONDONTWRITEBYTECODE:'1'};
  for(const variable of ['RESSECTIONLAB_DATA_ROOT','RESSECTIONLAB_DEV_URL','RESSECTIONLAB_PREPARED_PREFIX_CACHE_DIRECTORY',
    'RESSECTIONLAB_REHEARSAL_BUNDLE_ROOT','RESSECTIONLAB_NATIVE_DEFORMATION','RESSECTIONLAB_DEFORMATION_DIAGNOSTICS',
    'RESSECTIONLAB_REHEARSAL_TIMING'])
    delete env[variable];
  const diagnostic=phase==='diagnostic';
  if(diagnostic)env.RESSECTIONLAB_REHEARSAL_TIMING='1';
  const flags=phase==='recording'?[]:[phase==='cold'?'--cold-benchmark':'--benchmark'];
  const args=[path.join(root,'desktop/electron/verify-manual-practice.cjs'),workload.source,logicalDirectory,
    '--matched-performance',...workload.args,...flags,...extra];
  const attemptStart=Date.now();
  const receipt={phase,attempt,case:workload.id,build:which.revision,startUtc:new Date().toISOString(),
    executable:which.electron,args,environment:{RESSECTIONLAB_REHEARSAL_TIMING:diagnostic?'1':null,PYTHONDONTWRITEBYTECODE:'1'},
    before:loadSnapshot()};
  console.log(phase,attempt,workload.id,which.workspace);
  try {
    const {code,signal}=await new Promise((resolve,reject)=>{
      const child=spawn(which.electron,args,{cwd:which.workspace,env,stdio:['ignore',log,log],detached:true});
      const signalGroup=signal=>{
        try { process.kill(-child.pid,signal); } catch (error) { if(error.code!=='ESRCH') reject(error); }
      };
      let escalation;
      const timeout=setTimeout(()=>{
        signalGroup('SIGTERM');
        escalation=setTimeout(()=>signalGroup('SIGKILL'),5000);
      },300000);
      child.on('error',reject);
      child.on('close',(code,signal)=>{clearTimeout(timeout);clearTimeout(escalation);resolve({code,signal});});
    });
    receipt.code=code;receipt.signal=signal;
    const filename={recording:'interaction',cold:'benchmark-cold'}[phase] ?? 'benchmark';
    const report=path.join(physicalDirectory,filename+'.json');
    assert.equal(code,0,'Hidden verifier failed. Inspect '+path.join(physicalDirectory,phase+'.log'));
    const proof=read(report);
    assert.equal(proof.passed,true,report);
    if(proof.recordingSpool)assert(!fs.existsSync(proof.recordingSpool.path),'Disposable recording spool survived app shutdown');
    return report;
  } finally {
    receipt.wallMs=Date.now()-attemptStart;
    receipt.after=loadSnapshot();
    write(path.join(physicalDirectory,phase+'-invocation.json'),receipt);
    fs.closeSync(log);
    fs.rmSync(cache,{recursive:true,force:true});
  }
}

async function run() {
  const {values}=parseArgs({options:{
    baseline:{type:'string',default:path.join(root,'outputs/perf-baseline')},
    candidate:{type:'string',default:root},
    'baseline-revision':{type:'string',default:defaultBaseline},
    'candidate-revision':{type:'string'},
    output:{type:'string'}, pairs:{type:'string',default:'5'},
    pilot:{type:'boolean'}, acceptance:{type:'boolean'}, diagnostics:{type:'boolean'}, interactions:{type:'boolean'}, recordings:{type:'boolean'},
    case:{type:'string',multiple:true}, 'fixture-manifest':{type:'string'},
    help:{type:'boolean',short:'h'},
  }});
  if(values.help){
    console.log('Usage: node scripts/benchmark_practice_desktop.cjs --baseline ROOT --candidate ROOT --output FRESH_DIRECTORY');
    console.log('  [--pairs 5 | --pilot | --diagnostics | --acceptance | --interactions | --recordings] [--case tissue] [--case cranial]');
    console.log('  [--baseline-revision SHA] [--candidate-revision SHA] [--fixture-manifest LOCAL_JSON]');
    console.log('By default, use the verified personal HDD fixtures and performance evidence root.');
    console.log('A local JSON fixture manifest has tissue, cranial and tissueActions paths relative to itself.');
    console.log('With local fixtures, put the fresh output directory under this checkout’s ignored testcases/outputs/.');
    return;
  }
  assert(values.output,'Provide a fresh --output directory');
  assert([values.pilot,values.acceptance,values.diagnostics,values.interactions,values.recordings].filter(Boolean).length<=1,
    'Choose a single benchmark phase');
  const count=Number(values.pairs);
  assert(Number.isSafeInteger(count)&&count>=5,'Use at least five paired runs');
  const marker=values['fixture-manifest']?null:verifyStorage();
  const outputs=path.join(root,'testcases/outputs');
  const physicalRoot=marker?evidenceRoot:outputs;
  const localEvidence=marker?path.join(outputs,'rehearsal-performance-20261121'):outputs;
  const target=path.resolve(values.output);
  const relative=path.relative(physicalRoot,target);
  assert(relative && !relative.startsWith('..') && !path.isAbsolute(relative), 'Use a fresh performance evidence directory');
  if(!marker)fs.mkdirSync(outputs,{recursive:true});
  assert.equal(fs.realpathSync(localEvidence),physicalRoot);
  assert.equal(fs.realpathSync(path.dirname(target)),path.dirname(target),'Use a direct directory inside the selected output root');
  fs.mkdirSync(target);
  const allCases=workloads(values['fixture-manifest']);
  const requestedCases=values.case??(values.interactions?['cranial']:allCases.map(item=>item.id));
  const cases=allCases.filter(item=>requestedCases.includes(item.id));
  assert(cases.length&&requestedCases.length===cases.length,'Valid --case values are tissue and cranial');
  if(values.interactions)assert(cases.length===1&&cases[0].id==='cranial','High-cadence qualification uses the cranial control');
  const baseline=build(fs.realpathSync(values.baseline),values['baseline-revision']);
  const candidate=values.pilot?null:build(fs.realpathSync(values.candidate),values['candidate-revision']);
  assert(!candidate||candidate.workspace!==baseline.workspace);
  const plan={schema:'resectionlab-paired-practice-performance-v1', baseline, candidate, cases, marker,
    machine:{model:command('sysctl',['-n','hw.model']),macOS:command('sw_vers',['-productVersion']),osBuild:command('sw_vers',['-buildVersion'])},
    reports:[]};
  const runCase=async (variant,item,phase,round,directory,extra)=>{
    const local=path.join(localEvidence,relative,directory),absolute=path.join(target,directory);
    const report=await invoke(plan[variant],item,absolute,local,phase,round,extra);
    plan.reports.push({variant,phase,round,case:item.id,report:path.relative(target,report)});
    write(path.join(target,'plan.json'),plan);
  };
  write(path.join(target,'plan.json'),plan);
  if(values.pilot){
    for(const item of cases) await runCase('baseline',item,'pilot',0,path.join('baseline',item.id));
    return;
  }
  if(values.interactions){
    for(const variant of ['baseline','candidate'])
      await runCase(variant,cases[0],'interaction-cadence',0,variant,['--interaction-cadence']);
  } else if(values.recordings){
    for(const variant of ['baseline','candidate'])for(const item of cases)
      await runCase(variant,item,'recording',0,path.join('recordings',variant,item.id),['--local-recording-spool']);
  } else if(values.diagnostics){
    for(const variant of ['baseline','candidate']) for(const item of cases)
      await runCase(variant,item,'diagnostic',0,path.join(variant,item.id));
  } else if(values.acceptance){
    for(const variant of ['baseline','candidate']) for(const item of cases){
      await runCase(variant,item,'recording',0,path.join('recordings',variant,item.id));
      const directory=path.join('reopen',variant,item.id);
      await runCase(variant,item,'warm-save',0,directory,['--performance-stills']);
      await runCase(variant,item,'cold',0,directory);
    }
  } else {
    for(let round=0;round<=count;round++){
      const phase=round===0?'warmup':'paired';
      const order=round%2===1?['baseline','candidate']:['candidate','baseline'];
      for(const variant of order) for(const item of cases)
        await runCase(variant,item,phase,round,path.join(phase,String(round),variant,item.id));
    }
  }
  write(path.join(target,'summary.json'),summarize(target));
}

run().catch(error=>{console.error(error);process.exitCode=1;});
