'use strict';
const {familyAvailabilityRequest,familyRequest,checkedFamilyAvailability,validateFamilyResult}=require('./contact-family.cjs');
const {contactRequest,validateContactResult}=require('./public-contact.cjs');
const {comparisonRequest,validateComparisonResult}=require('./episode-comparison.cjs');
const {vascularRequest,validateVascularResult}=require('./episode-vascular.cjs');

const { app, BrowserWindow, dialog, ipcMain, session, Menu, screen } = require('electron');
const {secondaryWindowOptions,isSecondaryDisplayConnected}=require('./secondary-display.cjs');
const {surgicalTimeoutMs}=require('./surgical-timeout.cjs');
const {runFrozenCusa,loadFrozenWorkspace}=require('./frozen-cusa.cjs');
const {randomUUID}=require('node:crypto');
const path = require('node:path');
const fs = require('node:fs/promises');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
const { episodeRequest } = require('./development-episode.cjs');
const { persistentPassiveRequest } = require('./persistent-passive.cjs');
const { Sidecar } = require('./sidecar.cjs');
const { PrefixEngine } = require('./prefix-engine.cjs');
const { createLogger } = require('./logging.cjs');
const { assertSender, plainArgs } = require('./security.cjs');
const { importStructuralEvidence } = require('./structural-import.cjs');
const { importDisplaySeries } = require('./display-series-import.cjs');
const { importDiagnosticLayer } = require('./diagnostic-import.cjs');
const { inspectAxisPlanning } = require('./axis-inspection.cjs');
const { observedRequest, validateObservedResult } = require('./observed-landmark-contract.cjs');
const { CaseLibrarySelection } = require('./case-library-selection.cjs');

let log = message => process.stderr.write(String(message) + '\n');
let window;
let engine;
let allowedUrl;
let quitting = false;
let startupConsumed = false;
let caseLibrary;
let currentCasePath;
let pendingOpenPath;
let frozenCusaAbort;
let frozenCusaAdmission;
const { RehearsalUnsaved } = require('./rehearsal-unsaved.cjs');
const rehearsalUnsaved = new RehearsalUnsaved(async()=>{
  const answer = await dialog.showMessageBox(window, {type:'warning', title:'Unsaved rehearsal',
    message:'This rehearsal has unsaved changes.', detail:'Keep working to save the workspace, or discard these changes before opening another environment.',
    buttons:['Keep working','Discard unsaved rehearsal'],defaultId:0,cancelId:0,noLink:true});
  return answer.response === 1;
});
let currentCaseHash;
let currentCaseName = 'research-case';
const secondaryDisplayOnly=process.argv.includes('--secondary-display');
const automationHidden=process.argv.includes('--automation-hidden');
if((secondaryDisplayOnly||automationHidden)&&process.platform==='darwin')app.setActivationPolicy('accessory');
let secondaryDisplayId;
let secondaryDisplayUnavailable=false;
const casePaths = new Map();
const cliValue = name => { const index = process.argv.indexOf(name); return index >= 0 ? process.argv[index + 1] : undefined; };

async function startEngine() {
  const bundled = path.join(process.resourcesPath, 'research-engine', 'ressectionlab-engine');
  const repo = path.resolve(__dirname, '../..');
  caseLibrary = new CaseLibrarySelection({userData:app.getPath('userData'),defaultRoot:app.isPackaged?null:repo,
    preparedRoot:app.isPackaged?path.join(app.getPath('userData'),'prepared-cases'):path.join(repo,'outputs/poc-cases'),nativePaths:[path.join(app.getPath('downloads'),'RessectionLab-Native-Anatomy-sub-06/patient-anatomy.json')]});
  await caseLibrary.restore();
  const runDir = path.join(app.getPath('userData'), 'research-runs');
  const config = app.isPackaged
    ? { executable: bundled, cwd: app.getPath('userData'), runDir }
    : { python: path.join(repo, '.venv/bin/python'), cwd: repo, sourcePath: path.join(repo, 'src'), runDir };
  // Existing packaged engines do not accept the new argument. They remain
  // startable and refuse the unsupported observation operation independently.
  if (!app.isPackaged) config.observedSourceRoot = repo;
  await fs.mkdir(app.getPath('userData'), { recursive: true });
  await fs.access(config.executable || config.python);
  const createWorker = async ({prefixWorker} = {}) => {
    const transferDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-'));
    await fs.chmod(transferDir, 0o700);
    try { return new Sidecar({...config, transferDir, ...(prefixWorker ? {prefixWorker} : {})}); }
    catch (error) { await fs.rm(transferDir, {recursive:true, force:true}); throw error; }
  };
  // Qualification remains explicit while the default single-worker replay is
  // unchanged. Existing packaged engines do not implement the fixed roles.
  const pipelineEnabled = !app.isPackaged && process.env.RESSECTIONLAB_PREFIX_PIPELINE === '1';
  const maxWallMs = process.env.RESSECTIONLAB_PREFIX_PIPELINE_MAX_WALL_MS === undefined
    ? 7200000 : Number(process.env.RESSECTIONLAB_PREFIX_PIPELINE_MAX_WALL_MS);
  if (pipelineEnabled && (!Number.isSafeInteger(maxWallMs) || maxWallMs < 100 || maxWallMs > 7200000))
    throw new Error('Invalid prefix pipeline wall-time budget');
  engine = pipelineEnabled
    ? await PrefixEngine.create({createWorker, enabled:true, sourceForHash:hash => casePaths.get(hash), maxWallMs})
    : await createWorker();
  engine.on('event', payload => { if (window && !window.isDestroyed()) window.webContents.send('research:event', payload); });
  engine.on('diagnostic', message => log(message));
}

function handle(name, operation) {
  ipcMain.handle(`research:${name}`, async (event, ...args) => {
    assertSender(event, window, allowedUrl);
    if (!engine) throw new Error('Local research engine is unavailable');
    return operation(...args);
  });
}

async function pick(title, extensions) {
  const selected = await dialog.showOpenDialog(window, { title, properties: ['openFile'], filters: [{ name: 'Local research data', extensions }] });
  return selected.canceled ? null : selected.filePaths[0];
}

function bindOperations() {
  const openWorkspace = async filename => {
    if (!await rehearsalUnsaved.allowNavigation()) return null;
    const native = path.basename(filename)==='patient-anatomy.json'||filename.endsWith('.rsworkspace');
    const result=await engine.request(native?'anatomyOpen':'loadCase',{path:filename,...(filename.endsWith('.rsworkspace')?{workingRoot:path.join(app.getPath('userData'),'patient-studies')}:{})},300000);
    currentCasePath=filename;if(result?.caseHash)casePaths.set(result.caseHash,filename);await caseLibrary.remember(filename);return result;
  };
  handle('workspaceInstalled', args=>{args=plainArgs(args,['caseHash','caseId']);currentCaseHash=args.caseHash;currentCaseName=String(args.caseId||'research-case').replace(/[^a-zA-Z0-9._-]/g,'-');rehearsalUnsaved.installed();return true;});
  handle('openWorkspace',async()=>{
    const filename=pendingOpenPath||await pick('Open patient workspace',['rsworkspace','ressectionlab','rslab','json']);pendingOpenPath=null;
    return filename?openWorkspace(path.resolve(filename)):null;
  });
  handle('openListedWorkspace',async id=>openWorkspace(await caseLibrary.resolve(id)));
  handle('startupWorkspace',async()=>{
    if(startupConsumed)return null;startupConsumed=true;
    const filename=pendingOpenPath||cliValue('--anatomy')||cliValue('--case');pendingOpenPath=null;
    if(filename)return openWorkspace(path.resolve(filename));
    const cases=await caseLibrary.list();return cases.length?openWorkspace(await caseLibrary.resolve(cases[0].id)):null;
  });
  const openSaved = async filename => {
    if (!await rehearsalUnsaved.allowNavigation()) return null;
    const result = await engine.request('loadCase', {path: filename});
    currentCasePath = filename;
    casePaths.set(result.caseHash,filename);
    await caseLibrary.remember(filename);
    return result;
  };
  handle('anatomyCreate', async args => {
    if (!await rehearsalUnsaved.allowNavigation()) return null;
    plainArgs(args,['patientId','timepoint']);
    const selected=await dialog.showOpenDialog(window,{title:'Choose an empty patient collection directory',properties:['openDirectory','createDirectory']});
    if(selected.canceled)return null;const result=await engine.request('anatomyCreate',{...args,path:selected.filePaths[0]});await caseLibrary.remember(path.join(selected.filePaths[0],'patient-anatomy.json'));return result;
  });
  handle('anatomyOpen', async () => {
    const filename = cliValue('--anatomy') || await pick('Open native patient collection', ['json']);
    return filename ? engine.request('anatomyOpen', {path:path.resolve(filename)}) : null;
  });
  const anatomyKeys = {
    anatomySaveView: ['viewState','saveArchive'],
    anatomyComponents: ['layerId'],
    anatomyGraph: ['layerId','point'],
    anatomyReconstruct: ['seriesId','model','synthsegVersion','brainLayerId'],
    anatomyPlan: ['start','end','radiusMm'], anatomyRemoval: ['point','radiusMm','reset'],
    anatomySection: ['seriesId','center','normal','spacingMm','movingId','checkerboard','layers'],
    anatomyView: ['seriesId','layers','original','highlightLayer','highlightComponents','reviewId','previewAlignment','vesselScope'],
    anatomyEdit: ['action','layerId','author','points','radiusMm','threshold','reviewId','componentIds'],
    anatomySample: ['point'],
    anatomyRegister: ['action','seriesId','movingPoints','referencePoints','reviewer'],
  };
  for (const [operation, keys] of Object.entries(anatomyKeys))
    handle(operation, args => engine.request(operation, plainArgs(args, ['collectionId','snapshotId',...keys]), operation==='anatomyReconstruct'?1800000:300000));
  handle('anatomyImport', async args => {
    args=plainArgs(args,['collectionId','snapshotId','kind','modality','timepoint','patientId','seriesId','label','roleMap','sequenceBySeries']);
    const selected=await dialog.showOpenDialog(window,{title:'Import this patient’s imaging',properties:[args.kind==='dicom'?'openDirectory':'openFile']});
    if(selected.canceled)return null;
    return engine.request('anatomyImport',{...args,path:selected.filePaths[0]},300000);
  });
  handle('anatomySavePackage', async args => {
    args=plainArgs(args,['collectionId','snapshotId']);
    const selected=await dialog.showSaveDialog(window,{title:'Save portable patient study',defaultPath:'Patient-study.rsworkspace',filters:[{name:'Patient study',extensions:['rsworkspace']}]});
    if(selected.canceled||!selected.filePath)return null;
    const result=await engine.request('anatomySavePackage',{...args,path:selected.filePath},300000);
    await caseLibrary.remember(selected.filePath);return result;
  });
  handle('anatomyExport', async args => {
    const selected=await dialog.showOpenDialog(window,{title:'Select simulation snapshot destination',properties:['openDirectory','createDirectory']});
    if(selected.canceled)return null;
    return engine.request('anatomyExport',{...plainArgs(args,['collectionId','snapshotId']),path:selected.filePaths[0]},300000);
  });
  handle('ping', () => engine.request('ping'));
  handle('createSurgicalFixture', async () => {
    if (!await rehearsalUnsaved.allowNavigation()) return null;
    const result=await engine.request('createSurgicalFixture', {}); currentCasePath=null; return result;
  });
  handle('prepareSurgicalEnvironment', async args=>{
    if(!await rehearsalUnsaved.allowNavigation())return null;
    args=plainArgs(args,['collectionId','snapshotId','layers','allowUnreviewedRegistration']);
    const result=await engine.request('prepareSurgicalEnvironment',args,300000);currentCasePath=null;return result;
  });
  handle('importSurgicalScene', async args => {
    if (!await rehearsalUnsaved.allowNavigation()) return null;
    args=plainArgs(args,['frame','units','material','resolution','topology','thicknessMm']);
    const filename=await pick('Import segmented geometry', ['nii','gz','obj','stl']);
    if (!filename) return null;
    const result=await engine.request('importSurgicalScene', {...args,path:filename}, 120000);currentCasePath=null;return result;
  });
  handle('compareSurgicalAnatomy', args => engine.request('compareSurgicalAnatomy', plainArgs(args,['beforeCaseHash','afterCaseHash','changedLayers','prefix','decisions']),300000));
  handle('executeFrozenCusa', async args => {
    args=plainArgs(args,['caseHash','branchId','kind']);
    const kind=args.kind??'cusa';
    if(!['cusa','sector'].includes(kind))throw Error('Unknown frozen policy kind');
    if(app.isPackaged)throw Error('Frozen CUSA execution requires the local research checkout and matching Python runtime');
    if(frozenCusaAbort||frozenCusaAdmission)throw Error('A frozen CUSA episode is already running or awaiting display');
    if(args.caseHash!==currentCaseHash||typeof args.branchId!=='string'||!args.branchId||args.branchId.length>128)throw Error('Choose an access approach from the current case');
    frozenCusaAbort=new AbortController();
    try {
      const checkpoint=await pick(kind==='sector'?'Choose a compatible frozen sector checkpoint':'Choose a compatible frozen CUSA checkpoint',['pt']);
      if(!checkpoint)return null;
      const repo=path.resolve(__dirname,'../..');
      const folder=path.join(repo,kind==='sector'?'testcases/outputs/sector-desktop':'testcases/outputs/cusa-desktop',new Date().toISOString().replace(/[:.]/g,'-')+'-'+randomUUID().slice(0,8));
      await fs.mkdir(folder,{recursive:true});
      const source=path.join(folder,'input.ressectionlab');
      // Snapshot the current cached workspace, including unsaved approaches.
      // The original file and its navigation/save state remain untouched.
      await engine.request('saveCase',{caseHash:args.caseHash,path:source,overwrite:false},300000);
      const result=await runFrozenCusa({repo,source,branchId:args.branchId,checkpoint,kind,
        output:path.join(folder,'execution'),signal:frozenCusaAbort.signal});
      frozenCusaAbort.signal.throwIfAborted();
      const next=await loadFrozenWorkspace(filename=>engine.request('loadCase',{path:filename},300000),
        source,result.workspacePath,frozenCusaAbort.signal);
      const admissionId=randomUUID();
      frozenCusaAdmission={admissionId,source,result,caseHash:next.caseHash};
      return {case:next,...result,admissionId};
    } catch(error) {
      if(frozenCusaAbort.signal.aborted)throw Error('Operation cancelled · Frozen policy source and any experiment evidence were preserved');
      throw error;
    } finally {frozenCusaAbort=null;}
  });
  handle('finishFrozenCusa',async args=>{
    args=plainArgs(args,['admissionId','accepted']);
    const pending=frozenCusaAdmission;
    if(!pending||args.admissionId!==pending.admissionId||typeof args.accepted!=='boolean')throw Error('Frozen episode display admission is unavailable');
    if(args.accepted){
      currentCasePath=pending.result.workspacePath;casePaths.set(pending.caseHash,pending.result.workspacePath);
      // Library bookkeeping cannot invalidate an already displayed native case.
      try {await caseLibrary.remember(pending.result.workspacePath);}catch(error){log('Could not remember frozen episode: '+error);}
    }else await engine.request('loadCase',{path:pending.source},300000);
    frozenCusaAdmission=null;return true;
  });
  handle('cancelFrozenCusa',()=>{frozenCusaAbort?.abort();return true;});
  handle('surgicalRehearsal', async args => {
    args=plainArgs(args, ['caseHash','command','expectedStateId','action','mapping','position','frameIndex','cacheOnly','timelineToken','viewFieldTokens','prefixToken','prefixFieldTokens','branchId','source','sceneIndex','acknowledgeResearchUnknowns','opponent','allowModeledBrainCorridor','scenario','maxActions','initialAccessRadiusMm','exploreExteriorFaces','cumulativeReentry','policyInput','expectedComparisonSeal']);
    if(args.command==='attachProtection'){
      if(!currentCaseHash||args.caseHash!==currentCaseHash||typeof args.expectedStateId!=='string'||!args.expectedStateId||args.source)throw Error('Choose the accepted state in the current workspace before attaching protection');
      const caseHash=currentCaseHash,protectionPath=await pick('Choose surgical-rl-protection-v1 JSON',['json']);
      if(!protectionPath)return null;
      if(currentCaseHash!==caseHash)throw Error('Workspace changed while choosing protection');
      args={...args,protectionPath};
    }
    const result=await engine.request('surgicalRehearsal',args,surgicalTimeoutMs(args.command));rehearsalUnsaved.changed(args,result);return result;
  });
  for (const op of ['createRehearsal','inspectRehearsal','stepRehearsal','compareRehearsal']) {
    handle(op, async args => {const result=await engine.request(op, plainArgs(args, op === 'stepRehearsal'
      ? ['caseHash','expectedStateId','action'] : op === 'compareRehearsal'
      ? ['caseHash','expectedStateId'] : ['caseHash']), 120000);if(op!=='inspectRehearsal')rehearsalUnsaved.changed({command:'step'},result);return result;});
  }
  handle('listCases', () => caseLibrary.list());
  handle('caseLibraryInfo', () => caseLibrary.info());
  handle('chooseCaseLibrary', args => {
    plainArgs(args??{},[]);
    return caseLibrary.choose(()=>dialog.showOpenDialog(window,{title:'Choose case library',properties:['openDirectory'],...(caseLibrary.info().folder?{defaultPath:caseLibrary.info().folder}:{})}));
  });
  handle('openListedCase', async id => openSaved(await caseLibrary.resolve(id)));
  handle('startupCase', async () => {
    if (startupConsumed) return null;
    startupConsumed = true;
    const casePath = cliValue('--case');
    if (casePath) return openSaved(path.resolve(casePath));
    if (process.argv.includes('--demo')) return engine.request('createSyntheticCase');
    const cases = await caseLibrary.list();
    if (!cases.length) return null;
    return openSaved(await caseLibrary.resolve(cases[0].id));
  });
  handle('publicContactFamilyAvailability',async args=>checkedFamilyAvailability(await engine.request('publicContactFamilyAvailability',familyAvailabilityRequest(args),30000)));
  handle('executePublicContactFamilyEpisode',async args=>{const catalog=structuredClone(engine.contactFamilyAvailability);const request=familyRequest(args,catalog);return validateFamilyResult(await engine.request('executePublicContactFamilyEpisode',request,30000),request,catalog);});
  handle('executePublicSurfaceContactEpisode', async args => { const request=contactRequest(args); return validateContactResult(await engine.request('executePublicSurfaceContactEpisode',request,30000),request); });
  handle('inspectDevelopmentEpisodeComparison', async args => { const request=comparisonRequest(args); return validateComparisonResult(await engine.request('inspectDevelopmentEpisodeComparison',request,30000),request); });
  handle('executeDevelopmentEpisode', args => engine.request('executeDevelopmentEpisode', episodeRequest(args), 30000));
  handle('executePersistentPassiveEpisode', args => engine.request('executePersistentPassiveEpisode', persistentPassiveRequest(args), 30000));
  handle('evaluateDevelopmentEpisodeVascular', async args => { const request=vascularRequest(args); return validateVascularResult(await engine.request('evaluateDevelopmentEpisodeVascular',request),request); });
  handle('createSyntheticCase', async () => await rehearsalUnsaved.allowNavigation()?engine.request('createSyntheticCase'):null);
  handle('openCase', async () => {
    const selected = await pick('Open a saved research case', ['ressectionlab', 'rslab']);
    return selected ? openSaved(selected) : null;
  });
  handle('importNifti', async () => {
    if (!await rehearsalUnsaved.allowNavigation()) return null;
    const structuralPath = await pick('Select the structural MRI', ['nii', 'gz']);
    if (!structuralPath) return null;
    const tumorMaskPath = await pick('Select the tumor segmentation (Cancel to import MRI only)', ['nii', 'gz']);
    const result = await engine.request('importNifti', { structuralPath, ...(tumorMaskPath ? { tumorMaskPath } : {}) });
    currentCasePath = null;
    return result;
  });
  handle('importDiagnosticLayer', args => importDiagnosticLayer(args, { pick, request: (...requestArgs) => engine.request(...requestArgs) }));
  handle('importDisplaySeries', args => importDisplaySeries(args, { pick, request: (...requestArgs) => engine.request(...requestArgs) }));
  handle('importStructuralEvidence', args => importStructuralEvidence(args, { pick, request: (...requestArgs) => engine.request(...requestArgs) }));
  handle('saveCase', async args => {
    plainArgs(args, ['caseHash', 'workspace']);
    const selected = await dialog.showSaveDialog(window, { title: 'Save anatomy workspace', defaultPath: casePaths.get(args.caseHash) || currentCaseName+'.ressectionlab', filters: [{ name: 'RessectionLab case', extensions: ['ressectionlab'] }] });
    if (selected.canceled || !selected.filePath) return null;
    const result = await engine.request('saveCase', { ...args, path: selected.filePath, overwrite: true });
    currentCasePath = selected.filePath;
    casePaths.set(args.caseHash,selected.filePath);
    rehearsalUnsaved.saved();
    await caseLibrary.remember(selected.filePath);
    return result;
  });
  handle('generateRoutes', args => engine.request('generateRoutes', plainArgs(args, ['caseHash', 'toolIds', 'allowEstimatedSupport', 'config'])));
  handle('generateNativeRoutes', args => engine.request('generateNativeRoutes', plainArgs(args, ['caseHash'])));
  handle('inspectRefinement', args => engine.request('inspectRefinement', plainArgs(args, ['caseHash', 'routeId'])));
  handle('inspectAxisPlanning', args => inspectAxisPlanning(args, { request: (...requestArgs) => engine.request(...requestArgs) }));
  handle('inspectObservedLandmarkUpdate', async args => {
    const request = observedRequest(args);
    const result = await engine.request('inspectObservedLandmarkUpdate', request);
    return validateObservedResult(result, request);
  });
  handle('inspectEvidence', args => engine.request('inspectEvidence', plainArgs(args, ['caseHash'])));
  handle('trainPatient', args => engine.request('trainPatient', plainArgs(args, ['caseHash', 'budgetSeconds', 'seed', 'routeId', 'resumeRunId']), 300000));
  handle('listRuns', args => engine.request('listRuns', plainArgs(args, ['caseHash'])));
  handle('replayTraining', args => engine.request('replayTraining', plainArgs(args, ['caseHash', 'runId', 'step'])));
  handle('exportCandidate', async args => {
    plainArgs(args, ['caseHash', 'runId']);
    const selected = await dialog.showSaveDialog(window, { title: 'Export research candidate record', defaultPath: 'candidate.json', filters: [{ name: 'Research candidate', extensions: ['json'] }] });
    if (selected.canceled || !selected.filePath) return null;
    return engine.request('exportCandidate', { ...args, path: selected.filePath, overwrite: true });
  });
  handle('cancel', requestId => {
    if (typeof requestId !== 'string' || requestId.length > 128) throw new Error('Invalid request identifier');
    return engine.request('cancel', { requestId });
  });
  handle('readAsset', assetId => engine.assets.read(assetId));
}

async function createWindow() {
  const secondary=secondaryDisplayOnly&&!automationHidden?secondaryWindowOptions(screen.getAllDisplays(),screen.getPrimaryDisplay().id):null;
  secondaryDisplayId=secondary?.displayId;
  const devUrl = !app.isPackaged && process.env.RESSECTIONLAB_DEV_URL;
  if (devUrl && devUrl !== 'http://127.0.0.1:5173/') throw new Error('Development preview must use the configured local address');
  allowedUrl = devUrl || pathToFileURL(path.join(__dirname, '../dist/index.html')).href;
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  session.defaultSession.setPermissionCheckHandler(() => false);
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => callback({ responseHeaders: {
    ...details.responseHeaders,
    'Content-Security-Policy': ["default-src 'self'; script-src 'self'; worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'" + (devUrl ? ' ws://127.0.0.1:5173' : '') + "; object-src 'none'; base-uri 'none'; frame-src 'none'"],
  } }));
  window = new BrowserWindow({ width: 1460, height: 980, minWidth: 1050, minHeight: 720, ...secondary?.options, title: 'RessectionLab', backgroundColor: '#080909', show: false,
    webPreferences: { preload: path.join(__dirname, 'preload.cjs'), contextIsolation: true, nodeIntegration: false, sandbox: true, webSecurity: true, allowRunningInsecureContent: false, devTools: !app.isPackaged },
  });
  window.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  window.webContents.on('will-navigate', event => event.preventDefault());
  window.webContents.on('will-attach-webview', event => event.preventDefault());
  window.webContents.on('console-message', (_event, details) => { if (details.level === 'error' || details.level === 'warning') log(`Renderer ${details.level}: ${details.message}`); });
  window.webContents.on('render-process-gone', (_event, details) => log(`Renderer stopped: ${details.reason}`));
  const menuAction = action => { if (window && !window.isDestroyed()) window.webContents.send('research:event', {event: 'menuAction', action}); };
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    { role: 'appMenu' },
    { label: 'File', submenu: [
      { label: 'Open Case…', accelerator: 'CmdOrCtrl+O', click: () => menuAction('openCase') },
      { label: 'Save Workspace…', accelerator: 'CmdOrCtrl+S', click: () => menuAction('saveCase') },
      { type: 'separator' }, { role: 'close' },
    ] }, { role: 'editMenu' }, { role: 'viewMenu' }, { role: 'windowMenu' },
  ]));
  const secondaryAvailable=()=>!secondaryDisplayOnly||isSecondaryDisplayConnected(screen.getAllDisplays(),screen.getPrimaryDisplay().id,secondaryDisplayId);
  const guardSecondaryDisplay=()=>{
    if(secondaryDisplayOnly&&!secondaryAvailable()){
      secondaryDisplayUnavailable=true;window?.hide();log('Secondary display unavailable. Window hidden; restart with a connected secondary display to continue.');
    }
  };
  if(secondaryDisplayOnly){screen.on('display-removed',guardSecondaryDisplay);screen.on('display-metrics-changed',guardSecondaryDisplay);}
  window.once('ready-to-show', () => {if(!automationHidden&&secondaryAvailable()&&!secondaryDisplayUnavailable){if(secondaryDisplayOnly)window.showInactive();else window.show();}});
  window.on('close', event=>{
    if(!quitting&&rehearsalUnsaved.dirty){event.preventDefault();void rehearsalUnsaved.allowNavigation().then(allowed=>{if(allowed){rehearsalUnsaved.saved();window?.close();}});}
  });
  window.on('closed', () => {screen.removeListener('display-removed',guardSecondaryDisplay);screen.removeListener('display-metrics-changed',guardSecondaryDisplay);window = undefined;});
  await window.loadURL(allowedUrl);
}

const ownsAppInstance = app.requestSingleInstanceLock();
if (!ownsAppInstance) app.quit();
app.on('open-file',(event,filename)=>{event.preventDefault();pendingOpenPath=filename;if(window&&!window.isDestroyed())window.webContents.send('research:event',{event:'menuAction',action:'openCase'});});
app.on('second-instance', (_event,argv) => {
  if(automationHidden){log('Hidden automation instance remains hidden.');return;}
  if(argv.includes('--secondary-display')&&!secondaryDisplayOnly){log('Refused secondary-display request for an existing unrestricted window. Close the existing app and relaunch with --secondary-display.');return;}
  if(secondaryDisplayOnly&&(secondaryDisplayUnavailable||!isSecondaryDisplayConnected(screen.getAllDisplays(),screen.getPrimaryDisplay().id,secondaryDisplayId))){window?.hide();log('Secondary display unavailable; ignored second-instance request.');return;}
  const index=argv.findIndex(a=>a==='--case'||a==='--anatomy');
  const filename=index>=0?argv[index+1]:argv.find(a=>/\.(ressectionlab|rslab)$/.test(a)||a.endsWith('patient-anatomy.json')||a.endsWith('.rsworkspace'));
  if(filename){pendingOpenPath=path.resolve(filename);if(window&&!window.isDestroyed())window.webContents.send('research:event',{event:'menuAction',action:'openCase'});}

  if (window && !window.isDestroyed()) { if(secondaryDisplayOnly){window.showInactive();}else{if (window.isMinimized()) window.restore(); window.show(); window.focus();} }
});
if (ownsAppInstance) app.whenReady().then(async () => { log = createLogger(app.getPath('logs')); log(`Starting ${app.getVersion()} (${app.isPackaged ? 'packaged' : 'development'})`); await startEngine(); bindOperations(); await createWindow(); }).catch(error => {
  if(!secondaryDisplayOnly&&!automationHidden)dialog.showErrorBox('RessectionLab could not start', error.message); log(error.stack); app.quit();
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', event => {
  if (!quitting && engine) { event.preventDefault();void rehearsalUnsaved.allowNavigation().then(allowed=>{if(allowed){quitting = true;frozenCusaAbort?.abort();engine.stop().finally(() => app.quit());}}); }
});
