'use strict';
const {familyAvailabilityRequest,familyRequest,checkedFamilyAvailability,validateFamilyResult}=require('./contact-family.cjs');
const {contactRequest,validateContactResult}=require('./public-contact.cjs');
const {comparisonRequest,validateComparisonResult}=require('./episode-comparison.cjs');
const {vascularRequest,validateVascularResult}=require('./episode-vascular.cjs');

const { app, BrowserWindow, dialog, ipcMain, session, Menu } = require('electron');
const path = require('node:path');
const fs = require('node:fs/promises');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
const { episodeRequest } = require('./development-episode.cjs');
const { persistentPassiveRequest } = require('./persistent-passive.cjs');
const { Sidecar } = require('./sidecar.cjs');
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
const cliValue = name => { const index = process.argv.indexOf(name); return index >= 0 ? process.argv[index + 1] : undefined; };

async function startEngine() {
  const transferDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-'));
  await fs.chmod(transferDir, 0o700);
  const bundled = path.join(process.resourcesPath, 'research-engine', 'ressectionlab-engine');
  const repo = path.resolve(__dirname, '../..');
  caseLibrary = new CaseLibrarySelection({userData:app.getPath('userData'),defaultRoot:app.isPackaged?null:repo,
    preparedRoot:app.isPackaged?path.join(app.getPath('userData'),'prepared-cases'):path.join(repo,'outputs/poc-cases'),nativePaths:[path.join(app.getPath('downloads'),'RessectionLab-Native-Anatomy-sub-06/patient-anatomy.json')]});
  await caseLibrary.restore();
  const runDir = path.join(app.getPath('userData'), 'research-runs');
  const config = app.isPackaged
    ? { executable: bundled, cwd: app.getPath('userData'), transferDir, runDir }
    : { python: path.join(repo, '.venv/bin/python'), cwd: repo, sourcePath: path.join(repo, 'src'), transferDir, runDir };
  // Existing packaged engines do not accept the new argument. They remain
  // startable and refuse the unsupported observation operation independently.
  if (!app.isPackaged) config.observedSourceRoot = repo;
  await fs.mkdir(app.getPath('userData'), { recursive: true });
  await fs.access(config.executable || config.python);
  engine = new Sidecar(config);
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
    const native = path.basename(filename)==='patient-anatomy.json'||filename.endsWith('.rsworkspace');
    const result=await engine.request(native?'anatomyOpen':'loadCase',{path:filename,...(filename.endsWith('.rsworkspace')?{workingRoot:path.join(app.getPath('userData'),'patient-studies')}:{})},300000);
    currentCasePath=filename;await caseLibrary.remember(filename);return result;
  };
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
    const result = await engine.request('loadCase', {path: filename});
    currentCasePath = filename;
    await caseLibrary.remember(filename);
    return result;
  };
  handle('anatomyCreate', async args => {
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
    anatomyReconstruct: ['seriesId','model','synthsegVersion'],
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
    args=plainArgs(args,['collectionId','snapshotId','kind','modality','timepoint','patientId','seriesId','label']);
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
  for (const op of ['createRehearsal','inspectRehearsal','stepRehearsal','compareRehearsal']) {
    handle(op, args => engine.request(op, plainArgs(args, op === 'stepRehearsal'
      ? ['caseHash','expectedStateId','action'] : op === 'compareRehearsal'
      ? ['caseHash','expectedStateId'] : ['caseHash']), 120000));
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
  handle('createSyntheticCase', () => engine.request('createSyntheticCase'));
  handle('openCase', async () => {
    const selected = await pick('Open a saved research case', ['ressectionlab', 'rslab']);
    return selected ? openSaved(selected) : null;
  });
  handle('importNifti', async () => {
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
    const selected = await dialog.showSaveDialog(window, { title: 'Save anatomy workspace', defaultPath: currentCasePath || 'research-case.ressectionlab', filters: [{ name: 'RessectionLab case', extensions: ['ressectionlab'] }] });
    if (selected.canceled || !selected.filePath) return null;
    const result = await engine.request('saveCase', { ...args, path: selected.filePath, overwrite: true });
    currentCasePath = selected.filePath;
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
  const devUrl = !app.isPackaged && process.env.RESSECTIONLAB_DEV_URL;
  if (devUrl && devUrl !== 'http://127.0.0.1:5173/') throw new Error('Development preview must use the configured local address');
  allowedUrl = devUrl || pathToFileURL(path.join(__dirname, '../dist/index.html')).href;
  session.defaultSession.setPermissionRequestHandler((_contents, _permission, callback) => callback(false));
  session.defaultSession.setPermissionCheckHandler(() => false);
  session.defaultSession.webRequest.onHeadersReceived((details, callback) => callback({ responseHeaders: {
    ...details.responseHeaders,
    'Content-Security-Policy': ["default-src 'self'; script-src 'self'; worker-src 'self' blob:; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self'" + (devUrl ? ' ws://127.0.0.1:5173' : '') + "; object-src 'none'; base-uri 'none'; frame-src 'none'"],
  } }));
  window = new BrowserWindow({ width: 1460, height: 980, minWidth: 1050, minHeight: 720, title: 'RessectionLab', backgroundColor: '#071118', show: false,
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
  window.once('ready-to-show', () => window.show());
  window.on('closed', () => { window = undefined; });
  await window.loadURL(allowedUrl);
}

const ownsAppInstance = app.requestSingleInstanceLock();
if (!ownsAppInstance) app.quit();
app.on('open-file',(event,filename)=>{event.preventDefault();pendingOpenPath=filename;if(window&&!window.isDestroyed())window.webContents.send('research:event',{event:'menuAction',action:'openCase'});});
app.on('second-instance', (_event,argv) => {
  const index=argv.findIndex(a=>a==='--case'||a==='--anatomy');
  const filename=index>=0?argv[index+1]:argv.find(a=>/\.(ressectionlab|rslab)$/.test(a)||a.endsWith('patient-anatomy.json')||a.endsWith('.rsworkspace'));
  if(filename){pendingOpenPath=path.resolve(filename);if(window&&!window.isDestroyed())window.webContents.send('research:event',{event:'menuAction',action:'openCase'});}

  if (window && !window.isDestroyed()) { if (window.isMinimized()) window.restore(); window.show(); window.focus(); }
});
if (ownsAppInstance) app.whenReady().then(async () => { log = createLogger(app.getPath('logs')); log(`Starting ${app.getVersion()} (${app.isPackaged ? 'packaged' : 'development'})`); await startEngine(); bindOperations(); await createWindow(); }).catch(error => {
  dialog.showErrorBox('RessectionLab could not start', error.message); log(error.stack); app.quit();
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', event => {
  if (!quitting && engine) { event.preventDefault(); quitting = true; engine.stop().finally(() => app.quit()); }
});
