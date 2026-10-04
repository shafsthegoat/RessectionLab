'use strict';

const { app, BrowserWindow, dialog, ipcMain, session, Menu } = require('electron');
const path = require('node:path');
const fs = require('node:fs/promises');
const os = require('node:os');
const { pathToFileURL } = require('node:url');
const { Sidecar } = require('./sidecar.cjs');
const { createLogger } = require('./logging.cjs');
const { assertSender, plainArgs } = require('./security.cjs');

let log = message => process.stderr.write(String(message) + '\n');
let window;
let engine;
let allowedUrl;
let quitting = false;
let startupConsumed = false;
const cliValue = name => { const index = process.argv.indexOf(name); return index >= 0 ? process.argv[index + 1] : undefined; };

async function startEngine() {
  const transferDir = await fs.mkdtemp(path.join(os.tmpdir(), 'ressectionlab-'));
  await fs.chmod(transferDir, 0o700);
  const bundled = path.join(process.resourcesPath, 'research-engine', 'ressectionlab-engine');
  const repo = path.resolve(__dirname, '../..');
  const runDir = path.join(app.getPath('userData'), 'research-runs');
  const config = app.isPackaged
    ? { executable: bundled, cwd: app.getPath('userData'), transferDir, runDir }
    : { python: path.join(repo, '.venv/bin/python'), cwd: repo, sourcePath: path.join(repo, 'src'), transferDir, runDir };
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
  handle('ping', () => engine.request('ping'));
  handle('startupCase', async () => {
    if (startupConsumed) return null;
    startupConsumed = true;
    const casePath = cliValue('--case');
    if (casePath) return engine.request('loadCase', { path: path.resolve(casePath) });
    return process.argv.includes('--demo') ? engine.request('createSyntheticCase') : null;
  });
  handle('createSyntheticCase', () => engine.request('createSyntheticCase'));
  handle('openCase', async () => {
    const selected = await pick('Open a saved research case', ['ressectionlab', 'rslab']);
    return selected ? engine.request('loadCase', { path: selected }) : null;
  });
  handle('importNifti', async () => {
    const structuralPath = await pick('Select the structural MRI', ['nii', 'gz']);
    if (!structuralPath) return null;
    const tumorMaskPath = await pick('Select the tumor segmentation (Cancel to import MRI only)', ['nii', 'gz']);
    return engine.request('importNifti', { structuralPath, ...(tumorMaskPath ? { tumorMaskPath } : {}) });
  });
  handle('saveCase', async args => {
    plainArgs(args, ['caseHash', 'workspace']);
    const selected = await dialog.showSaveDialog(window, { title: 'Save research workspace', defaultPath: 'research-case.ressectionlab', filters: [{ name: 'RessectionLab case', extensions: ['ressectionlab'] }] });
    if (selected.canceled || !selected.filePath) return null;
    return engine.request('saveCase', { ...args, path: selected.filePath, overwrite: true });
  });
  handle('generateRoutes', args => engine.request('generateRoutes', plainArgs(args, ['caseHash', 'toolIds', 'allowEstimatedSupport', 'config'])));
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
app.on('second-instance', () => {
  if (window && !window.isDestroyed()) { if (window.isMinimized()) window.restore(); window.show(); window.focus(); }
});
if (ownsAppInstance) app.whenReady().then(async () => { log = createLogger(app.getPath('logs')); log(`Starting ${app.getVersion()} (${app.isPackaged ? 'packaged' : 'development'})`); await startEngine(); bindOperations(); await createWindow(); }).catch(error => {
  dialog.showErrorBox('RessectionLab could not start', error.message); log(error.stack); app.quit();
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', event => {
  if (!quitting && engine) { event.preventDefault(); quitting = true; engine.stop().finally(() => app.quit()); }
});
