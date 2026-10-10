'use strict';

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('resectionApi', Object.freeze({
  startupCase: () => ipcRenderer.invoke('research:startupCase'),
  ping: () => ipcRenderer.invoke('research:ping'),
  publicContactFamilyAvailability: args => ipcRenderer.invoke('research:publicContactFamilyAvailability',args),
  executePublicContactFamilyEpisode: args => ipcRenderer.invoke('research:executePublicContactFamilyEpisode',args),
  executePublicSurfaceContactEpisode: args => ipcRenderer.invoke('research:executePublicSurfaceContactEpisode', args),
  inspectDevelopmentEpisodeComparison: args => ipcRenderer.invoke('research:inspectDevelopmentEpisodeComparison', args),
  executeDevelopmentEpisode: args => ipcRenderer.invoke('research:executeDevelopmentEpisode', args),
  evaluateDevelopmentEpisodeVascular: args => ipcRenderer.invoke('research:evaluateDevelopmentEpisodeVascular', args),
  createSyntheticCase: () => ipcRenderer.invoke('research:createSyntheticCase'),
  openCase: () => ipcRenderer.invoke('research:openCase'),
  importNifti: () => ipcRenderer.invoke('research:importNifti'),
  importDisplaySeries: args => ipcRenderer.invoke('research:importDisplaySeries', args),
  importStructuralEvidence: args => ipcRenderer.invoke('research:importStructuralEvidence', args || {}),
  saveCase: args => ipcRenderer.invoke('research:saveCase', args || {}),
  generateRoutes: args => ipcRenderer.invoke('research:generateRoutes', args || {}),
  generateNativeRoutes: args => ipcRenderer.invoke('research:generateNativeRoutes', args || {}),
  inspectRefinement: args => ipcRenderer.invoke('research:inspectRefinement', args || {}),
  inspectAxisPlanning: args => ipcRenderer.invoke('research:inspectAxisPlanning', args || {}),
  inspectObservedLandmarkUpdate: args => ipcRenderer.invoke('research:inspectObservedLandmarkUpdate', args || {}),
  inspectEvidence: args => ipcRenderer.invoke('research:inspectEvidence', args || {}),
  trainPatient: args => ipcRenderer.invoke('research:trainPatient', args || {}),
  listRuns: args => ipcRenderer.invoke('research:listRuns', args || {}),
  replayTraining: args => ipcRenderer.invoke('research:replayTraining', args || {}),
  exportCandidate: args => ipcRenderer.invoke('research:exportCandidate', args || {}),
  cancel: requestId => ipcRenderer.invoke('research:cancel', requestId),
  readAsset: assetId => ipcRenderer.invoke('research:readAsset', assetId),
  onEvent: callback => {
    if (typeof callback !== 'function') throw new TypeError('Event callback required');
    const handler = (_event, payload) => callback(payload);
    ipcRenderer.on('research:event', handler);
    return () => ipcRenderer.removeListener('research:event', handler);
  },
}));
