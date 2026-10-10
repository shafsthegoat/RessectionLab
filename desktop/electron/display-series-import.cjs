'use strict';
const { plainArgs } = require('./security.cjs');
const MODALITIES = ['T1','T1CE','T2','FLAIR','CT','CTA','TOF-MRA','MRA','SWI','other-3D-scalar'];
async function importDisplaySeries(args, { pick, request }) {
  plainArgs(args, ['caseHash','modality','annotationKind']);
  if (!MODALITIES.includes(args.modality) || !['none','source-provided','estimated'].includes(args.annotationKind))
    throw new Error('Choose a supported 3D modality and annotation provenance');
  const imagePath = await pick('Select an additional 3D image for native-grid inspection', ['nii','gz']);
  if (!imagePath) return null;
  let annotationPath;
  if (args.annotationKind !== 'none') {
    annotationPath = await pick('Select its aligned integer-label annotation (source or estimate as declared)', ['nii','gz']);
    if (!annotationPath) return null;
  }
  return request('importDisplaySeries', { ...args, imagePath, ...(annotationPath ? { annotationPath } : {}) });
}
module.exports = { importDisplaySeries };
