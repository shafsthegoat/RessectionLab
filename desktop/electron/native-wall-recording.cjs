'use strict';

const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');

const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const near = (a, b, tolerance = 1e-4) => a.every((value, i) => Math.abs(value-b[i]) < tolerance);
const distance = camera => Math.hypot(...camera.position.map((value, i) => value-camera.target[i]));
const sight = camera => camera.position.map((value, i) => (value-camera.target[i])/distance(camera));

function nativeWallMotionReceipt(pane, view, scene) {
  const mechanics = view.vesselMechanics;
  const nativeVessels = pane.nativeVessels;
  const witnessErrors = {};
  const rendererModes = {
    'native-probability-wall': 'native-probability',
    'edited-native-wall': 'edited-native-wall',
    'accepted-source-cuboids': 'source-cell-boundary',
  };
  for (const region of scene.regions) {
    const mode = mechanics?.geometryModeByLabel?.[region.id];
    if (mode) {
      const wall = nativeVessels.find(surface => surface.layer === region.name);
      assert(rendererModes[mode] && wall?.mode === rendererModes[mode],
        'The GPU wall mode differs from accepted collision geometry');
      const vertexMotion = mechanics?.vertexDisplacementsByLabel?.[region.id];
      if (vertexMotion) assert(wall.vertexGeometryBinding === vertexMotion.geometryBinding && wall.vertices === vertexMotion.vertexCount,
        'The GPU vertices differ from the accepted source wall binding');
    }
  }
  for (const {layer, maxMotionWitness} of nativeVessels) {
    if (!maxMotionWitness) continue;
    const {sourceCell, vertex, restNativeIJK, displayedNativeIJK} = maxMotionWitness;
    const label = scene.regions.find(region => region.name === layer)?.id;
    const vertexMotion = mechanics?.vertexDisplacementsByLabel?.[label];
    const expected = vertexMotion ? vertexMotion.offsetsMm[vertexMotion.indices.indexOf(vertex)] :
      mechanics?.surfaceDisplacements?.[sourceCell];
    assert(expected, 'The moved GPU wall vertex has no accepted source displacement');
    const displacement = displayedNativeIJK.map((value, i) => value-restNativeIJK[i]);
    const ras = scene.affine.slice(0, 3).map(row => displacement.reduce((sum, value, i) => sum+value*row[i], 0));
    const error = Math.hypot(...ras.map((value, i) => value-expected[i]));
    assert(error < 1e-4, 'The moved GPU wall vertex differs from its accepted collision wall');
    witnessErrors[layer] = error;
  }
  return {
    nativeVessels,
    collisionGeometryByLabel: mechanics?.geometryModeByLabel ?? {},
    workingContacts: mechanics?.contactedCells ?? {},
    supportedRodCount: mechanics?.sites.filter(site => site.activeSupportCells.length > 0).length ?? 0,
    maximumRodDisplacementMm: Math.max(0, ...(mechanics?.sites ?? []).map(site =>
      Math.hypot(...site.positionMm.map((value, i) => value-site.restPositionMm[i])))),
    movedWallWitnessErrorMm: witnessErrors,
  };
}

function matchedNativeWallReceipt(plan, stateId, anatomyPane, operativePane, anatomyPixels, operativePixels) {
  const {camera, focusPlaneFieldMm} = plan.matchedAnatomy;
  assert(camera && Number.isFinite(focusPlaneFieldMm), 'A calibrated anatomy camera and field are required');
  assert(near(anatomyPane.camera.position, camera.position), 'The anatomy camera did not restore');
  assert(near(anatomyPane.camera.target, operativePane.camera.target), 'The matched focus axes moved');
  assert(near(anatomyPane.camera.up, operativePane.camera.up), 'The matched camera roll moved');
  assert(near(sight(anatomyPane.camera), sight(operativePane.camera)), 'The matched camera angle moved');
  assert(Math.abs(2*Math.tan(17*Math.PI/180)*distance(anatomyPane.camera)-operativePane.fieldMm) < 1e-3,
    'Anatomy and microscope must have the same physical frame height at the focus plane');
  assert.equal(operativePane.fieldMm, focusPlaneFieldMm);
  assert.equal(operativePane.view, 'surgeon');
  assert(!operativePane.inspection);
  const anatomy = anatomyPane.nativeVessels.find(surface => surface.mode === 'native-probability');
  const operative = operativePane.nativeVessels.find(surface => surface.layer === anatomy?.layer);
  assert(anatomy && operative, 'The exact native probability vessel wall must be visible in both views');
  assert.equal(operative.mode, anatomy.mode);
  assert.equal(operative.triangles, anatomy.triangles);
  assert.equal(operative.vertices, anatomy.vertices);
  assert.equal(operative.vertexGeometryBinding, anatomy.vertexGeometryBinding,
    'The matched views have different indexed source-wall geometry');
  assert.equal(operative.movedVertices, 0);
  assert.equal(anatomy.movedVertices, 0);
  return {
    sourceCaseHash: plan.caseHash, stateId, verticalFieldAtFocusMm: focusPlaneFieldMm,
    anatomy: {camera: anatomyPane.camera, gpuSerial: anatomyPane.gpuSerial, mesh: anatomy, sha256: digest(anatomyPixels)},
    operative: {camera: operativePane.camera, gpuSerial: operativePane.gpuSerial, mesh: operative, sha256: digest(operativePixels)},
  };
}

function nativeWallReviewReceipt(plan, boundary, pane, priorCamera, view, scene, pixels) {
  const mechanics = view.vesselMechanics;
  const editedWalls = Object.fromEntries(Object.entries(mechanics?.editedWallsByLabel ?? {}).map(([label, wall]) =>
    [label, {identity: wall.identity, triangles: wall.triangles.length, cuboidCount: wall.cuboidSourceCells?.length ?? 0}]));
  const expected = plan.reviewNativeWalls?.[boundary];
  if (expected) {
    const field = expected.fieldMm ?? plan.matchedAnatomy?.focusPlaneFieldMm;
    if (field !== undefined) assert.equal(pane.fieldMm, field);
    if (priorCamera) for (const axis of ['position','target','up']) {
      assert(near(priorCamera[axis], pane.camera[axis], 1e-5), 'The native wall comparison camera changed '+axis);
    }
    const shown = pane.nativeVessels.find(surface => surface.layer === expected.layer);
    assert(shown, 'The recorded native vessel is missing from the GPU frame');
    assert.equal(shown.mode, expected.mode);
    assert.equal(shown.triangles, expected.triangles);
    if (expected.identity) {
      const edited = editedWalls[expected.region];
      assert.equal(edited?.identity, expected.identity);
      assert.equal(edited?.triangles, expected.triangles);
      assert.equal(edited?.cuboidCount, 0);
      assert.equal(shown.geometryIdentity, edited.identity);
    }
  }
  return {
    sha256: digest(pixels), ...nativeWallMotionReceipt(pane, view, scene), editedWalls,
    transectedSourceFaces: view.transectedEdges ?? [],
  };
}

module.exports = {nativeWallMotionReceipt, matchedNativeWallReceipt, nativeWallReviewReceipt};
