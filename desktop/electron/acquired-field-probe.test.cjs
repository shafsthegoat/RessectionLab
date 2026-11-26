'use strict';

const assert = require('node:assert/strict');
const test = require('node:test');
const {installFieldProbe} = require('./acquired-field-probe.cjs');

test('a changed timeline with the previous tissue mesh cannot complete a geometric sample', t => {
  const timeline = {value: '3.20'};
  const frames = [];
  const globals = {
    window: {},
    document: {querySelector: () => timeline},
    requestAnimationFrame: callback => frames.push(callback)
  };
  for (const [key, value] of Object.entries(globals)) {
    const old = Object.getOwnPropertyDescriptor(globalThis, key);
    Object.defineProperty(globalThis, key, {value, configurable: true});
    t.after(() => old ? Object.defineProperty(globalThis, key, old) : delete globalThis[key]);
  }
  let fences = 0;
  const playback = {dataset: {operativePhase: 'tumor', operativeProgress: '1.000'}, querySelector: () => timeline};
  const host = {dataset: {aspirationPattern: 'overlapping', corticalSurface: 'registered-planning-surface'},
    closest: () => playback};
  const canvas = {width: 800, height: 600, closest: () => host, dataset: {camera: '{"fov":25}'}};
  const gl = {TIMEOUT_EXPIRED: 0, WAIT_FAILED: 1, SYNC_GPU_COMMANDS_COMPLETE: 2,
    fenceSync: () => ({id: ++fences}), flush() {}, clientWaitSync: () => 3, deleteSync() {}};
  const renderer = {isWebGLRenderer: true, domElement: canvas, getContext: () => gl,
    info: {render: {calls: 5, triangles: 36}}, render() {}};
  const mesh = {geometry: {attributes: {position: {count: 12}}, index: {count: 36}}};
  const scene = {getObjectByName: () => mesh};

  installFieldProbe();
  window.__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent('observe', {detail: renderer}));
  renderer.render(scene, {});
  window.acquiredField.start(4, () => { timeline.value = '4'; return timeline.value; });
  renderer.render(scene, {});
  assert.equal(fences, 0);
  assert.equal(window.acquiredField.sample, null);

  mesh.geometry = {attributes: {position: {count: 9}}, index: {count: 21}};
  renderer.render(scene, {});
  assert.equal(fences, 1);
  assert.equal(window.acquiredField.sample, null);
  frames.shift()();
  assert.equal(window.acquiredField.sample.position, 4);
  assert.deepEqual(window.acquiredField.sample.targetMesh, {vertices: 9, triangles: 7});
  assert.equal(window.acquiredField.current, null);
});
