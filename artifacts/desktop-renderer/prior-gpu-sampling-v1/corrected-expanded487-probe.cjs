'use strict';

// Run with the local Electron executable, not Node. This verifies numerical
// sampling in the actual WebGL2 shader, independently of screenshots or UI QA.
const { app, BrowserWindow } = require('electron');
const fs = require('node:fs');
const path = require('node:path');
const os = require('node:os');
const { createHash } = require('node:crypto');

const root = path.resolve(__dirname, '../..');
const outputFlag = process.argv.indexOf('--output');
if (outputFlag < 0 || !process.argv[outputFlag + 1]) throw new Error('Provide --output <new-receipt.json>');
const output = path.resolve(process.argv[outputFlag + 1]);
if (fs.existsSync(output)) throw new Error('Keep previous evidence: output already exists');
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'resectionlab-prior-gpu-'));
app.setPath('userData', profile);
const sha = value => createHash('sha256').update(value).digest('hex');

const priorPath = path.join(root, 'desktop/src/viewer/priorLayer.ts');
const coordinatesPath = path.join(root, 'desktop/src/viewer/coordinates.ts');
const shaderPath = path.join(root, 'desktop/src/viewer/shaders.ts');
// Electron's bundled Node strips types while loading the exact CPU helpers.
// Analytical expected values below do not depend on those helpers.
const prior = require(priorPath);
const coordinates = require(coordinatesPath);
const shaderSource = fs.readFileSync(shaderPath, 'utf8');
const fragment = shaderSource.match(/export const fragmentShader = `([\s\S]*?)`;/)?.[1];
if (!fragment) throw new Error('Production fragment shader source was not found');
const prelude = fragment.slice(0, fragment.indexOf('void main() {'));
const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
const wave = [0, 1, 0, 1, 0, 1, 0, 1];
const known = Array(8).fill(1);
const cases = [
  { name: 'functional_center', point: [.5, .5, .5], value: .5 },
  { name: 'covered_zero', point: [0, 0, 0], value: 0 },
  { name: 'incomplete_support', point: [0, 0, .25], coverage: [1, 0, 1, 1, 1, 1, 1, 1], covered: false },
  { name: 'binary_low', point: [0, 0, .49], kind: 1, value: 0 },
  { name: 'binary_high', point: [0, 0, .51], kind: 1, value: 1 },
  { name: 'tiny_positive', point: [.25, .5, .75], values: Array(8).fill(.00001), value: .00001 },
  { name: 'very_tiny_positive', point: [.25, .5, .75], values: Array(8).fill(1e-8), value: 1e-8 },
];
for (let n = 0; n < 32; n++) {
  const angle = (n + 1) * Math.PI / 67, c = Math.cos(angle), s = Math.sin(angle);
  const affine = [[2 * c, -3 * s, 0, 10], [2 * s, 3 * c, 0, -20], [0, 0, 4, 7], [0, 0, 0, 1]];
  for (const x of [0, 1]) cases.push({
    name: `coverage_boundary_${n}_${x}`, affine, point: [x, .3, .25],
    values: Array(8).fill(.4), coverage: x === 0 ? [1, 1, 1, 1, 0, 0, 0, 0] : [0, 0, 0, 0, 1, 1, 1, 1], value: .4,
  });
  cases.push({ name: `real_missing_support_${n}`, affine, point: [.001, .3, .25],
    values: Array(8).fill(.4), coverage: [1, 1, 1, 1, 0, 0, 0, 0], covered: false });
  for (const delta of [-.001, 0, .001]) cases.push({
    name: `binary_half_cell_${n}_${delta}`, affine, point: [.5 + delta, .3, .25], kind: 1,
    values: [0, 0, 0, 0, 1, 1, 1, 1], value: delta < 0 ? 0 : 1,
  });
  cases.push({ name: `binary_uncovered_tie_${n}`, affine, point: [.5, .3, .25], kind: 1,
    values: Array(8).fill(0), coverage: [1, 1, 1, 1, 0, 0, 0, 0], covered: false });
  cases.push({ name: `diagonal_missing_support_${n}`, affine, point: [.0002, .0002, 0],
    values: Array(8).fill(.4), coverage: [1, 1, 1, 1, 1, 1, 0, 1], covered: false });
  cases.push({ name: `oblique_scalar_ramp_${n}`, affine, point: [.21, .36, .64],
    values: [0, .1, .2, .3, .4, .5, .6, .7], value: .22 });
  for (const face of [-.5, 1.5]) for (const offset of [-.001, 0, .001]) cases.push({
    name: `outer_face_${n}_${face}_${offset}`, affine, point: [face + offset, .3, .25],
    values: Array(8).fill(.4), value: .4,
    covered: face === -.5 ? offset > 0 : offset < 0,
    reason: offset === 0 ? 'numerical-boundary-uncertainty' : undefined,
  });
}

for (const test of cases) {
  test.affine ??= identity;
  test.values ??= wave;
  test.coverage ??= known;
  test.kind ??= 0;
  test.covered ??= true;
  test.world = coordinates.transformPoint(test.affine, test.point);
  test.inverse = coordinates.inverseAffine(test.affine);
  const layer = { shape: [2, 2, 2], affine: test.affine, values: new Float32Array(test.values),
    coverage: new Uint8Array(test.coverage), mapKind: test.kind ? 'structural_mask' : 'functional_concordance' };
  test.tolerance = prior.priorSamplingTolerance(layer);
  if (!test.tolerance || test.tolerance.some(value => value >= .0001))
    throw new Error(`Fixture precision bound exceeds its independently chosen support offsets: ${test.name}`);
  test.cpu = prior.samplePriorVoxel(layer, coordinates.transformPoint(test.inverse, test.world));
}

async function renderProbe(prelude, cases) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 1;
  const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true });
  if (!gl || !gl.getExtension('EXT_color_buffer_float')) throw new Error('WebGL2 float framebuffer unavailable');
  function shader(kind, text) {
    const value = gl.createShader(kind);
    gl.shaderSource(value, text); gl.compileShader(value);
    if (!gl.getShaderParameter(value, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(value));
    return value;
  }
  const program = gl.createProgram();
  gl.attachShader(program, shader(gl.VERTEX_SHADER, '#version 300 es\nout vec2 vUv;out vec3 vWorld;void main(){vec2 p=gl_VertexID==0?vec2(-1.,-1.):gl_VertexID==1?vec2(3.,-1.):vec2(-1.,3.);vUv=vec2(0.);vWorld=vec3(0.);gl_Position=vec4(p,0.,1.);}'));
  gl.attachShader(program, shader(gl.FRAGMENT_SHADER, '#version 300 es\n' + prelude + '\nuniform vec3 uProbeWorld;void main(){float value;bool covered=priorAt(uProbeWorld,value);outColor=vec4(covered?1.:0.,value,0.,1.);}'));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
  gl.useProgram(program);
  function texture(unit, data, coverage) {
    const value = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_3D, value);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, coverage ? gl.R8 : gl.R32F, 2, 2, 2, 0, gl.RED, coverage ? gl.UNSIGNED_BYTE : gl.FLOAT, data);
    return value;
  }
  gl.uniform1i(gl.getUniformLocation(program, 'uPrior'), 0);
  gl.uniform1i(gl.getUniformLocation(program, 'uPriorCoverage'), 1);
  gl.uniform3f(gl.getUniformLocation(program, 'uPriorShape'), 2, 2, 2);
  const target = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, target);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
  const framebuffer = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, target, 0);
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) throw new Error('Incomplete framebuffer');
  const results = [];
  for (const test of cases) {
    const values = texture(0, new Float32Array(test.values), false);
    const coverage = texture(1, new Uint8Array(test.coverage), true);
    const matrix = new Float32Array([0, 1, 2, 3].flatMap(col => test.inverse.map(row => row[col])));
    gl.uniformMatrix4fv(gl.getUniformLocation(program, 'uWorldToPrior'), false, matrix);
    gl.uniform3fv(gl.getUniformLocation(program, 'uProbeWorld'), test.world);
    gl.uniform3fv(gl.getUniformLocation(program, 'uPriorVoxelTolerance'), test.tolerance);
    gl.uniform1i(gl.getUniformLocation(program, 'uPriorKind'), test.kind);
    gl.viewport(0, 0, 1, 1); gl.drawArrays(gl.TRIANGLES, 0, 3);
    const pixel = new Float32Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, pixel);
    results.push({ name: test.name, covered: pixel[0] === 1, value: pixel[1], glError: gl.getError() });
    gl.deleteTexture(values); gl.deleteTexture(coverage);
  }
  const debug = gl.getExtension('WEBGL_debug_renderer_info');
  return { renderer: gl.getParameter(gl.RENDERER), backend: debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : null, results };
}

app.whenReady().then(async () => {
  const window = new BrowserWindow({ show: false, webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false } });
  try {
    await window.loadURL('data:text/html,<html><body>Numerical shader audit</body></html>');
    const gpu = await window.webContents.executeJavaScript(`(${renderProbe.toString()})(${JSON.stringify(prelude)},${JSON.stringify(cases)})`);
    const agrees = (actual, test) => actual.covered === test.covered && (!test.covered ||
      Math.abs(actual.value - test.value) <= Math.max(2e-12, 2e-6 * Math.abs(test.value)));
    const results = cases.map((test, index) => ({ ...test, gpu: gpu.results[index],
      cpuAgrees: agrees(test.cpu, test) && (!test.reason || test.cpu.reason === test.reason),
      gpuAgrees: agrees(gpu.results[index], test) && gpu.results[index].glError === 0 }));
    const failures = results.filter(row => !row.cpuAgrees || !row.gpuAgrees).map(row => row.name);
    const report = { recordedAt: new Date().toISOString(), passed: failures.length === 0,
      scope: 'Actual WebGL2 production shader prefix and production CPU sampler on analytic fixtures; not native-app layout or clinical validation',
      renderer: gpu.renderer, backend: gpu.backend, caseCount: results.length, failedCaseNames: failures,
      expectedSemantics: 'Analytic affine landmarks and nonconstant scalar ramp; covered integer centers, upper-cell binary half ties, genuine missing support at +0.001 voxel and diagonal product 4e-8 remains unknown; exact outer faces abstain while offsets of 0.001 voxel preserve native inside/outside support',
      scalarTolerance: 'max(2e-12 absolute, 2e-6 times the analytic expected value)',
      sourceSha256: Object.fromEntries([__filename, shaderPath, priorPath, coordinatesPath].map(file => [path.relative(root, file), sha(fs.readFileSync(file))])),
      shaderPrefixSha256: sha(prelude), results };
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ passed: report.passed, caseCount: results.length, failures, output }));
    process.exitCode = report.passed ? 0 : 1;
  } catch (error) {
    console.error(String(error.stack)); process.exitCode = 1;
  } finally {
    window.destroy(); app.quit();
  }
});
setTimeout(() => { console.error('Prior GPU sampling probe timed out'); app.exit(2); }, 30000).unref();
