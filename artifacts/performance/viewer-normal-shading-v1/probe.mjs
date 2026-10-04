import { readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import os from "node:os";
import { maskSurface } from "../../../desktop/src/viewer/surface.ts";
import { surfaceNormals } from "../../../desktop/src/viewer/surfaceNormals.ts";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const read = (path) => readFileSync(new URL(path, new URL("../../../", import.meta.url)));
const digest = (data) => createHash("sha256").update(data).digest("hex");
const raw = (array) => new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
const source = JSON.parse(read("desktop/public/preview/manifest.json")).case;
const codePaths = [
  "desktop/src/viewer/surface.ts",
  "desktop/src/viewer/surfaceNormals.ts",
  "desktop/src/viewer/surface.worker.ts",
  "desktop/src/viewer/VolumeRenderer.ts",
  "desktop/src/viewer/coordinates.test.mjs",
  "artifacts/performance/viewer-normal-shading-v1/probe.mjs",
];
const report = {
  schema: "ressectionlab.viewer-normal-validation.v1",
  recordedAt: new Date().toISOString(),
  probe: "Single fresh Node process; first invocation per source compartment, in manifest order.",
  environment: { node: process.version, platform: process.platform, arch: process.arch, cpu: os.cpus()[0].model },
  code: {
    gitHead: execFileSync("git", ["rev-parse", "HEAD"], { cwd: root, encoding: "utf8" }).trim(),
    status: "Relevant working-tree files identified by exact content hashes; this is not a claim that all edits are committed.",
    sha256: Object.fromEntries(codePaths.map(path => [path, digest(read(path))])),
  },
  source: { caseId: source.caseId, caseHash: source.caseHash, frame: source.frame, shape: source.shape, affine: source.affine },
  change: "Area-weighted lighting normals at exactly coincident vertices; no positional smoothing, resampling or decimation.",
  unchanged: ["MRI", "source masks", "triangle positions and order", "cell-based metrics", "lighting", "material transparency", "camera"],
  compartments: [],
};
for (const layer of source.compartments) {
  const mask = new Uint8Array(read(`desktop/public/preview/${layer.array.assetId}.bin`));
  const sourceMaskSha256 = digest(mask);
  if (sourceMaskSha256 !== layer.array.sha256) throw new Error("Source mask digest mismatch");
  const meshStart = performance.now();
  const positions = maskSurface(mask, source.shape);
  const meshPreparationMs = performance.now() - meshStart;
  const positionsSha256Before = digest(raw(positions));
  const normalStart = performance.now();
  const normals = surfaceNormals(positions);
  const normalPreparationMs = performance.now() - normalStart;
  let nonfiniteNormals = 0, nonunitNormals = 0, maximumUnitLengthError = 0;
  for (let i = 0; i < normals.length; i += 3) {
    if (![normals[i], normals[i + 1], normals[i + 2]].every(Number.isFinite)) nonfiniteNormals++;
    const error = Math.abs(Math.hypot(normals[i], normals[i + 1], normals[i + 2]) - 1);
    if (error > 1e-6) nonunitNormals++;
    maximumUnitLengthError = Math.max(maximumUnitLengthError, error);
  }
  const positionsSha256After = digest(raw(positions));
  const checks = {
    positionsUnchanged: positionsSha256Before === positionsSha256After,
    maskUnchanged: digest(mask) === sourceMaskSha256,
    normalCountMatchesVertices: normals.length === positions.length,
    nonfiniteNormals, nonunitNormals, maximumUnitLengthError, unitTolerance: 1e-6,
  };
  if (!checks.positionsUnchanged || !checks.maskUnchanged || !checks.normalCountMatchesVertices || nonfiniteNormals || nonunitNormals) throw new Error(`Invariant failure: ${layer.name}`);
  report.compartments.push({
    name: layer.name, sourceMaskSha256, sourceAnnotationVolumeMm3: layer.volumeMm3,
    triangles: positions.length / 9, positionsSha256Before, positionsSha256After,
    normalsSha256: digest(raw(normals)), checks, meshPreparationMs, normalPreparationMs,
  });
}
const meshMs = report.compartments.reduce((sum, item) => sum + item.meshPreparationMs, 0);
const normalMs = report.compartments.reduce((sum, item) => sum + item.normalPreparationMs, 0);
report.timing = {
  sourceMeshPreparationMs: meshMs,
  additionalNormalPreparationMs: normalMs,
  combinedPreparationMs: meshMs + normalMs,
  additionalNormalCostRelativeToFullMesh: normalMs / meshMs,
  productionExecution: "Mesh and normal preparation both execute in surface.worker.ts; buffers are transferred to the renderer.",
  exclusions: "Node CPU preparation probe only. Excludes UI asset loading, worker startup/IPC, GPU upload, frame submission and GPU latency. One-run timings are not a throughput benchmark.",
};
report.visualEvidence = {
  comparison: "Qualitative native app comparison reviewed by root and viewer agent: reduced triangle lighting discontinuities; original voxel contour and source MRI retained.",
  controls: { caseId: source.caseId, allSourceLayersVisible: true, sourcePlaneVisible: false, overlayOpacity: 0.32, cursorRasMm: [-161.9, 84.9, 104.6], cameraPreset: "anatomy", selectedRoutes: 0 },
  screenshots: ["source-baseline.jpg", "source-normals-after.jpg"].map(name => {
    const path = `artifacts/electron-refinement-v2/${name}`;
    return { path, sha256: digest(read(path)), dimensions: [2920, 1736] };
  }),
};
writeFileSync(new URL("validation.json", import.meta.url), JSON.stringify(report, null, 2) + "\n");
console.log(JSON.stringify({ artifact: "artifacts/performance/viewer-normal-shading-v1/validation.json", allChecksPassed: true, ...report.timing }));
