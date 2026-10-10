import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

let server, cache, VolumeRenderer;
before(async () => {
  cache = await fs.mkdtemp(path.join(os.tmpdir(), "diagnostic-transition-"));
  server = await createServer({
    root: fileURLToPath(new URL("../../", import.meta.url)), cacheDir: cache,
    logLevel: "error", server: { middlewareMode: true, hmr: false, ws: false },
    appType: "custom",
  });
  VolumeRenderer = (await server.ssrLoadModule("/src/viewer/VolumeRenderer.ts")).VolumeRenderer;
});
after(async () => {
  await server?.close();
  if (cache) await fs.rm(cache, { recursive: true, force: true });
});

function fakeRenderer() {
  const calls = [];
  const uniforms = {
    uDiagnosticActive: { value: 1 },
    uDiagnosticState: { value: null },
    uDiagnosticCoverage: { value: null },
  };
  const engine = Object.create(VolumeRenderer.prototype);
  engine.materials = () => [{ uniforms }];
  engine.requestRender = () => {};
  engine.diagnosticStateTexture = { dispose() {} };
  engine.diagnosticCoverageTexture = { dispose() {} };
  for (const method of ["setReplay", "setStructuralProposal", "setPriorLayer",
    "setInspectionTool", "setPublicGoal", "updateRoutes"]) {
    engine[method] = (...args) => calls.push([method, ...args]);
  }
  return { engine, calls, uniforms };
}

test("invalid diagnostic request clears existing inspection modes before refusal", () => {
  const { engine, calls, uniforms } = fakeRenderer();
  assert.throws(() => engine.setDiagnosticLayer({}, null), /source SHA is unavailable/);
  assert.deepEqual(calls.map(([method]) => method), ["setReplay", "setStructuralProposal",
    "setPriorLayer", "setInspectionTool", "setPublicGoal", "updateRoutes"]);
  assert.ok(calls.every(([, argument]) => argument === null ||
    Array.isArray(argument) && argument.length === 0));
  assert.equal(uniforms.uDiagnosticActive.value, 0);
  assert.equal(engine.routeSignature, "[]");
});

test("hiding diagnostic alone leaves other modes intact", () => {
  const { engine, calls, uniforms } = fakeRenderer();
  assert.equal(engine.setDiagnosticLayer(null, null), null);
  assert.deepEqual(calls, []);
  assert.equal(uniforms.uDiagnosticActive.value, 0);
});
