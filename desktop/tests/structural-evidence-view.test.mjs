import assert from "node:assert/strict";
import { test, before, after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
let server, Inventory, ReplayStepControl, cacheDir;
before(async () => {
  cacheDir = await fs.mkdtemp(
    path.join(os.tmpdir(), "resection-renderer-test-"),
  );
  const root = fileURLToPath(new URL("../", import.meta.url));
  server = await createServer({
    root,
    cacheDir,
    logLevel: "error",
    server: { middlewareMode: true, hmr: false },
    appType: "custom",
  });
  Inventory = (
    await server.ssrLoadModule("/src/StructuralEvidenceInventory.tsx")
  ).StructuralEvidenceInventory;
  ReplayStepControl = (await server.ssrLoadModule("/src/ReplayStepControl.tsx"))
    .ReplayStepControl;
});
after(async () => {
  await server?.close();
  if (cacheDir) await fs.rm(cacheDir, { recursive: true, force: true });
});
function evidence(reviewStatus, reviewRequired) {
  return {
    evidenceId: "fixture",
    kind: "whole_brain_envelope",
    provenance: "estimated",
    reviewRequired,
    reviewStatus,
    metadata: {
      variant: "nocsf",
      current_target_annotation_outside_voxels: 214,
    },
    method: "fixture",
    modelHash: "a",
    sourceHash: "b",
  };
}
test("an explicitly rejected proposal is not relabeled as merely pending", () => {
  const html = renderToStaticMarkup(
    React.createElement(Inventory, { evidence: [evidence("rejected", true)] }),
  );
  assert.match(html, /Rejected/);
  assert.doesNotMatch(html, /Review required/);
  assert.match(html, /214/);
});
test("pending envelope remains an estimate with no cortical permission or approval action", () => {
  const html = renderToStaticMarkup(
    React.createElement(Inventory, {
      evidence: [evidence("review_required", true)],
    }),
  );
  assert.match(html, /Model estimate/);
  assert.match(html, /Review required/);
  assert.match(html, /does not certify cortex/);
  assert.doesNotMatch(html, /<button/);
});

test("pending replay announces the checked applied step, never labels an old overlay as the requested step", () => {
  const html = renderToStaticMarkup(
    React.createElement(ReplayStepControl, {
      requestedStep: 0,
      appliedStep: 1,
      stepCount: 2,
      disabled: false,
      onRequest: () => {},
    }),
  );
  assert.match(html, /Showing modeled step <span>1 \/ 2<\/span>/);
  assert.match(html, /Requested step 0; showing checked step 1 of 2/);
  assert.match(html, /Quantities and overlay still show step 1/);
  assert.match(html, /role="status"/);
});
test("zero-action replay is a labeled disabled timeline", () => {
  const html = renderToStaticMarkup(
    React.createElement(ReplayStepControl, {
      requestedStep: 0,
      appliedStep: 0,
      stepCount: 0,
      disabled: false,
      onRequest: () => {},
    }),
  );
  assert.match(html, /Showing checked step 0 of 0/);
  assert.match(html, /disabled=""/);
  assert.doesNotMatch(html, /Updating to step/);
});
