import assert from "node:assert/strict";
import { test, before, after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
let server, Inventory, cacheDir;
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
