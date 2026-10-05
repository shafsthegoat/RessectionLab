import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { neighboringFixture } from "./neighboring-paths-fixture.mjs";

// Isolated synthetic display review only: no engine, native preview or patient data.
let server, cache, Panel;
before(async () => {
  cache = await fs.mkdtemp(path.join(os.tmpdir(), "neighboring-independent-"));
  server = await createServer({
    root: fileURLToPath(new URL("../", import.meta.url)),
    cacheDir: cache,
    logLevel: "error",
    server: { middlewareMode: true, hmr: false, ws: false },
    appType: "custom",
  });
  Panel = (await server.ssrLoadModule("/src/NeighboringPathsPanel.tsx"))
    .NeighboringPathsPanel;
});
after(async () => {
  await server?.close();
  if (cache) await fs.rm(cache, { recursive: true, force: true });
});

function render(fixture, extra = {}) {
  return renderToStaticMarkup(React.createElement(Panel, {
    context: fixture.context,
    choices: fixture.choices,
    state: { kind: "complete", result: fixture.result },
    onChoices() {},
    onInspect() {},
    onCancel() {},
    ...extra,
  }));
}

test("additional source and preview uncertainty stays inspectable beyond fixed disclaimer text", () => {
  const f = neighboringFixture();
  const sourceWarning = "source registration correspondence has not been reviewed";
  const geometryWarning = "instrument articulation is outside this geometry model";
  f.result.inspection.unknowns.push(sourceWarning, geometryWarning);
  f.result.inspection.actions[1].native_preview.geometry_unknowns.push(geometryWarning);
  const html = render(f);
  assert.match(html, /Initial sampled paths/);
  assert.ok(html.includes(sourceWarning), "Additional source uncertainty was silently omitted");
  assert.ok(html.includes(geometryWarning), "Additional preview uncertainty was silently omitted");
});

test("unavailable engine capability withholds a previously complete inventory", () => {
  const f = neighboringFixture();
  const html = render(f, { unavailableReason: "The local engine stopped. Reopen to inspect again." });
  assert.match(html, /local engine stopped/);
  assert.equal(/Initial sampled paths|class="neighboring-path-row"/.test(html), false,
    "A previously complete inventory remained visible despite unavailable engine capability");
});
