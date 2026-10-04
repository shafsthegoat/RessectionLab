import assert from "node:assert/strict";
import { after, before, test } from "node:test";
import { createServer } from "vite";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { neighboringFixture } from "./neighboring-paths-fixture.mjs";
let server, cache, Panel;
before(async () => {
  cache = await fs.mkdtemp(path.join(os.tmpdir(), "neighboring-panel-test-"));
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
  if (cache)
    await fs.rm(cache, {
      recursive: true,
      force: true,
      maxRetries: 3,
      retryDelay: 30,
    });
});
function props(overrides = {}) {
  const { context, choices } = neighboringFixture();
  return {
    context,
    choices,
    state: { kind: "idle" },
    onChoices: () => {},
    onInspect: () => {},
    onCancel: () => {},
    ...overrides,
  };
}
const render = (input) =>
  renderToStaticMarkup(React.createElement(Panel, input));
function nodes(element) {
  if (!element || typeof element !== "object") return [];
  return [
    element,
    ...React.Children.toArray(element.props?.children).flatMap(nodes),
  ];
}

test("rendering and changing acknowledgments never trigger inspection; an explicit button launches a bounded request", () => {
  const f = neighboringFixture(),
    calls = [],
    choicesChanged = [];
  const input = props({
    choices: { ...f.choices, neighboringColumns: false },
    onChoices: (value) => choicesChanged.push(value),
    onInspect: (value) => calls.push(value),
  });
  let tree = Panel(input);
  assert.deepEqual(calls, []);
  const checkbox = nodes(tree).filter((node) => node.type === "input")[2];
  checkbox.props.onChange({ target: { checked: true } });
  assert.equal(choicesChanged[0].neighboringColumns, true);
  assert.deepEqual(calls, []);
  nodes(tree)
    .find((node) => node.type === "button")
    .props.onClick();
  assert.deepEqual(calls, []);
  tree = Panel({ ...input, choices: choicesChanged[0] });
  nodes(tree)
    .find((node) => node.type === "button")
    .props.onClick();
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].toolIds, ["native-fine-aspiration"]);
  assert.equal(calls[0].acknowledgeNeighboringColumns, true);
  assert.equal(calls[0].acknowledgeEstimatedSupport, true);
  assert.equal("anchorWindowRas" in calls[0], false);
});

test("blocked anatomy cannot be unlocked by either acknowledgment and no source is invented", () => {
  const f = neighboringFixture(),
    calls = [];
  const input = props({
    context: { ...f.context, support: "blocked" },
    onInspect: (value) => calls.push(value),
  });
  const html = render(input);
  assert.match(html, /Access support needs review/);
  assert.match(html, /neighboring-inspect" disabled=""/);
  nodes(Panel(input))
    .find((node) => node.type === "button")
    .props.onClick();
  assert.deepEqual(calls, []);
  assert.match(render(props({ context: null })), /Choose a source route/);
});

test("complete display preserves all thirteen slots and explicitly separates fallback/rejected geometry", () => {
  const f = neighboringFixture();
  const html = render(props({ state: { kind: "complete", result: f.result } }));
  assert.match(html, /13 of 13 declared path\/tool pairs accounted for/);
  assert.equal((html.match(/class="neighboring-path-row"/g) ?? []).length, 13);
  assert.match(html, /11<\/dd>/);
  assert.match(html, /Fallback after primary rejection/);
  assert.match(html, /first failure location is unavailable/);
  assert.match(html, /No endpoint generated for this omitted path/);
  assert.match(html, /No actions executed · no tissue changed/);
  assert.match(html, /bounded initial sample, not an exhaustive route search/);
  assert.doesNotMatch(
    html,
    /Export checked|Freeze assumptions &amp; train|clinical_deficit_probability|unexecuted_contact_cell_count|Removed target/,
  );
});

test("stale context or changed instruments withholds the entire inventory instead of displaying zero successes", () => {
  const f = neighboringFixture();
  const html = render(
    props({
      context: { ...f.context, routeId: "different-route" },
      state: { kind: "complete", result: f.result },
    }),
  );
  assert.match(html, /Inspection result withheld/);
  assert.doesNotMatch(
    html,
    /Initial sampled paths|No sampled path passed|neighboring-path-row/,
  );
  const changed = render(
    props({
      choices: { ...f.choices, estimatedSupport: false },
      state: { kind: "complete", result: f.result },
    }),
  );
  assert.match(changed, /Inspection assumptions changed/);
});

test("cancelled and loading states publish no partial inventory and do not claim the worker has stopped", () => {
  const f = neighboringFixture();
  const cancelled = render(
    props({ state: { kind: "cancelled", result: f.result } }),
  );
  assert.match(cancelled, /no result published/);
  assert.match(cancelled, /geometry check may still be finishing/);
  assert.doesNotMatch(cancelled, /Initial sampled paths|neighboring-path-row/);
  const loading = render(
    props({ state: { kind: "loading", result: f.result } }),
  );
  assert.match(loading, /Cancel inspection/);
  assert.match(loading, /No partial inventory is shown/);
  assert.doesNotMatch(loading, /Initial sampled paths|neighboring-path-row/);
});

test("unsupported windows remain plain refusals, without a manufactured ready or zero-action result", () => {
  const html = render(
    props({ state: { kind: "error", message: "UNSUPPORTED_NONAXIAL_ACCESS" } }),
  );
  assert.match(html, /not aligned with the source image axes/);
  assert.match(html, /selected window stays unchanged/);
  assert.doesNotMatch(
    html,
    /Initial sampled paths|No sampled path passed|13 of 13/,
  );
});
