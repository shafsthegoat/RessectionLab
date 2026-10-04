import assert from "node:assert/strict";
import { test, before, after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { routeComparisonPrompt } from "../src/route-comparison-prompt.ts";
import { researchSupportGate } from "../src/case-support.ts";
let server,
  Inventory,
  ReplayStepControl,
  PriorInventory,
  PriorCursorReadout,
  App,
  AnnotationCenterButton,
  WorkspaceBreadcrumb,
  RouteSelectionControls,
  RouteComparison,
  cacheDir;
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
  const priorModule = await server.ssrLoadModule("/src/PriorInventory.tsx");
  PriorInventory = priorModule.PriorInventory;
  PriorCursorReadout = priorModule.PriorCursorReadout;
  const appModule = await server.ssrLoadModule("/src/App.tsx");
  App = appModule.default;
  RouteSelectionControls = appModule.RouteSelectionControls;
  RouteComparison = appModule.RouteComparison;
  AnnotationCenterButton = (
    await server.ssrLoadModule("/src/AnnotationCenterButton.tsx")
  ).AnnotationCenterButton;
  WorkspaceBreadcrumb = (
    await server.ssrLoadModule("/src/WorkspaceBreadcrumb.tsx")
  ).WorkspaceBreadcrumb;
});
after(async () => {
  await server?.close();
  if (cacheDir)
    await fs.rm(cacheDir, {
      recursive: true,
      force: true,
      maxRetries: 3,
      retryDelay: 30,
    });
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

test("proposal controls explicitly view an estimate without accepting anatomy", () => {
  const item = evidence("review_required", true);
  const html = renderToStaticMarkup(
    React.createElement(Inventory, {
      evidence: [item],
      inspection: {
        onSelect: () => {},
        onClear: () => {},
        selectedId: undefined,
      },
    }),
  );
  assert.match(html, /View on MRI/);
  assert.match(html, /aria-pressed="false"/);
  assert.match(html, /Review required/);
  assert.doesNotMatch(html, />Accept|>Approve|Use as anatomy/);
});
test("selected proposal can clear and jump to a computed outside-annotation point", () => {
  const item = evidence("review_required", true);
  const html = renderToStaticMarkup(
    React.createElement(Inventory, {
      evidence: [item],
      inspection: {
        onSelect: () => {},
        onClear: () => {},
        selectedId: item.evidenceId,
        onOutside: () => {},
        outsideCount: 214,
      },
    }),
  );
  assert.match(html, /Hide estimate/);
  assert.match(html, /aria-pressed="true"/);
  assert.match(html, /Inspect annotation outside \(214\)/);
});

function priorView(kind = "functional_concordance", coverage = 1, value = 0) {
  return {
    mapKind: kind,
    shape: [1, 1, 1],
    affine: [
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1],
    ],
    values: new Float32Array([value]),
    coverage: new Uint8Array([coverage]),
  };
}
test("covered prior zero explicitly leaves patient function unknown", () => {
  const html = renderToStaticMarkup(
    React.createElement(PriorCursorReadout, {
      layer: priorView(),
      cursor: [0, 0, 0],
    }),
  );
  assert.match(html, /Within atlas field of view/);
  assert.match(html, /0.0000/);
  assert.match(
    html,
    /No signal in this released map; patient function unknown/,
  );
});
test("uncovered prior cell shows unknown without presenting a zero map value", () => {
  const html = renderToStaticMarkup(
    React.createElement(PriorCursorReadout, {
      layer: priorView("functional_concordance", 0),
      cursor: [0, 0, 0],
    }),
  );
  assert.match(html, /Outside atlas field of view — unknown/);
  assert.doesNotMatch(html, /0.0000|prior-cursor-value/);
});
test("prior inventory distinguishes all seven maps and does not claim patient function or language dominance", () => {
  const items = [
    { component: "motor", mapKind: "functional_concordance" },
    ...["phonology", "semantics", "speech_articulation"].flatMap((component) =>
      ["functional_concordance", "structural_mask"].map((mapKind) => ({
        component,
        mapKind,
      })),
    ),
  ].map((item, index) => ({ ...item, proposalId: String(index) }));
  const html = renderToStaticMarkup(
    React.createElement(PriorInventory, {
      items,
      selected: null,
      loadingId: null,
      disabled: false,
      onSelect: () => {},
      onClear: () => {},
      cursor: null,
    }),
  );
  assert.equal((html.match(/<option /g) ?? []).length, 8);
  assert.match(html, /Speech arrest \/ articulation · structural network mask/);
  assert.match(html, /Not used in route scoring/);
  assert.match(html, /Patient language dominance: unknown/);
  assert.doesNotMatch(html, />Accept|>Approve/);
});

test("a partly covered interpolation footprint is unknown rather than a covered zero", () => {
  const layer = priorView();
  layer.shape = [2, 1, 1];
  layer.values = new Float32Array([0, 0]);
  layer.coverage = new Uint8Array([1, 0]);
  const html = renderToStaticMarkup(
    React.createElement(PriorCursorReadout, { layer, cursor: [0.25, 0, 0] }),
  );
  assert.match(html, /Incomplete atlas sampling support — unknown/);
  assert.doesNotMatch(html, /0.0000|prior-cursor-value/);
});

test("tiny positive concordance cannot be displayed as a covered zero", () => {
  const html = renderToStaticMarkup(
    React.createElement(PriorCursorReadout, {
      layer: priorView("functional_concordance", 1, 0.00001),
      cursor: [0, 0, 0],
    }),
  );
  assert.match(html, /1.00e-5/);
  assert.doesNotMatch(html, /0.0000|No signal in this released map/);
});

test("atlas boundary precision abstention stays unknown in the shell readout", () => {
  const html = renderToStaticMarkup(
    React.createElement(PriorCursorReadout, {
      layer: priorView(),
      cursor: [-0.5, 0, 0],
    }),
  );
  assert.match(html, /Atlas field boundary — display precision unknown/);
  assert.doesNotMatch(
    html,
    /0.0000|prior-cursor-value|Outside atlas field of view/,
  );
});
test("excessive coordinate precision error cannot appear as a mapped value", () => {
  const layer = priorView();
  layer.affine[0][3] = 1e7;
  const html = renderToStaticMarkup(
    React.createElement(PriorCursorReadout, {
      layer,
      cursor: [1e7, 0, 0],
    }),
  );
  assert.match(html, /Atlas coordinates exceed display precision — unknown/);
  assert.doesNotMatch(
    html,
    /0.0000|prior-cursor-value|Outside atlas field of view/,
  );
});

test("the unloaded app owns one welcome message without mounting the viewer placeholder beneath it", () => {
  const previousWindow = globalThis.window;
  globalThis.window = { resectionApi: {} };
  try {
    const html = renderToStaticMarkup(React.createElement(App));
    assert.equal((html.match(/class="welcome-overlay"/g) ?? []).length, 1);
    assert.match(html, /From source imaging/);
    assert.match(html, /Open a case/);
    assert.match(html, /Local workspace · Open imaging to begin/);
    assert.doesNotMatch(html, /rl-viewer-empty|Your case, in perspective/);
    assert.doesNotMatch(html, /workspace-case-id|Current case:/);
    assert.doesNotMatch(
      html,
      /aria-label="Candidate category"|aria-label="Primary route A"|aria-label="Comparison route B"/,
    );
    assert.match(
      html,
      /<button[^>]*annotation-center-button[^>]*disabled=""[^>]*>[\s\S]*?Center on annotations<\/button>/,
    );
  } finally {
    if (previousWindow === undefined) delete globalThis.window;
    else globalThis.window = previousWindow;
  }
});

function routeControls(routes, available, overrides = {}) {
  return React.createElement(RouteSelectionControls, {
    routes,
    available,
    category: "pareto",
    routeA: "",
    routeB: "",
    onCategory: () => {},
    onRoute: () => {},
    ...overrides,
  });
}

function routeGuidance(routes, available, overrides = {}) {
  return React.createElement(RouteComparison, {
    selected: [],
    all: routes,
    onFailure: () => {},
    emptyPrompt: routeComparisonPrompt({
      hasCase: true,
      hasTargetAnnotations: true,
      candidateCount: routes.length,
      availableCount: available.length,
      support: {
        blocked: false,
        requiresEstimatedSupport: false,
        reason: null,
      },
      estimatedSupportChosen: false,
      readOnly: false,
      engineStopped: false,
      generating: false,
      ...overrides,
    }),
  });
}

test("cases without candidates show search or blocked guidance without empty category and A/B controls", () => {
  for (const blocked of [false, true]) {
    const state = blocked
      ? {
          support: researchSupportGate({
            brainMask: null,
            metadata: { structural_coverage: "full_head" },
          }),
        }
      : {};
    const html = renderToStaticMarkup(
      React.createElement(
        React.Fragment,
        null,
        routeControls([], []),
        routeGuidance([], [], state),
      ),
    );
    assert.doesNotMatch(
      html,
      /<select|Retained alternatives|Select a route|Add a comparison/,
    );
    assert.match(
      html,
      blocked ? /Access support needs review/ : /Generate routes to compare/,
    );
    if (blocked)
      assert.match(html, /Full-head MRI requires reviewed research support/);
  }
});

test("an empty filtered category keeps the category selector available when other candidates exist", () => {
  const routes = [
    {
      route_id: "rejected-route",
      tool_id: "generic_suction",
      category: "rejected",
    },
  ];
  const html = renderToStaticMarkup(
    React.createElement(
      React.Fragment,
      null,
      routeControls(routes, []),
      routeGuidance(routes, []),
    ),
  );
  const category = html.match(
    /<select aria-label="Candidate category"[^>]*>/,
  )?.[0];
  assert.ok(category);
  assert.doesNotMatch(category, /disabled/);
  assert.match(html, /Rejected candidates/);
  assert.match(html, /aria-label="Primary route A" disabled=""/);
  assert.match(html, /aria-label="Comparison route B" disabled=""/);
  assert.match(html, /No candidates in this category/);
  assert.match(html, /Choose another candidate category/);
});

test("populated candidates retain named A/B choices and their selected alternatives", () => {
  const routes = [
    { route_id: "route-a", tool_id: "generic_suction", category: "pareto" },
    { route_id: "route-b", tool_id: "generic_aspirator", category: "pareto" },
  ];
  const html = renderToStaticMarkup(
    routeControls(routes, routes, { routeA: "route-a", routeB: "route-b" }),
  );
  assert.match(html, /aria-label="Candidate category"/);
  assert.match(html, /aria-label="Primary route A"/);
  assert.match(html, /aria-label="Comparison route B"/);
  assert.doesNotMatch(html, /disabled=""/);
  assert.match(html, /value="route-a" selected="">Route 01 · Suction/);
  assert.match(html, /value="route-b" selected="">Route 02 · Aspirator/);
});

test("workspace breadcrumb identifies the loaded case and removes a previous case on replacement", () => {
  const first = renderToStaticMarkup(
    React.createElement(WorkspaceBreadcrumb, { caseId: "UCSF-PDGM-0004" }),
  );
  assert.match(first, /aria-label="Current case: UCSF-PDGM-0004"/);
  assert.match(first, />UCSF-PDGM-0004<\/span>/);
  assert.doesNotMatch(first, /Route comparison/);
  const next = renderToStaticMarkup(
    React.createElement(WorkspaceBreadcrumb, { caseId: "BTC-sub-PAT20" }),
  );
  assert.match(next, /aria-label="Current case: BTC-sub-PAT20"/);
  assert.doesNotMatch(next, /UCSF-PDGM-0004/);
  const unloaded = renderToStaticMarkup(
    React.createElement(WorkspaceBreadcrumb, { caseId: null }),
  );
  assert.match(unloaded, /Route comparison/);
  assert.doesNotMatch(unloaded, /workspace-case-id|Current case:/);
});

test("visually truncated case identifiers retain their complete accessible name and tooltip", () => {
  const caseId = `subject-${"long-id-".repeat(40)}PAT20`;
  const html = renderToStaticMarkup(
    React.createElement(WorkspaceBreadcrumb, { caseId }),
  );
  assert.ok(html.includes(`aria-label="Current case: ${caseId}"`));
  assert.ok(html.includes(`title="Current case: ${caseId}"`));
  assert.ok(html.includes(`>${caseId}</span>`));
  assert.match(html, /class="workspace-case-id" role="status"/);
});

test("centering annotations is an accessible navigation action that publishes only a copied MRI point", () => {
  const center = Object.freeze([-12, 2, 11]);
  const calls = [];
  const element = AnnotationCenterButton({
    center,
    onCenter: (point) => calls.push(point),
  });
  const html = renderToStaticMarkup(element);
  assert.match(html, /Center on annotations/);
  assert.match(
    html,
    /Move the linked MRI slices to the source annotation center/,
  );
  assert.doesNotMatch(html, /disabled=""/);
  element.props.onClick();
  assert.deepEqual(calls, [[-12, 2, 11]]);
  assert.notEqual(calls[0], center);
  calls[0][0] = 99;
  assert.deepEqual(center, [-12, 2, 11]);
});

test("centering without annotations stays disabled and cannot publish an invented MRI midpoint", () => {
  const calls = [];
  const element = AnnotationCenterButton({
    center: null,
    onCenter: (point) => calls.push(point),
  });
  assert.match(renderToStaticMarkup(element), /disabled=""/);
  element.props.onClick();
  assert.deepEqual(calls, []);
});
