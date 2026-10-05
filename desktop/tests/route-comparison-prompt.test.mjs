import assert from "node:assert/strict";
import { test } from "node:test";
import { routeComparisonPrompt } from "../src/route-comparison-prompt.ts";
import { researchSupportGate } from "../src/case-support.ts";

function loadedCase() {
  return {
    hasCase: true,
    hasTargetAnnotations: true,
    candidateCount: 0,
    availableCount: 0,
    support: researchSupportGate({ brainMask: null, metadata: {} }),
    estimatedSupportChosen: false,
    readOnly: false,
    engineStopped: false,
    generating: false,
  };
}

test("the photographed loaded-case state explains the undecided access assumption without opting in", () => {
  const state = loadedCase();
  const before = structuredClone(state);
  const prompt = routeComparisonPrompt(state);
  assert.equal(prompt.title, "Choose an access assumption");
  assert.match(prompt.description, /Search settings/);
  assert.match(prompt.description, /does not certify cortical access/);
  assert.deepEqual(state, before);
  assert.equal(
    routeComparisonPrompt({ ...state, estimatedSupportChosen: true }).title,
    "Generate routes to compare",
  );
});

test("a blocked full-head case does not suggest that the estimated-support option can unlock search", () => {
  const state = loadedCase();
  state.support = researchSupportGate({
    brainMask: null,
    metadata: { structural_coverage: "full_head" },
  });
  const prompt = routeComparisonPrompt(state);
  assert.equal(prompt.title, "Access support needs review");
  assert.equal(prompt.description, state.support.reason);
  assert.doesNotMatch(prompt.description, /decide whether|Generate candidate routes/);
});

test("cleared selections and empty categories ask for inspection choices instead of a new case", () => {
  const state = { ...loadedCase(), candidateCount: 54, availableCount: 12 };
  assert.equal(routeComparisonPrompt(state).title, "Choose a route to inspect");
  assert.equal(
    routeComparisonPrompt({ ...state, availableCount: 0 }).title,
    "No candidates in this category",
  );
  assert.equal(
    routeComparisonPrompt({ ...state, engineStopped: true }).title,
    "Choose a route to inspect",
  );
});

test("missing source inputs and unavailable execution give actionable messages without suggesting a runnable search", () => {
  const state = loadedCase();
  assert.equal(
    routeComparisonPrompt({ ...state, hasCase: false }).title,
    "Open a case to begin",
  );
  assert.equal(
    routeComparisonPrompt({ ...state, hasTargetAnnotations: false }).title,
    "Target annotations are needed",
  );
  assert.equal(
    routeComparisonPrompt({ ...state, readOnly: true }).title,
    "Route search runs in the Mac app",
  );
  assert.equal(
    routeComparisonPrompt({ ...state, engineStopped: true }).title,
    "Route search is unavailable",
  );
});
