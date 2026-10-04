import assert from "node:assert/strict";
import { test } from "node:test";
import { selectComparisonRoutes } from "../src/route-selection.ts";
const routes = [{ route_id: "first" }, { route_id: "second" }];
test("B-only route retains B identity instead of shifting to A", () => {
  assert.deepEqual(selectComparisonRoutes(routes, "", "second"), [
    { route_id: "second", comparisonSlot: "B" },
  ]);
});
test("A/B order is explicit and source search records remain unchanged", () => {
  assert.deepEqual(
    selectComparisonRoutes(routes, "second", "first").map((route) => [
      route.route_id,
      route.comparisonSlot,
    ]),
    [
      ["second", "A"],
      ["first", "B"],
    ],
  );
  assert.equal(routes[0].comparisonSlot, undefined);
});
test("same route is rendered once and missing choices do not replace slots", () => {
  assert.deepEqual(
    selectComparisonRoutes(routes, "first", "first").map(
      (route) => route.comparisonSlot,
    ),
    ["A"],
  );
  assert.deepEqual(
    selectComparisonRoutes(routes, "unknown", "second").map(
      (route) => route.comparisonSlot,
    ),
    ["B"],
  );
});
