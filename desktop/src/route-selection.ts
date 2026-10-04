import type { RouteCandidate } from "./types";

/** A/B are comparison identities, not positions in the compact visible array. */
export interface SelectedRoute extends RouteCandidate {
  comparisonSlot: "A" | "B";
}
export function selectComparisonRoutes(
  routes: RouteCandidate[],
  routeA: string,
  routeB: string,
): SelectedRoute[] {
  const selected: SelectedRoute[] = [];
  for (const [slot, id] of [
    ["A", routeA],
    ["B", routeB],
  ] as const) {
    if (!id || selected.some((route) => route.route_id === id)) continue;
    const route = routes.find((candidate) => candidate.route_id === id);
    if (route) selected.push({ ...route, comparisonSlot: slot });
  }
  return selected;
}
