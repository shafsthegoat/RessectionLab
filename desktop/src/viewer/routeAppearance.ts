export const COMPARISON_COLORS = { A: "#a3e5d3", B: "#e5c598" } as const;
export const FAILURE_COLOR = "#ff8580";

/** Slot identity survives compacting the visible route array or rejection. */
export function routeAppearance(
  route: { comparisonSlot?: "A" | "B" },
  visibleIndex: number,
) {
  const slot = route.comparisonSlot ?? (visibleIndex === 1 ? "B" : "A");
  return { slot, color: COMPARISON_COLORS[slot] };
}
