export interface PaneBounds {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** Actual image-area bounds, in CSS pixels; Three applies device pixel ratio. */
export function paneViewport(root: PaneBounds, pane: PaneBounds) {
  const width = Math.round(pane.width),
    height = Math.round(pane.height);
  if (width < 1 || height < 1) return null;
  return {
    left: Math.round(pane.left - root.left),
    bottom: Math.round(root.height - (pane.top + pane.height - root.top)),
    width,
    height,
  };
}
