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

/** Rehearse navigation layout. The operative field stays the main pane; the
 * three linked source slices share one column on its right, with the same
 * pane chrome as Anatomy (38 px heading, 27 px slice scrubber). The column
 * width is one CSS clamp() so the stylesheet and this geometry agree. */
export const REHEARSE_SLICE_COLUMN = {minPx: 232, percent: 33, maxPx: 420} as const;
export const SLICE_PANE_CHROME = {headingPx: 38, scrubberPx: 27} as const;

export function rehearseSliceColumnCss(): string {
  const {minPx, percent, maxPx} = REHEARSE_SLICE_COLUMN;
  return `clamp(${minPx}px, ${percent}%, ${maxPx}px)`;
}

export function rehearseSliceColumnPx(shellWidth: number): number {
  const {minPx, percent, maxPx} = REHEARSE_SLICE_COLUMN;
  return Math.min(maxPx, Math.max(minPx, shellWidth * percent / 100));
}

/** Pane rectangles for a viewer shell of the given size, in CSS pixels
 * relative to the shell. `toplinePx` and `transportPx` are the field's own
 * header and timeline rows; the slice column spans the full shell height. */
export function rehearseNavigationPanes(shell: {width: number; height: number}, rows: {toplinePx: number; transportPx: number}, gapPx = 4) {
  const column = rehearseSliceColumnPx(shell.width);
  const fieldWidth = Math.max(0, shell.width - column - gapPx);
  const sliceHeight = Math.max(0, (shell.height - 2 * gapPx) / 3);
  const slice = (index: number): PaneBounds => ({left: fieldWidth + gapPx, top: index * (sliceHeight + gapPx), width: column, height: sliceHeight});
  const image = (pane: PaneBounds): PaneBounds => ({left: pane.left, top: pane.top + SLICE_PANE_CHROME.headingPx,
    width: Math.max(0, pane.width - SLICE_PANE_CHROME.scrubberPx), height: Math.max(0, pane.height - SLICE_PANE_CHROME.headingPx)});
  const field: PaneBounds = {left: 0, top: rows.toplinePx, width: fieldWidth, height: Math.max(0, shell.height - rows.toplinePx - rows.transportPx)};
  const slices = {axial: slice(0), coronal: slice(1), sagittal: slice(2)};
  return {field, slices, sliceImages: {axial: image(slices.axial), coronal: image(slices.coronal), sagittal: image(slices.sagittal)}};
}
