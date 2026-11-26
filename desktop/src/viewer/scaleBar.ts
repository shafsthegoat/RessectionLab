/**
 * Physical scale bar for the perspective 3D pane. Coordinates are RAS
 * millimetres, so one world unit is one millimetre. The bar is exact only at
 * the orbit target's depth; nearer anatomy appears larger in perspective.
 */
export const NICE_SCALE_LENGTHS_MM = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500] as const;

/** Default longest bar, in CSS pixels; the chosen bar is 40-100% of it. */
export const SCALE_BAR_MAX_PX = 120;

/** Millimetres covered by one CSS pixel at `distanceMm` in front of the camera. */
export function perspectiveMmPerPixel(
  distanceMm: number,
  verticalFovDeg: number,
  paneHeightPx: number,
): number | null {
  if (!(distanceMm > 0) || !Number.isFinite(distanceMm)) return null;
  if (!(verticalFovDeg > 0 && verticalFovDeg < 180)) return null;
  if (!(paneHeightPx > 0) || !Number.isFinite(paneHeightPx)) return null;
  return (2 * distanceMm * Math.tan((verticalFovDeg * Math.PI) / 360)) / paneHeightPx;
}

/** Distance from the camera to its orbit target, both in RAS millimetres. */
export function cameraTargetDistance(
  position: readonly number[],
  target: readonly number[],
): number {
  return Math.hypot(position[0] - target[0], position[1] - target[1], position[2] - target[2]);
}

/**
 * The longest round length (…, 10, 20, 50, 100 mm, …) that fits in `maxPx`.
 * When even the shortest length is wider, that length is returned at its true width.
 */
export function niceScaleBar(
  mmPerPixel: number | null | undefined,
  maxPx = SCALE_BAR_MAX_PX,
): {mm: number; px: number} | null {
  if (mmPerPixel == null || !Number.isFinite(mmPerPixel) || mmPerPixel <= 0) return null;
  if (!(maxPx > 0)) return null;
  const longestMm = maxPx * mmPerPixel;
  let mm: number = NICE_SCALE_LENGTHS_MM[0];
  for (const length of NICE_SCALE_LENGTHS_MM) if (length <= longestMm * (1 + 1e-9)) mm = length;
  return {mm, px: mm / mmPerPixel};
}

export function formatScaleLength(mm: number): string {
  return `${Number(mm.toPrecision(3))} mm`;
}
