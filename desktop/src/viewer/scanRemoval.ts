import type {SurgicalView} from '../surgical-types';
import {bodyIsDisplaced} from './exposureAnatomy.ts';

/** Source cells the displayed scan volume leaves empty: accepted removal and
 * declared operative bodies (scalp, bone, dura) that have left their bed.
 * Display only; the engine's counts and the source grid are unchanged. */
export function displayRemovalCells(view: Pick<SurgicalView, 'removed' | 'operativeExposureV1' | 'scene'>): number[] {
  const cells = [...view.removed];
  const exposure = view.operativeExposureV1;
  if (exposure)
    for (const body of Object.values(exposure.bodies))
      if (bodyIsDisplaced(body, view.scene.shape, view.scene.affine)) cells.push(...body.cells);
  return cells;
}

export function maskCells(mask: Uint8Array): Uint32Array {
  let count = 0;
  for (let i = 0; i < mask.length; i++) if (mask[i]) count++;
  const cells = new Uint32Array(count);
  for (let i = 0, k = 0; i < mask.length; i++) if (mask[i]) cells[k++] = i;
  return cells;
}

/** Operative tissue whose walls are shaded by the case's own scan intensity. */
export const SCAN_SHADED_KINDS = new Set(['grayMatter', 'whiteMatter', 'tract', 'tumor']);

/** Intensity window over the brain and lesion cells of the scan: the 5th and
 * 95th percentiles of a regular sample. Undefined when the cells are too few
 * or the signal is flat. */
export function scanTissueWindow(signal: Float32Array, masks: readonly Uint8Array[], stride = 5): [number, number] | undefined {
  const values: number[] = [];
  for (let i = 0; i < signal.length; i += stride)
    if (masks.some((mask) => mask[i]) && Number.isFinite(signal[i])) values.push(signal[i]);
  if (values.length < 64) return;
  values.sort((a, b) => a - b);
  const low = values[Math.floor(0.05 * (values.length - 1))],
    high = values[Math.ceil(0.95 * (values.length - 1))];
  return high > low ? [low, high] : undefined;
}
