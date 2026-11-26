import assert from 'node:assert/strict';
import {test} from 'node:test';
import {
  NICE_SCALE_LENGTHS_MM,
  SCALE_BAR_MAX_PX,
  cameraTargetDistance,
  formatScaleLength,
  niceScaleBar,
  perspectiveMmPerPixel,
} from './scaleBar.ts';

const close = (actual, expected, tolerance = 1e-9) =>
  assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);

test('a perspective pane covers 2·d·tan(fov/2) millimetres over its height', () => {
  // 90° vertical field at 100 mm: 200 mm over 400 px.
  close(perspectiveMmPerPixel(100, 90, 400), 0.5);
  // The default 34° anatomy camera at 300 mm over 600 px.
  close(perspectiveMmPerPixel(300, 34, 600), (2 * 300 * Math.tan((17 * Math.PI) / 180)) / 600);
  for (const args of [[0, 34, 600], [300, 0, 600], [300, 180, 600], [300, 34, 0], [Number.NaN, 34, 600], [300, 34, Infinity]])
    assert.equal(perspectiveMmPerPixel(...args), null, `${args}`);
});

test('camera distance is measured to the orbit target in RAS millimetres', () => {
  close(cameraTargetDistance([3, 4, 12], [0, 0, 0]), 13);
  close(cameraTargetDistance([10, -20, 5], [10, -20, 5]), 0);
});

test('the bar is the longest round length that fits, and updates with zoom', () => {
  assert.deepEqual(niceScaleBar(1), {mm: 100, px: 100});
  assert.deepEqual(niceScaleBar(0.5), {mm: 50, px: 100});
  assert.deepEqual(niceScaleBar(0.25), {mm: 20, px: 80});
  assert.deepEqual(niceScaleBar(0.1), {mm: 10, px: 100});
  // Zooming in (fewer mm per pixel) never lengthens the physical bar.
  let previous = Infinity;
  for (let mmPerPixel = 2; mmPerPixel > 0.01; mmPerPixel *= 0.9) {
    const bar = niceScaleBar(mmPerPixel);
    assert.ok(bar.mm <= previous);
    assert.ok(bar.px <= SCALE_BAR_MAX_PX + 1e-9 && bar.px >= SCALE_BAR_MAX_PX * 0.4 - 1e-9, `${mmPerPixel}: ${bar.px}px`);
    previous = bar.mm;
  }
});

test('ordinary anatomy views use 10, 20, 50 or 100 mm', () => {
  for (const [distance, height] of [[250, 520], [400, 720], [650, 860], [900, 520]]) {
    const bar = niceScaleBar(perspectiveMmPerPixel(distance, 34, height));
    assert.ok([10, 20, 50, 100].includes(bar.mm), `${distance} mm over ${height} px gave ${bar.mm} mm`);
  }
});

test('an exact fit is accepted and the extremes stay finite', () => {
  assert.deepEqual(niceScaleBar(50 / SCALE_BAR_MAX_PX), {mm: 50, px: SCALE_BAR_MAX_PX});
  const tiny = niceScaleBar(1e-5);
  assert.equal(tiny.mm, NICE_SCALE_LENGTHS_MM[0]);
  const huge = niceScaleBar(100);
  assert.equal(huge.mm, NICE_SCALE_LENGTHS_MM.at(-1));
  for (const invalid of [null, undefined, 0, -1, Number.NaN, Infinity]) assert.equal(niceScaleBar(invalid), null);
  assert.equal(niceScaleBar(1, 0), null);
});

test('lengths are labelled in millimetres', () => {
  assert.equal(formatScaleLength(20), '20 mm');
  assert.equal(formatScaleLength(0.5), '0.5 mm');
  assert.equal(formatScaleLength(100), '100 mm');
});
