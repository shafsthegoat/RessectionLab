import assert from 'node:assert/strict';
import {test} from 'node:test';
import {VIEWPORT_HUD_HIDE_MS, ViewportHudController, viewportHudReading} from './viewportHud.ts';

const range = (ariaLabel, value, extra = {}) => ({ariaLabel, value, min: 0, max: 1, ...extra});

test('3D-image controls are recognised by name without markup changes', () => {
  assert.deepEqual(viewportHudReading(range('Tumour opacity', .42)), {label: 'Tumour opacity', value: '42%'});
  assert.deepEqual(viewportHudReading(range('Registered blend opacity', .4)), {label: 'Blend', value: '40%'});
  assert.deepEqual(viewportHudReading(range('Transfer opacity 2', .05)), {label: 'Transfer opacity 2', value: '5%'});
  assert.deepEqual(viewportHudReading({ariaLabel: 'Scan volume density', value: .85, min: 0, max: 2}), {label: 'Scan density', value: '0.85'});
});

test('slice, timeline and camera sliders are not 3D-image controls', () => {
  for (const name of ['Axial slice position', 'Surgical timeline', 'Viewing angle', 'Operated section slice', '', null])
    assert.equal(viewportHudReading({ariaLabel: name, value: 3, min: 0, max: 10}), null, String(name));
});

test('explicit HUD names, value text and units take precedence', () => {
  assert.deepEqual(viewportHudReading({ariaLabel: 'Cutaway position', hudLabel: 'Axial cutaway', valueText: 'Z 12.3 mm', value: 12.3, min: -80, max: 80}),
    {label: 'Axial cutaway', value: 'Z 12.3 mm'});
  assert.deepEqual(viewportHudReading({ariaLabel: 'Source MRI window width', value: 1.25, min: .4, max: 2}), {label: 'Window width', value: '1.25'});
  assert.deepEqual(viewportHudReading({hudLabel: 'Offset', hudUnit: 'mm', value: -4.25, min: -30, max: 30}), {label: 'Offset', value: '-4.25 mm'});
  assert.deepEqual(viewportHudReading({ariaLabel: 'Cutaway position', value: 112.34, min: 0, max: 200}), {label: 'Cutaway', value: '112'});
});

function fakeTimers() {
  let now = 0, next = 1;
  const pending = new Map();
  return {
    set(callback, ms) { const id = next++; pending.set(id, {at: now + ms, callback}); return id; },
    clear(id) { pending.delete(id); },
    advance(ms) {
      now += ms;
      for (const [id, entry] of [...pending].sort((a, b) => a[1].at - b[1].at))
        if (entry.at <= now && pending.has(id)) { pending.delete(id); entry.callback(); }
    },
  };
}

function harness() {
  const timers = fakeTimers(), events = [];
  const controller = new ViewportHudController({show: reading => events.push(['show', reading.value]), hide: () => events.push(['hide'])}, VIEWPORT_HUD_HIDE_MS, timers);
  return {timers, events, controller};
}

test('a drag shows the value throughout and hides about 600 ms after release', () => {
  const {timers, events, controller} = harness();
  controller.press();
  controller.change({label: 'Opacity', value: '40%'});
  timers.advance(2000);
  controller.change({label: 'Opacity', value: '45%'});
  assert.ok(controller.visible, 'a held pointer keeps the HUD up');
  controller.release();
  timers.advance(VIEWPORT_HUD_HIDE_MS - 1);
  assert.ok(controller.visible);
  timers.advance(1);
  assert.equal(controller.visible, false);
  assert.deepEqual(events, [['show', '40%'], ['show', '45%'], ['hide']]);
});

test('keyboard steps each restart the hide delay; a press without a change shows nothing', () => {
  const {timers, events, controller} = harness();
  controller.press();
  controller.release();
  timers.advance(1000);
  assert.deepEqual(events, []);
  controller.change({label: 'Cutaway', value: '1'});
  timers.advance(400);
  controller.change({label: 'Cutaway', value: '2'});
  timers.advance(400);
  controller.release();
  assert.ok(controller.visible);
  timers.advance(200);
  assert.equal(controller.visible, false);
  controller.dispose();
});
