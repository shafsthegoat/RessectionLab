import assert from 'node:assert/strict';
import {before, after, test} from 'node:test';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createServer} from 'vite';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';

let server, cache, ModeChip, ModeChipStack, resolveModes;
before(async () => {
  cache = await fs.mkdtemp(path.join(os.tmpdir(), 'mode-chip-'));
  server = await createServer({root: fileURLToPath(new URL('../', import.meta.url)), cacheDir: cache, logLevel: 'error',
    optimizeDeps: {noDiscovery: true, include: []}, server: {middlewareMode: true, hmr: false, ws: false}, appType: 'custom'});
  ({ModeChip, ModeChipStack} = await server.ssrLoadModule('/src/viewer/ModeChip.tsx'));
  ({resolveModes} = await server.ssrLoadModule('/src/viewer/viewerModes.ts'));
});
after(async () => {
  await server?.close();
  if (cache) await fs.rm(cache, {recursive: true, force: true});
});

test('a chip names its mode, keeps its provenance detail and offers a named exit', () => {
  const html = renderToStaticMarkup(React.createElement(ModeChip, {
    label: 'Section', detail: 'Source MRI · Axial · Z 12.0 mm', exitLabel: 'Exit section', escapable: true, onExit: () => {},
  }));
  assert.match(html, /role="group" aria-label="Section mode"/);
  assert.match(html, /<span class="rl-mode-chip-name">Section<\/span>/);
  assert.match(html, /<span class="rl-mode-chip-detail">Source MRI · Axial · Z 12.0 mm<\/span>/);
  assert.match(html, /<button type="button" class="rl-mode-chip-exit" aria-label="Exit section" title="Exit section \(Esc\)">/);
});

test('a mode without a way back here shows no exit button', () => {
  const html = renderToStaticMarkup(React.createElement(ModeChip, {label: 'Branch crop', detail: '3D only'}));
  assert.doesNotMatch(html, /<button/);
});

test('the stack orders chips by priority and keeps the verifier-facing exit names', () => {
  const modes = resolveModes([
    {id: 'prior-view', detail: 'Motor atlas · alignment review required', onExit: () => {}},
    {id: 'cta-cut', detail: 'Axial · Z 4.0 mm', exitLabel: 'Show full CTA head', onExit: () => {}},
    {id: 'vessel-slab', exitLabel: 'Exit vessel slab in 3D', onExit: () => {}},
  ]);
  const html = renderToStaticMarkup(React.createElement(ModeChipStack, {modes,
    actions: {'vessel-slab': React.createElement('button', {className: 'rl-mode-chip-action', 'aria-label': 'Faint vessel context outside slab', 'aria-pressed': false}, 'Faint context')}}));
  const order = [...html.matchAll(/data-mode="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(order, ['vessel-slab', 'cta-cut', 'prior-view']);
  for (const name of ['Exit vessel slab in 3D', 'Show full CTA head', 'Exit population prior', 'Faint vessel context outside slab'])
    assert.ok(html.includes(`aria-label="${name}"`), name);
  assert.match(html, /aria-live="polite"/);
});
