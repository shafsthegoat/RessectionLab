'use strict';
// Electron entry point. Run write on a disposable patient study, then reopen its
// saved package in a second process with the first receipt as the last argument.
const { app, dialog, ipcMain } = require('electron');
const assert = require('node:assert/strict'),
  fs = require('node:fs'),
  path = require('node:path'),
  os = require('node:os');
const { inputScript } = require('./verify-eden-planning-helpers.cjs');
const { studyScanSelectionScript } = require('./verify-study-ui-helpers.cjs');
const [sourceArg, outArg, phase = 'write', priorPath] = process.argv.slice(2);
assert(sourceArg && outArg && ['write', 'reopen'].includes(phase));
const root = path.resolve(__dirname, '../..'),
  allowed = fs.realpathSync(path.join(root, 'testcases/outputs')) + path.sep;
const source = path.resolve(sourceArg), out = path.resolve(outArg);
assert(source.startsWith(allowed) && out.startsWith(allowed) && !fs.existsSync(out));
// Writing exercises interactive plan edits on a disposable local patient.
// A persisted study may have been moved to the mounted archive behind a local link.
if (phase === 'write') assert(fs.realpathSync(source).startsWith(allowed));
const prior = priorPath ? JSON.parse(fs.readFileSync(priorPath)) : null;
assert(
  phase === 'write' ? !prior : prior?.passed && fs.realpathSync(source) === fs.realpathSync(prior.archive),
);
fs.mkdirSync(out);
const archive = path.join(out, 'paired-planning.rsworkspace');
const report = {
  phase,
  pid: process.pid,
  source,
  visible: false,
  requests: [],
  views: [],
  captures: [],
  errors: [],
};
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'rslab-paired-plan-'));
app.setPath('userData', profile);
if (process.platform === 'darwin') app.setActivationPolicy('accessory');
process.argv.push('--automation-hidden', '--anatomy', source);
delete process.env.RESSECTIONLAB_DEV_URL;
dialog.showSaveDialog = async () => ({ canceled: false, filePath: archive });
const bind = ipcMain.handle.bind(ipcMain);
ipcMain.handle = (channel, handler) =>
  bind(channel, async (...args) => {
    try {
      const r = await handler(...args);
      if (channel === 'research:startupWorkspace') {
        const d = r.document;
        report.patient = d.patient_id;
        report.snapshot = d.snapshot_id;
        report.binding = r.collectionId;
        report.reference = d.reference_series;
        report.sources = d.series.map((s) => ({
          id: s.id,
          modality: s.modality,
          sha256: s.original.sha256,
          status: d.transforms[s.id].status,
        }));
        report.initialView = r.viewState;
      }
      if (channel === 'research:anatomyView') {
        report.views.push({
          id: args[1].seriesId,
          ready: r.patientFrameReady,
          sourceHash: r.mri.sha256,
          alignment: r.registrationStatus,
          sourceAppearances: (r.appearanceSources ?? []).map((s) => ({
            id: s.seriesId,
            shape: s.signal.shape,
            signal: s.signal.sha256,
          })),
        });
      }
      if (channel === 'research:anatomyPlan') {
        if (args[1].action === 'operativeField') {
          report.field = {
            target: r.targetLayerId,
            route: r.approachId,
            hash: r.parameterHash,
            snapshot: r.snapshotId,
            labels: r.labels.sha256,
            intensity: r.intensity?.sourceSha256,
            excluded: r.excluded,
            stations: r.stations,
          };
        } else if (r.approaches) {
          report.requests.push({
            action: args[1].action,
            selected: r.selectedId,
            options: r.approaches.map((a) => a.id),
          });
        }
      }
      return r;
    } catch (e) {
      report.errors.push({ channel, error: String(e) });
      throw e;
    }
  });
const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
let done = false;
function finish(error) {
  if (done) return;
  done = true;
  report.passed = !error && !report.visible && !report.errors.length;
  if (error) {
    report.error = String(error.stack || error);
    console.error(report.error);
  }
  fs.writeFileSync(path.join(out, 'report.json'), JSON.stringify(report, null, 2));
  app.quit();
}
process.on('uncaughtException', finish);
process.on('unhandledRejection', finish);
app.once('will-quit', () => {
  fs.rmSync(profile, { recursive: true, force: true });
  if (!report.passed) app.exit(1);
});
app.on('browser-window-created', (_, win) => {
  win.on('show', () => {
    report.visible = true;
    finish(Error('Window became visible'));
  });
  win.webContents.once('did-finish-load', async () => {
    const js = (s) => win.webContents.executeJavaScript(s, true);
    const wait = async (s, limit = 360000) => {
      const start = Date.now();
      while (Date.now() - start < limit) {
        if (report.errors.length) throw Error(JSON.stringify(report.errors.at(-1)));
        if (await js(s)) return;
        await delay(200);
      }
      throw Error('Timed out waiting for ' + s);
    };
    const button = (label) =>
      `[...document.querySelectorAll('button')].find(b=>b.getAttribute('aria-label')===${
        JSON.stringify(label)
      }||b.textContent.trim()===${JSON.stringify(label)}||b.querySelector('span')?.textContent.trim()===${
        JSON.stringify(label)
      })`;
    const click = async (name) => {
      const b = button(name);
      await wait(`!!(${b})&&!(${b}).disabled`);
      await js(
        `(()=>{const b=${b};for(let p=b.parentElement;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;b.click()})()`,
      );
    };
    const ready = async (id) => {
      await wait(
        `JSON.parse(document.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity||'[]')[1]===${
          JSON.stringify(id)
        }&&document.querySelector('.rl-viewer')?.dataset.rendererReady==='true'&&document.querySelector('.rl-viewer')?.dataset.surfacesPending==='0'&&!document.querySelector('[data-layer-pending="true"]')`,
      );
      await delay(400);
    };
    const capture = async (name) => {
      assert(!win.isVisible());
      await delay(250);
      const shot = await win.webContents.capturePage(undefined, { stayHidden: true, stayAwake: true });
      const size = await js('({width:innerWidth,height:innerHeight})');
      const im = shot.getSize().width === size.width ? shot : shot.resize(size);
      fs.writeFileSync(path.join(out, name + '.png'), im.toPNG());
      report.captures.push(name);
      console.log('Captured', name);
      return im;
    };
    const scan = async (id) => {
      await js(studyScanSelectionScript(id));
      await ready(id);
    };
    const routes = () => js("JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes)");
    try {
      await wait("document.querySelector('.rl-viewer')?.dataset.rendererReady==='true'");
      const reference = report.reference,
        ct = report.sources.find((s) => s.modality === 'CT' && s.status === 'unreviewed').id;
      await click('Anatomy');
      if (phase === 'write') {
        await ready(reference);
        await js("document.querySelector('.anatomy-presets button:nth-child(2)').click()");
        await ready(reference);
        await click('Show structures only');
        await click('Layout');
        await js(inputScript('Camera preset', 'left'));
        await click('Close view controls');
        await capture('reference-reconstruction');
        await scan(ct);
        await capture('paired-ct');
        await scan(reference);
        await click('Plan');
        await wait(
          "!document.querySelector('.approach-computing')&&[...document.querySelectorAll('select[aria-label=\"Planning target\"] option')].some(o=>o.value==='tumor'&&!o.disabled)",
        );
        await js(inputScript('Planning target', 'tumor'));
        await click('Suggest approaches');
        await wait(
          "document.querySelectorAll('.approach-card').length>=2&&!document.querySelector('.approach-computing')",
        );
        const id = await js("document.querySelectorAll('.approach-card')[1].dataset.approachId");
        await js(
          `document.querySelector('.approach-card[data-approach-id="'+${JSON.stringify(id)}+'"]').click()`,
        );
        await wait(
          `document.querySelector('.approach-card.selected')?.dataset.approachId===${JSON.stringify(id)}`,
        );
        report.selected = id;
        await click('Anatomy');
        await ready(reference);
        await wait(
          `JSON.parse(document.querySelector('.rl-viewer').dataset.approachRoutes).some(r=>r.selected&&r.route_id===${
            JSON.stringify(id)
          })`,
        );
        report.route = (await routes()).find((r) => r.selected);
        await capture('reference-with-selected-route');
        await scan(ct);
      } else {
        assert.notEqual(prior.pid, process.pid);
        assert.equal(report.snapshot, prior.snapshot);
        assert.deepEqual(report.sources, prior.sources);
        assert.equal(report.initialView.seriesId, ct);
        assert.equal(report.initialView.selectedApproachId, prior.selected);
        report.selected = prior.selected;
        await ready(ct);
      }
      await wait('!!document.querySelector(\'[aria-label="Plan on reference scan"]\')');
      report.restriction = await js(
        'document.querySelector(\'[aria-label="Plan on reference scan"]\').innerText',
      );
      assert.deepEqual(await routes(), []);
      await capture('ct-with-retained-plan');
      if (phase === 'write') {
        await click('Scans');
        await click('Save patient study file…');
        await wait(
          "document.querySelector('footer')?.textContent.includes('Patient study file saved')",
          600000,
        );
        report.archive = archive;
        report.archiveBytes = fs.statSync(archive).size;
        await click('Anatomy');
      }
      await click('View plan on reference scan');
      await ready(reference);
      const route = (await routes()).find((r) => r.selected);
      assert.deepEqual(route, phase === 'write' ? report.route : prior.route);
      report.route = route;
      await click('Plan');
      await click('Preview operative steps');
      await wait(
        "!!document.querySelector('.operative-field-canvas canvas')&&!!document.querySelector('.has-operative-reference')",
      );
      const surround = () =>
        js("JSON.parse(document.querySelector('.operative-field-canvas').dataset.surroundingSources)");
      report.surrounding = await surround();
      assert(report.surrounding.some((s) => s.role === 'brain' && s.appearance?.seriesId === reference));
      assert(!report.surrounding.some((s) => s.role === 'bone'));
      assert(report.field.excluded.some((l) => l.id === 'bone' && /Registration/.test(l.reason)));
      assert.equal(report.field.route, report.selected);
      assert.equal(report.field.snapshot, report.snapshot);
      assert.equal(report.field.target, 'tumor');
      if (phase === 'reopen') {
        assert.deepEqual(report.field, prior.field);
        assert.deepEqual(report.surrounding, prior.surrounding);
      }
      const whole = await capture('operative-intact-reference');
      const rect = await js(
        "document.querySelector('.operative-field-canvas canvas').getBoundingClientRect().toJSON()",
      );
      const region = {
        x: Math.ceil(rect.x + rect.width * .1),
        y: Math.ceil(rect.y + rect.height * .1),
        width: Math.floor(rect.width * .8),
        height: Math.floor(rect.height * .8),
      };
      const pixels = whole.crop(region).toBitmap();
      let tissue = 0;
      for (let i = 0; i < pixels.length; i += 4) {
        if (pixels[i + 2] > 65 && pixels[i + 2] > pixels[i + 1] + 12) tissue++;
      }
      report.intactTissueCoverage = {
        region,
        pixels: tissue,
        fraction: tissue / (region.width * region.height),
      };
      assert(
        report.intactTissueCoverage.fraction > .06,
        'Surrounding source mesh is occluded or clipped away',
      );
      const slider = 'input[aria-label="Operative timeline"]';
      await js(inputScript('Operative timeline', await js(`document.querySelector('${slider}').max`)));
      await wait("Number(document.querySelector('[data-removed-cells]')?.dataset.removedCells)>0");
      report.previewRemovedCells = await js(
        "Number(document.querySelector('[data-removed-cells]').dataset.removedCells)",
      );
      if (prior) assert.equal(report.previewRemovedCells, prior.previewRemovedCells);
      await capture('geometric-endpoint');
      await js(
        "[...document.querySelectorAll('.operative-options label')].find(l=>l.innerText.trim()==='Inspection cutaway').querySelector('input').click()",
      );
      await capture('geometric-endpoint-cutaway');
      finish();
    } catch (e) {
      await capture('failure').catch(() => {});
      finish(e);
    }
  });
});
require('./main.cjs');
