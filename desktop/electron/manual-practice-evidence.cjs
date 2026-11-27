'use strict';

const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

function digest(data) {
  const canonical = value => value !== null && typeof value === 'object' && !Array.isArray(value)
    ? Object.fromEntries(Object.keys(value).sort().map(k => [k, canonical(value[k])]))
    : Array.isArray(value) ? value.map(canonical) : value;
  return createHash('sha256').update(JSON.stringify(canonical(data))).digest('hex');
}

function summary(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const percentile = q => sorted[Math.ceil(q * sorted.length) - 1] ?? null;
  const middle=Math.floor(sorted.length/2);
  const median=sorted.length===0?null:sorted.length%2?sorted[middle]:(sorted[middle-1]+sorted[middle])/2;
  return {samples:sorted.length, median, p95:percentile(.95)};
}

function stateWitness(result) {
  const {view, branchId, position, caseHash} = result;
  const fields = Object.fromEntries(['removed', 'moved', 'holds', 'fluid', 'debris', 'sources',
    'cuts', 'deformation', 'thermal', 'contact', 'slots'].map(k => [k, view[k]]));
  const tools = Object.fromEntries(Object.entries(view.slots).map(([slot, t]) =>
    [slot, {instrument: t.instrument, position: t.position, quaternion: t.quaternion, inserted: t.inserted}]));
  return {
    stateId: view.stateId, sceneId: view.scene.sceneId, caseHash, branchId, position,
    woundPorts:Object.fromEntries(Object.entries(view.vascularGraphV1?.ports??{}).map(([id,p])=>
      [id,{cell:p.cell,sealFraction:p.sealFraction}])),
    bleedingSources:(view.sources??[]).filter(s=>s.bleedingActive).length,
    emittedMm3:view.metrics.vesselPhysicsV2?.emittedMm3??0,
    reservoirMm3:(view.sources??[]).reduce((sum,s)=>sum+(s.reservoirMm3??0),0),
    vascularResidualMm3:view.metrics.vesselPhysicsV2?.conservationResidualMm3??null,
    freeField: Object.fromEntries(['fluid','debris'].flatMap(name=>{
      const quantities=Object.values(view[name]??{}).filter(x=>Number.isFinite(x)&&x>0);
      return [[name+'Cells',quantities.length],[name+'Mm3',quantities.reduce((sum,x)=>sum+x,0)]];
    })),
    geometryRevision: view.geometryRevision, physicalDigest: digest(fields), metricDigest: digest(view.metrics),
    tools, removedCells: view.removed.length, remaining: view.originalTargetResidualMm3,
    metrics: Object.fromEntries(['elapsedSeconds', 'targetRemovedMm3', 'nonTargetRemovedMm3', 'contactCellSeconds',
      'debrisMm3', 'debrisRemovedMm3', 'fluidAddedMm3', 'fluidRemovedMm3', 'fluidPresentMm3',
      'fluidEscapedMm3', 'brainMaxDisplacementMm', 'modeledVesselFluidGeneratedMm3', 'bipolarHeatInputJ',
      'vascularGraphGeneratedMm3'].filter(k=>view.metrics[k]!==undefined).map(k => [k, view.metrics[k]])),
  };
}

function checkAction(before, after, outcome, requested) {
  assert.equal(after.branchId, before.branchId);
  assert.equal(after.sceneId, before.sceneId);
  assert.equal(after.caseHash, before.caseHash);
  assert.equal(after.stateId, after.geometryRevision);
  if (outcome.status === 'rejected') {
    assert.deepEqual(after, before, 'Refusal changed committed quantities, pose, branch, cursor or native state');
    assert.equal(outcome.completedFraction, 0);
    return;
  }
  assert(['accepted', 'partial'].includes(outcome.status));
  assert.notEqual(after.stateId, before.stateId);
  assert.equal(after.position, before.position + 1);
  if (outcome.status === 'partial') {
    assert(outcome.completedFraction >= 0 && outcome.completedFraction < 1);
    const endpoint = requested.pose ?? requested.path?.at(-1);
    if (endpoint) assert.notDeepEqual(after.tools[requested.slot].position, endpoint.position);
  } else assert.equal(outcome.completedFraction, 1);
  assert(Math.abs((after.remaining - before.remaining) +
    (after.metrics.targetRemovedMm3 - before.metrics.targetRemovedMm3)) < 1e-6);
  const balance = state => {const m=state.metrics;return m.fluidPresentMm3 + m.fluidEscapedMm3 + m.fluidRemovedMm3 - m.fluidAddedMm3;};
  // fluidPresentMm3 already includes the free cells and any blocked outlet reservoir.
  assert(Math.abs(balance(after) - balance(before)) < 1e-6);
  assert(Math.abs(after.freeField.fluidMm3+after.reservoirMm3-after.metrics.fluidPresentMm3)<1e-6,
    'Free field and outlet reservoirs must account for all retained fluid');
  if(after.vascularResidualMm3!==null)assert(Math.abs(after.vascularResidualMm3)<1e-6);
}

// Every sample is a fresh hidden compositor capture. The variable frame rate follows
// the observed wall clock, even when the engine or frame capture stalls.
function startRecording(win, scratch, width, height, readState) {
  const rows = [];
  let running = true;
  const recording = (async () => {
    while (running) {
      const start = performance.now();
      assert(!win.isVisible());
      const before = await readState();
      const shot = await win.webContents.capturePage(undefined, {stayHidden: true, stayAwake: true});
      const at = performance.now();
      const after = await readState();
      const size = shot.getSize();
      assert(size.width >= width && size.height >= height && size.width * height === size.height * width);
      const name = 'frame-' + String(rows.length).padStart(5, '0') + '.jpg';
      const bytes = shot.resize({width, height, quality: 'good'}).toJPEG(87);
      await fs.promises.writeFile(path.join(scratch, name), bytes);
      rows.push({at,name,stateId:before && before===after ? after:null});
      await delay(Math.max(0, 100 - (performance.now() - start)));
    }
  })();
  // Let the driver report errors after disposing the hidden window.
  recording.catch(() => {});
  return async filename => {
    running = false;
    await recording;
    assert(rows.length >= 2);
    const manifest = [];
    rows.forEach((row, i) => {
      manifest.push("file '" + row.name + "'", 'option framerate 1000');
      if (rows[i+1]) manifest.push('duration ' + ((rows[i+1].at - row.at) / 1000).toFixed(6));
    });
    fs.writeFileSync(path.join(scratch, 'frames.txt'), manifest.join('\n') + '\n');
    execFileSync('ffmpeg', ['-hide_banner','-loglevel','error','-nostdin','-n','-f','concat','-safe','0',
      '-i','frames.txt','-fps_mode','vfr','-c:v','libx264','-threads','3','-preset','veryfast',
      '-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',filename], {cwd:scratch});
    const stream = JSON.parse(execFileSync('ffprobe',['-v','error','-show_entries',
      'stream=width,height,nb_frames,duration','-of','json',filename])).streams[0];
    assert.equal(Number(stream.nb_frames),rows.length);
    const wallSeconds = (rows.at(-1).at - rows[0].at) / 1000;
    assert(Math.abs(Number(stream.duration) - wallSeconds) < .1);
    return {filename: path.basename(filename), width, height, wallSeconds, firstCaptureEpochMs: performance.timeOrigin + rows[0].at,
      captureOffsetsMs: rows.map(row => Math.round(row.at - rows[0].at)),
      stateChanges:rows.flatMap((row,i)=>i===0||rows[i-1].stateId!==row.stateId?[{frame:i,stateId:row.stateId,offsetMs:Math.round(row.at-rows[0].at)}]:[]),
      frames: rows.length};
  };
}

module.exports = {digest, summary, stateWitness, checkAction, startRecording, delay};
