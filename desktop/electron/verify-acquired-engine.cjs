'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const {execFileSync} = require('node:child_process');
const {createHash} = require('node:crypto');
const {inputScript} = require('./verify-eden-planning-helpers.cjs');

// Diagnostic comparison only. During a draw on the bound microscope, suppress
// its registered-signal uniform, then restore it immediately after that draw.
function observeBoundSignal() {
  const proto=WebGL2RenderingContext.prototype, original=proto.drawElements, locations=new WeakMap();
  const control=window.boundSignalReview={suppress:false,enabledDraws:0,suppressedDraws:0};
  proto.drawElements=function(...args){
    const viewer=this.canvas.closest('.rl-viewer');
    if(!viewer?.querySelector('[data-accepted-tools][data-operative-view="surgeon"]'))return original.apply(this,args);
    const program=this.getParameter(this.CURRENT_PROGRAM);
    if(!locations.has(program))locations.set(program,this.getUniformLocation(program,'uCorticalSignalEnabled'));
    const location=locations.get(program);
    if(!location || !this.getUniform(program,location))return original.apply(this,args);
    control.enabledDraws++;
    if(!control.suppress)return original.apply(this,args);
    this.uniform1i(location,0);
    try{control.suppressedDraws++;return original.apply(this,args);}finally{this.uniform1i(location,1);}
  };
}

// Exercise one native study through its saved plan, illustrative exposure and
// admitted procedural recording. Each captured surgical frame is checked
// against the archive boundary, the IPC response and the drawn tool pose.
async function captureBoundStudy({episode, report, document, win, js, until, click, capture, field}) {
  const digest = () => createHash('sha256').update(fs.readFileSync(episode)).digest('hex');
  const originalDigest = digest();
  const artifact = JSON.parse(execFileSync('unzip', ['-p', episode, 'manifest.json'],
    {maxBuffer:256 * 1024 * 1024})).artifacts.surgical_toolbox;
  const exact = artifact.exactGridReplay, history = artifact.timeline[artifact.branchId];
  assert.equal(document.snapshot_id, exact.sourceManifest.snapshot_id);
  assert.equal(field.target, exact.targetLayerId);
  assert.equal(field.approach, exact.recordedApproach.id);
  const ablate=process.argv.includes('--compare-bound-signal');
  if(ablate)await js('('+observeBoundSignal+')()');
  const first = exact.observations.findIndex(o => o.residualTargetMm3 < exact.observations[0].residualTargetMm3);
  assert(first > 0);
  report.geometricRenderer=await js(`(()=>{const g=document.querySelector('.operative-field-canvas canvas').getContext('webgl2');
    const ext=g.getExtension('WEBGL_debug_renderer_info');return ext?g.getParameter(ext.UNMASKED_RENDERER_WEBGL):g.getParameter(g.RENDERER)})()`);
  const timeline = 'document.querySelector(\'[aria-label="Operative timeline"]\')';
  const brain = await js(`[...document.querySelectorAll('.operative-chapters button:not(:disabled)')].findIndex(b=>b.getAttribute('aria-label')==='Brain retraction')`);
  assert(brain >= 0);
  const fieldPosition = brain + .5;
  await js(inputScript('Operative timeline', fieldPosition));
  await until('Number('+timeline+'.value)==='+fieldPosition);
  await capture('geometric-brain-1440');
  await click('Open recorded engine episode');
  await until("!!document.querySelector('[data-engine-replay-layout] [data-accepted-tools]')", 240000);
  assert.equal(report.binding.sourceSnapshotId, document.snapshot_id);
  assert.equal(report.binding.targetLayerId, field.target);
  assert.equal(report.binding.recordedApproach.id, field.approach);
  assert.equal(report.binding.endpointStateId, history.at(-1).stateAfter);
  report.episode = {path:episode, sha256:originalDigest, firstEffectPosition:first, stages:{}};
  const pane = "document.querySelector('[data-engine-replay-layout] [data-accepted-tools]')";
  await until(pane+'.dataset.surgicalStateId==='+JSON.stringify(history.at(-1).stateAfter)+
    ` && document.querySelector('[aria-label="Surgical timeline"]')?.disabled===false`);
  const {imagingState}=JSON.parse(execFileSync('unzip',['-p',episode,'workspace.json']));
  const storedMode=imagingState?.states?.[imagingState.selectedSeriesId??'primary']?.viewer?.resultInspection;
  if(storedMode)await until("document.querySelector('.rl-viewer[data-result-inspection]')?.dataset.resultInspection==="+JSON.stringify(storedMode));
  const restored=await js("!!document.querySelector('.engine-result-orbit')");
  report.episode.initialInspection={position:history.length,stateId:history.at(-1).stateAfter,
    viewport:win.getContentSize(),storedMode:storedMode??null,dismissed:restored};
  if(restored){
    await click('Return to operative field');
    await until("!document.querySelector('.engine-result-orbit,.rl-viewer[data-result-inspection]')");
  }
  if (await js(pane+".dataset.operativeView") !== 'surgeon') await click('Surgeon / Microscope');
  for (const [width,height] of [[1440,900], [1050,720]]) {
    win.setContentSize(width,height);
    await until('innerWidth==='+width+' && innerHeight==='+height);
    for (const [stage, position] of [['contact',first], ['withdrawn',history.length]]) {
      const id = exact.observations[position].stateId;
      await until(`document.querySelector('[aria-label="Surgical timeline"]')?.disabled===false`);
      await js(inputScript('Surgical timeline', position));
      await until(pane+'.dataset.surgicalStateId==='+JSON.stringify(id)+
        ' && '+pane+".closest('.rl-viewer').dataset.surfacesPending==='0'");
      await until(pane+".dataset.operativeView==='surgeon' && !!"+pane+'.dataset.opticalCamera');
      assert.equal(await js("!!document.querySelector('.engine-result-orbit,.rl-viewer[data-result-inspection]')"), false,
        'A cavity inspection obscures the operative microscope');
      const name = stage+'-'+width;
      await capture(name);
      const drawn = await js(`(()=>{const p=${pane}, v=p.closest('.rl-viewer'), ref=document.querySelector('.engine-intact-reference');
        return {state:p.dataset.surgicalStateId, removed:Number(p.dataset.removedCells), tools:JSON.parse(p.dataset.acceptedTools),
          resultOverlay:!!document.querySelector('.engine-result-orbit,.rl-viewer[data-result-inspection]'),
          camera:JSON.parse(p.dataset.opticalCamera), viewport:p.getBoundingClientRect().toJSON(),
          sourceSnapshot:ref?.dataset.studySnapshot, nativeSeries:ref?.querySelector('.rl-viewer')?.dataset.sourceDisplayIdentity,
          operatedSource:v.dataset.sourceDisplayIdentity, appearance:p.dataset.operativeAppearance,
          practice:document.querySelector('.practice-entry')?{disabled:document.querySelector('.practice-entry button').disabled,
            text:document.querySelector('.practice-entry').innerText}:null,
          hud:p.querySelector('[data-operative-readout-state]')?.textContent};
      })()`);
      const response = report.boundaryResponses[id];
      assert(response);
      assert.equal(drawn.resultOverlay,false,'Inspection returned after settling the capture');
      assert.equal(drawn.sourceSnapshot, document.snapshot_id);
      assert.equal(drawn.removed, response.removed);
      assert.deepEqual(drawn.tools, response.tools);
      assert.equal(response.caseHash, report.binding.caseHash);
      assert.equal(response.position, position);
      if(process.argv.includes('--inspect-acquired-geometry')){
        drawn.diagnostic=await require('./verify-acquired-geometry.cjs').captureGeometry({win,js,until,capture,name,width,height});
        assert.equal(await js(pane+'.dataset.surgicalStateId'),drawn.state);
        assert.deepEqual(await js('JSON.parse('+pane+'.dataset.opticalCamera)'),drawn.camera);
      }
      report.episode.stages[name]=drawn;
      if(ablate && width===1440 && stage==='contact'){
        await js('window.boundSignalReview.suppress=true');
        win.setContentSize(width-1,height);await until('innerWidth==='+String(width-1));
        win.setContentSize(width,height);await until('innerWidth==='+width);
        await js('new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done)))');
        await capture('contact-1440-signal-suppressed');
        report.episode.signalControl=await js('({...window.boundSignalReview})');
        assert.equal(await js(pane+'.dataset.surgicalStateId'),drawn.state);
        assert.deepEqual(await js('JSON.parse('+pane+'.dataset.opticalCamera)'),drawn.camera);
        await js('window.boundSignalReview.suppress=false');
      }
    }
  }
  assert.equal(digest(), originalDigest);
}

module.exports = {captureBoundStudy};
