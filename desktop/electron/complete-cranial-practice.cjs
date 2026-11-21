'use strict';

// One declared generated task. All commands go through the visible instrument
// controls; the read-only native view supplies the next decision and its checks.
const assert = require('node:assert/strict');
const {checkAction, delay, digest} = require('./manual-practice-evidence.cjs');

const parentPosition = 99;
const parentStateId = 'sha256:9a9ea1706db7f7911005a927c426239c05fa9327204055d645754093f2ab1d5c';
const endpoint = {originalTargetMm3:27, residualTargetMm3:0, kerfDebrisMm3:236, residualDebrisMm3:0,
  membraneExtractionMm3:49, preserveRetainedRim:true, noAdditionalNonTargetWork:true, withdrawBoth:true};

// Surface stations of the checked square kerf, in millimetres RAS. Rim stations
// stand 1.5 mm above retained material. Their destinations are proposed anew;
// the measured field controls whether a second stationary application is needed.
const stations = [
  [26,32,60],[27,28,60],[30,26,60],[35,26,60],[38,28,60],[38,33,60],[37,37,60],
  [33,38,60],[28,38,60],[26,35,60],[26,26,60],[25,38,58.5],[38,25,58.5],
  [38,30,60],[33,28,60],[31,36,60],[38,39,58.5],[35,39,58.5],[31,39,58.5],
  [39,36,58.5],[33,25,58.5],[25,29,58.5],[28,32,60],[28,25,58.5],[36,33,60],
  [39,26,58.5],[26,39,58.5],[25,35,58.5],[25,32,58.5],[25,26,58.5],[33,36,60],
  [39,38,58.5],[39,33,58.5],[39,30,58.5],[37,30,58.5],[36,26,60],[32,25,58.5],
  [26,25,58.5],[29,38,60],
];

function fieldAtTip(view, position) {
  const {affine,shape} = view.scene;
  let total = 0;
  for (const [id, mass] of Object.entries(view.debris)) {
    const cell = Number(id);
    const index = [Math.floor(cell/(shape[1]*shape[2])), Math.floor(cell/shape[2])%shape[1],cell%shape[2]];
    const point = affine.slice(0,3).map(row => row[3] + index.reduce((sum,n,i)=>sum + n*row[i],0));
    // Native voxel footprint includes its half-diagonal around the stock tip.
    if (Math.hypot(...point.map((n,i)=>n-position[i])) <= 1.5 + Math.sqrt(3)/2 + 1e-8) total += mass;
  }
  return total;
}

async function inspectApproach({click, js, capture, recorded}) {
  await click('Case stage');
  const caseText = await js("document.querySelector('[aria-label=\"Case anatomy and evidence\"]').innerText");
  assert.match(caseText,/target/i);
  await js("document.querySelector('[aria-label=\"Case anatomy and evidence\"]').scrollIntoView({block:'center'})");
  await capture('01-source-target');
  if (recorded) await delay(1500);
  await click('Review constraints and plan');
  const plan = await js("document.querySelector('[data-workflow-stage=plan]').innerText");
  await capture('02-planned-opening');
  if (recorded) await delay(1600);
  await click('Compare stage');
  const legend='Recorded paths and generated anatomy legend';
  if(await js(`[...document.querySelectorAll('summary')].some(s=>s.textContent.trim()===${JSON.stringify(legend)} && s.getClientRects().length)`)) await click(legend);
  const comparison = await js("document.querySelector('[data-workflow-stage=compare]').innerText.slice(0,4000)");
  await capture('02-checked-approach');
  if (recorded) await delay(1600);
  await click('Rehearse/edit stage');
  return {caseText,plan,comparison,normalEntryRasMm:[32,32,55],targetWorkAxisRas:[0,0,1],targetDepthRasMm:68,
    source:'Previously checked superior approach in the frozen square-shell control. Scalp exposure, bone opening and membrane perimeter cut predate the manual task.'};
}

// Observe every native step during one genuine, bounded pointer hold. Keeping
// the pointer down triggers normal practice scheduling; releasing it stops that
// schedule. The driver never issues a native action or starts a playback clock.
async function holdSteps({label,count,click,byName,point,send,wait,read,observe,snapshot,report}) {
  await click('Hold controls');
  const start = read().witness;
  const p = await point(byName(label));
  const coords = {x:p.x,y:p.y};
  const rows = [];
  let before = start;
  const firstInput = Date.now();
  observe((action,outcome,witness,received,ipcMs) => {
    const result = {requested:action,status:outcome.status,reason:outcome.reason,reasonCode:outcome.reasonCode,
      completedFraction:outcome.completedFraction,eventKinds:[...new Set(outcome.events.map(e=>e.kind))],witness,
      backendResponseEpochMs:received,ipcMs};
    checkAction(before,witness,result,action);
    const row = {label,before:before.stateId,inputEpochMs:firstInput,result,heldInput:true};
    rows.push(row);report.actions.push(row);before=witness;
  });
  await send('dispatchMouseEvent',{type:'mouseMoved',...coords});
  await send('dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,...coords});
  try {
    await wait(()=>rows.length>=count || rows.some(row=>row.result.status!=='accepted'),label,10000);
  } finally {
    await send('dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,clickCount:1,...coords});
  }
  await delay(350);
  observe(null);
  assert(rows.length>=count && rows.every(row=>row.result.status==='accepted'));
  rows.at(-1).display=await snapshot(before);
  report.continuousGesture={label,inputEpochMs:firstInput,steps:rows.length,startStateId:start.stateId,endStateId:before.stateId,
    acceptedPoses:rows.map(row=>row.result.witness.tools.left.position),stoppedAfterRelease:true};
  (report.continuousGestures??=[]).push(report.continuousGesture);
  return before;
}

async function jogToStation(ui, target) {
  const pad=ui.byName('Instrument jog pad');
  const readDraft=()=>ui.js("JSON.parse(document.querySelector('[data-practice-proposed-position]').dataset.practiceProposedPosition)");
  const equal=(left,right)=>left.every((n,i)=>Math.abs(n-right[i])<1e-7);
  let draft=await readDraft();
  let gestures=0;
  while(gestures<12 && Math.hypot(target[0]-draft[0],target[1]-draft[1])>1e-7){
    // Opening the proposed-tip readout can move the pad in the scroll panel.
    // Each gesture starts at its actual rectangle and at the displayed draft.
    const p=await ui.point(pad);
    draft=await readDraft();
    const dx=Math.max(-2,Math.min(2,target[0]-draft[0]));
    const dy=Math.max(-2,Math.min(2,target[1]-draft[1]));
    const wanted=[draft[0]+dx,draft[1]+dy,draft[2]];
    const endpoint={x:p.x+20*dx,y:p.y-20*dy};
    await ui.send('dispatchMouseEvent',{type:'mouseMoved',x:p.x,y:p.y});
    await ui.send('dispatchMouseEvent',{type:'mousePressed',button:'left',buttons:1,clickCount:1,x:p.x,y:p.y});
    try {
      for(let j=1;j<=4;j++){
        await ui.send('dispatchMouseEvent',{type:'mouseMoved',buttons:1,x:p.x+5*j*dx,y:p.y-5*j*dy});
        if(ui.recorded) await delay(18);
      }
      // Chromium may coalesce a last pointermove while the proposal is being
      // drawn. Leave the pointer down and observe the actual accepted UI input.
      for(let n=0;n<5 && !equal(await readDraft(),wanted);n++){
        await delay(20);
        await ui.send('dispatchMouseEvent',{type:'mouseMoved',buttons:1,...endpoint});
      }
    } finally {
      await ui.send('dispatchMouseEvent',{type:'mouseReleased',button:'left',buttons:0,clickCount:1,...endpoint});
    }
    const next=await readDraft();
    gestures++;
    if(equal(draft,next)) break;
    draft=next;
  }
  // Use the same visible pose settings to refine any sub-step rounding or a
  // clipped gesture in the compact pad. Verify the draft before any commit.
  draft=await readDraft();
  const refined=[];
  for(let i=0;i<2;i++) if(Math.abs(draft[i]-target[i])>1e-7) {
    const axis=['R','A'][i];
    await ui.number('Tool position '+axis,target[i]);
    refined.push(axis);
  }
  await ui.point(pad);
  await ui.js(`(${pad}).focus({preventScroll:true})`);
  const depth=target[2]-(await readDraft())[2];
  assert(Number.isInteger(depth*2));
  for(let j=0;j<Math.abs(depth)*2;j++) {
    if(depth>0) await ui.key('PageDown','PageDown',34);
    else await ui.key('PageUp','PageUp',33);
  }
  if(Math.abs((await readDraft())[2]-target[2])>1e-7) {
    await ui.number('Tool position S',target[2]);
    refined.push('S');
  }
  assert(equal(await readDraft(),target),'Jog pad proposal must coincide with the source-bound station');
  (ui.report.jogStations??=[]).push({target,gestures,refined});
}

async function run(ui) {
  const {number,press,choose,capture,click,report,read,recorded} = ui;
  const state = () => read().witness;
  const view = () => read().nativeView;
  const mark = async name => {
    await capture(name);
    if(recorded) await delay(800);
  };
  const pose = async xyz => {
    for (const [i,axis] of ['R','A','S'].entries()) await number('Tool position '+axis,xyz[i]);
  };
  const duration = seconds=>number('Action duration',seconds);
  const actuate = async (button,xyz,seconds,allowed='accepted') => {
    if(xyz) await pose(xyz);
    await duration(seconds);
    return press(button,allowed);
  };
  assert.equal(state().stateId,parentStateId);
  const opening = view().cranialStages;
  const previouslyRemoved = new Set(view().removed);
  assert.equal(opening.boundaryCutComplete,true);
  assert.equal(opening.patchExtracted,false);
  assert.equal(state().remaining,27);
  assert.equal(state().metrics.debrisMm3,236);
  const baseline = state().metrics;
  const inspectTask = async (expected, name) => {
    const card='details[aria-label="Practice task"]';
    await ui.wait(()=>ui.js(`document.querySelector('${card}')?.dataset.taskState===${JSON.stringify(state().stateId)}`),expected);
    await click('Practice task');
    const text=await ui.js(`document.querySelector('${card}').innerText`);
    assert(text.includes(expected));
    await mark(name);
    await click('Practice task');
    return text;
  };
  report.task={endpoint,startStages:opening,originalToolStates:view().slots};
  report.task.atEntry=await inspectTask('Grasp and extract the cut membrane patch','02-task-at-entry');
  await duration(.1);
  await choose('membrane_forceps');
  // Select the existing microscope range using native keyboard input. The
  // optical camera remains fixed through contact, treatment and cleanup.
  for (const [label,value] of [['Field / magnification',28],['Viewing angle',24]]) {
    const slider=ui.byName(label);
    await ui.point(slider);
    await ui.js(`(${slider}).focus({preventScroll:true})`);
    for(let i=0;i<80;i++) {
      const current=Number(await ui.js(`(${slider}).value`));
      if(current===value) break;
      if(current>value) await ui.key('ArrowLeft','ArrowLeft',37);
      else await ui.key('ArrowRight','ArrowRight',39);
    }
    assert.equal(Number(await ui.js(`(${slider}).value`)),value);
  }
  report.task.fieldWidthMm=28;
  report.task.angleDeg=24;
  await number('Jaw opening · mm',0);
  await number('Tool roll',0);
  await actuate('Place tool',[32,32,55],.1);
  await actuate('Grasp tissue',[32,32,63.5],.1);
  assert.equal(view().holds.left.cells.length,49);
  const heldCells = new Set(view().holds.left.cells);
  await mark('03-grasped-membrane');
  const held = await holdSteps({...ui,label:'Hold to retract',count:18});
  assert(held.tools.left.position[2]<=54.5 && view().holds.left.cells.length===49);
  await mark('04-continuous-membrane-lift');
  await click('Jog and commit');
  assert.equal(view().holds.left.cells.length,49);
  await mark('05-exterior-membrane');
  await press('Extract tissue');
  assert.equal(view().cranialStages.patchExtracted,true);
  await press('Withdraw tool');
  const prepared = state().metrics;
  assert.equal(prepared.nonTargetRemovedMm3-baseline.nonTargetRemovedMm3,49);
  assert.deepEqual(new Set(view().removed.filter(id=>!previouslyRemoved.has(id))),heldCells);

  await choose('cusa');
  await actuate('Place tool',[32,32,55],.1);
  await press('Activate tool');
  for(const depth of [66.5,68]) {
    for(let attempt=0;attempt<14 && state().remaining>0;attempt++){
      const at = state().tools.left.position[2];
      if(Math.abs(at-depth)<1e-5) break;
      await actuate('Apply tool',[32,32,depth],depth===66.5?30:.1,['accepted','partial']);
      if(report.actions.at(-1).result.status==='partial'){
        assert.equal(report.actions.at(-1).result.reasonCode??report.actions.at(-1).result.reason,'TISSUE_ASPIRATION_IN_PROGRESS');
        if(!report.task.firstPartial){
          report.task.firstPartial=state().stateId;
          await mark('06-finite-contact-stop');
          await holdSteps({...ui,label:'Hold to apply',count:6});
          await click('Jog and commit');
        }
        // The UI snaps the draft to the computed partial endpoint. Apply finite
        // capacity at that accepted point before attempting more axial travel.
        await actuate('Apply tool',null,3);
      }
    }
  }
  assert.equal(state().remaining,0,'Original target still has residual volume');
  assert.equal(state().metrics.targetRemovedMm3,27);
  assert.equal(state().metrics.nonTargetRemovedMm3,prepared.nonTargetRemovedMm3);
  await mark('07-treated-cavity');
  await press('Release tool');await press('Withdraw tool');

  await choose('frazier_suction');
  await actuate('Place tool',[26,32,55],.2);
  await number('Tip radius · mm',1.5);
  await number('Flow · mm³/s',20);
  for(const [i,xyz] of stations.entries()){
    if(state().metrics.debrisMm3<1e-8) break;
    await jogToStation(ui,xyz);
    await actuate('Apply tool',null,.2);
    const mass=fieldAtTip(view(),xyz);
    if(mass>1e-8) await actuate('Apply tool',null,Math.min(3,Math.ceil(mass/20*10)/10+.1));
    if(i===0||i===8) await mark('08-field-uptake-'+(i+1));
  }
  assert(state().metrics.debrisMm3<1e-8,'Kerf still contains suspended debris');
  assert(Math.abs(state().metrics.debrisRemovedMm3-236)<1e-8);
  await press('Release tool');
  const [lastR,lastA]=state().tools.left.position;
  await actuate('Move tool',[lastR,lastA,55],.4);
  await press('Withdraw tool');
  const final=view();
  assert(Object.values(final.slots).every(slot=>!slot.inserted&&!slot.active&&!slot.grip.length));
  assert.equal(final.cranialStages.membrane.retainedRimIntact,true);
  assert.equal(final.cranialStages.wholeExitComplete,true);
  assert.equal(state().metrics.nonTargetRemovedMm3,prepared.nonTargetRemovedMm3);
  const sourceChanges = final.removed.filter(id=>!previouslyRemoved.has(id));
  const aspiratedCells = new Set(report.actions.flatMap(a=>a.result.cusaRemovedCells??[]));
  assert.equal(aspiratedCells.size,27);
  assert.deepEqual(new Set(sourceChanges),new Set([...heldCells,...aspiratedCells]));
  report.task.sourceRemoval={priorCells:previouslyRemoved.size,heldMembraneCells:[...heldCells].sort((a,b)=>a-b),
    aspiratedTargetCells:[...aspiratedCells].sort((a,b)=>a-b),finalRemovedSourceDigest:digest([...final.removed].sort((a,b)=>a-b))};
  report.task.finalStages=final.cranialStages;
  report.task.finalProtection=final.protection;
  report.task.finalToolStates=final.slots;
  await mark('09-clean-field-withdrawn');
  report.task.atGoal=await inspectTask('Modeled endpoint reached','09-verified-task-endpoint');
  await click('Inspection cutaway');
  await delay(350);
  await mark('10-final-inspection-cutaway');
  await click('Surgeon / Microscope');
  await delay(350);
  await mark('11-final-optical-field');
}

module.exports={parentPosition,parentStateId,endpoint,inspectApproach,run,holdSteps};
