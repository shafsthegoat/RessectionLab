import assert from 'node:assert/strict';
import test from 'node:test';
import {readFileSync} from 'node:fs';
import {analyticShift,analyticAmplitude,sampleRatios,TaperInterpolator,shiftAt,brainShiftGrid,brainShiftSummary} from './brainShiftField.ts';

// Engine output on the generated convexity case after the shift settles
// (written by the engine's own brain_shift.view() and the bridge display fields).
const fixture=JSON.parse(readFileSync(new URL('./fixtures/brain-shift-generated.json',import.meta.url),'utf8'));
const {view,probes,affine}=fixture;
const shift=analyticShift(view);
const norm=v=>Math.hypot(...v);

test('the ported analytic amplitude matches the engine at its cells',()=>{
  assert.ok(shift&&shift.peak===5);
  for(const probe of probes)
    assert.ok(Math.abs(analyticAmplitude(shift,probe.positionRasMm)-probe.analyticUnitAmplitude)<1e-9);
});

test('analytic amplitude times the engine taper reproduces the engine displacement exactly',()=>{
  for(const probe of probes){
    const magnitude=shift.peak*analyticAmplitude(shift,probe.positionRasMm)*probe.taperTimesPullBack;
    const expected=shift.direction.map(c=>c*magnitude);
    probe.displacementMm.forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-6,`${v} vs ${expected[i]}`));
  }
});

test('sample ratios recover the taper and the interpolated field stays close to the engine',()=>{
  const ratios=sampleRatios(shift,view.latticeSamples);
  assert.ok(ratios.every(r=>r>=0&&r<=1));
  const taper=new TaperInterpolator(view.latticeSamples,ratios);
  // At the engine's own samples the field reproduces their displacement within a small fraction of the peak.
  for(const sample of view.latticeSamples.slice(0,80)){
    const d=shiftAt(shift,taper,sample.positionRasMm);
    assert.ok(norm(d.map((v,i)=>v-sample.displacementMm[i]))<.35*shift.peak);
  }
  // At independent engine cells away from rigid support (taper ~1) the error is small.
  const interior=probes.filter(p=>p.taperTimesPullBack>.98);
  assert.ok(interior.length>5);
  const errors=interior.map(p=>norm(shiftAt(shift,taper,p.positionRasMm).map((v,i)=>v-p.displacementMm[i])));
  assert.ok(errors.reduce((a,b)=>a+b,0)/errors.length<.12*shift.peak,`mean error ${errors}`);
  // Far outside the sampled domain nothing moves.
  assert.deepEqual(shiftAt(shift,taper,[1e4,1e4,1e4]).map(Math.abs),[0,0,0]);
});

test('the voxel lattice encodes the field in source voxel units',()=>{
  const grid=brainShiftGrid(view,affine);
  assert.ok(grid&&grid.size.every(n=>n>2&&n<=160));
  assert.ok(grid.maxMm<=shift.peak+1e-9&&grid.maxMm>.5*shift.peak);
  assert.equal(grid.data.length,grid.size[0]*grid.size[1]*grid.size[2]*4);
});

test('no field before onset or without display geometry, and a labelled summary',()=>{
  assert.equal(analyticShift({...view,started:false}),null);
  assert.equal(analyticShift({...view,display:undefined}),null);
  assert.equal(brainShiftGrid({...view,currentPeakMm:0},affine),null);
  const summary=brainShiftSummary(view);
  assert.equal(summary.peakMm,5);
  assert.ok(summary.targetMovedMm>0&&summary.targetMovedMm<=5);
});
