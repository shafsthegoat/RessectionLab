// Opt-in real-case acceptance; called by tests/poc_acceptance.py, never the unit suite.
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createRequire} from 'node:module';
import {hydrateWorkspace} from '../src/workspace-session.ts';
import {workspaceDisplay} from '../src/workspace-imaging-data.ts';
const require=createRequire(import.meta.url);
const {AssetRegistry}=require('../electron/assets.cjs');
const {validateWorkspaceResult}=require('../electron/workspace-session.cjs');
const fixture=JSON.parse(await readFile(process.argv[2],'utf8'));
validateWorkspaceResult(fixture.payload);
const assets=new AssetRegistry(fixture.transferRoot);
const payload=await assets.expose(fixture.payload);
assert.equal('path' in payload.mri,false);
const hydrated=await hydrateWorkspace(payload,{readAsset:id=>assets.read(id)});
assert.deepEqual(hydrated.imagingState,payload.workspaceSession.imagingState);
assert.deepEqual(hydrated.volume.compartments.map(x=>x.name).sort(),fixture.expectedLayers.sort());
assert.equal(hydrated.episodeView,null);
assert.equal(hydrated.passiveView,null);
for(const layer of hydrated.volume.compartments) {
  assert.ok(layer.mask.some(x=>x===1),`${layer.name} must be a nonempty saved mask`);
  assert.ok(Number.isFinite(layer.volumeMm3)&&layer.volumeMm3>0);
}
for(const series of hydrated.series) {
  assert.equal(series.descriptor.registration.status,'source-provided');
  assert.equal(series.descriptor.registration.overlayPermitted,true);
  assert.equal(series.descriptor.planningEligible,false);
  assert.deepEqual(series.volume.affine,hydrated.volume.affine);
  assert.equal(workspaceDisplay(hydrated.volume,series).primaryOverlaysPermitted,false);
}
process.stdout.write(JSON.stringify({caseId:payload.caseId,layers:hydrated.volume.compartments.map(x=>x.name),
  series:hydrated.series.length,settingsExact:true,hostAssetBoundary:true,frontendHydration:true}));
