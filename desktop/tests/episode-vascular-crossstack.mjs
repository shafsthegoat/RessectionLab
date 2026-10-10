// Generated scalar report from actual backend execution/evaluation; no reference arrays.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createRequire} from 'node:module';
import {checkedVascularEvaluation} from '../src/episode-vascular-data.ts';
const require=createRequire(import.meta.url),{validateVascularResult}=require('../electron/episode-vascular.cjs');
const {episode,result}=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const request={caseHash:episode.caseHash,episodeId:episode.episodeId};
validateVascularResult(result,request);
const checked=await checkedVascularEvaluation(result,episode);
assert.deepEqual(checked.actionIds,episode.history.map(row=>row.action_id));
assert.equal(checked.perAction.length,episode.history.length);
for(const row of checked.perAction){assert.equal(row.sweepCount,row.interactionMode==='stop'?0:1);if(row.sweepCount)assert.equal(row.wholeTool.clinical_injury_probability,null)}
assert.equal(checked.patientAdmission,false);assert.equal(checked.removedOverlap.outcomes,null);
process.stdout.write(JSON.stringify({status:'actual_backend_scalar_evaluation_matches_same_episode',selector:episode.selector,
 episodeId:episode.episodeId,evaluationId:checked.evaluationId,actions:checked.perAction.length,wholeTool:checked.wholeTool})+'\n');
