import test from 'node:test';
import assert from 'node:assert/strict';
import {currentSensitivity,failedCommandPoint,executionIssueText,executionFailureText} from '../src/strategy-sensitivity-data.ts';

test('only the exact comparison and declared offset display a saved sensitivity',()=>{
 const study={comparisonReportSeal:'current',assessment:{controls:{offsetMm:1}}};
 const report={reportSeal:'current',executionSensitivity:{study}};
 assert.equal(currentSensitivity(report,1),study);
 assert.equal(currentSensitivity(report,0),null);
 assert.equal(currentSensitivity({...report,reportSeal:'new'},1),null);
 assert.equal(currentSensitivity(report,NaN),null);
 assert.equal(currentSensitivity({reportSeal:'current'},1),null);
});
test('failed-command marker detaches the intended RAS pose and ignores missing or nonfinite coordinates',()=>{
 const outcome={failure:{actionIndex:1},actionProgram:[{}, {pose:{position:[-83,118,66]}}]};
 const point=failedCommandPoint(outcome);assert.deepEqual(point,[-83,118,66]);point[0]=99;
 assert.equal(outcome.actionProgram[1].pose.position[0],-83);
 assert.equal(failedCommandPoint({...outcome,failure:null}),null);
 assert.equal(failedCommandPoint({...outcome,failure:{actionIndex:0}}),null);
 assert.equal(failedCommandPoint({...outcome,actionProgram:[{}, {pose:{position:[NaN,0,0]}}]}),null);
});
test('visible failure text explains the task and geometry without raw diagnostic codes',()=>{
 assert.equal(executionIssueText({meetsTask:true,violations:[]}), 'Task met');
 assert.equal(executionIssueText({meetsTask:false,violations:['INCOMPLETE_EXECUTION','PROGRESS_THRESHOLD']}),'Sequence did not complete; Progress below the declared threshold');
 assert.equal(executionFailureText('STRATEGY_SENSITIVITY: execution refused: RETAINED_TISSUE_COLLISION'),'The commanded tool intersects retained tissue.');
 assert.equal(executionFailureText('STRATEGY_SENSITIVITY: new non-target removal exceeds task'),'new non-target removal exceeds task');
});
