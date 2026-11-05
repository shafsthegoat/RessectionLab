import test from 'node:test';
import assert from 'node:assert/strict';
import {currentAnatomicalSensitivity,anatomicalIssueText,contextCellPoint} from '../src/anatomical-sensitivity-data.ts';

test('label study requires exact schema, comparison, integer shift and current controls',()=>{
 const study={schema:'resectionlab.two-tool-context-label-sensitivity.v1',comparisonReportSeal:'current',assessment:{controls:{contextVoxelShift:1}}};
 const report={reportSeal:'current',anatomicalSensitivity:{study}};
 assert.equal(currentAnatomicalSensitivity(report,1),study);
 for(const step of [0,.5,NaN,3])assert.equal(currentAnatomicalSensitivity(report,step),null);
 assert.equal(currentAnatomicalSensitivity({...report,reportSeal:'new'},1),null);
 assert.equal(currentAnatomicalSensitivity({reportSeal:'current',anatomicalSensitivity:{study:{...study,schema:'execution-offset'}}},1),null);
 assert.equal(currentAnatomicalSensitivity({reportSeal:'current'},1),null);
});
test('hypothetical manipulation and unknown context remain distinct from proximity',()=>{
 assert.equal(anatomicalIssueText([]),'Declared task met');
 assert.equal(anatomicalIssueText(['NEW_CONTEXT_MANIPULATION','UNKNOWN_CONTEXT_COVERAGE']), 'New CUSA work in hypothetical context; Context coverage unknown');
 assert.match(anatomicalIssueText(['ANATOMICAL_CONTEXT_LIMIT']),/proximity/);
});
test('unknown and manipulated cell markers use destination C-order source cells and RAS affine',()=>{
 const scene={shape:[3,4,5],affine:[[0,-2,0,10],[.5,0,0,-20],[0,0,-1,40],[0,0,0,1]]};
 // Cell (1,2,3), with z fastest, has flat index33.
 assert.deepEqual(contextCellPoint(33,scene),[6,-19.5,37]);
 for(const cell of [undefined,-1,60,NaN,.5])assert.equal(contextCellPoint(cell,scene),null);
 assert.equal(contextCellPoint(0,{...scene,affine:[[NaN,0,0,0]]}),null);
});
