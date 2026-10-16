import {test} from 'node:test';
import assert from 'node:assert/strict';
import {scanFitPoints} from './scanCameraFit.ts';
test('camera support excludes padded air while preserving original scalar data and physical coordinates',()=>{
 const mri=new Float32Array(1000);mri[555]=100;
 const volume={mri,shape:[10,10,10],frame:'RAS+',affine:[[.5,0,0,2],[0,.5,0,-4],[0,0,.5,7],[0,0,0,1]]};
 const original=mri.slice();assert.deepEqual([...scanFitPoints(volume,[0,100])],[4.5,-1.5,9.5]);assert.deepEqual(mri,original);
});
