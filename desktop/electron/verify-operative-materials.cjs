'use strict';
const assert = require('node:assert/strict');

// The saved lateral BITE-09 study view has visible cortex beneath faint
// adjacent estimates. At this site, a failed peel loses almost all red signal.
function compareCorticalPatch(isolated, combined) {
  const region = {x:780, y:316, width:155, height:108};
  const before=isolated.crop(region).toBitmap();
  const after=combined.crop(region).toBitmap();
  assert.equal(before.length,region.width*region.height*4);
  assert.equal(after.length,before.length);
  let corticalPixels=0, darkenedPixels=0;
  // Electron's native bitmap is BGRA on this little-endian platform.
  for(let i=2;i<before.length;i+=4){
    if(before[i]<=115)continue;
    corticalPixels++;
    if(after[i]<57)darkenedPixels++;
  }
  assert(corticalPixels>5000,'Expected the saved BITE-09 cortical crown at this camera and size');
  assert(darkenedPixels<corticalPixels*.005,'Faint adjacent labels erased the visible cortex');
  return {region,corticalPixels,darkenedPixels};
}

async function verifyOperativeMaterials({js,ready,presentation,capture,report,surfaceFrame}) {
  assert.equal(report.patient,'MNI-BITE-09');
  const visible=await js(`Array.from(document.querySelectorAll('.native-structure-row'),row=>({
    label:row.dataset.structureLabel,id:row.dataset.structureId,
    visible:row.querySelector('.native-structure-visibility').getAttribute('aria-pressed')==='true'
  })).filter(row=>row.visible)`);
  const cortex=visible.filter(row=>/^(Gray matter|Cerebellar gray matter)/i.test(row.label));
  const csf=visible.filter(row=>/Cerebrospinal/i.test(row.label));
  const vessels=visible.filter(row=>/vascular tissue/i.test(row.label));
  const head=visible.filter(row=>/other head tissue/i.test(row.label));
  assert(cortex.length&&csf.length&&vessels.length&&head.length);
  report.materialViewport=await js("document.querySelector('.rl-viewer-anatomy-image').getBoundingClientRect().toJSON()");
  report.materialIsolation={};
  let current=new Set(visible.map(row=>row.id));
  const isolate=async (name,rows)=>{
    const next=new Set(rows.map(row=>row.id));
    for(const row of visible){
      if(current.has(row.id)===next.has(row.id))continue;
      const selector=`.native-structure-row[data-structure-id="${row.id}"] .native-structure-visibility`;
      await js(`document.querySelector(${JSON.stringify(selector)}).click()`);
      await js("new Promise(resolve=>requestAnimationFrame(resolve))");
    }
    await ready();
    const frame=await presentation();
    assert.deepEqual(frame.layers,[...next].sort());
    assert.deepEqual(frame.frame.cameraState,surfaceFrame.cameraState);
    assert.equal(frame.frame.sourceSignalFingerprint,surfaceFrame.sourceSignalFingerprint);
    report.materialIsolation[name]=frame;
    current=next;
    return capture(name);
  };
  const isolated=await isolate('isolated-cortex',cortex);
  const fluid=await isolate('cortex-with-csf',[...cortex,...csf]);
  await isolate('isolated-vascular',vessels);
  await isolate('isolated-head',head);
  const combined=await isolate('restored-layers',visible);
  report.corticalContact={
    reference:'isolated-cortex',
    cortexWithCSF:compareCorticalPatch(isolated,fluid),
    combined:compareCorticalPatch(isolated,combined)
  };
}
module.exports={verifyOperativeMaterials};
