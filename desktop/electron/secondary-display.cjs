'use strict';
/** Select before window creation so no first frame appears on the primary. */
function secondaryWindowOptions(displays, primaryId) {
  const display=displays.find(display=>display.id!==primaryId);
  if(!display)throw new Error('--secondary-display requires a connected nonprimary display. No window was opened.');
  const area=display.workArea;
  if(!area||![area.x,area.y,area.width,area.height].every(Number.isFinite)||area.width<=0||area.height<=0)
    throw new Error('The secondary display has no usable work area. No window was opened.');
  const width=Math.min(1460,area.width),height=Math.min(980,area.height);
  return {displayId:display.id,options:{x:Math.round(area.x+(area.width-width)/2),y:Math.round(area.y+(area.height-height)/2),width,height,minWidth:Math.min(1050,width),minHeight:Math.min(720,height)}};
}
function isSecondaryDisplayConnected(displays,primaryId,displayId) {
  return displays.some(display=>display.id===displayId&&display.id!==primaryId);
}
module.exports={secondaryWindowOptions,isSecondaryDisplayConnected};
