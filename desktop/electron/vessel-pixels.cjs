'use strict';
// Pixel classes for hidden vessel-display captures (BGRA bitmaps from capturePage).
// Vessel walls: artery red or pink source colours (red well above green and blue),
// or vein blue (well above red and green; the cyan tumour is not). The review
// overlay is amber. Skull, brain and tumour colours match neither.
function pixelClass(r,g,b){
  if(r>60&&r>g*1.8&&r>b*1.6||b>60&&b>r*1.25&&b>g*1.25)return 'vessel';
  // Amber #ffad54 keeps green near 0.68 and blue near 0.33 of red when lit or shaded;
  // a specular highlight on a red wall (about 255,150,120) is not amber.
  if(r>120&&g>r*.6&&g<r*.8&&b<r*.42)return 'amber';
  return null;
}
function countVesselPixels(bgra){
  let vessel=0,amber=0;
  for(let i=0;i<bgra.length;i+=4){const kind=pixelClass(bgra[i+2],bgra[i+1],bgra[i]);if(kind==='vessel')vessel++;else if(kind==='amber')amber++;}
  return {vessel,amber};
}
module.exports={pixelClass,countVesselPixels};
