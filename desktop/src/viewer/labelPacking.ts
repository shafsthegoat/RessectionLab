import type {ViewerVolume} from './contracts';
export const MAX_ANATOMY_LAYERS=128;
/** Four byte channels: each holds eight independent masks or 255 categorical
 * regions. Region membership stays exact, including overlaps between groups. */
export function packAnatomyLabels(volume:Pick<ViewerVolume,'shape'|'compartments'|'labelSets'>){
  const count=volume.shape.reduce((a,b)=>a*b,1);
  if(volume.compartments.length>MAX_ANATOMY_LAYERS)throw Error('Too many anatomical layers');
  const codes=Array.from({length:MAX_ANATOMY_LAYERS},()=>[0,0] as [number,number]);
  const data=new Uint8Array(count*4);let channel=0;
  const sets=new Map((volume.labelSets??[]).map(s=>[s.id,s.data]));
  for(const id of new Set(volume.compartments.flatMap(l=>l.labelSet?[l.labelSet]:[]))){
    const source=sets.get(id);if(!source||source.length!==count)throw Error('Missing categorical source grid');
    if(channel>=4)throw Error('The selected label groups exceed the GPU texture budget');
    const palette=new Map<number,number>();
    volume.compartments.forEach((l,i)=>{if(l.labelSet===id){
      if(!Number.isInteger(l.labelValue)||l.labelValue!<=0||l.labelValue!>65535)throw Error('Invalid categorical region');
      if(!palette.has(l.labelValue!))palette.set(l.labelValue!,palette.size+1);
      if(palette.size>255)throw Error('Categorical palette exceeds byte budget');
      codes[i]=[channel,palette.get(l.labelValue!)!];
    }});
    const lut=new Uint8Array(65536);for(const [value,code] of palette)lut[value]=code;
    for(let i=0;i<count;i++)data[i*4+channel]=lut[source[i]];
    channel++;
  }
  let bit=0;
  volume.compartments.forEach((l,index)=>{if(l.labelSet)return;
    if(l.mask.length!==count)throw Error('A source compartment does not match the MRI grid');
    if(bit===8){channel++;bit=0;}
    if(channel>=4)throw Error('The selected independent masks exceed the GPU texture budget');
    const flag=1<<bit++;codes[index]=[channel,-flag];
    for(let i=0;i<count;i++)if(l.mask[i])data[i*4+channel]|=flag;
  });
  return {data,codes};
}
export function materializeLayerMask(volume:ViewerVolume,layer:ViewerVolume['compartments'][number]):Uint8Array{
  if(!layer.labelSet)return layer.mask;
  const source=volume.labelSets?.find(s=>s.id===layer.labelSet)?.data;
  if(!source)throw Error('Missing categorical source');
  return Uint8Array.from(source,v=>v===layer.labelValue?1:0);
}
export const labelMembershipShader=`
uniform int uLayerCount;
uniform vec2 uLayerCodes[128];
uniform float uLayerEnabled[128];
bool hasLabel(ivec4 packed,int layer){
  ivec2 code=ivec2(uLayerCodes[layer]);int value=packed[code.x];
  return code.y<0 ? (value & (-code.y))!=0 : code.y>0 && value==code.y;
}
`;
