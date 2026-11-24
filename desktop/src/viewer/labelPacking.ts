import type {ViewerVolume} from './contracts';
export const MAX_ANATOMY_LAYERS=128;
/** Up to four byte channels, each holding eight independent masks or 255
 * categorical regions. Membership stays exact, including overlapping masks. */
export function packAnatomyLabels(volume:Pick<ViewerVolume,'shape'|'compartments'|'labelSets'>,compact=false){
  const count=volume.shape.reduce((a,b)=>a*b,1);
  if(volume.compartments.length>MAX_ANATOMY_LAYERS)throw Error('Too many anatomical layers');
  const codes=Array.from({length:MAX_ANATOMY_LAYERS},()=>[0,0] as [number,number]);
  const groupIds=new Set(volume.compartments.flatMap(l=>l.labelSet?[l.labelSet]:[]));
  const independent=volume.compartments.flatMap((l,index)=>l.labelSet?[]:[index]);
  const required=groupIds.size+Math.ceil(independent.length/8);
  // WebGL2 uses R8, RG8 or RGBA8. Modeled sparse updates retain RGBA8.
  const channels=compact&&required<=2?Math.max(1,required):4;
  const data=new Uint8Array(count*channels);let channel=0;
  const sets=new Map((volume.labelSets??[]).map(s=>[s.id,s.data]));
  for(const id of groupIds){
    const source=sets.get(id);if(!source||source.length!==count)throw Error('Missing categorical source grid');
    if(channel>=channels)throw Error('The selected label groups exceed the GPU texture budget');
    const palette=new Map<number,number>();
    volume.compartments.forEach((l,i)=>{if(l.labelSet===id){
      if(!Number.isInteger(l.labelValue)||l.labelValue!<=0||l.labelValue!>65535)throw Error('Invalid categorical region');
      if(!palette.has(l.labelValue!))palette.set(l.labelValue!,palette.size+1);
      if(palette.size>255)throw Error('Categorical palette exceeds byte budget');
      codes[i]=[channel,palette.get(l.labelValue!)!];
    }});
    const lut=new Uint8Array(65536);for(const [value,code] of palette)lut[value]=code;
    for(let i=0;i<count;i++)data[i*channels+channel]=lut[source[i]];
    channel++;
  }
  for(const index of independent)if(volume.compartments[index].mask.length!==count)throw Error('A source compartment does not match the MRI grid');
  const categorical=new Set<number>();
  if(independent.length>(channels-channel)*8){
    // Losslessly compact mutually disjoint masks when a full-head inventory
    // exceeds32 independent bits. Overlapping masks remain separate. This is
    // storage encoding only: original masks, membership and visibility survive.
    const groups:Array<{indices:number[];values:Uint8Array}>=[];
    for(const index of independent){
      const mask=volume.compartments[index].mask;
      let selected:typeof groups[number]|undefined;
      for(const group of groups){
        let overlap=false;for(let i=0;i<count;i++)if(mask[i]&&group.values[i]){overlap=true;break;}
        if(!overlap&&group.indices.length<255){selected=group;break;}
      }
      if(!selected){selected={indices:[],values:new Uint8Array(count)};groups.push(selected);}
      selected.indices.push(index);const code=selected.indices.length;
      for(let i=0;i<count;i++)if(mask[i])selected.values[i]=code;
    }
    groups.sort((a,b)=>b.indices.length-a.indices.length);
    let remaining=independent.length;
    for(const group of groups){
      if(Math.ceil(remaining/8)<=channels-channel)break;
      if(group.indices.length<=8||channel>=channels)throw Error('The selected independent masks exceed the GPU texture budget');
      group.indices.forEach((index,i)=>{codes[index]=[channel,i+1];categorical.add(index);});
      for(let i=0;i<count;i++)data[i*channels+channel]=group.values[i];
      remaining-=group.indices.length;channel++;
    }
  }
  let bit=0;
  volume.compartments.forEach((l,index)=>{if(l.labelSet||categorical.has(index))return;
    if(l.mask.length!==count)throw Error('A source compartment does not match the MRI grid');
    if(bit===8){channel++;bit=0;}
    if(channel>=channels)throw Error('The selected independent masks exceed the GPU texture budget');
    const flag=1<<bit++;codes[index]=[channel,-flag];
    for(let i=0;i<count;i++)if(l.mask[i])data[i*channels+channel]|=flag;
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
