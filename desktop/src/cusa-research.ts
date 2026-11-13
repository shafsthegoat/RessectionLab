import type {SurgicalView} from './surgical-types';

export function cusaContactModel(view:SurgicalView|null|undefined):Record<string,unknown>|null {
  const value=(view?.physics??view?.scene.physics)?.cusaV3;
  if(!value||typeof value!=='object'||Array.isArray(value))return null;
  const row=value as Record<string,unknown>,profile=row.profile;
  return row.enabled===true&&profile&&typeof profile==='object'&&
    (profile as Record<string,unknown>).schema==='resectionlab.cusa-contact.v3'?row:null;
}

export function cusaActionParameters(view:SurgicalView|null|undefined,slot:string,parameters:Record<string,number>):Record<string,number> {
  if(view?.slots[slot]?.instrument!=='cusa'||(view.physics??view.scene.physics)?.profile?.schema==='resectionlab.tissue-physics.v2')return parameters;
  // Historical native requests have no duty parameter; retain their admission.
  const {duty:_,...supported}=parameters;
  return supported;
}

export const cusaParameterSpecs={
  radiusMm:{label:'Fixed CUSA radius · mm',min:1.5,max:1.5,default:1.5},
  power:{label:'CUSA activation fraction',min:0,max:1,default:1},
  duty:{label:'CUSA duty fraction',min:0,max:1,default:1},
};
