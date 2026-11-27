/** Readout text for the engine's opt-in operative scenarios. Every line names
 * what it is: a scenario, a model estimate, navigation as the system reports
 * it, rehearsal truth (audit data a surgeon would not have), or UNKNOWN.
 * Nothing here is computed beyond formatting the engine's own fields, except
 * the renderer's tip proximity, which is labelled estimated. */
import type {SurgicalView} from './surgical-types';
import {brainShiftSummary,type BrainShiftViewV1} from './viewer/brainShiftField.ts';

type Point3=[number,number,number];
export type ScenarioTone='scenario'|'navigation'|'truth'|'estimate'|'unknown'|'injury'|'bleeding';
export interface ScenarioLine {key:string;tone:ScenarioTone;text:string}

export interface NavigationReading {kind:'navigation';reportedImageRasMm:Point3;imageRegion:{regionId:number;name:string;material:string|null}|null;
  distanceToPlannedTumourMm?:number;nearestPlannedTumourImageRasMm?:Point3;registration?:{declaredTargetErrorRmsMm:number}}
export interface NavigationTruth {tipRasMm:Point3;registrationErrorMm:Point3;navigationErrorMm:number;
  nearestTumourEdge?:{discrepancyMm:number};trueDistanceToRemainingTumourMm?:number|null;
  tissueUnderTip?:{name:string;shiftMm:Point3}|null}
export interface MappingFamily {result:string;nearestTract?:string;estimatedDistanceMm?:number;marginMm?:number;reason?:string;decisionWithinUncertainty?:boolean}
export interface MappingObservation {kind:'model-estimate';category:'mapping';probe:string;position:Point3;result:string;
  parameters:Record<string,number>;effectiveCurrentMa:number|null;estimatedReachMm:number|null;uncertaintyMm:number;
  families:Record<string,MappingFamily>;unknownFamilies:string[];tracts:Array<{id:string;regionName:string;family:string;estimatedDistanceMm:number|null}>}
export interface BleedingWound {id:string;segment:string;kind:string;kindStatus?:string;calibreMm?:number;flowCalibreMm?:number;calibre?:string;
  openingFraction?:number;sealFraction:number;rateMm3PerSecond:number;lossMm3:number;cells:number[];openedAtSeconds:number}
export interface VesselBleedingState {wounds:Record<string,BleedingWound>;bloodLossMm3:number;lossByKindMm3:Record<string,number>;
  tumourOozeMm3:number;rateMm3PerSecond:number;depositedMm3:number}
export interface InjuryDisplay {contused:number[];lacerated:number[];vesselInjured:number[];totalsMm3:{contused:number;lacerated:number;vessel:number};
  avulsions:number;eventCount:number;events:Array<{kind:string;code:string;structures:string[];regions:string[];instrument:string}>}
export interface TipProximity {tip:Point3;slot:string;instrument:string;
  tract:{name:string;distanceMm:number}|null;vessel:{name:string;distanceMm:number}|null;searchMm:number}

const mm=(n:number|null|undefined,digits=1)=>typeof n==='number'&&Number.isFinite(n)?`${n.toFixed(digits)} mm`:'unknown';
const ml=(mm3:number)=>mm3<100?`${mm3.toFixed(1)} mm³`:`${(mm3/1000).toFixed(mm3<1000?2:1)} mL`;
const familyName:Record<string,string>={motor:'motor',language:'language',vision:'visual',other:'other'};

export function latestNavigation(view:SurgicalView|null|undefined):{reading:NavigationReading;truth:NavigationTruth|null;index:number}|null {
  const observations=view?.observations??[];
  const readings=observations.filter(o=>o.kind==='navigation');
  if(!readings.length)return null;
  const truths=((view as {navigationV1?:{privilegedTruth?:{readings?:NavigationTruth[]}}}).navigationV1?.privilegedTruth?.readings)??[];
  return {reading:readings.at(-1) as unknown as NavigationReading,truth:truths.length===readings.length?truths.at(-1)!:null,index:readings.length};
}

export function latestMapping(view:SurgicalView|null|undefined):MappingObservation|null {
  const row=[...(view?.observations??[])].reverse().find(o=>o.kind==='model-estimate'&&o.category==='mapping');
  return (row as unknown as MappingObservation)??null;
}

/** "Model estimate: motor response likely, CST within ~X mm (±U)" or an UNKNOWN reason; no-response is never stated as safe. */
export function mappingText(row:MappingObservation):string {
  const uncertainty=`±${row.uncertaintyMm.toFixed(1)} mm`;
  const current=row.effectiveCurrentMa!=null?` at ${row.effectiveCurrentMa.toFixed(1)} mA`:'';
  const responding=Object.entries(row.families).filter(([,f])=>f.result==='model_estimate_response');
  if(responding.length){
    const [family,f]=responding.sort((a,b)=>(a[1].estimatedDistanceMm??0)-(b[1].estimatedDistanceMm??0))[0];
    return `Model estimate${current}: ${familyName[family]??family} response likely, ${tractLabel(row,f.nearestTract)} within ~${(f.estimatedDistanceMm??0).toFixed(1)} mm (${uncertainty})`;
  }
  const silent=Object.entries(row.families).filter(([,f])=>f.result==='model_estimate_no_response');
  if(silent.length){
    const [family,f]=silent.sort((a,b)=>(a[1].estimatedDistanceMm??0)-(b[1].estimatedDistanceMm??0))[0];
    return `Model estimate${current}: no ${familyName[family]??family} response; nearest ${tractLabel(row,f.nearestTract)} ~${(f.estimatedDistanceMm??0).toFixed(1)} mm (${uncertainty}). No response is not evidence of safety`;
  }
  const motor=row.families.motor;
  const reason=motor?.reason??Object.values(row.families).find(f=>f.reason)?.reason;
  if(/no tract evidence/i.test(reason??'')||!row.tracts.length)return 'UNKNOWN: no tract evidence';
  if(/outside declared tract coverage/i.test(reason??''))return 'UNKNOWN: tip outside the declared tract coverage';
  if(/awake/i.test(reason??''))return 'UNKNOWN: this tract needs an awake patient performing tasks';
  if(/cortical/i.test(reason??''))return 'UNKNOWN: a cortical probe without a tract response is not evidence';
  return `UNKNOWN: ${reason??'no model estimate'}`;
}

function tractLabel(row:MappingObservation,id:string|undefined){
  const tract=row.tracts.find(t=>t.id===id);
  const name=(tract?.regionName??id??'tract').replace(/\s*\(tract\)\s*$/i,'');
  return /corticospinal|\bcst\b/i.test(name)?'CST':name;
}

export function bleedingSummary(state:VesselBleedingState|undefined) {
  if(!state)return null;
  const wounds=Object.values(state.wounds??{});
  const open=wounds.filter(w=>w.sealFraction<1);
  const count=(kind:string)=>open.filter(w=>w.kind===kind).length;
  return {open:open.length,total:wounds.length,artery:count('artery'),vein:count('vein')+count('sinus'),unknown:count('uncertain'),
    lossMm3:state.bloodLossMm3,rateMm3PerSecond:state.rateMm3PerSecond,oozeMm3:state.tumourOozeMm3,
    unknownLossMm3:wounds.filter(w=>w.kind==='uncertain').reduce((s,w)=>s+w.lossMm3,0)};
}

export function scenarioReadout(view:SurgicalView|null|undefined,proximity?:TipProximity|null):ScenarioLine[] {
  if(!view)return [];
  const lines:ScenarioLine[]=[];
  const shift=brainShiftSummary(view.brainShiftV1 as BrainShiftViewV1|undefined);
  if(shift)lines.push({key:'shift',tone:'scenario',text:shift.started?
    `Brain shift scenario · peak ${mm(shift.peakMm)}${shift.level<shift.steps?` (settling to ${mm(shift.declaredPeakMm)})`:''} · target moved ${mm(shift.targetMovedMm)}`:
    'Brain shift scenario · begins at dural opening'});
  const nav=latestNavigation(view);
  if(nav){
    const {reading,truth}=nav;
    lines.push({key:'nav',tone:'navigation',text:`Navigation · image shows ${reading.imageRegion?.name??'outside image'} · ${mm(reading.distanceToPlannedTumourMm)} to planned tumour`});
    if(truth)lines.push({key:'nav-truth',tone:'truth',text:`Rehearsal truth · navigation off ${mm(truth.navigationErrorMm)}`+
      (truth.trueDistanceToRemainingTumourMm!=null?` · tumour actually ${mm(truth.trueDistanceToRemainingTumourMm)} away`:'')+
      (truth.nearestTumourEdge?` · edge moved ${mm(truth.nearestTumourEdge.discrepancyMm)}`:'')});
  }
  const mapping=latestMapping(view);
  if(mapping)lines.push({key:'map',tone:mapping.result==='unknown'?'unknown':'estimate',text:mappingText(mapping)});
  const bleeding=bleedingSummary(view.vesselBleedingV1 as VesselBleedingState|undefined);
  if(bleeding&&(bleeding.total||bleeding.lossMm3>0)){
    const kinds=[bleeding.artery&&`${bleeding.artery} arterial`,bleeding.vein&&`${bleeding.vein} venous`,bleeding.unknown&&`${bleeding.unknown} kind UNKNOWN`].filter(Boolean).join(', ');
    const mixed=bleeding.unknownLossMm3>0&&bleeding.unknownLossMm3<bleeding.lossMm3;
    lines.push({key:'bleed',tone:'bleeding',text:`Bleeding · ${bleeding.open?`${bleeding.open} open wound${bleeding.open>1?'s':''} (${kinds})`:'wounds sealed'} · loss ${ml(bleeding.lossMm3)}`+
      (bleeding.rateMm3PerSecond>0?` · ${(bleeding.rateMm3PerSecond*.06).toFixed(1)} mL/min`:'')+
      (bleeding.unknown||bleeding.unknownLossMm3>0?` · UNKNOWN kind bled at the arterial law${mixed?` (${ml(bleeding.unknownLossMm3)})`:''}`:'')});
  }
  const injury=view.surgicalInjuryV1 as InjuryDisplay|undefined;
  if(injury&&injury.eventCount){
    const structures=[...new Set(injury.events.flatMap(e=>e.structures))];
    lines.push({key:'injury',tone:'injury',text:`Injury recorded · contused ${injury.totalsMm3.contused.toFixed(0)} mm³ · lacerated ${injury.totalsMm3.lacerated.toFixed(0)} mm³`+
      (injury.vesselInjured.length?` · ${injury.vesselInjured.length} vessel cells`:'')+(injury.avulsions?` · ${injury.avulsions} avulsion${injury.avulsions>1?'s':''}`:'')+
      (structures.length?` · ${structures.join(', ')}`:'')});
  }
  if(proximity&&(proximity.tract||proximity.vessel))lines.push({key:'near',tone:'estimate',text:`Estimated from labels, ${proximity.instrument.replaceAll('_',' ')} tip · `+
    [proximity.tract?`${proximity.tract.name} ${mm(proximity.tract.distanceMm)}`:`no tract within ${proximity.searchMm} mm`,
     proximity.vessel?`${proximity.vessel.name} ${mm(proximity.vessel.distanceMm)}`:`no vessel within ${proximity.searchMm} mm`].join(' · ')});
  return lines;
}
