import {useEffect,useRef,useState} from 'react';
import {loadEpisodeComparison} from './episode-comparison-data';
import type {EpisodeComparisonView} from './episode-comparison-data';
import type {EpisodeView} from './episode-data';
import type {EpisodeComparisonArm} from './episode-comparison-types';
import type {ResectionApi} from './types';
export function useEpisodeComparison(actor:EpisodeView|null,api:ResectionApi|null,available:boolean,caseHash:string|null){
 const [result,setResult]=useState<EpisodeComparisonView|null>(null),[arm,setArm]=useState<EpisodeComparisonArm>('actor'),[step,setStep]=useState(0),[pending,setPending]=useState(false),[error,setError]=useState<string|null>(null);
 const generation=useRef(0),current=useRef({actor,caseHash});current.current={actor,caseHash};
 useEffect(()=>{generation.current++;setResult(null);setArm('actor');setStep(0);setPending(false);setError(null);return()=>{generation.current++}},[actor,caseHash]);
 const checked=result?.actor===actor&&actor?.source.caseHash===caseHash?result:null;
 const selected=checked&&arm==='search';
 const inspect=async()=>{
  if(pending||!available||!api||!actor||actor.source.caseHash!==caseHash)return;
  const expected=actor,token=++generation.current;setPending(true);setError(null);
  const valid=()=>generation.current===token&&current.current.actor===expected&&current.current.caseHash===caseHash;
  try{const value=await loadEpisodeComparison(api,expected,valid);if(valid())setResult(value)}
  catch(failure){if(valid())setError(failure instanceof Error?failure.message:String(failure))}
  finally{if(valid())setPending(false)}
 };
 return {result:checked,arm:selected?'search' as const:'actor' as const,activeView:selected?checked.companion:actor,companionStep:step,pending,error,inspect,
  select:(value:EpisodeComparisonArm)=>{if(checked&&!pending){setArm(value);setStep(0)}},
  setStep:(value:number)=>{if(selected&&Number.isInteger(value)&&value>=0&&value<checked.companion.frames.length)setStep(value)}};
}
