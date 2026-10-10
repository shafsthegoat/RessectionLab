import {useEffect,useRef,useState} from 'react';
import type {DevelopmentEpisode} from './episode-types';
import type {EpisodeVascularApi,EpisodeVascularEvaluation,VascularContactCount} from './episode-vascular-types';
import {loadVascularEvaluation} from './episode-vascular-data';
function contactText(value:VascularContactCount){
 return value.annotated_positive_encounter===true?'Annotated encounter':value.annotated_positive_encounter===null?'Unknown where coverage is absent':'No annotated encounter within evaluated coverage';
}
export function EpisodeVascularReadout({evaluation,actionIndex}:{evaluation:EpisodeVascularEvaluation;actionIndex:number}){
 const action=evaluation.perAction[actionIndex];
 return <div className="episode-vascular-readout">
  <p className="instrument-note">Completed strategy evaluated against a generated vessel annotation. Planning did not use these annotations. No injury probability or biological clearance is assessed.</p>
  {action ? <><strong>Action {action.actionIndex+1} · {action.interactionMode==='stop'?'STOP':action.interactionMode==='probe'?'Probe':'Aspiration'}</strong>
   <p className="muted-note">Entire recorded action and return path. These counts are not instantaneous contact at the displayed frame.</p>
   {action.sweepCount===0?<p>STOP has no tool sweep. No contact was evaluated for this action.</p>:<dl className="episode-quantities">{(['shaft','tip','wholeTool'] as const).map(part=>{
    const row=action[part]!;return <div key={part}><dt>{part==='wholeTool'?'Full tool':part==='shaft'?'Shaft':'Tip'}</dt><dd>{contactText(row)}<br/>{row.positive_reference_cells} positive annotation cells · {row.unknown_reference_cells} unknown in-grid cells<br/>{row.outside_reference_fov?'Extends outside reference field of view. ':''}{row.annotation_coverage_complete_for_sweep?'Annotation coverage complete for this sweep.':'Annotation coverage incomplete.'}</dd></div>;
   })}</dl>}</>:<p className="muted-note">Initial tissue has no executed action. Select an action in this replay to inspect its complete tool path.</p>}
  <details className="episode-evidence"><summary>Evaluation binding & limits</summary><p>Tool encounters are separate from removed tissue. Vessel removal overlap was not evaluated.</p><dl className="episode-identities"><dt>Evaluation</dt><dd>{evaluation.evaluationId}</dd><dt>Episode</dt><dd>{evaluation.episodeId}</dd><dt>Reference binding</dt><dd>{evaluation.referenceBindingHash}</dd></dl></details>
 </div>;
}
export function EpisodeVascularPanel({episode,actionIndex,api,busy}:{episode:DevelopmentEpisode;actionIndex:number;api:EpisodeVascularApi|null;busy:boolean}){
 const [evaluation,setEvaluation]=useState<EpisodeVascularEvaluation|null>(null),[pending,setPending]=useState(false),[error,setError]=useState<string|null>(null);
 const generation=useRef(0),current=useRef(episode);current.current=episode;
 useEffect(()=>{generation.current++;setEvaluation(null);setError(null);setPending(false);return()=>{generation.current++}},[episode]);
 const evaluate=async()=>{
  if(busy||pending||!api?.evaluateDevelopmentEpisodeVascular)return;
  const selected=episode,token=++generation.current;setPending(true);setError(null);
  try{const checked=await loadVascularEvaluation(api,selected,()=>token===generation.current&&current.current===selected);
   if(token===generation.current&&current.current===selected)setEvaluation(checked);
  }catch(failure){if(token===generation.current&&current.current===selected)setError(failure instanceof Error?failure.message:String(failure));}
  finally{if(token===generation.current&&current.current===selected)setPending(false)}
 };
 const visible=evaluation?.episodeId===episode.episodeId?evaluation:null;
 return <section aria-label="Generated vessel annotation evaluation" className="episode-evidence">
  <h3>Vessel annotation encounters</h3><p className="eyebrow">GENERATED REFERENCE · EVALUATOR ONLY</p>
  <button className="outline-button" onClick={evaluate} disabled={busy||pending||!api?.evaluateDevelopmentEpisodeVascular}>{pending?'Evaluating recorded strategy…':'Evaluate annotated encounters'}</button>
  {!api?.evaluateDevelopmentEpisodeVascular&&<p className="muted-note">This engine does not provide the generated vessel evaluator.</p>}
  {!visible&&<p className="instrument-note">Available after execution. Evaluates the sealed strategy without changing its actions, tissue state or replay.</p>}
  {error&&<p role="alert">{error}</p>}
  {visible&&<EpisodeVascularReadout evaluation={visible} actionIndex={actionIndex}/>}
 </section>;
}
