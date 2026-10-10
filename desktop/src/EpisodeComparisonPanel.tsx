import type {EpisodeView} from './episode-data';
import type {EpisodeComparisonView} from './episode-comparison-data';
import type {EpisodeComparisonArm} from './episode-comparison-types';
export interface EpisodeComparisonControls {
 actor:EpisodeView|null;result:EpisodeComparisonView|null;arm:EpisodeComparisonArm;
 available:boolean;pending:boolean;error:string|null;inspect:()=>void;select:(arm:EpisodeComparisonArm)=>void;
}
export function EpisodeComparisonPanel({controls,busy}:{controls:EpisodeComparisonControls;busy:boolean}){
 const {actor,result,arm,available,pending,error,inspect,select}=controls;
 if(actor?.episode.selector!=='RL256_ASPIRATION_TRANSFER')return null;
 const live=actor.origin==='live'&&actor.authorship?.status==='verified_live_backend_run';
 return <section aria-label="Matched aspiration comparison" className="episode-evidence">
  <h3>Matched aspiration comparison</h3>
  <p>Same initial generated task and aspiration/STOP rule. The branches can reach different states. This single bounded comparison does not establish superiority or generalization.</p>
  {!result&&<button className="outline-button" disabled={busy||pending||!available||!live} onClick={inspect}>{pending?'Checking the recorded comparator…':'Inspect matched aspiration search'}</button>}
  {!live?<p className="muted-note">The complete matched pair is not retained after reopen. This recorded actor remains available.</p>:!available&&<p className="muted-note">This engine does not provide matched-pair inspection.</p>}
  {error&&<p role="alert">{error}</p>}
  {result&&<>
   <div className="episode-replay-buttons" role="group" aria-label="Displayed comparison branch">
    <button className="outline-button" aria-pressed={arm==='actor'} disabled={busy||pending} onClick={()=>select('actor')}>Trained actor</button>
    <button className="outline-button" aria-pressed={arm==='search'} disabled={busy||pending} onClick={()=>select('search')}>Matched aspiration search</button>
   </div>
   <table><caption>Modeled outcomes of the two recorded branches</caption><thead><tr><th>Outcome</th><th>Trained actor</th><th>Matched search</th></tr></thead><tbody>
    <tr><th>Modeled return</th><td>{result.actorOutcome.modeledReturn.toFixed(4)}</td><td>{result.searchOutcome.modeledReturn.toFixed(4)}</td></tr>
    <tr><th>Target removed</th><td>{result.actorOutcome.targetRemoved.toFixed(2)} mm³</td><td>{result.searchOutcome.targetRemoved.toFixed(2)} mm³</td></tr>
    <tr><th>Other tissue removed</th><td>{result.actorOutcome.normalRemoved.toFixed(2)} mm³</td><td>{result.searchOutcome.normalRemoved.toFixed(2)} mm³</td></tr>
    <tr><th>Actions / termination</th><td>{result.actorOutcome.actions} / {result.actorOutcome.termination}</td><td>{result.searchOutcome.actions} / {result.searchOutcome.termination}</td></tr>
   </tbody></table>
   <p className="instrument-note">Inspection replays saved native actions. It runs neither the actor nor a new search. The comparator is temporary; return to the actor replay before saving the workspace.</p>
   <details><summary>Original search budget & binding</summary><p>Recorded search accounting was not remeasured by this replay.</p><pre>{JSON.stringify(result.companion.episode.planning.searchAccounting,null,2)}</pre><dl className="episode-identities"><dt>Original pair</dt><dd>{result.binding.pairSeal}</dd></dl></details>
  </>}
 </section>;
}
