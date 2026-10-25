import type {SurgicalController} from './use-surgical';
import {surgicalReplayRoutes} from './surgical-authority';
import {rescueLearningReport,learningEndpointVerified} from './rescue-learning-data';

export function RescueLearningPanel({c,report,pause}:{c:SurgicalController;report:unknown;pause:()=>void}) {
  const data=rescueLearningReport(report,c.result);
  if(!data)return null;
  const quantity=(value:number)=>value>0&&value<.001?value.toExponential(1):value.toFixed(3);
  const names={untrained:'Untrained PPO',scripted:'Scripted',trained:'Trained PPO','search-selected':'Search'};
  return <details className="research-planning rescue-learning"><summary>Patient rescue experiment · PPO and baselines</summary>
    {data.status&&data.status!=='completed'&&<p role="status">The experiment stopped before completion. Only saved assessments are listed; missing comparisons remain unevaluated.</p>}
    <p>PPO: {data.settings.train_decisions} real option decisions per seed. Search: {data.settings.search_decisions}. Scripted and untrained baselines use no fitting.</p>
    <p className="surgical-note">Saved outcomes · select a row to inspect its trajectory, then verify full replay to confirm the state.</p>
    <div className="rescue-comparison"><table aria-label="Patient rescue learning outcomes"><caption>Modeled fluid · mm³</caption><thead><tr><th>Method / seed</th><th>Retained</th><th>Emitted</th></tr></thead><tbody>{data.assessments.map(row=>{
      const selected=c.result?.branchId===row.branchId,verified=learningEndpointVerified(row,c.result);
      return <tr key={`${row.method}:${row.seed}`} aria-selected={selected}><th><button disabled={c.pending} aria-pressed={selected} onClick={()=>{pause();void c.run(surgicalReplayRoutes(c.result).branch,{branchId:row.branchId});}}>{names[row.method]} · {row.seed}</button><small>Cleanup {row.success?'complete':'incomplete'}</small>{verified&&<small>State replay verified</small>}</th><td>{quantity(row.modeledField[0])}</td><td>{quantity(row.modeledField[3])}</td></tr>;
    })}</tbody></table></div>
    <details><summary>Recorded returns and limits</summary><p>Identical physical histories share a native branch; policy records remain separate. Historical preparation and task waypoints were supplied. This tests rescue choices on one recorded injury.</p><ul>{data.assessments.map(row=><li key={`${row.method}:${row.seed}`}>{names[row.method]} · seed {row.seed}: return {row.reward.toFixed(3)}, {row.decisions} decisions, {row.reason.replaceAll('_',' ')}.</li>)}</ul><ul>{data.limitations.map((text,index)=><li key={index}>{text}</li>)}</ul><p>Fluid quantities and rewards are uncalibrated model outputs. Endpoint verification confirms recorded physics; it does not rerun policy inference or validate clinical outcomes.</p></details>
  </details>;
}
