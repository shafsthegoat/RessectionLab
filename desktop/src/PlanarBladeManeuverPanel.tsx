import type {SurgicalController} from './use-surgical';
import type {SurgicalAction} from './surgical-types';

const quantity=(value:number|null)=>value===null?'Unknown':value.toPrecision(5);
const detailValue=(value:unknown):string=>typeof value==='string'?value:JSON.stringify(value,null,2)??'Unknown';

/** Native actions and quantities come from the accepted engine declaration.
 * Mounting or opening evidence never starts or prepares a maneuver. */
export function PlanarBladeManeuverPanel({c,disabled=false,pause}:{c:SurgicalController;disabled?:boolean;pause:()=>void}){
  const view=c.preview??c.view,data=view?.planarBladeManeuver;
  if(!view||!data)return null;
  const committed=c.view?.planarBladeManeuver??data;
  const blocked=disabled||c.pending||!!c.result?.readOnly||!!c.result?.playback||!!c.result?.verification?.readOnly||!committed.available;
  const run=(command:'preview'|'step',action:SurgicalAction)=>{pause();void c.run(command,{action});};
  return <section className="planar-blade-maneuver" aria-label="Planar blade maneuver" data-planar-blade-state-id={view.stateId} data-planar-blade-accepted-state-id={c.view?.stateId} data-planar-blade-phase={data.phase} aria-busy={c.pending}>
    <h3>Fixed blade maneuver{c.preview?' · preview':''}</h3>
    <p role="status"><strong>{data.inContact?'Contact':'No contact'}</strong> · {data.activePhase??data.phase}</p>
    <div className="surgical-metrics">
      <div>Modeled reaction · N<strong data-planar-blade-metric="normalReactionN" data-value={data.normalReactionN??undefined}>{quantity(data.normalReactionN)}</strong></div>
      <div>Maximum displacement · mm<strong data-planar-blade-metric="maxDisplacementMm" data-value={data.maxDisplacementMm??undefined}>{quantity(data.maxDisplacementMm)}</strong></div>
    </div>
    {data.refusal&&<p className="surgical-warning" role="alert">{data.refusal}</p>}
    <p className="surgical-note">Research model · cycle-work sensitivity unqualified.</p>
    <div className="planar-blade-actions">{committed.nextActions.map(({label,action})=><div className="surgical-actions" key={action.maneuverPhase}>
      <button disabled={blocked} data-planar-blade-command="preview" data-planar-blade-phase={action.maneuverPhase} onClick={()=>run('preview',action)}>Preview {label.toLowerCase()}</button>
      <button className="primary-button" disabled={blocked} data-planar-blade-command="step" data-planar-blade-phase={action.maneuverPhase} onClick={()=>run('step',action)}>{label}</button>
    </div>)}</div>
    <details><summary>Blade model and numerical evidence</summary>
      <dl className="surgical-declared-fields">
        <div><dt>Contact area · mm²</dt><dd data-planar-blade-metric="contactAreaMm2" data-value={data.contactAreaMm2??undefined}>{quantity(data.contactAreaMm2)}</dd></div>
        <div><dt>Normal offset · mm</dt><dd data-planar-blade-metric="offsetMm" data-value={data.offsetMm??undefined}>{quantity(data.offsetMm)}</dd></div>
        <div><dt>Declaration</dt><dd>{data.declarationId}</dd></div>
        {Object.entries(data.details).map(([key,value])=><div key={key}><dt>{key.replace(/([a-z])([A-Z])/g,'$1 $2').replaceAll('_',' ')}</dt><dd>{detailValue(value)}</dd></div>)}
      </dl>
      <p className="surgical-warning">Prior 5% cycle signed-work gate: FAIL · 10.08% / 11.41%. This maneuver does not establish qualified cycle-work sensitivity.</p>
    </details>
  </section>;
}
