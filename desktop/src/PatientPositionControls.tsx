import {useState} from 'react';
import type {SurgicalController} from './use-surgical';
import type {PositionEffects} from './surgical-types';

export function PatientPositionControls({c,readOnly}:{c:SurgicalController;readOnly:boolean}) {
  const [preset,setPreset]=useState('supine'),[tilt,setTilt]=useState(0),[rotation,setRotation]=useState(0),[height,setHeight]=useState<number|null>(null),[before,setBefore]=useState<{effects?:PositionEffects;shift:number;fluid:number}|null>(null);
  const v=c.view,e=v?.positionEffects;
  const disabled=c.pending||readOnly||!v||Object.values(v.slots).some(s=>s.inserted||s.active)||Object.keys(v.holds??{}).length>0;
  async function apply(){if(!v)return;setBefore({effects:e,shift:Number(v.metrics.brainMaxDisplacementMm??0),fluid:Number(v.metrics.fluidPresentMm3??0)});await c.run('step',{action:{slot:c.slot,operation:'position',duration:1,position:{preset,bodyTiltDeg:tilt,headRotationRasDeg:[0,0,rotation],...(height===null?{}:{siteHeightAboveHeartMm:height,heightEstimateSource:'user-entered estimated operative-site height'})}}});}
  return <details className="patient-position"><summary>Patient position and gravity</summary>
    <p className="surgical-note">Position changes head/body orientation, gravity loading and fluid drainage in physical RAS. Source scans and camera stay fixed. Withdraw and deactivate both tools before setup. Use Branch to compare the same source in different positions.</p>
    <fieldset disabled={disabled}>
      <label>Setup preset<select aria-label="Patient position preset" value={preset} onChange={event=>{setPreset(event.target.value);setTilt(event.target.value==='semisitting'?60:0);}}>{[['supine','Supine'],['prone','Prone'],['left_lateral','Left lateral'],['right_lateral','Right lateral'],['park_bench','Park-bench'],['semisitting','Semisitting']].map(([id,label])=><option key={id} value={id}>{label}</option>)}</select></label>
      <label>Body tilt · °<input aria-label="Body tilt degrees" type="number" min={-45} max={85} value={tilt} onChange={e=>setTilt(Number(e.target.value))}/></label>
      <label>Head rotation about S · °<input aria-label="Head rotation degrees" type="number" min={-90} max={90} value={rotation} onChange={e=>setRotation(Number(e.target.value))}/></label>
      <label>Estimated site height above heart · mm<input aria-label="Estimated site height mm" type="number" min={-2000} max={2000} placeholder="Derived from assumed 300 mm heart-to-site vector" value={height??''} onChange={e=>setHeight(e.target.value===''?null:Number(e.target.value))}/></label>
      <button onClick={()=>void apply()}>Apply position and settle 1 s</button>
    </fieldset>
    {disabled&&<p className="surgical-note">Setup requires an editable recorded boundary, V2 mechanics and both instruments withdrawn.</p>}
    {e&&<section aria-label="What changed and why" data-position-preset={e.preset}><h4>What changed and why</h4>
      <dl><dt>Patient preset</dt><dd>{e.preset}</dd><dt>Gravity RAS · m/s²</dt><dd>{e.gravityRasMPerS2.map(n=>n.toFixed(3)).join(', ')}</dd><dt>Site above heart · estimated mm</dt><dd>{e.siteHeightAboveHeartEstimateMm.toFixed(1)}</dd><dt>Static column: site minus heart</dt><dd>{e.hydrostaticSiteMinusHeartPa.toFixed(1)} Pa / {e.hydrostaticSiteMinusHeartMmHg.toFixed(2)} mmHg</dd><dt>Maximum local modeled shift</dt><dd>{before?before.shift.toFixed(4)+' → ':''}{Number(v?.metrics.brainMaxDisplacementMm??0).toFixed(4)} mm</dd><dt>Fluid in field</dt><dd>{before?before.fluid.toFixed(2)+' → ':''}{Number(v?.metrics.fluidPresentMm3??0).toFixed(2)} mm³</dd></dl>
      <p className="surgical-note">Gravity applies body force to the exposed local lattice; fluid moves down available free cells. Hydrostatic pressure uses −ρgh and an assumed blood density of 1060 kg/m³. Height is an estimate, never a measured vital. {e.mechanicsAssumptions}.</p>
      <p className="surgical-note">Venous outflow: neck geometry, jugular patency and venous pressure are unknown. ICP, perfusion and neurological-deficit probabilities remain unknown.</p>
    </section>}
  </details>;
}
