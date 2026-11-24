import {useState} from 'react';
import type {Binding,EnvironmentOptions,EnvironmentRole} from './native-anatomy-data';
import {rehearsalAvailability} from './NativeStudyEvidence';
import type {Point3} from './viewer/coordinates';
import './native-planning-presentation.css';

const roles:EnvironmentRole[]=['brain','tumor','bone','vessels'];
const roleNames:Record<EnvironmentRole,string>={brain:'Brain',tumor:'Tumor',bone:'Bone',vessels:'Vessels'};
export interface EnvironmentApproach {approachId:string;measurementHash?:string;parameterHash:string;targetLayerId:string;snapshotId:string}
export interface EnvironmentPreparation {approach?:EnvironmentApproach;collectionId:string;snapshotId:string;layers:Record<string,string>;allowUnreviewedRegistration:boolean;nativeWallCenter?:Point3}
export function EnvironmentPreparationPanel({binding,approach,routeName,measurements,options,busy,patientCursor,onPrepare,onAnatomy}:{
  routeName?:string;measurements?:import('./native-approaches').NativeApproach['measurements'];
  binding:Binding;approach?:EnvironmentApproach;options?:EnvironmentOptions;busy:boolean;patientCursor?:Point3|null;onPrepare:(options:EnvironmentPreparation)=>Promise<void>;onAnatomy:()=>void;
}) {
  const [layers,setLayers]=useState<Record<string,string>>(()=>Object.fromEntries(roles.map(role=>[
    role,role==='tumor'&&approach?approach.targetLayerId:options?.layers.find(layer=>layer.role===role&&layer.status==='ready')?.layerId??'',
  ])));
  const [allow,setAllow]=useState(false),[preparing,setPreparing]=useState(false),[error,setError]=useState<string|null>(null);
  const selected=options?.layers.filter(layer=>layers[layer.role]===layer.layerId)??[];
  const unavailable=options?.layers.filter(layer=>layer.status==='unavailable')??[];
  const unreviewed=selected.filter(layer=>layer.status==='estimated_alignment');
  const label=(id:string)=>binding.document.layers.find(layer=>layer.id===id)?.label??id;
  const canPrepare=!!options&&!options.referenceIssue&&selected.length>0&&(!approach||selected.some(layer=>layer.role==='tumor'))&&
    selected.every(layer=>layer.status==='ready'||layer.status==='estimated_alignment'&&allow);
  const missing=roles.filter(role=>!selected.some(layer=>layer.role===role));
  const wallIssue=selected.find(layer=>layer.nativeWallIssue)?.nativeWallIssue;
  const canPrepareNative=canPrepare&&!!patientCursor&&!!layers.vessels&&!!layers.brain&&!unreviewed.length&&!wallIssue&&!approach;
  const forecast=approach?options?.selectedRouteForecast:undefined;
  const crossing=measurements?.normalTraversal?.supportedNonTargetVoxelCenterVolumeMm3;
  const alignmentRequired=unreviewed.length>0&&!allow;
  const readiness=canPrepare?'ready':options&&!options.referenceIssue&&alignmentRequired?'review':'unavailable';
  const status=canPrepare?'Ready to prepare':!options||options.referenceIssue?'Preparation unavailable':alignmentRequired?'Alignment choice required':'Select source anatomy';
  async function prepare(nativeWallCenter?:Point3){
    setPreparing(true);setError(null);
    try{
      await onPrepare({collectionId:binding.collectionId,snapshotId:binding.document.snapshot_id,
        layers:Object.fromEntries(selected.map(layer=>[layer.role,layer.layerId])),allowUnreviewedRegistration:allow,
        ...(approach?{approach}:{}),...(nativeWallCenter?{nativeWallCenter}:{})});
    }catch(e){setError(e instanceof Error?e.message:String(e));}
    finally{setPreparing(false);}
  }
  return <section className="surgical-panel environment-preparation" aria-label="Prepare standard rehearsal environment" aria-busy={preparing}>
    <h2>{approach?'Inspect planned environment':'Limited research rehearsal'}</h2>
    <div className="environment-preparation-status" data-readiness={readiness} role="status">
      <strong>{preparing?'Preparing environment…':status}</strong>
      <p className="surgical-note">{rehearsalAvailability(options)}</p>
    </div>
    <fieldset className="surgical-controls environment-preparation-controls" disabled={busy||preparing}>
      <legend>Source anatomy <span>{roles.length-missing.length} / {roles.length} supplied</span></legend>
      <div className="environment-layer-list">{roles.map(role=>{
        const source=selected.find(layer=>layer.role===role);
        return <label className="environment-layer-row" data-layer-status={source?.status??'unknown'} key={role}>
          <span className="environment-layer-role">{roleNames[role]}<small>{source?.status==='ready'?'Ready':source?'Estimated':'Unknown'}</small></span>
          <select disabled={role==='tumor'&&!!approach} aria-label={'Environment '+role+' layer'} value={layers[role]} onChange={e=>setLayers({...layers,[role]:e.target.value})}>
            <option value="">Unknown / not supplied</option>{options?.layers.filter(layer=>layer.role===role&&layer.status!=='unavailable').map(layer=>
              <option key={layer.layerId} value={layer.layerId}>{label(layer.layerId)}{layer.status==='estimated_alignment'?' · estimated alignment':''}</option>)}
          </select>
        </label>;
      })}</div>
      {missing.length>0&&<p className="environment-unassessed">{missing.map(role=>roleNames[role]).join(', ')} · not supplied and unassessed.</p>}
      {!!unavailable.length&&<details className="environment-excluded"><summary>Excluded layers · {unavailable.length}</summary>{unavailable.map(layer=><p key={layer.layerId}>{label(layer.layerId)} · {layer.reason}</p>)}</details>}
      {unreviewed.length>0&&<div className="environment-alignment-choice">
        <p className="surgical-warning">Estimated alignment for {unreviewed.map(layer=>label(layer.layerId)).join(', ')}.</p>
        <label><input aria-label="Allow estimated alignment for research" type="checkbox" checked={allow} onChange={e=>setAllow(e.target.checked)}/><span>Use estimated alignment for this research derivative.</span></label>
      </div>}
      {forecast&&approach&&<section aria-label="Selected route forecast" className="environment-route-decision">
        <h3>Predicted decision · STOP before access</h3>
        <p>{binding.document.patient_id} · {routeName} · {label(approach.targetLayerId)}</p>
        {measurements&&<p>{measurements.lengthMm.toFixed(2)} mm path{typeof crossing==='number'&&` · ${crossing.toLocaleString()} mm³ mapped non-target brain by voxel centers in the corridor`}. Physical access and withdrawal costs for treatment are unknown.</p>}
        <p>{forecast.reason} Stopping now predicts {forecast.preparationActions} preparation actions, {forecast.nonTargetRemovedMm3} mm³ removed and {forecast.withdrawalActions} withdrawal actions. The entire sampled target remains.</p>
        <details><summary>Admission and search limits</summary>
          <p>Physical treatment candidates examined: {forecast.physicalCandidatesExamined}. Other approaches remain untested.</p>
          <ul>{forecast.requiredEvidence.map(item=><li key={item}>{item}</li>)}</ul>
          <p>Source revision <code>{approach.snapshotId}</code></p>
        </details>
      </section>}
      <div className="environment-preparation-actions"><button className="primary-button" disabled={!canPrepare} onClick={()=>void prepare()}>{preparing?'Preparing environment…':approach?'Inspect approach in engine':'Prepare standard environment'}</button>
      {!approach&&<button disabled={!canPrepareNative} onClick={()=>patientCursor&&void prepare(patientCursor)}>Prepare native wall at cursor</button>}
      <button onClick={onAnatomy}>Review anatomy and alignment</button>
      </div>
      {!approach&&wallIssue&&<p className="surgical-note">{wallIssue}</p>}
    </fieldset>
    {error&&<div role="alert" className="surgical-error environment-preparation-error"><strong>Preparation failed</strong><p>{error}</p></div>}
    <details className="environment-preparation-method"><summary>Preparation checks & limitations</summary>
      <p className="surgical-note">The standard environment uses a 1 mm grid and V2 mechanics. Unavailable anatomy remains unassessed.</p>
      {!approach&&<p className="surgical-note">For a local native wall, place the cursor on a vessel in the reviewed patient frame. Select the brain and native angiography score layers. The engine checks local source coverage, vessel continuity and wall clearance before opening a roughly 24 mm working region.</p>}
    </details>
  </section>;
}
