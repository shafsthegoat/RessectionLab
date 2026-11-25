import {useEffect,useRef,useState} from 'react';
import type {Binding,EnvironmentOptions,EnvironmentRole,NativeWallPair} from './native-anatomy-data';
import {rehearsalAvailability} from './NativeStudyEvidence';
import type {Point3} from './viewer/coordinates';
import './native-planning-presentation.css';

const roles:EnvironmentRole[]=['brain','tumor','bone','vessels'];
const roleNames:Record<EnvironmentRole,string>={brain:'Brain',tumor:'Tumor',bone:'Bone',vessels:'Vessels'};
export interface EnvironmentApproach {approachId:string;measurementHash?:string;parameterHash:string;targetLayerId:string;snapshotId:string}
export interface EnvironmentPreparation {approach?:EnvironmentApproach;collectionId:string;snapshotId:string;layers:Record<string,string>;allowUnreviewedRegistration:boolean;nativeWallCenter?:Point3}
const pairId=(pair:NativeWallPair)=>JSON.stringify([pair.brainLayerId,pair.vesselLayerId]);

export function EnvironmentPreparationPanel({binding,approach,routeName,measurements,options,busy,sourceEditing,patientCursor,onPrepare,onAnatomy}:{
  routeName?:string;measurements?:import('./native-approaches').NativeApproach['measurements'];
  binding:Binding;approach?:EnvironmentApproach;options?:EnvironmentOptions;busy:boolean;sourceEditing?:boolean;patientCursor?:Point3|null;
  onPrepare:(options:EnvironmentPreparation,isCurrent:()=>boolean)=>Promise<void>;onAnatomy:()=>void;
}) {
  const [layers,setLayers]=useState<Record<string,string>>(()=>Object.fromEntries(roles.map(role=>[
    role,role==='tumor'&&approach?approach.targetLayerId:options?.layers.find(layer=>layer.role===role&&layer.status==='ready')?.layerId??'',
  ])));
  const [allow,setAllow]=useState(false),[preparing,setPreparing]=useState(false),[error,setError]=useState<string|null>(null);
  const [mode,setMode]=useState<'native-wall'|'standard'|null>(null);
  const [chosenPair,setChosenPair]=useState('');
  const active=useRef(true);
  const editing=useRef(sourceEditing);
  editing.current=sourceEditing;
  useEffect(()=>{active.current=true;return ()=>{active.current=false;};},[]);
  const wall=options?.nativeWall;
  const pairs=wall?.pairs??[];
  const wallLayers=options?.layers.filter(layer=>layer.role==='brain'||layer.role==='vessels')??[];
  const wallExclusions=wallLayers.filter(layer=>layer.nativeWallIssue);
  const brainLayers=wallLayers.filter(layer=>layer.role==='brain');
  const brainIssue=brainLayers.length&&brainLayers.every(layer=>layer.nativeWallIssue)?brainLayers[0].nativeWallIssue:undefined;
  const wallIssue=wall?.referenceIssue??brainIssue??
    'A local wall requires compatible brain tissue and intact angiography scores with reviewed alignment.';
  const local=!approach&&(mode??(pairs.length?'native-wall':'standard'))==='native-wall';
  const pair=pairs.length===1?pairs[0]:pairs.find(candidate=>pairId(candidate)===chosenPair);
  const reviewed=pair?.scope==='reviewed-crop';
  const center=reviewed?pair.centerRasMm:patientCursor;
  const wallCenter=center?.length===3&&center.every(Number.isFinite)?center:null;
  const canPrepareNative=!!wall&&!wall.referenceIssue&&!!pair&&!!wallCenter;
  const selected=options?.layers.filter(layer=>layers[layer.role]===layer.layerId)??[];
  const unavailable=options?.layers.filter(layer=>layer.status==='unavailable')??[];
  const unreviewed=selected.filter(layer=>layer.status==='estimated_alignment');
  const label=(id:string)=>binding.document.layers.find(layer=>layer.id===id)?.label??id;
  const sourceLabel=(id:string)=>{
    const sourceId=binding.document.layers.find(layer=>layer.id===id)?.source_series;
    const series=binding.document.series.find(scan=>scan.id===sourceId);
    return series?`${label(id)} · ${series.label??series.modality}`:label(id);
  };
  const canPrepare=!!options&&!options.referenceIssue&&selected.length>0&&(!approach||selected.some(layer=>layer.role==='tumor'))&&
    selected.every(layer=>layer.status==='ready'||layer.status==='estimated_alignment'&&allow);
  const missing=roles.filter(role=>!selected.some(layer=>layer.role===role));
  const forecast=approach?options?.selectedRouteForecast:undefined;
  const crossing=measurements?.normalTraversal?.supportedNonTargetVoxelCenterVolumeMm3;
  const alignmentRequired=unreviewed.length>0&&!allow;
  const readiness=local?(canPrepareNative?'ready':pairs.length?'review':'unavailable'):
    canPrepare?'ready':options&&!options.referenceIssue&&alignmentRequired?'review':'unavailable';
  const status=local?(canPrepareNative?'Ready to check local wall':pair?'Patient cursor required':pairs.length?'Select a source pair':'Local wall unavailable'):
    canPrepare?'Ready to prepare':!options||options.referenceIssue?'Preparation unavailable':alignmentRequired?'Alignment choice required':'Select source anatomy';
  const explanation=local?(wall?.referenceIssue??(pairs.length?
    'Preparation checks local source coverage, vessel continuity, and nearby patient tissue.':
    wallIssue)):rehearsalAvailability(options);
  async function prepare(choice:Pick<EnvironmentPreparation,'layers'|'allowUnreviewedRegistration'|'approach'|'nativeWallCenter'>){
    setPreparing(true);setError(null);
    try{
      await onPrepare({collectionId:binding.collectionId,snapshotId:binding.document.snapshot_id,...choice},()=>active.current&&!editing.current);
    }catch(e){if(active.current)setError(e instanceof Error?e.message:String(e));}
    finally{if(active.current)setPreparing(false);}
  }
  return <section className="surgical-panel environment-preparation" aria-label="Prepare standard rehearsal environment" aria-busy={preparing}>
    <h2>{approach?'Inspect planned environment':'Limited research rehearsal'}</h2>
    <div className="environment-preparation-status" data-readiness={readiness} role="status">
      <strong>{preparing?'Preparing environment…':status}</strong>
      <p className="surgical-note">{explanation}</p>
    </div>
    {!approach&&<label className="environment-layer-row environment-mode">
      <span className="environment-layer-role">Environment</span>
      <select aria-label="Rehearsal environment" value={local?'native-wall':'standard'} disabled={busy||preparing}
        onChange={e=>{setMode(e.target.value==='native-wall'?'native-wall':'standard');setError(null);}}>
        <option value="native-wall">Local native wall</option><option value="standard">Standard 1 mm environment</option>
      </select>
    </label>}
    {!approach&&!local&&wall&&!pairs.length&&<p className="surgical-note">Local native wall unavailable. {wallIssue}</p>}
    {local?<fieldset className="surgical-controls environment-preparation-controls" disabled={busy||preparing}>
      <legend>Local wall sources</legend>
      {!pairs.length&&wallExclusions.length>0&&<details className="environment-excluded">
        <summary>Local wall evidence by source</summary>
        {wallExclusions.map(layer=><p key={layer.layerId}>{sourceLabel(layer.layerId)} · {layer.nativeWallIssue}</p>)}
      </details>}
      {pairs.length>1&&<label className="environment-layer-row">
        <span className="environment-layer-role">Source pair</span>
        <select aria-label="Native wall source pair" value={pair?pairId(pair):''} onChange={e=>{setChosenPair(e.target.value);setError(null);}}>
          <option value="">Choose a pair</option>{pairs.map(candidate=><option key={pairId(candidate)} value={pairId(candidate)}>
            {sourceLabel(candidate.brainLayerId)} + {sourceLabel(candidate.vesselLayerId)}{candidate.scope==='reviewed-crop'?' · reviewed field':''}
          </option>)}
        </select>
      </label>}
      {pair&&<><dl className="environment-layer-list environment-native-pair">
        <div className="environment-layer-row"><dt className="environment-layer-role">Brain</dt><dd>{sourceLabel(pair.brainLayerId)}</dd></div>
        <div className="environment-layer-row"><dt className="environment-layer-role">Vessels</dt><dd>{sourceLabel(pair.vesselLayerId)}</dd></div>
      </dl>
      <p className="surgical-note">{wallCenter?`${reviewed?'Reviewed field centre':'Patient cursor'} [${wallCenter.map(n=>n.toFixed(1)).join(', ')}] RAS mm.`:
        'Place the patient cursor near a vessel in Anatomy.'}</p></>}
      <div className="environment-preparation-actions">
        <button className="primary-button" disabled={!canPrepareNative} onClick={()=>pair&&wallCenter&&void prepare({
          layers:{brain:pair.brainLayerId,vessels:pair.vesselLayerId},allowUnreviewedRegistration:false,nativeWallCenter:wallCenter,
        })}>{reviewed?'Prepare reviewed local wall':'Prepare native wall at cursor'}</button>
        <button onClick={onAnatomy}>Review anatomy and alignment</button>
      </div>
    </fieldset>:<fieldset className="surgical-controls environment-preparation-controls" disabled={busy||preparing}>
      <legend>Source anatomy <span>{roles.length-missing.length} / {roles.length} supplied</span></legend>
      <div className="environment-layer-list">{roles.map(role=>{
        const source=selected.find(layer=>layer.role===role);
        return <label className="environment-layer-row" data-layer-status={source?.status??'unknown'} key={role}>
          <span className="environment-layer-role">{roleNames[role]}<small>{source?.status==='ready'?'Ready':source?'Estimated':'Unknown'}</small></span>
          <select disabled={role==='tumor'&&!!approach} aria-label={'Environment '+role+' layer'} value={layers[role]} onChange={e=>setLayers({...layers,[role]:e.target.value})}>
            <option value="">Unknown / not supplied</option>{options?.layers.filter(layer=>layer.role===role&&layer.status!=='unavailable').map(layer=>
              <option key={layer.layerId} value={layer.layerId}>{label(layer.layerId)}{layer.nativeWallAlignment?' · local wall review':layer.status==='estimated_alignment'?' · estimated alignment':''}</option>)}
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
      <div className="environment-preparation-actions"><button className="primary-button" disabled={!canPrepare} onClick={()=>void prepare({
        layers:Object.fromEntries(selected.map(layer=>[layer.role,layer.layerId])),allowUnreviewedRegistration:allow,...(approach?{approach}:{}),
      })}>{preparing?'Preparing environment…':approach?'Inspect approach in engine':'Prepare standard environment'}</button>
      <button onClick={onAnatomy}>Review anatomy and alignment</button>
      </div>
    </fieldset>}
    {error&&<div role="alert" className="surgical-error environment-preparation-error"><strong>Preparation failed</strong><p>{error}</p></div>}
    <details className="environment-preparation-method"><summary>Preparation checks & limitations</summary>
      <p className="surgical-note">{local?'The roughly 24 mm native field uses the selected brain and vessel pair. Tumor and bone are unassessed. The reviewed field centre applies only to that saved local alignment.':
        'The standard environment uses a 1 mm grid and V2 mechanics. Unavailable anatomy remains unassessed.'}</p>
    </details>
  </section>;
}
