import {useEffect,useState} from 'react';
import type {IntakeItem,IntakeKind,IntakeSelection} from './native-anatomy-data';
import {scanLabel,studyLabel} from './native-study-state';
import type {NativeWorkspace} from './use-native-workspace';

export const importSequences=['T1','T1CE','T2','FLAIR','CT','CTA','MRA','TOF','MRV','SWI','SWI_MAGNITUDE','SWI_PHASE'] as const;
export const importRoles=['tumor','brain','bone','vessels','context'] as const;
const sequenceLabels:Record<string,string>={T1CE:'T1 post-contrast',SWI:'Processed SWI',SWI_MAGNITUDE:'SWI magnitude',SWI_PHASE:'SWI phase'};
const roleLabels:Record<string,string>={tumor:'Lesion candidate',bone:'Skull / bone',context:'Context / unclassified'};
type LabelRow={value:string;role:string;name?:string};
type ReviewRow=IntakeSelection&{modality:string;source:string;roles:LabelRow[]};

export function segmentationRoleMap(rows:LabelRow[]):NonNullable<IntakeItem['roleMap']> {
  const result:NonNullable<IntakeItem['roleMap']>={};
  for(const row of rows){
    const value=Number(row.value);
    if(!/^\d+$/.test(row.value)||!Number.isSafeInteger(value)||value<1||value>65535)
      throw Error('Each label must be an integer from 1 to 65535. Background 0 is preserved.');
    if(!importRoles.includes(row.role as typeof importRoles[number]))throw Error('Choose a meaning for every listed label.');
    if(Object.hasOwn(result,String(value)))throw Error('Label '+value+' is listed more than once.');
    const name=row.name?.trim();
    result[String(value)]=name?{kind:row.role,label:name}:row.role;
  }
  if(!rows.length)throw Error('List every nonzero label and its meaning.');
  return result;
}

export function NativeImportPanel({workspace:w,timepoint}:{workspace:NativeWorkspace;timepoint:string}){
  const [rows,setRows]=useState<ReviewRow[]>([]);
  const [confirmed,setConfirmed]=useState(false);
  const [error,setError]=useState('');
  const [sourcePatientId,setSourcePatientId]=useState('');
  const b=w.binding,api=w.api;
  useEffect(()=>{setRows([]);setConfirmed(false);setError('');setSourcePatientId('')},[b?.collectionId,b?.document.snapshot_id]);
  if(!b||!api)return null;
  const blocked=w.busy||!!w.draft||w.planning.pending;
  const chosenSources=rows.filter(row=>row.kind!=='labels'&&row.storage!=='source-asset'&&row.supported!==false);
  const hasIncomplete=rows.some(row=>row.supported===false||(
    row.kind==='labels'?!row.source:row.storage!=='source-asset'&&!row.modality
  ));
  const knownIds=[...new Set(rows.flatMap(row=>row.patientId?[row.patientId]:[]))];
  const needsAlias=knownIds.length===1&&knownIds[0]!==b.document.patient_id;
  const associationReady=knownIds.length<2&&(!needsAlias||sourcePatientId.trim()===knownIds[0]);
  const update=(id:string,change:Partial<ReviewRow>)=>{
    setError('');
    setRows(all=>all.map(row=>row.selectionId===id?{...row,...change}:row));
  };
  function pick(kind:IntakeKind){
    void w.run(async()=>{
      w.setMessage('Inspecting imaging inputs…');
      setError('');
      try{
        const found=await api!.anatomyPickIntake({...w.args(),kind});
        if(!found){w.setMessage('File selection cancelled');return;}
        setRows(all=>[...all,...found.items.filter(item=>!all.some(row=>row.selectionId===item.selectionId)).map(item=>({
          ...item,modality:(item.candidates?.length??0)<2&&importSequences.includes(item.suggestedRole as typeof importSequences[number])?item.suggestedRole!:'',
          source:'',roles:[{value:'1',role:''}],
        }))]);
        setConfirmed(false);
        w.setMessage('Review source roles and confirm the study association.');
      }catch(failure){
        setError(failure instanceof Error?failure.message:String(failure));
        w.setMessage('Imaging inspection failed');
      }
    });
  }
  function remove(id:string){
    setConfirmed(false);setError('');
    setRows(all=>all.filter(row=>row.selectionId!==id).map(row=>row.source===id?{...row,source:''}:row));
  }
  function importReviewed(){
    if(!confirmed||!associationReady||hasIncomplete)return;
    try{
      const items=rows.map((row):IntakeItem=>{
        const identity={selectionId:row.selectionId,kind:row.kind};
        if(row.kind!=='labels')return {...identity,...(row.modality?{modality:row.modality}:{})};
        const sourceIndex=rows.findIndex(scan=>scan.selectionId===row.source);
        return {...identity,...(sourceIndex<0?{seriesId:row.source}:{sourceInputIndex:sourceIndex}),
          label:row.label,roleMap:segmentationRoleMap(row.roles)};
      });
      setError('');
      void w.change(async()=>{
        w.setMessage('Importing reviewed imaging…');
        try{
          const result=await api!.anatomyImportBatch({
            ...w.args(),patientId:b!.document.patient_id,timepoint,items,confirmedAssociation:true,
            ...(needsAlias?{sourcePatientId:sourcePatientId.trim()}:{}),
          });
          setRows([]);setConfirmed(false);setSourcePatientId('');
          w.setMessage(result.document.snapshot_id===b!.document.snapshot_id?'Selected imaging is already in this study':'Imaging imported');
          return result;
        }
        catch(failure){setError(failure instanceof Error?failure.message:String(failure));throw failure;}
      },{align:false});
    }catch(failure){setError(failure instanceof Error?failure.message:String(failure));}
  }
  return <details className="unified-details import-data" open><summary>Import into this study</summary>
    <div className="tool-row">
      <button disabled={blocked} onClick={()=>pick('nifti')}>Add NIfTI files…</button>
      <button disabled={blocked} onClick={()=>pick('dicom')}>Add DICOM folders…</button>
      <button disabled={blocked} onClick={()=>pick('labels')}>Add segmentations…</button>
    </div>
    {!!rows.length&&<fieldset className="intake-review" disabled={blocked}><legend>Review selected data</legend>
      {rows.map(row=><div className="intake-item" key={row.selectionId}>
        <div className="intake-heading"><strong>{row.label}</strong><button aria-label={'Remove '+row.label} onClick={()=>remove(row.selectionId)}>Remove</button></div>
        {row.patientId&&<p>Source patient {row.patientId}</p>}
        {row.timepoint&&<p>Source timepoint {studyLabel(row.timepoint)}</p>}
        {row.studyId&&<p title={row.studyId}>Source study {row.studyId}</p>}
        {row.warning&&<p>{row.warning}</p>}
        {(row.candidates?.length??0)>1&&<p>Possible sequences · {row.candidates!.join(', ')}</p>}
        {row.supported===false?<p>Unsupported input. Remove this row to import the other files.</p>
        :row.kind==='labels'?<>
          <label>Source scan<select aria-label={'Source scan for '+row.label} value={row.source} onChange={e=>update(row.selectionId,{source:e.target.value})}>
            <option value="">Choose the scan this mask annotates</option>
            {chosenSources.map(scan=><option key={scan.selectionId} value={scan.selectionId}>{scan.label}</option>)}
            {b.document.series.map(scan=><option key={scan.id} value={scan.id}>{scanLabel(scan)}</option>)}
          </select></label>
          <p className="subtle">Use the supplied label definitions for every nonzero value.</p>
          {row.roles.map((entry,index)=>{
            const replace=(change:Partial<LabelRow>)=>update(row.selectionId,{roles:row.roles.map((label,i)=>i===index?{...label,...change}:label)});
            return <div className="intake-label" key={index}>
              <label>Value<input aria-label={'Segmentation label '+(index+1)} type="number" min="1" max="65535" step="1" value={entry.value} onChange={e=>replace({value:e.target.value})}/></label>
              <label>Meaning<select aria-label={'Segmentation role '+(index+1)} value={entry.role} onChange={e=>replace({role:e.target.value})}>
                <option value="">Choose</option>{importRoles.map(role=><option key={role} value={role}>{roleLabels[role]??role}</option>)}
              </select></label>
              <label>Supplied name<input aria-label={'Segmentation name '+(index+1)} placeholder="Optional" maxLength={128} value={entry.name??''} onChange={e=>replace({name:e.target.value})}/></label>
              <button aria-label={'Remove segmentation label '+(index+1)} onClick={()=>update(row.selectionId,{roles:row.roles.filter((_,i)=>i!==index)})}>Remove label</button>
            </div>;
          })}
          <button onClick={()=>update(row.selectionId,{roles:[...row.roles,{value:'',role:''}]})}>Add label</button>
        </>:row.storage==='source-asset'?<p>{row.suggestedRole??'Acquisition'} · preserved source file only</p>
        :<label>Acquired sequence<select aria-label={'Sequence for '+row.label} value={row.modality} onChange={e=>update(row.selectionId,{modality:e.target.value})}>
          <option value="">Choose the acquired sequence</option>{importSequences.map(sequence=><option key={sequence} value={sequence}>{sequenceLabels[sequence]??sequence}</option>)}
        </select></label>}
      </div>)}
      {rows.some(row=>!row.patientId)&&<p className="subtle">Confirm the source association for files without a recorded patient ID.</p>}
      {knownIds.length>1&&<p role="alert">These files list multiple patient IDs. Review one source patient ID at a time.</p>}
      {needsAlias&&<label>Source patient ID
        <span className="subtle">Enter the recorded ID to link a de-identified source to {b.document.patient_id}.</span>
        <input aria-label="Source patient ID" spellCheck={false} value={sourcePatientId}
          onChange={e=>{setSourcePatientId(e.target.value);setConfirmed(false);setError('')}}/>
      </label>}
      <label className="intake-confirm"><input type="checkbox" aria-label="Confirm patient and timepoint" checked={confirmed} onChange={e=>setConfirmed(e.target.checked)}/>
        All selected files belong to {b.document.patient_id} and this {studyLabel(timepoint)} study visit.</label>
      <button disabled={!confirmed||!associationReady||hasIncomplete} onClick={importReviewed}>Import selected data</button>
    </fieldset>}
    {error&&<p role="alert">{error}</p>}
    {w.series&&<details><summary>Attach tractography or functional maps</summary>
      <p className="subtle">Source scan {b.document.series.find(s=>s.id===w.series)?.label??w.series}</p>
      <div className="tool-row">{[['tractography','Add tractography…'],['functional','Add functional map…']].map(([kind,label])=><button key={kind} disabled={blocked}
        onClick={()=>w.change(()=>api.anatomyImport({...w.args(),kind,seriesId:w.series,label:kind}),{align:false})}>{label}</button>)}</div>
    </details>}
  </details>;
}
