import {useState} from 'react';
import type {NativeWorkspace} from './use-native-workspace';
import {studyLabel} from './native-study-state';

export const importSequences=['T1','T1CE','T2','FLAIR','CT','MRA','CTA','TOF'] as const;
export const importRoles=['tumor','brain','bone','vessels','context'] as const;
type RoleRow={value:string;role:string};

export function segmentationRoleMap(rows:RoleRow[]):Record<string,string> {
  const result:Record<string,string>={};
  for(const row of rows){
    const value=Number(row.value);
    if(!/^\d+$/.test(row.value)||!Number.isSafeInteger(value)||value<1||value>65535)
      throw Error('Each segmentation label must be an integer from 1 to 65535. Background 0 is preserved.');
    if(!importRoles.includes(row.role as typeof importRoles[number]))throw Error('Choose an operative role for every listed label.');
    if(Object.hasOwn(result,String(value)))throw Error(`Label ${value} is listed more than once.`);
    result[String(value)]=row.role;
  }
  if(!rows.length)throw Error('List the nonzero segmentation labels and their roles.');
  return result;
}

export function NativeImportPanel({workspace:w,timepoint,sourceLabel}:{workspace:NativeWorkspace;timepoint:string;sourceLabel:string}){
  const [modality,setModality]=useState('T1');
  const [roles,setRoles]=useState<RoleRow[]>([{value:'1',role:''}]);
  const [error,setError]=useState('');
  const b=w.binding,api=w.api;if(!b||!api)return null;
  const blocked=w.busy||!!w.draft;
  function importLabels(){
    try{
      const roleMap=segmentationRoleMap(roles);setError('');
      void w.change(()=>api!.anatomyImport({...w.args(),kind:'labels',seriesId:w.series,label:'Imported segmentation',roleMap}));
    }catch(e){setError(e instanceof Error?e.message:String(e));}
  }
  return <details className="unified-details import-data" open><summary>Import into this study</summary>
    <label>Image sequence<select aria-label="Import image sequence" value={modality} disabled={blocked} onChange={e=>setModality(e.target.value)}>
      {importSequences.map(m=><option key={m}>{m}</option>)}
    </select></label>
    <p className="subtle">Adding to {studyLabel(timepoint).toLowerCase()} · {b.document.patient_id.replace(/^ZENODO-17486320-/,'')}</p>
    <div className="tool-row">
      <button disabled={blocked} onClick={()=>w.change(()=>api.anatomyImport({...w.args(),kind:'nifti',modality,timepoint,patientId:b.document.patient_id}))}>Import NIfTI…</button>
      <button disabled={blocked} onClick={()=>w.change(()=>api.anatomyImport({...w.args(),kind:'dicom',modality,timepoint}))}>Import DICOM series…</button>
    </div>
    <p className="subtle">Choose the acquired sequence explicitly. For DICOM MRI, select a folder containing one series. MR alone does not identify T1, contrast T1, T2 or FLAIR.</p>
    <p className="subtle">Attach annotations to {sourceLabel}. List every nonzero label in the selected segmentation; choose roles from its source documentation.</p>
    <fieldset disabled={blocked||!w.series}><legend>Segmentation label roles</legend>
      {roles.map((row,index)=><div className="tool-row" key={index}>
        <label>Label<input aria-label={`Segmentation label ${index+1}`} type="number" min="1" max="65535" step="1" value={row.value} onChange={e=>setRoles(roles.map((r,i)=>i===index?{...r,value:e.target.value}:r))}/></label>
        <label>Role<select aria-label={`Segmentation role ${index+1}`} value={row.role} onChange={e=>setRoles(roles.map((r,i)=>i===index?{...r,role:e.target.value}:r))}>
          <option value="">Choose role</option>{importRoles.map(role=><option key={role} value={role}>{role==='bone'?'skull / bone':role==='context'?'context / unclassified':role}</option>)}
        </select></label>
        <button aria-label={`Remove segmentation label ${index+1}`} onClick={()=>setRoles(roles.filter((_,i)=>i!==index))}>Remove</button>
      </div>)}
      <button onClick={()=>setRoles([...roles,{value:'',role:''}])}>Add label</button>
      <button onClick={importLabels}>Import segmentation…</button>
    </fieldset>
    {error&&<p role="alert">{error}</p>}
    <div className="tool-row">{['tractography','functional'].map(kind=><button key={kind} disabled={blocked||!w.series} onClick={()=>w.change(()=>api.anatomyImport({...w.args(),kind,seriesId:w.series,label:kind}))}>{kind}</button>)}</div>
  </details>;
}
