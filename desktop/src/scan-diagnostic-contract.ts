const HASH=/^sha256:[a-f0-9]{64}$/,HEX=/^[a-f0-9]{64}$/;
function requireValue(v:unknown,message:string):asserts v {if(!v)throw Error(`Diagnostic import withheld: ${message}`)}
function object(value:unknown):Record<string,any>{requireValue(value!==null&&typeof value==='object'&&!Array.isArray(value),'expected object');return value as Record<string,any>}
function exact(value:unknown,keys:string[]){const o=object(value);requireValue(JSON.stringify(Object.keys(o).sort())===JSON.stringify([...keys].sort()),'unexpected fields');return o}
export function diagnosticRequest(value:unknown){const r=exact(value,['caseHash','seriesId']);requireValue(HASH.test(r.caseHash)&&HASH.test(r.seriesId),'current workspace/image identity required');return {caseHash:r.caseHash as string,seriesId:r.seriesId as string}}
export function validateDiagnosticResult(value:unknown,request:unknown):void{
  const r=diagnosticRequest(request),v=exact(value,['caseHash','seriesId','descriptor','sourceSha256','stateGrid','coverageGrid','loadedStateSha256','loadedCoverageSha256','state','coverage']);
  requireValue(v.caseHash===r.caseHash&&v.seriesId===r.seriesId,'response belongs to another image');
  const d=object(v.descriptor),shape=d.shape_xyz;
  requireValue(d.schema==='case4_unreviewed_diagnostic_display_layer_v1'&&d.scope==='display_only_atlas_native_grid'&&
    d.output_origin==='model_generated_from_observed_case4_preoperative_scans'&&d.model_output_verified===true&&d.same_grid_only===true&&
    d.anatomical_qc==='unreviewed_inferior_mask_omission'&&d.training_overlap_status==='unknown'&&d.planning_eligible===false&&d.evaluation_eligible===false&&d.clinical_evidence===false,'unreviewed display scope changed');
  requireValue(HEX.test(v.sourceSha256)&&d.source_sha256?.t1c===v.sourceSha256,'source binding missing');
  requireValue(Array.isArray(shape)&&shape.length===3&&shape.every(n=>Number.isSafeInteger(n)&&n>0&&n<=4096),'invalid grid');
  const count=shape.reduce((a:number,b:number)=>a*b,1);requireValue(count<=64*1024**2,'display grid too large');
  for(const [name,dtype]of [['state','int8'],['coverage','uint8']]){
    const a=object(v[name]);requireValue(Object.keys(a).every(k=>['path','assetId','dtype','shape','byteOrder','order','sha256','byteLength'].includes(k))&&
      (typeof a.path==='string')!==(typeof a.assetId==='string')&&a.dtype===dtype&&a.byteOrder==='little'&&a.order==='C'&&
      JSON.stringify(a.shape)===JSON.stringify(shape)&&a.byteLength===count&&HEX.test(a.sha256),'invalid state/coverage transfer');
    const file=exact(d.outputs?.[name],['path','sha256','dtype']);requireValue(file.path===name+'.nii.gz'&&file.dtype===dtype&&HEX.test(file.sha256),'saved file binding missing');
    requireValue(v[name==='state'?'loadedStateSha256':'loadedCoverageSha256']===file.sha256,'decoded file identity changed');
    const grid=exact(v[name+'Grid'],['shape','affineRAS','sformCode']);requireValue(JSON.stringify(grid.shape)===JSON.stringify(shape)&&Number.isSafeInteger(grid.sformCode)&&grid.sformCode>0,'invalid decoded grid');
  }
  // AssetRegistry recursively visits results; no transfer may hide in metadata.
  const noAssets=(x:unknown):void=>{if(Array.isArray(x)){x.forEach(noAssets);return}if(x&&typeof x==='object'){const o=object(x);requireValue(!('assetId'in o)&&!(typeof o.path==='string'&&'byteLength'in o),'asset outside diagnostic slots');Object.values(o).forEach(noAssets)}};
  noAssets(d);noAssets(v.stateGrid);noAssets(v.coverageGrid);
}
