from idc_index import IDCClient
import idc_index_data,hashlib,json,urllib.request,urllib.parse,xml.etree.ElementTree as E
from pathlib import Path
from datetime import datetime,timezone
import openpyxl
base=Path('data/remind_source/metadata')
w=openpyxl.load_workbook(base/'ReMIND-Dataset-Clinical-Data-September-2023.xlsx',read_only=True,data_only=True)
rows=list(w.worksheets[0].values);head=rows[0];eligible=[dict(zip(head,r)) for r in rows[1:] if r[8] in ('Astrocytoma','Glioblastoma','Oligodendroglioma')]
selected=min(eligible,key=lambda x:str(x['Case Number']).zfill(3));assert str(selected['Case Number']).zfill(3)=='001'
c=IDCClient();d=c.sql_query("SELECT * FROM index WHERE collection_id='remind' AND PatientID='ReMIND-001' AND StudyDescription='Preop'")
roles={'3D_AX_T1_postcontrast':'structural_t1ce','3D_SAG_T2_SPACE':'structural_t2','cerebrum seg - MR ref: 3D_AX_T1_postcontrast':'cerebrum_annotation','tumor seg - MR ref: 3D_SAG_T2_SPACE':'tumor_annotation'}
series=[]
for row in json.loads(d.to_json(orient='records')):
 if row['SeriesDescription'] not in roles:continue
 row['role']=roles[row['SeriesDescription']]
 u='https://idc-open-data.s3.amazonaws.com/?'+urllib.parse.urlencode({'list-type':'2','prefix':row['crdc_series_uuid']+'/'})
 b=urllib.request.urlopen(u,timeout=30).read();r=E.fromstring(b);ns={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
 assert r.find('s:IsTruncated',ns).text=='false'
 objects=[]
 for o in r.findall('s:Contents',ns):
  key=o.find('s:Key',ns).text;etag=o.find('s:ETag',ns).text.strip('"');size=int(o.find('s:Size',ns).text)
  if key==row['crdc_series_uuid']+'/':assert size==0;continue
  assert '-' not in etag and len(etag)==32
  objects.append({'key':key,'url':'https://idc-open-data.s3.amazonaws.com/'+key,'bytes':size,'etag_md5':etag})
 assert len(objects)==row['instanceCount'];assert sum(x['bytes'] for x in objects)==round(row['series_size_MB']*1e6)
 row['objects']=sorted(objects,key=lambda x:x['key']);row['listing_sha256']=hashlib.sha256(b).hexdigest();series.append(row)
assert len(series)==4
m={'schema':'resectionlab.remind-development-acquisition.v1','created_utc':datetime.now(timezone.utc).isoformat(),'selected_before_image_access':True,'patient_id':'ReMIND-001','patient_group':'ReMIND:001','role':'development_annotation_assisted_geometry','excluded_roles':['external_final_evaluation','clinical_safety_calibration'],'selection_rule':'Lowest case number among official clinical workbook Astrocytoma/Glioblastoma/Oligodendroglioma with preoperative 3D MRI and cerebrum/tumor SEG in the official index. No image pixels, outcomes, tumor volume, or model performance used. First eligible case 001 has required metadata.','clinical_eligibility':{'histopathology':selected['Histopathology'],'source_field':'Histopathology','availability_at_planning':'unknown; eligibility only, excluded from planner inputs/rewards'},'metadata_sources':[{'url':'https://www.cancerimagingarchive.net/wp-content/uploads/'+p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(base.glob('*.xlsx'))],'source_release':'TCIA ReMIND version 1 (2023-09-26); IDC v24','idc_index_version':'0.12.5','idc_index_data_version':idc_index_data.__version__,'idc_parquet_sha256':hashlib.sha256(Path(idc_index_data.IDC_INDEX_PARQUET_FILEPATH).read_bytes()).hexdigest(),'license':{'name':'CC BY 4.0','url':'https://creativecommons.org/licenses/by/4.0/','citation':'ReMIND: The Brain Resection Multimodal Imaging Database, https://doi.org/10.7937/3rag-d070'},'series':sorted(series,key=lambda x:x['role']),'expected_total_bytes':sum(x['bytes'] for s in series for x in s['objects']),'expected_total_instances':sum(len(s['objects']) for s in series),'unavailable_or_unreviewed':['patient functional mapping','patient directional diffusion/tractography','vascular coverage','expert-reviewed cortical access','expert geometry approval','cross-sequence frame reconciliation pending conversion QC'],'postoperative_and_intraoperative_images_selected':False,'molecular_or_outcome_fields_as_inputs':False}
p=Path('manifests/experiments/remind-001-structural-development-v1.json');p.write_text(json.dumps(m,indent=2)+'\n');print(p,m['expected_total_bytes'],m['expected_total_instances'],hashlib.sha256(p.read_bytes()).hexdigest())
