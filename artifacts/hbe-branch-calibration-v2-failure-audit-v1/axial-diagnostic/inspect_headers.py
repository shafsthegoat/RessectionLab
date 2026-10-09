"""One bounded diagnostic of two explicitly permitted axial members; no curve fit."""
from datetime import datetime, timezone
from pathlib import Path
import csv,hashlib,io,json,zipfile,zlib
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
ALLOWED=('HBE_01/HBE_01_03/compression_c3.csv','HBE_01/HBE_01_03/tension_c3.csv')
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  while b:=f.read(1024**2):h.update(b)
 return h.hexdigest()
def load(path):return json.loads((ROOT/path).read_text())
def binding(path):return {'path':path,'sha256':sha(ROOT/path)}
def save(name,obj):
 with (OUT/name).open('x') as f:f.write(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
release_path='build/hbe-branch-calibration-v2/release.json'
release=load(release_path);study=load(release['study']['path']);roles=load(study['roles']['path'])
assert sha(ROOT/release['study']['path'])==release['study']['sha256']
assert sha(ROOT/study['roles']['path'])==study['roles']['sha256']
assert tuple(release['permitted_members'])==ALLOWED
assert tuple(x['path'] for x in roles['calibration']['members'])==ALLOWED
base=study['output_root']+'/experiment'
metadata=[release_path,release['study']['path'],study['roles']['path'],base+'/state.json',base+'/result.json',base+'/access.jsonl',base+'/supervision/supervision.json',base+'/supervision/combined.log',base+'/publication-check.json']
metadata += [b['path'] for b in release['source_bindings'].values()]
before={p:sha(ROOT/p) for p in metadata}
plan={'schema':'hbe-axial-csv-header-inspection-v1','authorized_scope':'root-directed diagnostic of released axial framing only','written_before_member_reads':True,'allowed_members':list(ALLOWED),'forbidden_member_content':'all other archive members, including both torsion branches','member_reads':1,'fits':0,'native_calls':0,'new_source_edits':0,'numeric_response_conversion':False,'original_inputs':{p:s for p,s in before.items()},'started_utc':datetime.now(timezone.utc).isoformat()}
save('inspection-declaration.json',plan)
source=roles['source'];archive=ROOT/source['archive_path'];assert archive.stat().st_size==source['archive_bytes'];assert sha(archive)==source['archive_sha256']
rows=[]
with archive.open('rb') as stream,zipfile.ZipFile(stream) as bundle:
 infos=bundle.infolist();assert len({x.filename for x in infos})==len(infos)
 for branch,item in zip(('compression','tension'),roles['calibration']['members'],strict=True):
  assert item['path'] in ALLOWED
  info=bundle.getinfo(item['path']);assert info.file_size==item['bytes'] and f'{info.CRC:08x}'==item['crc32']
  data=bundle.read(info);assert len(data)==item['bytes'] and f'{zlib.crc32(data):08x}'==item['crc32']
  text=data.decode('utf-8');records=list(csv.reader(io.StringIO(text),delimiter=','))
  header=records[0];line=text.splitlines(keepends=True)[0]
  assert len(data)<1024**2
  rows.append({'branch':branch,'member':item['path'],'bytes':len(data),'crc32':f'{info.CRC:08x}','sha256':hashlib.sha256(data).hexdigest(),'encoding':'UTF-8 strict decode succeeded','utf8_bom_present':data.startswith(b'\xef\xbb\xbf'),'raw_first_line':line,'header_fields':header,'header_field_count':len(header),'total_csv_records':len(records),'following_records':len(records)-1,'record_widths':sorted(set(map(len,records))),'blank_record_count':sum(not x for x in records),'CRLF_count':text.count('\r\n'),'LF_count':text.count('\n'),'trailing_newline':text.endswith('\n'),'v2_declared_schema':study['csv_schemas'][branch],'diagnostic_used_numeric_values':False,'content_retained':'first header line only; full source bytes not copied'})
assert sha(archive)==source['archive_sha256']
assert all(sha(ROOT/p)==s for p,s in before.items())
result={'schema':'hbe-axial-csv-header-diagnosis-v1','release':binding(release_path),'study':release['study'],'roles':study['roles'],'archive':{'path':source['archive_path'],'bytes':source['archive_bytes'],'sha256':source['archive_sha256']},'opened_member_names':[x['member'] for x in rows],'members':rows,'original_inputs_unchanged':True,'archive_unchanged':True,'torsion_member_content_opened':False,'fitted':False,'native_calls':0,'frozen_v2_state':load(base+'/state.json'),'access_ledger_records':[json.loads(x) for x in (ROOT/(base+'/access.jsonl')).read_text().splitlines()],'finished_utc':datetime.now(timezone.utc).isoformat()}
save('diagnosis.json',result)
print(json.dumps(rows,indent=2))
