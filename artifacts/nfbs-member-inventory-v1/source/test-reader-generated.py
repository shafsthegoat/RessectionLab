from pathlib import Path
import gzip,hashlib,io,json,runpy,tarfile,tempfile
P=Path(__file__).resolve().parent;N=runpy.run_path(str(P/'inventory.py'),run_name='generated');F=N['inventory_archive'];L={'members':2048,'member_name_bytes':1024,'single_member_bytes':1024**3,'uncompressed_bytes':64*1024**3,'wall_seconds':120}
results=[]
def make(items):
 raw=io.BytesIO()
 with tarfile.open(fileobj=raw,mode='w',format=tarfile.USTAR_FORMAT) as t:
  for name,body,kind in items:
   m=tarfile.TarInfo(name);m.size=len(body);m.type=kind;t.addfile(m,io.BytesIO(body) if kind==tarfile.REGTYPE else None)
 return gzip.compress(raw.getvalue())
def run(body,limits=L,sha=None):
 with tempfile.TemporaryFile() as f:
  f.write(body);f.seek(0)
  return F(f,expected_sha256=sha or hashlib.sha256(body).hexdigest(),expected_md5=hashlib.md5(body).hexdigest(),expected_bytes=len(body),limits=limits)
a=make([('subject/a.nii.gz',b'not-an-image',tarfile.REGTYPE)]);r=run(a);assert r['members']==[{'name':'subject/a.nii.gz','bytes':12,'type':'file'}] and r['extracted_files']==0
results.append({'case':'regular_names_sizes_only','status':'pass'})
for case,body,limits,expected,hash_override in [
 ('traversal',make([('../bad',b'x',tarfile.REGTYPE)]),L,'unsafe_or_overlong_member_name',None),
 ('duplicate',make([('same',b'x',tarfile.REGTYPE),('same',b'y',tarfile.REGTYPE)]),L,'duplicate_member_name',None),
 ('link',make([('link',b'',tarfile.SYMTYPE)]),L,'unsupported_link_special_or_sparse_member',None),
 ('member_cap',a,{**L,'members':0},'member_count_cap',None),
 ('decompression_cap',a,{**L,'uncompressed_bytes':100},'uncompressed_byte_cap',None),
 ('checksum',a,L,'original_checksum','0'*64),
 ('deadline',a,{**L,'wall_seconds':0},'inventory_wall_deadline',None),
 ('invalid_gzip',b'not gzip',L,None,None),
 ('crc_corrupt',a[:-5]+bytes([a[-5]^1])+a[-4:],L,None,None)]:
 try:run(body,limits,hash_override)
 except (ValueError,TimeoutError,OSError,EOFError,tarfile.TarError) as e:
  if expected:assert str(e)==expected,(case,str(e))
 else:raise AssertionError(case+' accepted')
 results.append({'case':case,'status':'pass'})
# A huge declared PAX extension must be refused before its body is requested.
m=tarfile.TarInfo('pax');m.type=tarfile.XHDTYPE;m.size=1024**3
body=gzip.compress(m.tobuf(format=tarfile.USTAR_FORMAT)+bytes(1024))
try:run(body)
except ValueError as e:assert str(e)=='unsupported_link_special_or_sparse_member'
else:raise AssertionError('PAX extension accepted')
results.append({'case':'pax_huge_declared_body_refused_before_read','status':'pass'})
raw=gzip.decompress(a);prefix=raw[:1024]
malformed_cases=[
 ('nonzero_trailer_in_read_ahead',raw[:2048]+b'TAIL'+raw[2052:],'nonzero_unparsed_tar_trailer'),
 ('malformed_later_header',prefix+b'x'*512+bytes(1024),'invalid_or_missing_tar_header:InvalidHeaderError'),
 ('truncated_later_header',prefix+b'x'*13,'invalid_or_missing_tar_header:TruncatedHeaderError'),
 ('missing_end_records',prefix,'invalid_or_missing_tar_header:EmptyHeaderError'),
 ('only_one_end_record',prefix+bytes(512),'missing_second_zero_end_record'),
 ('nonzero_second_end_record',prefix+bytes(512)+b'x'*512,'missing_second_zero_end_record')]
for case,rawbytes,expected in malformed_cases:
 try:run(gzip.compress(rawbytes))
 except ValueError as e:assert str(e)==expected,(case,str(e))
 else:raise AssertionError(case+' accepted')
 results.append({'case':case,'status':'pass'})
# Monkeypatch extraction and image-style access, retaining only metadata APIs.
old=tarfile.TarFile.extractfile;tarfile.TarFile.extractfile=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('extractfile forbidden'))
try:run(a)
finally:tarfile.TarFile.extractfile=old
results.append({'case':'extractfile_forbidden','status':'pass'})
receipt={'status':'pass','reader_sha256':hashlib.sha256((P/'inventory.py').read_bytes()).hexdigest(),'actual_NFBS_archive_opens':0,'generated_archive_controls':results}
(P/'generated-controls.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
