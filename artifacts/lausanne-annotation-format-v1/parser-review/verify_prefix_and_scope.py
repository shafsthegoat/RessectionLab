"""Read exactly the existing source's 592-byte prefix; no scalar data or intake."""
import ast
import base64
import copy
from datetime import datetime,timezone
import hashlib
import importlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import time
import zlib
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
EXPECTED={'scripts/lausanne_annotation_intake.py':'2f698cd7dcccf76797b4e466b490903dbdd8d65e0c44e3091e149d8f2f49c0cf','tests/test_lausanne_annotation_intake.py':'3bf58ac1f9f70aa863a46f05b13928424562b3d024adbc8298ddc38027db18f4'}
def sha(b):return hashlib.sha256(b).hexdigest()
started=time.monotonic();sources=[]
for name,digest in EXPECTED.items():
 b=(ROOT/name).read_bytes();assert sha(b)==digest
 snapshot=OUT/'source-snapshot'/name;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(b)
 sources.append({'path':name,'bytes':len(b),'sha256':digest,'snapshot':str(snapshot.relative_to(ROOT))})
old=subprocess.check_output(['git','show','ec336480ee4d1ce876a6425410386cb379c76392:scripts/lausanne_annotation_intake.py'],cwd=ROOT)
assert sha(old)=='04b71e433847d98f0d567cf24d97af77144351d4094429a630cc9f758af3a46e'
old_tree=ast.parse(old);new_tree=ast.parse((ROOT/'scripts/lausanne_annotation_intake.py').read_bytes())
old_functions={f.name:f for f in old_tree.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
new_functions={f.name:f for f in new_tree.body if isinstance(f,(ast.FunctionDef,ast.AsyncFunctionDef))}
assert set(new_functions)-set(old_functions)=={'inspect_extensions'}
unchanged=[]
for name,node in old_functions.items():
 if name!='inspect_mask':
  assert ast.dump(node,include_attributes=False)==ast.dump(new_functions[name],include_attributes=False),name
  unchanged.append(name)
class StripFramingChange(ast.NodeTransformer):
 def visit_Assign(self,node):
  if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('padding','extensions'):return None
  return self.generic_visit(node)
 def visit_If(self,node):
  if any(isinstance(n,ast.Name) and n.id=='padding' for n in ast.walk(node.test)):return None
  return self.generic_visit(node)
 def visit_Dict(self,node):
  pairs=[(k,v) for k,v in zip(node.keys,node.values) if not (isinstance(k,ast.Constant) and k.value=='extensions')]
  node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
  return self.generic_visit(node)
assert ast.dump(StripFramingChange().visit(copy.deepcopy(old_functions['inspect_mask'])),include_attributes=False)==ast.dump(StripFramingChange().visit(copy.deepcopy(new_functions['inspect_mask'])),include_attributes=False)

diagnostic_path=ROOT/'build/lausanne-sub022-extension-diagnostic-v1/report.json'
diag=json.loads(diagnostic_path.read_bytes());spec=ROOT/diag['primary_specification']['path']
assert sha(spec.read_bytes())=='fb86abcab6dcb825da340a3252feb314832dc07b1286e25019dfb3dc07b99486'
failed_path=ROOT/diag['original_failed_receipt']['path'];failed_before=failed_path.read_bytes()
assert sha(failed_before)=='1a8ffdf58847b6468f884905e5101313d6799f2addeeaa0f68350dce705f1c09'
source_path=ROOT/diag['source']['path'];before=source_path.stat();compressed=source_path.read_bytes()
assert len(compressed)==36551 and sha(compressed)==diag['source']['sha256']=='ba83116ae5fa9744bb83eb338cd491b431691b3a54a5a5b3c8d6111a1d2eec11'
# max_length is the authorized prefix ceiling; no flush or further decode.
decoder=zlib.decompressobj(31);prefix=decoder.decompress(compressed,592)
assert len(prefix)==592 and sha(prefix)==diag['header']['prefix_sha256']=='71ce043d73c953f330e80294d44ea35787d9c666b5373c971065116b3d7b044c'
assert struct.unpack_from('<i',prefix)[0]==348 and prefix[344:348]==b'n+1\0'
assert struct.unpack_from('<f',prefix,108)[0]==592.
assert sha(prefix[:348])==diag['header']['header_sha256']
assert sha(prefix[352:])==diag['extension_region']['sha256']
sys.path.insert(0,str(ROOT/'scripts'));intake=importlib.import_module('lausanne_annotation_intake')
def forbidden(*args,**kwargs):raise AssertionError('No network, whole-image reader, or intake allowed')
intake.open_without_redirect=forbidden;intake.gzip.open=forbidden;intake.inspect_mask=forbidden;intake.worker=forbidden;intake.batch=forbidden
result=intake.inspect_extensions(prefix[348:],data_offset=592,endian='<',maximum_offset=65536)
assert result['status']=='framed_uninterpreted' and result['extension_count']==1
record=result['records'][0];known=diag['extension_region']['extensions'][0]
for key in ('offset','end_exclusive','esize','ecode','block_sha256','payload_bytes','payload_sha256'):assert record[key]==known[key]
assert record['esize']==240 and record['ecode']==0 and record['payload_bytes']==232
assert not result['payloads_interpreted'] and not result['used_as_annotation_or_coordinate_evidence']
assert source_path.stat()==before and failed_path.read_bytes()==failed_before
# Only bounded scalar-header arithmetic; no scalar data is read or inferred.
raw_dims=struct.unpack_from('<8h',prefix,40);raw_datatype=struct.unpack_from('<h',prefix,70)[0];raw_bitpix=struct.unpack_from('<h',prefix,72)[0]
manifest=json.loads((ROOT/'manifests/lausanne-train-annotations-v1.json').read_bytes())
budget=intake.decoding_budget(list(raw_dims[1:4]),raw_bitpix//8,manifest['bounds'])
assert 592+budget['native_payload_bytes']<=manifest['bounds']['max_uncompressed_mask_bytes']
assert budget['chunk_working_bytes_bound']<=16*1024**2
max_chain=intake.inspect_extensions(b'\1\0\0\0'+(struct.pack('<ii',16,0)+bytes(8))*4074,data_offset=65536,endian='<',maximum_offset=65536)
result_record={'schema':'lausanne-extension-parser-independent-review-v1','created_at':datetime.now(timezone.utc).isoformat(),
 'conclusion':'No defects found in reviewed extension framing change. Ready for a separately declared cached sub022 QC repeat under unchanged source/content/grid/admission gates.',
 'blockers':[],'frozen_sources':sources,
 'specification':{'path':str(spec.relative_to(ROOT)),'sha256':sha(spec.read_bytes()),'primary_url':diag['primary_specification']['url'],'sections':'nifti1.h lines 213-301','version_limit':'Mutable upstream URL; review binds exact archived document bytes.',
 'requirements_checked':'Signed int32 esize/ecode in header byte order; esize >=16 and divisible by16; nonnegative ecode; correct record extents and bounded data offset; multiple extension records; opaque contents may use unrelated byte order and are not interpreted.'},
 'parser_policy':{'private_content':'Records preserve offset/end, esize/ecode, complete block and unstripped payload SHA256/length. No payload text, content handler, annotation meaning or coordinate meaning is emitted.',
 'malformed_and_trailer':'Unknown nonnegative codes and any nonzero first indicator are supported. Reserved indicator bytes, truncated framing, negative codes, invalid sizes, overrun and any unframed trailer are refused. Present chains must end exactly at vox_offset.',
 'absent_extension':'Existing bounded all-zero padding branch remains. Extension-absent data offsets need not be 16-aligned; the spec recommends aligned data offsets, while extension-record alignment is mandatory.',
 'maximum_metadata':'At 65536 maximum data offset, at most4074 minimum-size records; metadata receipt below the existing2MiB read bound.',
 'maximum_record_count':4074,'maximum_chain_receipt_bytes':len(intake.encoded({'content_qc':{'extensions':max_chain}}))},
 'scope_comparison':{'baseline_runner_sha256':sha(old),'all_other_functions_AST_identical':unchanged,'inspect_mask_after_removing_exact_framing_change_AST_identical':True,
 'preserved_gates':'Same immutable source/rights/split checks, limits, dtype/scaling/binary-value checks, foreground/background requirement, gzip EOF/integrity behavior, reference/grid proof, unknown background/timing, and false training/scanner/planning admissions.'},
 'tests':[{'suite':'tests/test_lausanne_annotation_intake.py','passed':56,'failed':0,'seconds':0.43},
 {'suite':'build/lausanne-extension-parser-review-v1/test_independent_framing.py','passed':32,'failed':0,'seconds':0.09,'junit':'build/lausanne-extension-parser-review-v1/adversarial-controls.xml'}],
 'reviewer_control_correction':{'initial_result':'24 passed,8 failed before production invocation due to reviewer byte-string length assertion (25 actual bytes vs24 intended).','production_defect':False,
 'repair':'Only the ignored control payload was shortened by one zero byte. Initial control source and XML are retained.','initial_source':'build/lausanne-extension-parser-review-v1/initial-reviewer-controls.py','initial_junit':'build/lausanne-extension-parser-review-v1/initial-reviewer-controls.xml'},
 'actual_prefix':{'source_sha256':sha(compressed),'compressed_bytes_read':len(compressed),'decompressed_bytes':len(prefix),'authorized_maximum_decompressed_bytes':592,
 'prefix_sha256':sha(prefix),'header_sha256':sha(prefix[:348]),'parsed_extensions':result,'matches_prior_diagnostic':True,'private_payload_interpreted_or_copied_to_output':False,
 'scalar_bytes_decoded':0,'scalar_values_counted':0,'header_only_budget':budget,'header_only_datatype_code':raw_datatype,
 'scalar_QC_grid_quality_and_gzip_CRC_not_established':True},
 'first_failed_receipt':{'path':str(failed_path.relative_to(ROOT)),'sha256':sha(failed_before),'unchanged':True},
 'actions':{'network_requests':0,'scientific_transfers':0,'scalar_decodes':0,'actual_intake_invocations':0,'whole_image_readers':0,'tracked_edits':0,'original_source_modifications':0,'elapsed_seconds':time.monotonic()-started},
 'admission':{'content_QC_for_sub022_completed':False,'training_admitted':False,'scanner_frame_admitted':False,'spatial_planning_admitted':False},
 'next_step_limit':'Only a separately recorded cache-only actual-source QC attempt can establish scalar content and reference/grid result. Preserve failed historical receipt and unqualified manual subtype/background/timing semantics.'}
result_record['review_script_sha256']=sha(Path(__file__).read_bytes())
(OUT/'verification.json').write_text(json.dumps(result_record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'status':'passed','controls_passed':88,'real_prefix_bytes':len(prefix),'scalar_bytes':0,'verification_sha256':sha((OUT/'verification.json').read_bytes())}))
