"""Display-only derivative validation; synthetic flags are not saved GUI evidence."""
from pathlib import Path
from zipfile import ZipFile
import sys,json,hashlib,copy,importlib.util
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'src'))
from resectionlab.workspace_bundle import encoded,session_digest,load_workspace
B=ROOT/'testcases/outputs/full-case-rehearsal-20261020';R=B/'review'
folder=R/'display-smoothing-validation-20261020';folder.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('display_helper',R/'apply_rehearsal_display.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
source=ROOT/'testcases/cases/meningioma-sub-06-standard-v2.ressectionlab'
native=B/'vascular-v2/vascular.ressectionlab';reference=B/'gui/display-reference-field.ressectionlab'
inputs={str(p):helper.sha(p) for p in (source,native,reference)}
assert inputs[str(source)]=='ff5a49b8706a8ccda08d8c3e82dad3e079f85dea98cb6bd40783d2262eec9838'

def derivative(original,output,value):
 with ZipFile(original) as src:
  manifest=json.loads(src.read('manifest.json'));doc=json.loads(src.read('workspace.json'))
  doc['imagingState']['states']['primary'].setdefault('viewer',{})['smoothSurfaces']=value
  doc['sessionHash']=session_digest(doc);workspace_bytes=encoded(doc)
  manifest['workspace_sha256']=hashlib.sha256(workspace_bytes).hexdigest()
  with ZipFile(output,'x') as dst:
   for member in src.infolist():
    payload=encoded(manifest) if member.filename=='manifest.json' else workspace_bytes if member.filename=='workspace.json' else src.read(member.filename)
    dst.writestr(member,payload)
 return output

def payload_check(original,derivative):
 with ZipFile(original) as a,ZipFile(derivative) as b:
  assert a.namelist()==b.namelist()
  ma=json.loads(a.read('manifest.json'));mb=json.loads(b.read('manifest.json'))
  assert ma['artifacts']['surgical_toolbox']==mb['artifacts']['surgical_toolbox']
  hashes={}
  for n in a.namelist():
   if n not in ('manifest.json','workspace.json'):
    data=a.read(n);assert data==b.read(n);hashes[n]=hashlib.sha256(data).hexdigest()
 return hashes

# The existing reference omits this field. Reuse its identical content for the
# omitted case; only distinct modified references need new derivative archives.
with ZipFile(reference) as z:ref_doc=json.loads(z.read('workspace.json'))
assert 'smoothSurfaces' not in ref_doc['imagingState']['states']['primary']['viewer']
target=derivative(native,folder/'target-explicit-true.ressectionlab',True)
ref_false=derivative(reference,folder/'synthetic-reference-false.ressectionlab',False)
ref_bad=derivative(reference,folder/'synthetic-reference-malformed.ressectionlab','false')
rows=[]
for label,ref,expected in [('explicit-false',ref_false,False),('omitted',reference,True)]:
 output=folder/(label+'-merged.ressectionlab')
 report=helper.apply(target,ref,output)
 case,artifacts,workspace=load_workspace(output)
 assert workspace['imagingState']['states']['primary']['viewer']['smoothSurfaces'] is expected
 merged='primary.viewer.smoothSurfaces' in report['displayFieldsMerged']
 assert merged is (label=='explicit-false')
 payloads=payload_check(native,output)
 assert report['surgicalToolboxEncodedArtifactUnchanged'] and report['physicsCalls']==0 and report['guiVerified'] is False
 rows.append({'case':label,'output':str(output),'smoothSurfaces':expected,'reportedSmoothFieldMerged':merged,
  'sourceArrayZipPayloadSha256':payloads,'encodedSurgicalToolboxPreserved':True,'physicsCalls':0})
output=folder/'malformed-must-not-publish.ressectionlab'
try:helper.apply(target,ref_bad,output)
except ValueError as error:message=str(error)
else:raise AssertionError('Malformed explicit smoothing flag was accepted')
assert not output.exists() and not output.with_suffix(output.suffix+'.display-provenance.json').exists()
assert 'surface' in message.lower() and ('flag' in message.lower() or 'boolean' in message.lower())
rows.append({'case':'malformed-string','rejected':True,'error':message,'outputAndProvenanceAbsent':True})
for p in (source,native,reference):assert helper.sha(p)==inputs[str(p)]
for original,deriv in ((native,target),(reference,ref_false),(reference,ref_bad)):payload_check(original,deriv)
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'passed','helperSha256':helper.sha(R/'apply_rehearsal_display.py'),
 'originalAndReferenceSha256Unchanged':inputs,'rows':rows,
 'scope':'Ignored display-only validation derivatives. False/malformed reference fields were synthetically inserted for testing, not saved from GUI. No original/reference/native file modified, no final artifact or GUI persistence claim, no SurgicalEngine construction/restore/preview/step/physics.'}
p=R/'display-smoothing-merge-validation-receipt.json';p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
