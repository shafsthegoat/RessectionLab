"""Independent process/receipt controls; zero scientific payload reads."""
import copy,importlib,json,os,pathlib,sys,tempfile,time
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
qc=importlib.import_module('lausanne_deferred_qc')
@pytest.fixture(scope='module')
def metadata():return qc.preflight()
@pytest.fixture(autouse=True)
def prevent_payloads(monkeypatch):
 def forbidden(*a,**kw):raise AssertionError('Independent controls forbid scientific payloads/network')
 monkeypatch.setattr(qc.gzip,'open',forbidden)
 monkeypatch.setattr(qc,'inspect_original',forbidden)
 monkeypatch.setattr(qc,'verify_source_file',forbidden)
 monkeypatch.setattr(qc.annotations,'open_without_redirect',forbidden)
 original=pathlib.Path.open
 def opened(p,*a,**kw):
  if p.name.endswith(('.nii','.nii.gz','.nii.gz.partial')):forbidden()
  return original(p,*a,**kw)
 monkeypatch.setattr(pathlib.Path,'open',opened)

def test_parent_rejects_completed_worker_accounted_after_batch_deadline(metadata,monkeypatch):
 m,am,sessions=metadata;row=m['originals'][0]
 with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='deferred-parent-boundary-') as folder:
  cache=pathlib.Path(folder);offset=[0];clock=time.monotonic
  monkeypatch.setattr(qc,'CACHE',cache)
  monkeypatch.setattr(qc,'preflight',lambda:metadata)
  monkeypatch.setattr(qc,'execution_source',lambda m:{'files':{}})
  monkeypatch.setattr(qc,'validate_execution',lambda *a,**kw:None)
  monkeypatch.setattr(qc.time,'monotonic',lambda:clock()+offset[0])
  def supervisor(command,log,**kw):
   kw['on_start'](123)
   trial=log.parent;intent=qc.bound_json(trial/'intent.json')
   # Process-state control receipt: no scientific positive fields.
   qc.save(trial/'receipt.json',{'schema':'lausanne-file-review-v1','manifest_sha256':qc.MANIFEST_SHA,'path':row['path'],'subject':row['subject'],'session':row['session'],'role':'TRAIN','intent_sha256':qc.digest(qc.encode(intent)),'declaration_sha256':intent['declaration_sha256'],'status':'review_failed',**qc.CLAIMS})
   offset[0]=31
   return 'completed',0
  monkeypatch.setattr(qc,'supervise',supervisor)
  report=qc.batch('metadata-deadline-control','originals',30,only_path=row['path'])
  assert report['elapsed_seconds']>=30
  assert report['status']=='failed_or_incomplete',report
  item=next(r for r in report['outcomes'] if r['path']==row['path'])
  assert item['status']!='completed'
