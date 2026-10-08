"""Metadata-only interruption evidence; no positive scientific receipt."""
import json,pathlib,sys,tempfile
import pytest
ROOT=pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import lausanne_deferred_qc as qc

@pytest.mark.parametrize('failure',[KeyboardInterrupt,OSError])
def test_interruption_binds_existing_failed_worker_receipt(monkeypatch,failure):
 metadata=qc.preflight();m,am,sessions=metadata;row=m['originals'][0]
 with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='deferred-interruption-control-') as folder:
  cache=pathlib.Path(folder)
  monkeypatch.setattr(qc,'CACHE',cache)
  monkeypatch.setattr(qc,'preflight',lambda:metadata)
  monkeypatch.setattr(qc,'execution_source',lambda m:{'files':{}})
  monkeypatch.setattr(qc,'validate_execution',lambda *a,**kw:None)
  def supervisor(command,log,**kw):
   kw['on_start'](123)
   trial=log.parent;intent=qc.bound_json(trial/'intent.json')
   qc.save(trial/'receipt.json',{'schema':'lausanne-file-review-v1','manifest_sha256':qc.MANIFEST_SHA,'path':row['path'],'subject':row['subject'],'session':row['session'],'role':'TRAIN','intent_sha256':qc.digest(qc.encode(intent)),'declaration_sha256':intent['declaration_sha256'],'status':'failed_or_incomplete',**qc.CLAIMS})
   raise failure('metadata-only interruption')
  monkeypatch.setattr(qc,'supervise',supervisor)
  report=qc.batch('metadata-interruption-control','originals',30,only_path=row['path'])
  item=next(r for r in report['outcomes'] if r['path']==row['path'])
  assert report['status']=='failed_or_incomplete' and item['status']=='failed_or_incomplete'
  path=cache/'runs/metadata-interruption-control/items'/qc.review_key(row['path'])/'receipt.json'
  assert path.exists()
  assert item.get('receipt_sha256')==qc.digest(path.read_bytes()),item
  assert item.get('review_status')=='failed_or_incomplete'
