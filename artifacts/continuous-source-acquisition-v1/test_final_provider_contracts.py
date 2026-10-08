"""Independent pure HTTP timing controls; no image creation or network."""
import importlib
from pathlib import Path
import sys
import time
import io
from email.message import Message
from email.utils import formatdate
from urllib.error import HTTPError
from urllib.request import Request
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
q=importlib.import_module('continuous_source_acquisition')

@pytest.mark.parametrize('status',[403,404,409,413,429,503])
@pytest.mark.parametrize('kind',['numeric','date'])
def test_retry_after_holds_provider_even_if_object_terminal(status,kind):
 now=1000
 value='7200' if kind=='numeric' else formatdate(8200,usegmt=True)
 event={'status':status,'observed_utc_epoch':now,'retry_after_utc_epoch':q.retry_after(value,now)}
 error=HTTPError('https://osf.io/control',status,'control',Message(),io.BytesIO())
 outcome={'status':q.failure_kind(error,event),'provider':'osf','network_attempted':True,'http_failure':event}
 q.apply_backoff(outcome,[],{},now)
 providers=q.provider_state({'control':[outcome]})
 assert providers.get('osf',{}).get('not_before',0)>=8200

@pytest.mark.parametrize('status',[301,302,303,307,308])
def test_redirect_retry_after_requires_cooldown_before_next_request(status,tmp_path):
 # A transport-only HTTP response. No payload or scientific record.
 headers=Message();headers['Retry-After']='7200';headers['Location']='https://osf.io/control-next'
 def redirect(request,**kwargs):
  raise HTTPError(request.full_url,status,'control',headers,io.BytesIO())
 # Metadata writer requires ignored path under project root.
 import tempfile
 with tempfile.TemporaryDirectory(dir=ROOT/'build',prefix='http-redirect-control-') as name:
  opener=q.ObservedOpener(redirect,Path(name),time.monotonic()+10)
  with pytest.raises(q.Refusal, match='redirect_retry_after_deferred'):
   opener.open(Request('https://osf.io/control'),timeout=5)
  assert opener.failure is not None and opener.failure['retry_after_utc_epoch']>=time.time()+7198
