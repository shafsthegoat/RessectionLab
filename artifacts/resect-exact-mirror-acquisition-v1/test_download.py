import importlib.util
import hashlib
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('exact_mirror_download',HERE/'download.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_exact_scope_matches_original_authority_and_rights():
    rows,retained,rights=m.prepare()
    assert len(rows)==24 and sum(r['expected_bytes'] for r in rows)==1561749
    assert len({r['source_id'] for r in rows})==24
    assert all(r['role']=='TRAIN' and r['source_authority']['file_revision']==2 for r in rows)
    assert not any(r['source_id']=='Case3-during-mask' for r in rows)
    assert rights['noncommercial_only'] and not rights['commercial_clearance']

@pytest.mark.parametrize('change',['http','host','file','commit','auth','fragment','other_object'])
def test_route_refusals(change):
    row=m.prepare()[0][0];url=row['mirror_url']
    choices={'http':url.replace('https:','http:'),'host':url.replace('huggingface.co','example.com'),
        'file':url.replace('Case2-US-during-resection','Case4-US-during-resection'),
        'commit':url.replace(m.REVISION,'0'*40),'auth':url.replace('https://','https://user:secret@'),
        'fragment':url+'#secret','other_object':'https://cas-bridge.xethub.hf.co/xet-bridge-us/'+'1'*24+'/'+'0'*64}
    with pytest.raises(m.Refusal):m.route(row,choices[change],initial=change!='other_object')

def test_exact_content_object_redirect_and_sanitized_query():
    row=m.prepare()[0][0]
    url='https://cas-bridge.xethub.hf.co/xet-bridge-us/'+'1'*24+'/'+row['mirror_xet_hash']+'?Signature=PRIVATE&Expires=1'
    value=m.route(row,url)
    assert value['path'].endswith(row['mirror_xet_hash'])
    assert 'PRIVATE' not in json.dumps(value)
    assert value['query_field_names']==['Expires','Signature']

def test_real_cached_pilot_exact_body_verification():
    manifest=json.loads((m.ROOT/'manifests/resect-train-cavity-acquisition-v1.json').read_text())
    source=next(s for s in manifest['sources'] if s['id']=='Case3-during-mask')
    row={'expected_bytes':source['bytes'],'sha256':source['sha256'],'md5':source['expected_md5']}
    path=m.ROOT/'data/annotations/resect-seg-v1'/source['path']
    verified=m.verify(path,row)
    assert verified['sha256']==source['sha256']
    row['sha256']='0'*64
    with pytest.raises(m.Refusal,match='fixity'):m.verify(path,row)

def test_bounds_before_transport():
    with pytest.raises(m.Deadline):m.check({'deadline':0,'bytes_read':0,'requests':0})
    import time
    with pytest.raises(m.Refusal):m.check({'deadline':time.monotonic()+10,'bytes_read':m.CAP_BYTES+1,'requests':0})
    with pytest.raises(m.Refusal):m.check({'deadline':time.monotonic()+10,'bytes_read':0,'requests':m.CAP_REQUESTS})
