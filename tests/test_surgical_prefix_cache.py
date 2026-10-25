"""Generated authenticated prepared-prefix packages; no replay/engine actions."""
import copy
import hashlib
import hmac
import json
import os
from pathlib import Path
import shutil
from types import MappingProxyType
import zlib

import pytest

from resectionlab import surgical_prefix_cache as cache


def digest(value):
    return 'sha256:' + hashlib.sha256(value.encode()).hexdigest()


def binding():
    return {'schema':cache.BINDING_SCHEMA,'sceneId':digest('scene'),
            'originSnapshotHash':digest('snapshot'),'actionLogHash':digest('log'),
            'protectionContractHash':digest('protection'),
            'implementation':digest('implementation')[7:],'prefixImplementation':digest('prefix')[7:]}


def payload(index=0):
    before, after = digest('before-'+str(index)), digest('after-'+str(index))
    initial, final = zlib.compress(b'{"x":0}'), zlib.compress(b'{"x":1}')
    frames = [dict(stateId=state_id,state=state,kind=kind,observedTimeSeconds=float(number),
                   effectiveActivation={'left':False,'right':False},
                   fieldTokens=MappingProxyType({'flapComponents':'a'*64}))
              for number,state_id,state,kind in ((0,before,initial,'begin'),(1,after,final,'end'))]
    return {'actionHash':digest('action-'+str(index)), 'beforeStateId':before,'afterStateId':after,
            'receipt':(zlib.compress(b'{"status":"partial"}'),20,zlib.compress(b'{"context":true}')),
            'checkpoint':{'position':index+1,'stateId':after,'state':final},'frames':frames}


def publish(store, value=None, *, identity=None, index=0):
    return store.publish_action(binding() if identity is None else identity,index,
                                payload(index) if value is None else value,_authority=cache._PUBLISH_AUTHORITY)


def entry(store, identity=None, index=0):
    return store._entry(binding() if identity is None else identity,index)


def rewrite_manifest(store, mutate, *, domain=None):
    directory=entry(store);path=directory/'manifest.json'
    raw=json.loads(path.read_text());mutate(raw)
    encoded=json.dumps(raw,sort_keys=True,separators=(',',':')).encode('ascii')
    path.write_bytes(encoded)
    key=(store._namespace/'key').read_bytes()
    (directory/'manifest.mac').write_bytes(hmac.digest(key,(cache._DOMAIN if domain is None else domain)+encoded,'sha256').hex().encode())


def test_complete_roundtrip_detaches_maps_preserves_receipt_tuple_and_deduplicates(tmp_path):
    store=cache.PreparedPrefixStore(tmp_path)
    value=payload();assert publish(store,value)
    loaded=store.load_action(binding(),0)
    assert type(loaded['receipt']) is tuple and loaded['receipt']==value['receipt']
    assert loaded['checkpoint']==value['checkpoint']
    assert loaded['frames']==value['frames']
    assert type(loaded['frames'][0]['fieldTokens']) is dict
    # End frame and checkpoint share complete compressed bytes on disk.
    manifest=json.loads((entry(store)/'manifest.json').read_text())
    assert manifest['payload']['checkpoint']['state']==manifest['payload']['frames'][-1]['state']
    assert (entry(store)/'payload.bin').stat().st_size==sum(item['bytes'] for item in manifest['blobs'])
    loaded['frames'][0]['effectiveActivation']['left']=True
    loaded['frames'][0]['fieldTokens']['flapComponents']='b'*64
    loaded['checkpoint']['position']=123
    again=store.load_action(binding(),0)
    assert again['frames']==value['frames'] and again['checkpoint']==value['checkpoint']
    stats=store.statistics
    assert stats['hits']==2 and stats['publications']==1 and stats['lastSkip'] is None
    assert Path(stats['directory'])==tmp_path
    assert Path(stats['namespaceDirectory'])==tmp_path/'prepared-prefix-v1'
    assert stats['diskBytes']<=stats['maxDiskBytes']
    assert not hasattr(store,'restore') and not hasattr(store,'owner')


def test_private_authority_precedes_io_and_separate_namespaces(tmp_path):
    store=cache.PreparedPrefixStore(tmp_path)
    assert store.publish_action(binding(),0,payload(),_authority=object()) is False
    assert not tmp_path.joinpath('prepared-prefix-v1').exists()
    assert 'private prefix authority' in store.lastSkip
    # A canonical-looking key in the supplied base is never reused.
    (tmp_path/'key').write_bytes(b'legacy'*5+b'!!')
    assert publish(store)
    assert (store._namespace/'key').read_bytes()!=(tmp_path/'key').read_bytes()
    assert (tmp_path/'key').read_bytes()==b'legacy'*5+b'!!'


@pytest.mark.parametrize('field', ['sceneId','originSnapshotHash','actionLogHash','protectionContractHash',
                                 'implementation','prefixImplementation'])
def test_exact_binding_mismatch_cannot_load_even_at_copied_location(tmp_path,field):
    store=cache.PreparedPrefixStore(tmp_path);assert publish(store)
    changed=binding();changed[field]=digest('different-'+field)
    shutil.copytree(entry(store),entry(store,changed))
    assert store.load_action(changed,0) is None
    assert store.load_action(binding(),1) is None
    assert store.load_action(binding(),0) is not None


@pytest.mark.parametrize('damage', ['binary','manifest','signature','domain','ordinal','state-chain','part-bounds'])
def test_tampered_or_authenticated_wrong_domain_ordinal_bounds_miss(tmp_path,damage):
    store=cache.PreparedPrefixStore(tmp_path);assert publish(store)
    directory=entry(store)
    if damage=='binary':
        p=directory/'payload.bin';p.write_bytes(p.read_bytes()+b'x')
    if damage=='manifest':
        p=directory/'manifest.json';p.write_bytes(p.read_bytes()+b' ')
    if damage=='signature':(directory/'manifest.mac').write_bytes(b'0'*64)
    if damage=='domain':rewrite_manifest(store,lambda row:None,domain=b'resectionlab.observed-surgical-frames.v1\0')
    if damage=='ordinal':rewrite_manifest(store,lambda row:row.update(index=1))
    if damage=='state-chain':rewrite_manifest(store,lambda row:row['payload'].update(afterStateId=digest('poison')))
    if damage=='part-bounds':rewrite_manifest(store,lambda row:row['blobs'][0].update(bytes=2**62))
    assert store.load_action(binding(),0) is None
    assert store.statistics['misses']==1 and store.lastSkip


@pytest.mark.parametrize('damage', ['bool-index','extra-binding','wrong-schema','receipt-list','checkpoint',
                                 'binary-type','cycle','huge-string','nonfinite'])
def test_invalid_package_is_rejected_before_publication(tmp_path,damage):
    store=cache.PreparedPrefixStore(tmp_path);identity=binding();value=payload();index=0
    if damage=='bool-index':index=False
    if damage=='extra-binding':identity['canonicalAuthority']=True
    if damage=='wrong-schema':identity['schema']='resectionlab.verified-surgical-cache.v2'
    if damage=='receipt-list':value['receipt']=list(value['receipt'])
    if damage=='checkpoint':value['checkpoint']['position']=3
    if damage=='binary-type':value['checkpoint']['state']=bytearray(b'x')
    if damage=='cycle':value['frames'][0]['cycle']=value['frames'][0]
    if damage=='huge-string':value['frames'][0]['huge']='x'*(cache.MAX_MANIFEST_BYTES+1)
    if damage=='nonfinite':value['frames'][0]['observedTimeSeconds']=float('nan')
    assert store.publish_action(identity,index,value,_authority=cache._PUBLISH_AUTHORITY) is False
    assert store.lastSkip
    assert not store._namespace.exists()


def test_action_and_manifest_caps_and_hard_limits(tmp_path):
    value=payload();store=cache.PreparedPrefixStore(tmp_path/'small',max_action_bytes=1)
    assert publish(store,value) is False
    assert not store._namespace.exists()
    for kwargs in ({'max_action_bytes':cache.MAX_ACTION_BYTES+1},{'max_disk_bytes':cache.MAX_DISK_BYTES+1},
                   {'max_disk_bytes':True},{'max_action_bytes':0}):
        with pytest.raises(ValueError):cache.PreparedPrefixStore(tmp_path,**kwargs)
    store=cache.PreparedPrefixStore(tmp_path/'metadata')
    value['frames']=[{**value['frames'][0],'extra':'x'*2000} for _ in range(1000)]
    value['frames'][-1]={**value['frames'][-1],'stateId':value['afterStateId']}
    assert publish(store,value) is False
    assert not store._namespace.exists()


def test_pending_remnants_consume_quota_and_cannot_be_evicted_as_entries(tmp_path):
    store=cache.PreparedPrefixStore(tmp_path,max_disk_bytes=8192)
    assert publish(store)
    remnant=store._namespace/'.pending-abandoned';remnant.mkdir()
    (remnant/'partial.bin').write_bytes(b'x'*7900)
    assert store.statistics['diskBytes']>7900
    assert publish(store,index=1) is False
    assert (remnant/'partial.bin').stat().st_size==7900
    assert store.statistics['diskBytes']<=8192
    assert 'quota' in store.lastSkip.lower()


@pytest.mark.parametrize('where', ['ancestor','base','namespace','entry','pending','leaf','key','lock'])
def test_symlinks_are_rejected_without_touching_targets(tmp_path,monkeypatch,where):
    outside=tmp_path/'outside';outside.mkdir();marker=outside/'marker';marker.write_bytes(b'keep')
    base=tmp_path/'base';store=cache.PreparedPrefixStore(base)
    if where=='ancestor':
        link=tmp_path/'ancestor';link.symlink_to(outside,target_is_directory=True)
        store=cache.PreparedPrefixStore(link/'child')
    elif where=='base':base.symlink_to(outside,target_is_directory=True)
    elif where=='namespace':base.mkdir();store._namespace.symlink_to(outside,target_is_directory=True)
    elif where=='pending':
        base.mkdir();store._namespace.mkdir();link=store._namespace/'.pending-redirect'
        def redirected(**kwargs):
            link.symlink_to(outside,target_is_directory=True)
            return str(link)
        monkeypatch.setattr(cache.tempfile,'mkdtemp',redirected)
    else:
        assert publish(store)
        if where=='entry':
            shutil.rmtree(entry(store));entry(store).symlink_to(outside,target_is_directory=True)
        else:
            path=entry(store)/'payload.bin' if where=='leaf' else store._namespace/where
            path.unlink();path.symlink_to(marker)
    before_mode=outside.stat().st_mode
    if where in ('entry','leaf','key','lock'):assert store.load_action(binding(),0) is None
    else:assert publish(store) is False
    assert marker.read_bytes()==b'keep' and outside.stat().st_mode==before_mode


@pytest.mark.parametrize('kind', ['fifo','hardlink','oversize'])
def test_nonregular_linked_and_oversized_files_are_misses(tmp_path,kind):
    store=cache.PreparedPrefixStore(tmp_path);assert publish(store)
    path=entry(store)/'payload.bin';original=path.read_bytes();path.unlink()
    if kind=='fifo':os.mkfifo(path)
    if kind=='hardlink':
        target=tmp_path/'external.bin';target.write_bytes(original);os.link(target,path)
    if kind=='oversize':
        with path.open('wb') as stream:stream.truncate(cache.MAX_ACTION_BYTES+1)
    assert store.load_action(binding(),0) is None


@pytest.mark.parametrize('exception', [InterruptedError('interrupted'),OSError('rename failed'),MemoryError('allocation failed')])
def test_failed_staging_and_atomic_rename_leave_prior_action_intact(tmp_path,monkeypatch,exception):
    store=cache.PreparedPrefixStore(tmp_path);assert publish(store)
    original=store.load_action(binding(),0)
    with monkeypatch.context() as changes:
        if isinstance(exception,InterruptedError):
            changes.setattr(cache.os,'fsync',lambda fd:(_ for _ in ()).throw(exception))
        else:changes.setattr(cache.os,'replace',lambda *args:(_ for _ in ()).throw(exception))
        assert publish(store,index=1) is False
        assert not entry(store,index=1).exists()
        assert not list(store._namespace.glob('.pending-*'))
    assert store.load_action(binding(),0)==original


def test_explicit_cancellation_propagates_after_cleanup(tmp_path,monkeypatch):
    store=cache.PreparedPrefixStore(tmp_path)
    monkeypatch.setattr(cache.os,'replace',lambda *args:(_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):publish(store)
    assert not list(store._namespace.glob('.pending-*'))
    assert store.load_action(binding(),0) is None


def test_153_full_descriptors_fit_manifest_cap_with_independent_compressed_states(tmp_path):
    store=cache.PreparedPrefixStore(tmp_path);value=payload()
    first,last=value['frames']
    frames=[first]
    for index in range(1,152):
        frames.append({'stateId':digest('microstate-'+str(index)),
            'state':zlib.compress(('{"microstate":'+str(index)+'}').encode()),
            'kind':'microstep','observedTimeSeconds':index*.2,
            'effectiveActivation':{'left':False,'right':True},
            'fieldTokens':MappingProxyType({'flapComponents':'a'*64})})
    frames.append({**last,'observedTimeSeconds':30.4})
    value['frames']=frames
    assert len(frames)==153 and publish(store,value)
    loaded=store.load_action(binding(),0)
    assert loaded['frames']==frames and loaded['receipt']==value['receipt']
    assert (entry(store)/'manifest.json').stat().st_size<=cache.MAX_MANIFEST_BYTES


@pytest.mark.parametrize('failure', ['enter','body','exit'])
def test_plain_context_releases_underlying_lock_on_all_exception_boundaries(tmp_path,monkeypatch,failure):
    import fcntl
    store=cache.PreparedPrefixStore(tmp_path)
    actual=store._scan;calls=0
    def scan():
        nonlocal calls
        calls+=1
        if failure=='enter' and calls==2:raise OSError('entry validation failed')
        if failure=='exit' and calls==3:raise OSError('exit validation failed')
        return actual()
    with monkeypatch.context() as changes:
        changes.setattr(store,'_scan',scan)
        expected=KeyboardInterrupt if failure=='body' else OSError
        with pytest.raises(expected):
            with store._locked():
                if failure=='body':raise KeyboardInterrupt()
    fd=os.open(store._namespace/'lock',os.O_RDWR)
    try:
        # A leaked flock would fail immediately, rather than hanging a test.
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(fd,fcntl.LOCK_UN)
    finally:os.close(fd)
    assert publish(store)


def test_corrupt_existing_is_repaired_and_identical_valid_publication_is_idempotent(tmp_path):
    store=cache.PreparedPrefixStore(tmp_path);value=payload();assert publish(store,value)
    before=store.load_action(binding(),0)
    assert publish(store,value) and store.statistics['publications']==1
    path=entry(store)/'payload.bin';path.write_bytes(path.read_bytes()+b'corrupt')
    assert store.load_action(binding(),0) is None
    assert publish(store,value)
    assert store.load_action(binding(),0)==before
    assert store.statistics['publications']==2


def test_authenticated_different_payload_is_a_determinism_conflict(tmp_path):
    store=cache.PreparedPrefixStore(tmp_path);assert publish(store)
    before=store.load_action(binding(),0)
    different=payload();different['receipt']=(b'different-compressed-receipt',123,b'context')
    assert publish(store,different) is False
    assert 'determinism conflict' in store.lastSkip
    assert store.load_action(binding(),0)==before


def test_repair_reserves_both_packages_and_staging_failure_preserves_corrupt_entry(tmp_path,monkeypatch):
    store=cache.PreparedPrefixStore(tmp_path);value=payload();assert publish(store,value)
    path=entry(store)/'payload.bin';path.write_bytes(path.read_bytes()+b'corrupt')
    original={p.name:p.read_bytes() for p in entry(store).iterdir()}
    with monkeypatch.context() as changes:
        changes.setattr(cache.os,'fsync',lambda fd:(_ for _ in ()).throw(InterruptedError('stage failed')))
        assert publish(store,value) is False
    assert {p.name:p.read_bytes() for p in entry(store).iterdir()}==original
    # Existing corrupt bytes cannot be purged to make staging headroom.
    current=store.statistics['diskBytes']
    limited=cache.PreparedPrefixStore(tmp_path,max_disk_bytes=current+1)
    assert publish(limited,value) is False
    assert 'quota' in limited.lastSkip
    assert {p.name:p.read_bytes() for p in entry(store).iterdir()}==original
    assert publish(store,value) and store.load_action(binding(),0) is not None


def test_directory_environment_is_optional_and_explicit_base_wins(tmp_path,monkeypatch):
    configured=tmp_path/'configured';explicit=tmp_path/'explicit'
    monkeypatch.setenv(cache.DIRECTORY_ENV,str(configured))
    assert cache.PreparedPrefixStore().directory==configured
    assert cache.PreparedPrefixStore(explicit).directory==explicit
    monkeypatch.delenv(cache.DIRECTORY_ENV)
    assert cache.PreparedPrefixStore().directory==cache.DEFAULT_DIRECTORY


def test_busy_lock_is_prompt_optional_miss_for_load_and_publish_then_recovers(tmp_path):
    import fcntl
    import subprocess
    import sys
    store=cache.PreparedPrefixStore(tmp_path);store._ready()
    fd=os.open(store._namespace/'lock',os.O_RDWR)
    value=payload()
    for frame in value['frames']:frame['fieldTokens']=dict(frame['fieldTokens'])
    script=f'''
import time
from resectionlab import surgical_prefix_cache as cache
store=cache.PreparedPrefixStore({str(tmp_path)!r})
binding={binding()!r};value={value!r}
began=time.monotonic()
assert store.load_action(binding,0) is None
assert store.publish_action(binding,0,value,_authority=cache._PUBLISH_AUTHORITY) is False
assert store.lastSkip
assert time.monotonic()-began<1.,'Optional lock operation waited'
'''
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True,timeout=10)
        assert result.returncode==0,result.stdout+result.stderr
    finally:
        fcntl.flock(fd,fcntl.LOCK_UN);os.close(fd)
    assert publish(store) and store.load_action(binding(),0) is not None
