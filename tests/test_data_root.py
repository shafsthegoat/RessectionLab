import json
import plistlib
from pathlib import Path

import pytest

from resectionlab.data_root import DataRootError, atomic_json, checked_root, safe_path


def test_data_paths_cannot_escape_through_relative_or_symlink_paths(tmp_path):
    root = tmp_path / "data root with spaces"
    root.mkdir()
    assert safe_path(root, "cases/one.ressectionlab") == root / "cases/one.ressectionlab"
    with pytest.raises(DataRootError):
        safe_path(root, "../outside")
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(DataRootError):
        safe_path(root, "escape/target")


def test_missing_root_does_not_create_mount_or_fallback(tmp_path):
    absent = tmp_path / "missing external volume" / "data"
    with pytest.raises(DataRootError, match="unavailable"):
        checked_root(absent)
    assert not absent.parent.exists()


def test_volume_identity_and_free_space_are_checked_on_each_write(monkeypatch, tmp_path):
    import resectionlab.data_root as module
    info = {"volume_uuid": "expected", "total_bytes": 1000 * 1024**3, "free_bytes": 200 * 1024**3}
    monkeypatch.setattr(module, "volume_info", lambda root: info)
    atomic_json(tmp_path / module.MARKER, {"schema": "ressectionlab-data-root-v1", "volume_uuid": "expected"})
    assert checked_root(tmp_path, required_bytes=99 * 1024**3) == tmp_path
    with pytest.raises(DataRootError, match="reserve"):
        checked_root(tmp_path, required_bytes=101 * 1024**3)
    info["volume_uuid"] = "replacement"
    with pytest.raises(DataRootError, match="identity"):
        checked_root(tmp_path)


def test_live_uuid_never_hides_a_missing_or_reused_device_mount(monkeypatch,tmp_path):
    import resectionlab.data_root as module
    root=tmp_path/'data';root.mkdir()
    state={'mounted':True,'device':'/dev/disk9s1','uuid':'one','queries':0}
    monkeypatch.setattr(module.os.path,'ismount',lambda p:state['mounted'] and Path(p)==tmp_path)
    def output(command,**kwargs):
        if command[0]=='/sbin/mount':return f"{state['device']} on {tmp_path} (exfat, local)\n"
        pytest.fail('Storage verification must not invoke diskutil')
    def native(*_):
        state['queries']+=1
        return {'MountPoint':str(tmp_path),'Internal':False,'WritableVolume':True,'VolumeUUID':state['uuid']}
    monkeypatch.setattr(module,'_native_volume_info',native)
    monkeypatch.setattr(module.subprocess,'check_output',output)
    assert module.volume_info(root)['volume_uuid']=='one'
    assert module.volume_info(root)['volume_uuid']=='one' and state['queries']==2
    state.update(uuid='two') # Same device number and inode are not a UUID attestation.
    assert module.volume_info(root)['volume_uuid']=='two' and state['queries']==3
    state['mounted']=False
    with pytest.raises(DataRootError,match='not mounted'):
        module.volume_info(root)
