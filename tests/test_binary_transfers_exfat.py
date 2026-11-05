import os
import errno
from pathlib import Path

import numpy as np
import pytest

from resectionlab.desktop_bridge import BinaryTransfers


def test_cleanup_handles_companion_disappearing_when_its_data_file_is_removed(tmp_path, monkeypatch):
    transfers=BinaryTransfers(tmp_path)
    item=transfers.array(np.arange(8,dtype=np.float32),'float32')
    source=Path(item['path']);companion=source.with_name('._'+source.name)
    companion.write_bytes(b'AppleDouble metadata')
    unlink=os.unlink
    def macos_unlink(path,*args,**kwargs):
        result=unlink(path,*args,**kwargs)
        if Path(path).name==source.name:
            unlink(companion)
        return result
    monkeypatch.setattr(os,'unlink',macos_unlink)
    transfers.close()
    assert not transfers.root.exists()
    transfers.close()


def test_cleanup_keeps_unexpected_filesystem_errors_visible(tmp_path, monkeypatch):
    transfers=BinaryTransfers(tmp_path)
    transfers.array(np.arange(8,dtype=np.float32),'float32')
    def denied(*args,**kwargs):
        raise PermissionError(errno.EACCES,'external volume became read-only')
    monkeypatch.setattr(os,'unlink',denied)
    with pytest.raises(PermissionError,match='read-only'):
        transfers.close()
