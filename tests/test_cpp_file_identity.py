"""Native source identity checks preserve stat fields and failure behavior."""
import os
import errno

import pytest


@pytest.fixture
def identities():
    return pytest.importorskip("resectionlab._geometry_cpp").FileIdentities


def stamp(path):
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def test_live_binding_changes_fail_closed():
    from types import ModuleType
    native=pytest.importorskip('resectionlab._geometry_cpp')
    module=ModuleType('generated_binding_control')
    original=object(); module.entry=original
    expected=(('entry',original),)
    assert native.bindings_unchanged(module,expected)
    module.entry=object()
    assert not native.bindings_unchanged(module,expected)
    del module.entry
    assert not native.bindings_unchanged(module,expected)


def test_file_guard_constructor_changes_fail_closed(monkeypatch):
    from resectionlab import native_backend as backend
    if backend._module is None:
        pytest.skip('Compiled backend required')
    original=backend._module.FileIdentities.__init__
    monkeypatch.setattr(backend._module.FileIdentities,'__init__',lambda self,*args:original(self,[],[]))
    with pytest.raises(backend.NativeIntegrityError,match='file guard changed'):
        backend._check_live()


def test_changes_are_ordered_and_baseline_is_immutable(identities, tmp_path):
    paths = [tmp_path / str(index) for index in range(3)]
    for path in paths:
        path.write_bytes(b"original")
    before = [stamp(path) for path in paths]
    tracked = identities([str(path) for path in paths], before)
    assert tracked.changed() == []
    paths[2].write_bytes(b"new contents")
    os.utime(paths[0], ns=(before[0][1], before[0][1] + 1_000_000))
    assert paths[0].read_bytes() == b"original"
    assert tracked.changed() == [0, 2]
    assert tracked.changed() == [0, 2]
    assert identities([str(path) for path in paths], [stamp(path) for path in paths]).changed() == []


def test_replaced_same_bytes_and_missing_file(identities, tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"same")
    before = stamp(path)
    tracked = identities([str(path)], [before])
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"same")
    os.utime(replacement, ns=(before[1], before[1]))
    replacement.replace(path)
    assert tracked.changed() == [0]
    path.unlink()
    with pytest.raises(FileNotFoundError) as missing:
        tracked.changed()
    assert missing.value.filename == str(path)


def test_signed_and_large_nanoseconds_do_not_overflow(identities, tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"x")
    original = stamp(path)
    for nanos in [-1, -1_000_000_001, 2**64, -(2**64)]:
        altered = (original[0], nanos, nanos, original[3])
        assert identities([str(path)], [altered]).changed() == [0]


def test_filesystem_surrogate_path(identities, tmp_path):
    if os.name != "posix":
        pytest.skip("POSIX filesystem bytes")
    path = tmp_path / os.fsdecode(b"source-\xff")
    try:
        path.write_bytes(b"x")
    except OSError as unavailable:
        if unavailable.errno != errno.EILSEQ:
            raise
        with pytest.raises(OSError) as reference:
            path.stat()
        with pytest.raises(OSError) as error:
            identities([str(path)], [(0, 0, 0, 0)]).changed()
        assert error.value.errno == reference.value.errno
        assert error.value.filename == str(path)
    else:
        assert identities([str(path)], [stamp(path)]).changed() == []


@pytest.mark.parametrize("paths,stamps", [
    (("x",), [(0, 0, 0, 0)]), (["x"], ((0, 0, 0, 0),)),
    (["x"], []), (["x"], [[0, 0, 0, 0]]),
    (["x"], [(0, 0, 0)]), (["x"], [(False, 0, 0, 0)]),
    (["x"], [(0, 0.0, 0, 0)]), ([object()], [(0, 0, 0, 0)]),
])
def test_rejects_noncanonical_metadata(identities, paths, stamps):
    with pytest.raises(TypeError):
        identities(paths, stamps)


def test_rejects_subclasses_without_hooks(identities):
    class CustomString(str):
        def __str__(self):
            raise AssertionError("custom string conversion")

    class CustomInt(int):
        def __index__(self):
            raise AssertionError("custom integer conversion")

    with pytest.raises(TypeError):
        identities([CustomString("x")], [(0, 0, 0, 0)])
    with pytest.raises(TypeError):
        identities(["x"], [(CustomInt(0), 0, 0, 0)])
    with pytest.raises(ValueError, match="embedded null"):
        identities(["x\0"], [(0, 0, 0, 0)])
