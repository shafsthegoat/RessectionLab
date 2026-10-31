"""Guarded local data storage; never create a missing volume mount point."""
from __future__ import annotations

import hashlib
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT_ENV = "RESSECTIONLAB_DATA_ROOT"
MARKER = ".ressectionlab-data-root.json"
LAYOUT = ("sources", "prepared", "cases/real", "cases/synthetic", "cases/controls",
          "catalog", "qc", "quarantine", "archive", "staging", "cache")


class DataRootError(ValueError):
    pass


def configured_root() -> Path:
    value = os.environ.get(ROOT_ENV)
    if not value or not Path(value).is_absolute():
        raise DataRootError(f"Set {ROOT_ENV} to the mounted data directory.")
    return Path(value)


def _native_volume_info(mount,device):
    """Read current Disk Arbitration properties and the filesystem's own UUID.

    diskutil additionally probes filesystem statistics and can stall under
    ExFAT writes. These read-only system APIs avoid that unnecessary probe.
    """
    pointer=ctypes.c_void_p
    cf=ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    da=ctypes.CDLL('/System/Library/Frameworks/DiskArbitration.framework/DiskArbitration')
    def bind(library,name,args,result):
        function=getattr(library,name);function.argtypes=args;function.restype=result;return function
    release=bind(cf,'CFRelease',[pointer],None)
    session=disk=description=None
    try:
        session=bind(da,'DASessionCreate',[pointer],pointer)(None)
        disk=bind(da,'DADiskCreateFromBSDName',[pointer,pointer,ctypes.c_char_p],pointer)(None,session,os.fsencode(device))
        if not session or not disk:raise DataRootError('Cannot inspect the mounted device.')
        description=bind(da,'DADiskCopyDescription',[pointer],pointer)(disk)
        if not description:raise DataRootError('Mounted device description is unavailable.')
        get=bind(cf,'CFDictionaryGetValue',[pointer,pointer],pointer)
        def value(name):
            result=get(description,pointer.in_dll(da,'kDADiskDescription'+name+'Key'))
            if not result:raise DataRootError('Mounted device property is unavailable: '+name)
            return result
        boolean=bind(cf,'CFBooleanGetValue',[pointer],ctypes.c_bool)
        def string(ref):
            buffer=ctypes.create_string_buffer(4096)
            if not bind(cf,'CFStringGetCString',[pointer,ctypes.c_char_p,ctypes.c_long,ctypes.c_uint32],ctypes.c_bool)(ref,buffer,len(buffer),0x08000100):
                raise DataRootError('Cannot decode mounted device property.')
            return buffer.value.decode('utf8')
        actual_device=string(value('MediaBSDName'))
        if '/dev/'+actual_device!=device:raise DataRootError('Mounted device changed during verification.')
        uuid_ref=bind(cf,'CFUUIDCreateString',[pointer,pointer],pointer)(None,value('VolumeUUID'))
        try:volume_uuid=string(uuid_ref)
        finally:release(uuid_ref)
        path_ref=bind(cf,'CFURLCopyFileSystemPath',[pointer,ctypes.c_long],pointer)(value('VolumePath'),0)
        try:actual_mount=string(path_ref)
        finally:release(path_ref)
        class Attributes(ctypes.Structure):
            _fields_=[('count',ctypes.c_uint16),('reserved',ctypes.c_uint16),('common',ctypes.c_uint32),
                      ('volume',ctypes.c_uint32),('directory',ctypes.c_uint32),('file',ctypes.c_uint32),('fork',ctypes.c_uint32)]
        attributes=Attributes(5,0,0,0x80040000,0,0,0) # ATTR_VOL_INFO | ATTR_VOL_UUID; sys/attr.h
        buffer=ctypes.create_string_buffer(20)
        getattrlist=bind(ctypes.CDLL(None,use_errno=True),'getattrlist',
            [ctypes.c_char_p,ctypes.POINTER(Attributes),pointer,ctypes.c_size_t,ctypes.c_ulong],ctypes.c_int)
        if getattrlist(os.fsencode(mount),ctypes.byref(attributes),buffer,len(buffer),0)!=0 or int.from_bytes(buffer.raw[:4],sys.byteorder)!=20:
            raise DataRootError('Cannot verify the live filesystem UUID.')
        if str(uuid.UUID(bytes=buffer.raw[4:20])).upper()!=volume_uuid.upper() or actual_mount!=str(mount):
            raise DataRootError('Live filesystem identity differs from the mounted-device description.')
        return {'MountPoint':actual_mount,'Internal':boolean(value('DeviceInternal')),
            'WritableVolume':boolean(value('MediaWritable')) and os.access(mount,os.W_OK),
            'VolumeUUID':volume_uuid,'DeviceIdentifier':actual_device,'FilesystemType':string(value('VolumeKind'))}
    finally:
        for reference in [description,disk,session]:
            if reference:release(reference)


def volume_info(root: Path) -> dict:
    """Inspect the existing mount without mkdir or an internal-disk fallback."""
    root = Path(root)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise DataRootError("External data directory is unavailable. Connect its volume.")
    if sys.platform != "darwin":
        raise DataRootError("External-volume verification currently requires macOS.")
    mount_candidate = root.resolve()
    while mount_candidate.parent != mount_candidate and not os.path.ismount(mount_candidate):
        mount_candidate = mount_candidate.parent
    # Query the currently mounted device directly. Resolving an ExFAT pathname
    # through Disk Arbitration can stall during heavy writes; the kernel mount
    # table supplies its current device without retaining an old device number.
    device = str(mount_candidate)
    mounts = subprocess.check_output(["/sbin/mount"], text=True, timeout=5)
    for line in mounts.splitlines():
        if " on " in line:
            source, location = line.split(" on ", 1)
            if location.split(" (", 1)[0] == str(mount_candidate) and source.startswith("/dev/"):
                device = source
                break
    # Check the live mount and UUID on every call, including device-number reuse.
    root_stat, mount_stat = root.stat(), mount_candidate.stat()
    if mount_candidate == Path('/') or not os.path.ismount(mount_candidate) or not device.startswith('/dev/'):
        raise DataRootError('The configured external volume is not mounted.')
    info=_native_volume_info(mount_candidate,device)
    mount = Path(info.get("MountPoint", "/"))
    if (not info.get("MountPoint") or info.get("Internal", True) or mount == Path("/")
            or not os.path.ismount(mount) or not root.resolve().is_relative_to(mount)
            or root.stat().st_dev != mount.stat().st_dev
            or not info.get("WritableVolume") or not info.get("VolumeUUID")):
        raise DataRootError("The data root must be on a mounted, writable external volume.")
    usage = shutil.disk_usage(root)
    return {"volume_uuid": info["VolumeUUID"], "mount_point": str(mount),
            "device": info.get("DeviceIdentifier"), "filesystem": info.get("FilesystemType"),
            "total_bytes": usage.total, "free_bytes": usage.free}


def checked_root(root=None, *, required_bytes=0, expected_uuid=None, reserve_bytes=None) -> Path:
    root = configured_root() if root is None else Path(root)
    info = volume_info(root)
    if expected_uuid and info["volume_uuid"] != expected_uuid:
        raise DataRootError("A different external volume is mounted at the configured path.")
    marker = root / MARKER
    if not marker.is_file() or marker.is_symlink() or marker.stat().st_size > 16384:
        raise DataRootError("The external data-root identity marker is missing or invalid.")
    document = json.loads(marker.read_text())
    if document.get("schema") != "ressectionlab-data-root-v1" or document.get("volume_uuid") != info["volume_uuid"]:
        raise DataRootError("External volume identity differs from the data-root record.")
    reserve = max(50 * 1024**3, info["total_bytes"] // 10) if reserve_bytes is None else reserve_bytes
    if required_bytes < 0 or reserve < 0 or info["free_bytes"] - required_bytes < reserve:
        raise DataRootError("Insufficient external free space after the configured reserve.")
    if not os.access(root, os.W_OK):
        raise DataRootError("External data root is not writable.")
    return root.resolve()


def initialize(root, *, expected_uuid) -> dict:
    root = Path(root)
    info = volume_info(root)
    if info["volume_uuid"] != expected_uuid:
        raise DataRootError("External volume UUID does not match the requested destination.")
    marker = root / MARKER
    if not marker.exists():
        atomic_json(marker, {"schema": "ressectionlab-data-root-v1", "volume_uuid": expected_uuid,
                             "root_id": str(uuid.uuid4()), "mount_point": info["mount_point"]})
    checked_root(root, expected_uuid=expected_uuid)
    for relative in LAYOUT:
        destination = safe_path(root, relative)
        destination.mkdir(parents=True, exist_ok=True)
    return info


def safe_path(root, relative) -> Path:
    root = Path(root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise DataRootError("Data path must be relative to the configured root.")
    candidate = root / relative
    if not candidate.resolve().is_relative_to(root):
        raise DataRootError("Data path escapes the configured root.")
    return candidate


def sha256_file(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, document):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__=='__main__':
    if len(sys.argv)!=3 or sys.argv[1]!='--volume-info':raise SystemExit('Usage: python -m resectionlab.data_root --volume-info ROOT')
    print(json.dumps(volume_info(Path(sys.argv[2]))))
