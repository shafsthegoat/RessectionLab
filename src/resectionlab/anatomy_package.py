"""Portable native study archives; originals and immutable arrays travel together."""
import json
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from .patient_anatomy import PatientImagingCollection, file_hash

MAX_BYTES = 16 * 1024**3


def referenced_files(value):
    if isinstance(value, dict):
        if isinstance(value.get('path'), str) and value.get('sha256') and not Path(value['path']).is_absolute():
            yield value['path'], value['sha256']
        for item in value.values():
            yield from referenced_files(item)
    elif isinstance(value, list):
        for item in value:
            yield from referenced_files(item)


def save_package(collection, destination, *, check=lambda: None, progress=lambda *x: None, commit=lambda: None):
    destination = Path(destination)
    files = {'patient-anatomy.json': None}
    for name, checksum in referenced_files(collection.document):
        source = collection.path(name)
        if not source.is_file() or file_hash(source) != checksum:
            raise ValueError(f'Cannot package changed or missing source: {name}')
        files[name] = checksum
    if (collection.root / 'workspace-view.json').exists():
        files['workspace-view.json'] = None
    if (collection.root / 'surgical-toolbox.json').exists():
        if (collection.root / 'surgical-toolbox.json').stat().st_size > 32 * 1024**2:
            raise ValueError('Surgical history exceeds archive budget')
        files['surgical-toolbox.json'] = file_hash(collection.root / 'surgical-toolbox.json')
    if sum(collection.path(name).stat().st_size for name in files) > MAX_BYTES:
        raise ValueError('Patient study exceeds the 16 GiB archive limit')
    fd, temporary = tempfile.mkstemp(dir=destination.parent, suffix='.rsworkspace')
    os.close(fd)
    try:
        with ZipFile(temporary, 'w', compression=ZIP_DEFLATED, compresslevel=1) as archive:
            for index, name in enumerate(sorted(files)):
                check()
                progress(index / len(files), f"Saving patient study · file {index + 1} of {len(files)}")
                if name == 'patient-anatomy.json':
                    archive.writestr(name, json.dumps(collection.document, allow_nan=False))
                    continue
                checksum = hashlib.sha256()
                with collection.path(name).open('rb') as source, archive.open(name, 'w', force_zip64=True) as target:
                    while block := source.read(4 * 1024**2):
                        check()
                        checksum.update(block)
                        target.write(block)
                if files[name] and checksum.hexdigest() != files[name]:
                    raise ValueError(f'Source changed while saving: {name}')
        check()
        commit()
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {'path': str(destination), 'files': len(files)}


def open_package(filename, working_root, *, check=lambda: None):
    filename, working_root = Path(filename), Path(working_root)
    identity = file_hash(filename)
    destination = working_root / identity
    if destination.exists():
        # Reopen this archive's editable working copy, including later revisions.
        PatientImagingCollection(destination)
        return destination / 'patient-anatomy.json'
    working_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=working_root))
    try:
        with ZipFile(filename) as archive:
            members = archive.infolist()
            names = [m.filename for m in members]
            if (len(members) > 10000 or len(names) != len(set(names)) or
                    sum(m.file_size for m in members) > MAX_BYTES or 'patient-anatomy.json' not in names):
                raise ValueError('Invalid or oversized patient study file')
            if any(m.file_size > (16 * 1024**2 if m.filename == 'patient-anatomy.json' else 256 * 1024)
                   for m in members if m.filename in {'patient-anatomy.json', 'workspace-view.json'}):
                raise ValueError('Oversized patient study metadata')
            for member in members:
                check()
                parts = PurePosixPath(member.filename)
                if (parts.is_absolute() or '..' in parts.parts or '\\' in member.filename or
                        stat.S_ISLNK(member.external_attr >> 16) or member.is_dir()):
                    raise ValueError('Unsafe patient study member')
                target = temporary / member.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open('wb') as out:
                    while block := source.read(4 * 1024**2):
                        check()
                        out.write(block)
        collection = PatientImagingCollection(temporary)
        for name, checksum in referenced_files(collection.document):
            check()
            if file_hash(collection.path(name)) != checksum:
                raise ValueError('Patient study source checksum differs')
        check()
        temporary.rename(destination)
        return destination / 'patient-anatomy.json'
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
