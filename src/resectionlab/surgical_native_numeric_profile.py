"""The two qualified ARM64 numeric images, independent of their code signatures.

Layout follows the public SDK mach-o/loader.h. Preserve every image byte except
the signature and its dependent sizes; sections, ABI and dependency names remain
bound. The adapter must register the returned whole-file digest for live checks.
"""
import hashlib
import importlib
from pathlib import Path
import platform
import struct
import sys


# Profiles derived once from the qualified installed images (whole-file hashes
# 3432cd90...193bf85d and 5a5913d8...73f04bba respectively).
_PROFILES = (
    ('numpy._core._multiarray_umath', 'c8327393fc46485a481e2a1d079f13bd89a63a8feb870082ffca38c4d934f4e5'),
    ('scipy.spatial.transform._rotation_cy', 'e6f7fd1c92158a6c6e9f583d59958d02a78282936a529adf402b1dc51a819ec8'),
)


def _page_size(size):
    return (size + 16383) & ~16383


def _macho_profile(data):
    """Hash semantic bytes of this narrow little-endian ARM64 Mach-O64 layout."""
    if len(data) < 32:
        raise ValueError('truncated Mach-O header')
    magic, cpu, subtype, kind, count, size, flags, reserved = struct.unpack_from('<8I', data)
    if magic != 0xFEEDFACF or cpu != 0x100000C or kind != 8:
        raise ValueError('requires an ARM64 Mach-O64 bundle')
    end = 32 + size
    if end > len(data) or not 0 < count <= size // 8:
        raise ValueError('invalid load-command bounds')
    offset, linkedit, signature, segments = 32, None, None, []
    for _ in range(count):
        if offset + 8 > end:
            raise ValueError('truncated load command')
        command, length = struct.unpack_from('<II', data, offset)
        if length < 8 or length % 8 or offset + length > end:
            raise ValueError('invalid load-command size')
        if command == 0x19:  # LC_SEGMENT_64, followed by section_64 records.
            if length < 72:
                raise ValueError('truncated segment')
            segment = struct.unpack_from('<II16s4Q4I', data, offset)
            name, address, memory, start, extent = segment[2:7]
            sections = segment[9]
            if length != 72 + sections * 80 or start + extent > len(data) or memory < extent:
                raise ValueError('invalid segment bounds')
            if name.rstrip(b'\0') == b'__LINKEDIT':
                if linkedit is not None or sections:
                    raise ValueError('invalid link-edit segment')
                linkedit = (offset, start, extent, memory)
            else:
                segments.append((start, extent))
            for index in range(sections):
                section = struct.unpack_from('<16s16sQQ8I', data, offset + 72 + index * 80)
                _, owner, location, amount, position, alignment, relocations, reloc_count, attributes, *_ = section
                if owner != name or location < address or location + amount > address + memory:
                    raise ValueError('invalid section memory bounds')
                if (attributes & 0xFF) not in (1, 0xC, 0x12) and amount:
                    if position < start or position + amount > start + extent:
                        raise ValueError('invalid section file bounds')
                if reloc_count and relocations + reloc_count * 8 > len(data):
                    raise ValueError('invalid section relocations')
        elif command == 0x1D:  # LC_CODE_SIGNATURE.
            if length != 16 or signature is not None:
                raise ValueError('invalid signature command')
            position, amount = struct.unpack_from('<II', data, offset + 8)
            signature = (offset, position, amount)
        offset += length
    if offset != end or linkedit is None or signature is None:
        raise ValueError('incomplete Mach-O layout')
    command, position, amount = signature
    link_command, link_start, link_extent, link_memory = linkedit
    if (not amount or position < end or position + amount != len(data)
            or link_start < end or link_start > position or link_start + link_extent != len(data)
            or link_memory != _page_size(link_extent)
            or any(start + extent > position for start, extent in segments)):
        raise ValueError('signature is not the bounded link-edit tail')
    semantic_extent = position - link_start
    patches = sorted(((command + 12, struct.pack('<I', 0)),
        (link_command + 32, struct.pack('<Q', _page_size(semantic_extent))),
        (link_command + 48, struct.pack('<Q', semantic_extent))))
    digest, cursor = hashlib.sha256(), 0
    view = memoryview(data)
    for offset, replacement in patches:
        digest.update(view[cursor:offset])
        digest.update(replacement)
        cursor = offset + len(replacement)
    digest.update(view[cursor:position])
    return digest.hexdigest()


def qualified_modules():
    """Return ((module, path, actual whole SHA256), ...), reason; None declines."""
    if platform.system() != 'Darwin' or platform.machine() != 'arm64' or sys.version_info[:2] != (3, 12):
        return None, 'unsupported-numeric-platform'
    captures = []
    for name, expected in _PROFILES:
        try:
            module = importlib.import_module(name)
            path = Path(module.__file__).resolve(strict=True)
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            if any(getattr(before, field) != getattr(after, field) for field in
                    ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
                return None, 'numeric-image-changed-during-read:' + name
            if _macho_profile(data) != expected:
                return None, 'numeric-image-profile-mismatch:' + name
        except (AttributeError, ImportError, OSError, TypeError, ValueError, struct.error) as error:
            return None, 'numeric-image-unavailable:' + name + ':' + type(error).__name__ + ':' + str(error)
        captures.append((module, path, hashlib.sha256(data).hexdigest()))
    return tuple(captures), None
