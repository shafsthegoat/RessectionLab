"""Small generated Mach-O images; no compiler, signing service or numeric imports."""
import hashlib
import struct
from types import SimpleNamespace

import pytest

from resectionlab import surgical_native_numeric_profile as profile


def image(signature=b'signed metadata'):
    def segment(name, address, offset, size, sections=()):
        raw_name = name.encode().ljust(16, b'\0')
        memory = profile._page_size(size) if name == '__LINKEDIT' else size
        body = struct.pack('<II16s4Q4I', 0x19, 72 + 80 * len(sections), raw_name,
                           address, memory, offset, size, 7, 3, len(sections), 0)
        for section, location in sections:
            body += struct.pack('<16s16sQQ8I', section.encode().ljust(16, b'\0'), raw_name,
                                location, 16, location, 3, 0, 0, 0, 0, 0, 0)
        return body
    text = segment('__TEXT', 0, 0, 1024, [('__text', 768), ('__const', 784)])
    data = segment('__DATA', 1024, 1024, 1024, [('__data', 1024)])
    link = segment('__LINKEDIT', 2048, 2048, 256 + len(signature))
    name = b'/usr/lib/libSystem.B.dylib\0'
    length = (24 + len(name) + 7) & ~7
    dependency = struct.pack('<6I', 0xC, length, 24, 0, 0x10000, 0x10000) + name
    dependency = dependency.ljust(length, b'\0')
    sign = struct.pack('<4I', 0x1D, 16, 2304, len(signature))
    commands = text + data + link + dependency + sign
    header = struct.pack('<8I', 0xFEEDFACF, 0x100000C, 0, 8, 5, len(commands), 133, 0)
    result = bytearray((header + commands).ljust(2304, b'\0') + signature)
    result[768:800] = b'T' * 16 + b'C' * 16
    result[1024:1040] = b'D' * 16
    result[2100:2116] = b'L' * 16
    return result


def test_signature_payload_and_rounded_sizes_can_change_at_a_new_path(tmp_path):
    original = image()
    signed = image(b'new signing identity' * 1200)
    assert original != signed
    relocated = tmp_path / 'new-location.so'
    relocated.write_bytes(signed)
    assert profile._macho_profile(original) == profile._macho_profile(relocated.read_bytes())


@pytest.mark.parametrize('position', [768, 784, 1024, 2100])
def test_text_constants_data_and_linkedit_changes_do_not_qualify(position):
    original = image()
    modified = original.copy()
    modified[position] ^= 1
    assert profile._macho_profile(modified) != profile._macho_profile(original)


def test_abi_and_dependency_names_remain_bound():
    original = image()
    flags = original.copy()
    struct.pack_into('<I', flags, 24, 132)
    dependency = original.replace(b'libSystem.B.dylib', b'libChangedB.dylib')
    assert profile._macho_profile(flags) != profile._macho_profile(original)
    assert profile._macho_profile(dependency) != profile._macho_profile(original)


@pytest.mark.parametrize('change', ['short', 'cpu', 'command', 'section', 'signature'])
def test_unsupported_or_malformed_images_decline(change):
    data = image()
    if change == 'short':
        data = data[:20]
    elif change == 'cpu':
        struct.pack_into('<I', data, 4, 0x1000007)
    elif change == 'command':
        struct.pack_into('<I', data, 36, 7)
    elif change == 'section':
        struct.pack_into('<I', data, 32 + 72 + 48, 3000)
    else:
        data.pop()
    with pytest.raises(ValueError):
        profile._macho_profile(data)


def test_qualified_modules_reports_actual_file_digest_and_mismatch_reason(monkeypatch, tmp_path):
    path = tmp_path / 'numeric.so'
    data = image()
    path.write_bytes(data)
    module = SimpleNamespace(__file__=str(path))
    monkeypatch.setattr(profile, 'platform', SimpleNamespace(system=lambda: 'Darwin', machine=lambda: 'arm64'))
    monkeypatch.setattr(profile, 'sys', SimpleNamespace(version_info=(3, 12)))
    monkeypatch.setattr(profile, '_PROFILES', (('generated.numeric', profile._macho_profile(data)),))
    monkeypatch.setattr(profile, 'importlib', SimpleNamespace(import_module=lambda name: module))
    captures, reason = profile.qualified_modules()
    assert reason is None and captures == ((module, path, hashlib.sha256(data).hexdigest()),)
    data = image(b'another valid arithmetic image signature')
    path.write_bytes(data)
    assert profile.qualified_modules() == (((module, path, hashlib.sha256(data).hexdigest()),), None)
    data[768] ^= 1
    path.write_bytes(data)
    assert profile.qualified_modules() == (None, 'numeric-image-profile-mismatch:generated.numeric')
    monkeypatch.setattr(profile.platform, 'machine', lambda: 'x86_64')
    assert profile.qualified_modules() == (None, 'unsupported-numeric-platform')
