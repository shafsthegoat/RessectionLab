"""Generated tiny-file tests; never opens HBE/patient/measured inputs."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import mechanics_hbe_v5_n8_one_shot as io

SOURCE = Path(__file__).with_name("hash_hint.py")
SPEC = importlib.util.spec_from_file_location("hbe_nocache_hash_hint", SOURCE)
assert SPEC is not None and SPEC.loader is not None
hint = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hint)


def audit():
    return {"eligible_bytes": 0, "eligible_files": 0,
            "hinted_file_opens": 0, "hinted_file_paths": []}


class TinyNoCacheTests(unittest.TestCase):
    def test_exact_original_sha_with_real_f_nocache(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "tiny.bin"
            raw = b"same SHA256 data\x00" * 17
            file.write_bytes(raw)
            note = audit()
            result = hint.hinted_hash(file, maximum=4096, original_hash=io.file_hash,
                                      allowed_directories=[file.parent], audit=note,
                                      minimum_bytes=1)
            self.assertEqual(result, hashlib.sha256(raw).hexdigest())
            self.assertEqual(result, io.file_hash(file, maximum=4096))
            self.assertEqual(note["eligible_bytes"], len(raw))
            self.assertEqual(note["eligible_files"], 1)
            self.assertEqual(note["hinted_file_opens"], 1)
            self.assertEqual(note["hinted_file_paths"], [str(file)])

    def test_noneligible_or_unlisted_file_uses_unchanged_hash_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / "small.bin"
            file.write_bytes(b"bounded fixture")
            note = audit()
            expected = io.file_hash(file, maximum=4096)
            self.assertEqual(hint.hinted_hash(
                file, maximum=4096, original_hash=io.file_hash,
                allowed_directories=[root], audit=note), expected)
            self.assertEqual(hint.hinted_hash(
                file, maximum=4096, original_hash=io.file_hash,
                allowed_directories=[root / "another"], audit=note,
                minimum_bytes=1), expected)
            self.assertEqual(note["hinted_file_opens"], 0)
            self.assertEqual(note["eligible_files"], 0)

    def test_size_cap_and_symlink_rejection_remain_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            file = root / "large.bin"
            file.write_bytes(b"a" * 65)
            symlink = root / "link.bin"
            symlink.symlink_to(file)
            for target in (file, symlink):
                with self.assertRaisesRegex(ValueError, "special or oversized"):
                    hint.hinted_hash(target, maximum=64, original_hash=io.file_hash,
                                     allowed_directories=[root], audit=audit(),
                                     minimum_bytes=1)

    def test_fcntl_error_fails_closed_without_returning_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "tiny.bin"
            file.write_bytes(b"read-only input")
            note = audit()
            with patch.object(hint.fcntl, "fcntl", side_effect=OSError("hint refused")):
                with self.assertRaisesRegex(OSError, "hint refused"):
                    hint.hinted_hash(file, maximum=4096, original_hash=io.file_hash,
                                     allowed_directories=[file.parent], audit=note,
                                     minimum_bytes=1)
            self.assertEqual(note["hinted_file_opens"], 0)
            self.assertEqual(file.read_bytes(), b"read-only input")

    def test_original_lstat_identity_check_still_runs_with_proxy(self):
        with tempfile.TemporaryDirectory() as temporary:
            file = Path(temporary) / "tiny.bin"
            file.write_bytes(b"stable")
            note = audit()

            def mutating_original(proxy, *, maximum):
                before = proxy.lstat()
                with proxy.open("rb") as stream:
                    self.assertEqual(stream.read(), b"stable")
                file.write_bytes(b"changed length")
                after = proxy.lstat()
                self.assertNotEqual((before.st_size, before.st_mtime_ns),
                                    (after.st_size, after.st_mtime_ns))
                raise ValueError("Bound file changed while hashing")

            with self.assertRaisesRegex(ValueError, "changed while hashing"):
                hint.hinted_hash(file, maximum=4096, original_hash=mutating_original,
                                 allowed_directories=[file.parent], audit=note,
                                 minimum_bytes=1)
            self.assertEqual(note["hinted_file_opens"], 1)


if __name__ == "__main__":
    unittest.main()
