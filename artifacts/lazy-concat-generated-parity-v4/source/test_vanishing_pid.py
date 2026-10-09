"""Generated/mock-only PID-race controls; no checkpoint or patient data."""
import ctypes
import errno
from types import SimpleNamespace
import unittest
from unittest import mock

from darwin_fast_sampler import FastDarwinSampler


PID = 23001
PGID = 23001


def sampler_with_failed_pidinfo(error_code):
    group_function = object()
    child_function = object()

    def failed_pidinfo(*_args):
        ctypes.set_errno(error_code)
        return 0

    sampler = object.__new__(FastDarwinSampler)
    sampler.proc = SimpleNamespace(proc_listpgrppids=group_function,
                                   proc_listchildpids=child_function,
                                   proc_pidinfo=failed_pidinfo)
    sampler._pid_list = lambda function, _identifier: ([PID] if function is group_function else [])
    return sampler


class VanishingPidTests(unittest.TestCase):
    def test_eperm_after_verified_exit_is_skipped(self):
        sampler = sampler_with_failed_pidinfo(errno.EPERM)
        with mock.patch("darwin_fast_sampler.os.getpgid", side_effect=[
                PGID, ProcessLookupError(errno.ESRCH, "gone")]) as getpgid:
            result = sampler.group(PGID, PID)
        self.assertEqual(getpgid.call_count, 2)
        self.assertEqual(result["pids"], [])
        self.assertEqual(result["process_group_resident_bytes"], 0)

    def test_eperm_for_still_live_member_remains_fail_closed(self):
        sampler = sampler_with_failed_pidinfo(errno.EPERM)
        with mock.patch("darwin_fast_sampler.os.getpgid", side_effect=[PGID, PGID]):
            with self.assertRaises(PermissionError):
                sampler.group(PGID, PID)

    def test_esrch_for_reused_or_still_live_pid_remains_fail_closed(self):
        sampler = sampler_with_failed_pidinfo(errno.ESRCH)
        with mock.patch("darwin_fast_sampler.os.getpgid", side_effect=[PGID, PGID]):
            with self.assertRaises(ProcessLookupError):
                sampler.group(PGID, PID)

    def test_unverifiable_pid_remains_fail_closed(self):
        sampler = sampler_with_failed_pidinfo(errno.EPERM)
        with mock.patch("darwin_fast_sampler.os.getpgid", side_effect=[
                PGID, PermissionError(errno.EPERM, "unverifiable")]):
            with self.assertRaises(PermissionError):
                sampler.group(PGID, PID)


if __name__ == "__main__":
    unittest.main()
