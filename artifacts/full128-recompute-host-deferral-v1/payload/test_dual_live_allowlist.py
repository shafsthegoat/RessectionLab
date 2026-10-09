"""Generated process-list checks for two exact live acquisitions; no model call."""

import json
from pathlib import Path
import unittest
from unittest import mock

import slow_inventory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


class DualLiveTests(unittest.TestCase):
    def setUp(self):
        self.allow = slow_inventory.load_acquisition_allowlist(HERE / "dual-live-allowlist.json")
        self.tracto = self.allow["entries"]["tracto"]
        self.ixi = self.allow["entries"]["ixi"]
        self.commands = {
            "tracto": json.loads((ROOT / "build/tractoinferno-train-preparation-v1/live-monitor-identity.json").read_text())["ps_command"],
            "ixi": json.loads((ROOT / "build/ixi-paired-intake-preparation-v1/live-monitor-identity.json").read_text())["ps_command_stripped"],
        }

    def collect_fixture(self, missing=(), changed_command=(), changed_kernel=(), high_rss=(), unrelated=False):
        lines = []
        for role in ("tracto", "ixi"):
            if role in missing:
                continue
            manifest = self.allow["entries"][role]
            command = self.commands[role] if role not in changed_command else "python3.12 /repo/build/unrelated.py"
            rss = manifest["max_rss_kib"] + 1 if role in high_rss else 30000
            lines.append("%d 1 %d %d python3.12 %s" %
                         (manifest["pid"], manifest["pgid"], rss, command))
        if unrelated:
            lines.append("88888 1 88888 10000 python3.12 /repo/build/compute.py")

        def fake_run(args):
            if args[:2] == ["ps", "-axww"]:
                return "\n".join(lines) + "\n"
            if args[:2] == ["ps", "-p"]:
                pid = int(args[2])
                role = next(role for role, item in self.allow["entries"].items() if item["pid"] == pid)
                if role in missing:
                    raise slow_inventory.subprocess.CalledProcessError(1, args)
                return self.allow["entries"][role]["process_start_lstart"] + "\n"
            if args == ["vm_stat"]:
                return ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                        "Pages occupied by compressor: 1.\nPageouts: 2.\nSwapouts: 3.\n")
            raise AssertionError(args)

        class FakeSampler:
            def process_start_identity(inner, pid):
                role = next(role for role, item in self.allow["entries"].items() if item["pid"] == pid)
                if role in missing:
                    return None
                identity = self.allow["kernel_start_identities"][role]
                return (identity[0], identity[1] + 1) if role in changed_kernel else identity

        with mock.patch.object(slow_inventory, "run", side_effect=fake_run), \
                mock.patch.object(slow_inventory, "FastDarwinSampler", return_value=FakeSampler()):
            return slow_inventory.collect(999999, acquisition_allowlist=self.allow)["project_process_inventory"]

    def test_both_exact_live_processes_and_only_them_are_admitted(self):
        rows = self.collect_fixture()
        matched = [r for r in rows if r.get("acquisition_role")]
        self.assertEqual({r["acquisition_role"] for r in matched}, {"tracto", "ixi"})
        self.assertTrue(all(r["identity_matched"] and r["kernel_start_identity_matched"] and
                            r["rss_within_cap"] and not r["blocking_overlap"] for r in matched))
        rows = self.collect_fixture(unrelated=True)
        self.assertTrue(any(r["pid"] == 88888 and r["blocking_overlap"] for r in rows))

    def test_absence_is_always_blocking_with_no_completion_exception(self):
        self.assertNotIn("completion", json.dumps(self.allow))
        for missing in (("ixi",), ("tracto",)):
            rows = self.collect_fixture(missing=missing)
            role = missing[0]
            row = next(r for r in rows if r.get("acquisition_role") == role)
            self.assertTrue(row["blocking_overlap"])
            self.assertEqual(row["reason"], "allowlisted_live_pid_absent")

    def test_rss_command_and_kernel_identity_mismatches_block(self):
        for kwargs in ({"high_rss": ("ixi",)}, {"changed_command": ("ixi",)},
                       {"changed_kernel": ("ixi",)}, {"changed_kernel": ("tracto",)}):
            rows = self.collect_fixture(**kwargs)
            self.assertTrue(any(r.get("acquisition_role") and r["blocking_overlap"] for r in rows))


if __name__ == "__main__":
    unittest.main()
