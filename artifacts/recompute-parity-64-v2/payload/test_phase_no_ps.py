"""Generated-only phase logger control; no model import or checkpoint."""
from contextlib import redirect_stdout
import io
import json
import subprocess
import unittest
from unittest import mock

import pair_worker


class PhaseLoggerTests(unittest.TestCase):
    def test_phase_reads_self_rss_without_spawning_ps(self):
        output = io.StringIO()
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("child spawned")):
            with redirect_stdout(output):
                pair_worker.phase("generated_only_phase")
        record = json.loads(output.getvalue().strip())
        self.assertEqual(record["phase"], "generated_only_phase")
        self.assertEqual(record["phase_rss_source"],
                         "direct_libproc_self_pid_no_subprocess")
        self.assertGreater(record["self_rss_kib"], 0)


if __name__ == "__main__":
    unittest.main()
