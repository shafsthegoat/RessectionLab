"""Root-run scalar regression only; no source arrays/tasks/model/native imports."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('frontier_independent_scalar',Path(__file__).with_name('scalar_checks.py'))
checks=importlib.util.module_from_spec(spec);spec.loader.exec_module(checks)


def outcome(reward,cells):
    return {'total_reward':reward,'target_removed_mm3':float(cells),
        'positive_target_source_cells_removed':cells,'normal_removed_mm3':0.,
        'complete_tool_path_length_mm':12.,'nonstop_actions':12,'tool_changes':3,
        'cumulative_contacted_tissue_upper_bound_mm3':50.}


class NetSuffixRegression(unittest.TestCase):
    def test_positive_absolute_return_cannot_hide_nonpositive_suffix(self):
        old=outcome(48.,51)
        for value in (47.9,48.):
            with self.subTest(full_return=value):
                result=checks.useful_gate(old,outcome(value,52),admissible=True)
                self.assertTrue(result['additional_target_cells'])
                self.assertFalse(result['positive_net_suffix_return'])
                self.assertFalse(result['useful_gain'])

    def test_both_target_and_value_gain_require_complete_admission(self):
        old=outcome(48.,51)
        self.assertTrue(checks.useful_gate(old,outcome(48.1,52),admissible=True)['useful_gain'])
        self.assertFalse(checks.useful_gate(old,outcome(48.1,51),admissible=True)['useful_gain'])
        self.assertFalse(checks.useful_gate(old,outcome(48.1,52),admissible=False)['useful_gain'])

    def test_fixed_prefix_is_incumbent_for_negative_suffix(self):
        result=checks.best_fixed_prefix([4.]*12+[-.2,-.3,0.],prefix_length=12,horizon=24)
        self.assertEqual(result['best_absolute_prefix_length'],12)
        self.assertEqual(result['best_full_return'],48.)
        self.assertEqual(result['best_net_suffix_return'],0.)


if __name__=='__main__':unittest.main()
