"""Prepared generated scalar controls only; no imports from patient runtime."""
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_focused_independent_oracles', HERE/'scalar_oracles.py')
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)


class IndependentScalars(unittest.TestCase):
    def test_reward_exactly_once_and_negative_tail_is_not_penalty(self):
        ids = ('STOP', 'first', 'second')
        # Correct 2+4=6 versus5+0=5 selects first; double reward would select second.
        self.assertEqual(o.choose_positive(ids, (0.,2.,5.), (0.,4.,-3.)), 'first')
        self.assertEqual(o.choose_positive(ids, (0.,2.,5.), (0.,-9.,-3.)), 'second')
        self.assertEqual(o.choose_positive(ids, (0.,2.,5.), (0.,3.,0.)), 'first')
        self.assertEqual(o.choose_positive(ids, (0.,-1.,0.), (0.,100.,100.)), 'STOP')

    def test_base_duplicates_count_once_and_extra_keeps_owning_page(self):
        def row(i,x):return {'action_id':i,'tool_id':'tool','entry_mm':[0.,0.,0.],'tip_mm':[x,0.,1.]}
        a,b,c,d = (row('a',1.),row('b',2.),row('c',3.),row('d',4.))
        rows,owners = o.union_rows(((a,b),(a,b,c),(a,b,d)))
        self.assertEqual([r['action_id'] for r in rows], ['a','b','c','d'])
        self.assertEqual(owners, {'a':0,'b':0,'c':1,'d':2})
        bad = dict(a, tip_mm=[99.,0.,1.])
        with self.assertRaises(ValueError):o.union_rows(((a,b),(bad,c)))


if __name__ == '__main__':unittest.main()
