import copy
import json
from types import SimpleNamespace
import importlib.util
import unittest
from unittest.mock import patch
from pathlib import Path
spec = importlib.util.spec_from_file_location('storage_guard', Path(__file__).with_name('storage_guard.py'))
g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)

def ledger():
    return {'reservations': {}, 'growth_highwater_bytes': 0,
            'latest_measurement': {'free_bytes': 100*1024**3}}

class GuardControls(unittest.TestCase):
    def test_shared_growth_and_reservations_cap(self):
        x=ledger(); x['growth_highwater_bytes']=g.CAP-10*g.MIB
        with patch.object(g,'du_scope',return_value=0):
            g.reserve(x,'first',6,'agent',['build/first'],True)
            with self.assertRaises(ValueError): g.reserve(x,'second',5,'agent',['build/second'],True)
        self.assertEqual(set(x['reservations']), {'first'})
    def test_free_floor_includes_outstanding(self):
        x=ledger(); x['latest_measurement']['free_bytes']=g.FREE_FLOOR+g.MIB
        with patch.object(g,'du_scope',return_value=0):
            with self.assertRaises(ValueError): g.reserve(x,'fail',2,'agent',['build/fail'],True)
    def test_only_one_running_pilot(self):
        x=ledger()
        with patch.object(g,'du_scope',return_value=0):
            for k in ('first','second'): g.reserve(x,k,1,'agent',['build/'+k],True)
        g.begin(x,'first')
        with self.assertRaises(ValueError): g.begin(x,'second')
        self.assertEqual(x['reservations']['second']['status'],'prepared')
    def test_no_duplicate_or_overlapping_allocation(self):
        x=ledger()
        with patch.object(g,'du_scope',return_value=0):
            g.reserve(x,'first',1,'agent',['build/first'],True)
            with self.assertRaises(ValueError): g.reserve(x,'first',1,'agent',['build/else'],True)
            with self.assertRaises(ValueError): g.reserve(x,'nested',1,'agent',['build/first/child'],True)
    def test_scope_escape_refused(self):
        for s in ('../escape','/tmp/escape','build/../../escape'):
            with self.assertRaises(ValueError): g.safe_scope(s)
    def test_per_scope_highwater_no_shrink_credit(self):
        x=ledger(); b={'allocated_bytes':{'build':100,'data':100}}
        with patch.object(g,'du_scope',return_value=0):
            g.update_measurement(x,b,{'allocated_bytes':{'build':150,'data':80},'free_bytes':100*1024**3})
            self.assertEqual(x['growth_highwater_bytes'],50)
            g.update_measurement(x,b,{'allocated_bytes':{'build':120,'data':110},'free_bytes':100*1024**3})
            self.assertEqual(x['growth_highwater_bytes'],60)
    def test_unregistered_new_top_scope_counts(self):
        x=ledger(); b={'allocated_bytes':{'build':100}}
        g.update_measurement(x,b,{'allocated_bytes':{'build':100,'new-cache':456},'free_bytes':100*1024**3})
        self.assertEqual(x['growth_highwater_bytes'],456)
    def test_scope_overrun_blocks_next_allocation(self):
        x=ledger()
        with patch.object(g,'du_scope',return_value=0): g.reserve(x,'first',1,'agent',['build/first'],True)
        with patch.object(g,'du_scope',return_value=2*g.MIB):
            g.update_measurement(x,{'allocated_bytes':{}},{'allocated_bytes':{},'free_bytes':100*1024**3})
        self.assertEqual(g.capacity(x)['reservation_scope_overruns'],['first'])
        self.assertFalse(g.capacity(x)['allocation_allowed'])
    def test_close_requires_explicit_clean_reap(self):
        x=ledger()
        with patch.object(g,'du_scope',return_value=0): g.reserve(x,'first',1,'agent',['build/first'],True)
        for receipt in ({'worker_termination_confirmed':True}, {'worker_termination_confirmed':True,'remaining_owned_pids':[123]}):
            with patch.object(Path,'stat',return_value=SimpleNamespace(st_size=100)), patch.object(Path,'read_bytes',return_value=json.dumps(receipt).encode()):
                with self.assertRaises(ValueError): g.close(x,'first','build/receipt.json')
        self.assertEqual(x['reservations']['first']['status'],'prepared')
    def test_negative_scientific_result_can_release_storage(self):
        x=ledger()
        with patch.object(g,'du_scope',return_value=0): g.reserve(x,'first',1,'agent',['build/first'],True)
        receipt={'status':'failed','exit_code':1,'worker_termination_confirmed':True,'remaining_owned_pids':[]}
        with patch.object(Path,'stat',return_value=SimpleNamespace(st_size=100)), patch.object(Path,'read_bytes',return_value=json.dumps(receipt).encode()):
            g.close(x,'first','build/receipt.json')
        self.assertEqual(x['reservations']['first']['status'],'closed')
        self.assertEqual(g.capacity(x)['outstanding_reserved_bytes'],0)
    def test_full_pending_reservation_is_conservative(self):
        x=ledger(); x['growth_highwater_bytes']=g.MIB
        with patch.object(g,'du_scope',return_value=0): g.reserve(x,'first',2,'agent',['build/first'],True)
        self.assertEqual(g.capacity(x)['conservative_growth_plus_reservations_bytes'],3*g.MIB)

if __name__=='__main__': unittest.main(verbosity=2)
