"""Generated B=4 parity against the archived policy; no acquired inputs."""
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import signal
import sys
import time
import unittest
import weakref

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
ROOT = next(p for p in HERE.parents if (p/'src/resectionlab').is_dir())
sys.path.insert(0, str(STAGE))
import paths
import numpy as np
import torch
from gymnasium import spaces
from torch.nn import functional as F

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
signal.alarm(55)
OLD = ROOT/'artifacts/train045-local-goal-curriculum-interruption-v1/original-policy.py'
NEW = STAGE/'model/policy.py'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


old = load('memory_archived_policy', OLD)
new = load('memory_candidate_policy', NEW)
CHECKS = []


def exact(left, right):
    if torch.is_tensor(left):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
                           right.detach().contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right): exact(a, b)
    else:
        assert left == right


def fixture(frozen):
    torch.manual_seed(503)
    b, n, shape = 4, 153, (7, 8, 6)
    x = {
        'images': torch.randn(b, 6, *shape),
        'coverage': torch.randint(0, 2, (b, 6, *shape), dtype=torch.uint8),
        'available': torch.tensor([[1,0,1,1,0,1],[0,1,1,0,1,1],[1,1,1,1,1,1],[0,0,0,0,0,0]], dtype=torch.uint8),
        'geometry': torch.randn(b, n, 16), 'grid': torch.rand(b, n, 5, 3)*3.-1.5,
        'inside': torch.randint(0, 2, (b, n, 5)).float(),
        'state': torch.randn(b, 10), 'spacing': torch.rand(b, 3)+.1,
        'public_target': torch.randn(b, 16), 'public_relation': torch.randn(b, n, 4),
        'goal_global': torch.randn(b, 4), 'goal_relation': torch.randn(b, n, 4),
        'action_mask': torch.zeros(b, n, dtype=torch.uint8),
    }
    for row, count in enumerate((6, 4, 1, 3)): x['action_mask'][row, :count] = 1
    x['images'][0, 0, 0, 0, :2] = torch.tensor([-0., 0.])
    space = spaces.Dict({k: spaces.Box(0, 1, tuple(v.shape[1:]), np.uint8) if v.dtype == torch.uint8
        else spaces.Box(-np.inf, np.inf, tuple(v.shape[1:]), np.float32) for k,v in x.items()})
    a = old.LocalGoalPolicy(space, spaces.Discrete(n), lambda _: 3e-4, freeze_encoder=frozen)
    b = new.LocalGoalPolicy(space, spaces.Discrete(n), lambda _: 3e-4, freeze_encoder=frozen)
    b.load_state_dict(a.state_dict(), strict=True)
    return a, b, x


def ppo_loss(policy, batch):
    actions = torch.tensor([1, 2, 0, 1])
    values, log_prob, entropy = policy.evaluate_actions(batch, actions, batch['action_mask'])
    advantages = torch.tensor([1., -.5, .2, .8])
    advantages = (advantages-advantages.mean())/(advantages.std()+1e-8)
    ratio = torch.exp(log_prob-torch.tensor([-1.6, -1.4, 0., -1.1]))
    policy_loss = -torch.minimum(advantages*ratio, advantages*ratio.clamp(.8,1.2)).mean()
    value_loss = F.mse_loss(values.flatten(), torch.tensor([.4,-.2,1.,.3]))
    return policy_loss+.5*value_loss-.01*entropy.mean()


class InputMemoryParity(unittest.TestCase):
    def parity(self, frozen):
        a, b, x = fixture(frozen)
        before = {k:v.clone() for k,v in x.items()}
        observed = {}
        handles = []
        for name, policy in (('old',a),('new',b)):
            handles.append(policy.spatial.encoder.register_forward_pre_hook(
                lambda module, args, key=name: observed.__setitem__(key, args[0].detach().clone())))
        try:
            la, va = a.logits_values(x); lb, vb = b.logits_values(x)
            exact(observed['old'], observed['new']); exact(la,lb); exact(va,vb)
            exact(before,x)
            assert observed['new'].dtype == torch.float32 and observed['new'].is_contiguous()
            xa = {k:v.clone() for k,v in x.items()}; xb = {k:v.clone() for k,v in x.items()}
            xa['images'].requires_grad_(True); xb['images'].requires_grad_(True)
            loss_a = ppo_loss(a, xa); loss_b = ppo_loss(b, xb); exact(loss_a,loss_b)
            a.optimizer.zero_grad(); b.optimizer.zero_grad(); loss_a.backward(); loss_b.backward()
            compared = 0
            for (name,pa),(other,pb) in zip(a.named_parameters(),b.named_parameters()):
                assert name == other and (pa.grad is None) == (pb.grad is None)
                if pa.grad is not None: exact(pa.grad,pb.grad); compared += 1
            if frozen: assert xa['images'].grad is None and xb['images'].grad is None
            else: exact(xa['images'].grad,xb['images'].grad)
            torch.nn.utils.clip_grad_norm_(a.parameters(), .5)
            torch.nn.utils.clip_grad_norm_(b.parameters(), .5)
            a.optimizer.step(); b.optimizer.step()
            exact(a.state_dict(),b.state_dict()); exact(a.optimizer.state_dict(),b.optimizer.state_dict())
            assert a.costs['optimizer_completed'] == b.costs['optimizer_completed'] == 1
            CHECKS.append({'case':'frozen' if frozen else 'trainable', 'batch':4,
                'volume_bytes_equal':True,'logits_values_equal':True,'loss_equal':True,
                'parameter_gradients_equal':compared,'one_Adam_state_equal':True,'inputs_unchanged':True})
        finally:
            for handle in handles: handle.remove()

    def test_frozen_exact_volume_forward_gradient_adam(self): self.parity(True)
    def test_trainable_exact_volume_forward_gradient_adam(self): self.parity(False)

    def test_frozen_dense_references_released_before_heads(self):
        a, b, x = fixture(True)
        states = {}
        for label, policy in (('old',a),('new',b)):
            refs = {}; seen = []
            h1 = policy.spatial.encoder.register_forward_pre_hook(
                lambda module,args: refs.__setitem__('volume',weakref.ref(args[0])))
            h2 = policy.spatial.encoder.register_forward_hook(
                lambda module,args,out: refs.__setitem__('encoded',weakref.ref(out)))
            h3 = policy.spatial.actor.register_forward_pre_hook(
                lambda module,args: seen.append({k:v() is not None for k,v in refs.items()}))
            try: policy.logits_values(x)
            finally:
                for h in (h1,h2,h3): h.remove()
            assert len(seen) == 1; states[label] = seen[0]
        assert states == {'old':{'volume':True,'encoded':True},'new':{'volume':False,'encoded':False}}
        CHECKS.append({'case':'frozen_reference_lifetimes',**states})

    def test_validation_and_hidden_payload_unchanged(self):
        a,b,x = fixture(True)
        hidden = ~(x['coverage'].bool() & x['available'][:,:,None,None,None].bool())
        y = {k:v.clone() for k,v in x.items()}; y['images'][hidden] = 12345.
        for policy in (a,b):
            l,v=policy.logits_values(x); ll,vv=policy.logits_values(y); exact(l,ll); exact(v,vv)
            bad={**x,'images':x['images'].clone()}; bad['images'].flatten()[0]=float('nan')
            with self.assertRaisesRegex(ValueError,'Nonfinite public image'): policy.logits_values(bad)
            wrong=x['action_mask'].clone(); wrong[0,-1]=1
            with self.assertRaisesRegex(ValueError,'mask mismatch'): policy.logits_values(x,wrong)
        CHECKS.append({'case':'validation_and_unobserved_payload','equal':True})


if __name__ == '__main__':
    started=time.perf_counter()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(InputMemoryParity))
    record={'status':'pass' if result.wasSuccessful() else 'failed','tests':result.testsRun,
        'failures':len(result.failures),'errors':len(result.errors),'checks':CHECKS,
        'elapsed_seconds':time.perf_counter()-started,'peak_self_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD,NEW,Path(__file__))},
        'acquired_inputs':False,'native_calls':0,'acquired_checkpoint_loads':0}
    print(json.dumps(record,sort_keys=True))
    sys.exit(0 if result.wasSuccessful() else 1)
