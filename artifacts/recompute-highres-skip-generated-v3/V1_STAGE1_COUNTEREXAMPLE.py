"""Generated-only guard counterexample; never loads weights or patient data."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "build/limited-input-guard-design/recompute-highres-skip-v1"))

import torch
from test_recompute_skip import tiny_network
from recompute_skip import attach_recomputed_stage_zero


class MutatingLaterStage(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, values):
        output = self.inner(values)
        values.add_(0.25)  # A later stage changes the retained first skip.
        return output


network = tiny_network()
network.encoder.stages[1] = MutatingLaterStage(network.encoder.stages[1]).eval()
torch.manual_seed(333)
values = torch.randn((1, 2, 16, 16, 32))
with torch.inference_mode():
    baseline = network(values)
attached = attach_recomputed_stage_zero(network, expected_encoder_stages=3)
with torch.inference_mode():
    recomputed = network(values)
maximum_error = float((recomputed - baseline).abs().max().item())
assert attached["state_parameter_module_identities_preserved"] is True
assert maximum_error > 0.001
print(json.dumps({"scope": "generated three-stage counterexample",
                  "attachment_accepted": True,
                  "exact_logits": bool(torch.equal(recomputed, baseline)),
                  "max_absolute_logit_error": maximum_error}, sort_keys=True))
