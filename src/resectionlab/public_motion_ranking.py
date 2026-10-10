"""Training-only public nominal ranking labels; never a policy input.

A frozen owned-run manifest authenticates the score/trace/plan bytes. Hashes and
strict schemas prevent accidental rebinding; they do not authenticate a hostile
caller's falsely labelled data. Native source/replay qualification stays in the
existing collector/cache. This module performs no previews, loading or scoring.
"""
from dataclasses import dataclass
import math
import re

from .core import freeze_json, semantic_digest, thaw_json

VERSION = 'public_nominal_motion_gap_ranking_v1'
SCOPE = 'same_fixed_TRAIN_teacher_states_public_nominal_immediate_rewards'
OBJECTIVE_SOURCE = 'permitted_nominal_target_and_frozen_geometric_costs'
SUBJECTS = ('ReMIND-002', 'ReMIND-015', 'ReMIND-018', 'ReMIND-045')
STEPS = (1, 14, 1, 13)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r'sha256:[0-9a-f]{64}', value) is not None


def supervision_record(corpus_hash):
    if not _hash(corpus_hash):
        raise ValueError('Exact frozen public score corpus digest required')
    return {'version': VERSION, 'corpus_hash': corpus_hash, 'state_count': 29,
            'motion_states': 25, 'STOP_states': 4}


@dataclass(frozen=True)
class PublicMotionRankingCorpus:
    """Immutable labels bound to the frozen teacher worlds and observations."""
    _record: object
    fingerprint: str

    @classmethod
    def admit(cls, record, *, expected_hash):
        value = freeze_json(record)
        result = cls(value, expected_hash)
        result.require()
        return result

    def require(self):
        r = self._record
        if not _hash(self.fingerprint) or semantic_digest(r) != self.fingerprint:
            raise ValueError('Public motion score corpus identity differs')
        if (set(r) != {'version','scope','objective_source','private_reference_used','subjects'}
                or r['version'] != VERSION or r['scope'] != SCOPE
                or r['objective_source'] != OBJECTIVE_SOURCE or r['private_reference_used'] is not False
                or len(r['subjects']) != 4):
            raise ValueError('Only exact public nominal TRAIN score schema is admitted')
        for case, subject, count in zip(r['subjects'], SUBJECTS, STEPS):
            if (set(case) != {'subject','role','source_hash','decision_model_hash','input_sha256','decisions'}
                    or case['subject'] != subject or case['role'] != 'TRAIN'
                    or not _hash(case['source_hash']) or not _hash(case['decision_model_hash'])
                    or set(case['input_sha256']) != {'scores','trace','plan','replay'}
                    or any(not isinstance(h,str) or re.fullmatch(r'[0-9a-f]{64}',h) is None
                           for h in case['input_sha256'].values())
                    or len(case['decisions']) != count):
                raise ValueError('Exact four TRAIN worlds and complete teacher source pins required')
            for step, row in enumerate(case['decisions']):
                if (set(row) != {'step','observation_hash','source_state_hash','action_ids','action_mask',
                                 'teacher_action','rewards'}
                        or type(row['step']) is not int or row['step'] != step
                        or not _hash(row['observation_hash']) or not _hash(row['source_state_hash'])):
                    raise ValueError('Exact teacher state label schema required')
                ids, mask, rewards = row['action_ids'], row['action_mask'], row['rewards']
                if (not 1 <= len(ids) <= 121 or len(set(ids)) != len(ids) or ids[0] != 'STOP'
                        or any(not isinstance(a,str) or not a for a in ids)
                        or len(mask) != len(ids) or len(rewards) != len(ids)
                        or any(type(v) is not bool for v in mask) or mask[0] is not True
                        or any((type(v) not in (float,int) or not math.isfinite(v)) if legal else v is not None
                               for legal,v in zip(mask,rewards)) or rewards[0] != 0.):
                    raise ValueError('Exact complete legal action score vector required')
                legal = [i for i,v in enumerate(mask) if v]
                winner = max(legal, key=lambda i: rewards[i])
                if (row['teacher_action'] != ids[winner]
                        or (row['teacher_action'] == 'STOP') != (step == count-1)):
                    raise ValueError('Saved insertion-order nominal greedy teacher or terminal STOP differs')
        return self

    def record(self):
        self.require()
        return thaw_json(self._record)

    def require_trace(self, trace):
        # Exact trace admission and role/source checks are retained. Corpus
        # labels bind observation identity, not a stale old protocol context.
        self.require(); trace.require()
        record = trace.context.record()
        case = next((c for c in self._record['subjects'] if c['subject'] == record['subject']), None)
        if (case is None or record['role'] != 'TRAIN'
                or record['source_hash'] != case['source_hash']
                or record['decision_model_hash'] != case['decision_model_hash']
                or trace.behavior_parameter_hash is not None
                or len(trace.transitions) != len(case['decisions'])):
            raise ValueError('Foreign world, role or on-policy trace cannot supply ranking teachers')
        for expected, row, native in zip(case['decisions'], trace.transitions, trace.history):
            self.require_observation(expected, row.observation, row.action_id)
            chosen = expected['action_ids'].index(row.action_id)
            if (expected['rewards'][chosen] != row.reward
                    or (row.action_id != 'STOP' and
                        native.get('source_state_hash') != expected['source_state_hash'])):
                raise ValueError('Teacher score differs from replayed public state/reward')
        return case['decisions']

    @staticmethod
    def require_observation(labels, observation, teacher_action):
        observation.assert_intact()
        if (observation.fingerprint != labels['observation_hash']
                or tuple(observation.action_ids) != tuple(labels['action_ids'])
                or tuple(bool(v) for v in observation.action_mask) != tuple(labels['action_mask'])
                or teacher_action != labels['teacher_action']):
            raise ValueError('Ranking labels differ from the actual teacher observation/actions')


def motion_ranking_loss(logits, *, action_ids, action_mask, rewards, teacher_action):
    """STOP CE unchanged; motion gate + normalized reward-gap pairwise loss.

For motion rows, the old CE decomposes into -log P(motion) and conditional
winner CE. Only the latter is replaced. STOP's direct logit gradient is unchanged
at fixed logits. Shared-encoder gradients and later stopping behavior can change.
Exact ties receive zero weight. Small gaps are downweighted only relative to
other gaps in the SAME state; an all-roundoff-gap state would still receive a
full normalized term. No tuned tie threshold is applied. An all-tie/single-motion
state has zero ranking term. Rewards are labels, not actor features. Logits must
already mask illegal actions to -inf, as the unchanged SpatialPolicy does.
"""
    import torch
    if (logits.ndim != 1 or len(logits) != len(action_ids)
            or len(action_mask) != len(action_ids) or len(rewards) != len(action_ids)
            or action_ids[0] != 'STOP' or not action_mask[0]):
        raise ValueError('Exact masked-logit action alignment required')
    legal = [i for i,v in enumerate(action_mask) if v]
    if (teacher_action not in action_ids or action_ids.index(teacher_action) not in legal
            or not torch.isfinite(logits[legal]).all()):
        raise ValueError('Finite legal logits and legal teacher required')
    if teacher_action == 'STOP':
        return -logits.log_softmax(-1)[0]
    moves = [i for i in legal if i != 0]
    if not moves or any(type(rewards[i]) not in (int,float) or not math.isfinite(rewards[i]) for i in moves):
        raise ValueError('Finite public motion scores required')
    gate = torch.logsumexp(logits[legal],0)-torch.logsumexp(logits[moves],0)
    scale = max(1., max(abs(rewards[i]) for i in moves))
    pairs = [(i,j,(rewards[i]/scale-rewards[j]/scale))
             for i in moves for j in moves if rewards[i] > rewards[j]]
    denominator = math.fsum(gap for _,_,gap in pairs)
    if denominator == 0.:
        return gate
    # Each state contributes one normalized ranking term, irrespective of its
    # candidate/pair count. Max 120*119/2 scalar pairs, one existing forward.
    high = [i for i,_,_ in pairs]; low = [j for _,j,_ in pairs]
    weights = logits.new_tensor([gap/denominator for _,_,gap in pairs])
    return gate+(torch.nn.functional.softplus(logits[low]-logits[high])*weights).sum()
