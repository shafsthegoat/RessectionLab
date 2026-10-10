"""Pure readout math and fixed teacher-state indexing; no project/ML imports."""
import math
import statistics


def terminal_outcome(elapsed, stop_reason, code, errors, result_exists):
    # A successful exit during the final polling interval cannot waive the cap.
    if elapsed >= 30.:
        stop_reason = stop_reason or 'hard_total_wall_cap'
    return stop_reason is None and code == 0 and not errors and result_exists, stop_reason


def index_teacher_states(teachers, updates, digest):
    if len(teachers) != 24 or len(updates) != 32:
        raise ValueError('Require all24 TRAIN teachers and all32 IL receipts')
    observations = {}
    for i, update in enumerate(updates, 1):
        if update['method'] != 'IL' or update['update'] != i:
            raise ValueError('IL receipt order changed')
        for sample in update['samples']:
            key = (sample['binding_hash'], sample['action_id'])
            previous = observations.setdefault(key, sample['observation_hash'])
            if previous != sample['observation_hash']:
                raise ValueError('Teacher label has conflicting observation fingerprints')
    states = []
    for i, teacher in enumerate(teachers):
        if (teacher['teacher_index'] != i or teacher['role'] != 'TRAIN'
                or teacher['status'] != 'complete_bounded_teacher'):
            raise ValueError('Require original ordered complete TRAIN teachers')
        binding = teacher['binding']
        if (binding['role'] != 'TRAIN' or binding['layout_id'] != teacher['layout_id']
                or binding['goal_id'] != teacher['goal_id']):
            raise ValueError('Teacher binding differs from TRAIN identity')
        strategy = teacher['strategy']['strategy']
        if digest(strategy) != teacher['strategy']['strategySeal']:
            raise ValueError('Saved teacher seal differs')
        if teacher['label_count'] != len(strategy['actions']):
            raise ValueError('Teacher label count differs')
        binding_hash = digest(binding)
        for step, action in enumerate(strategy['actions']):
            states.append({'teacher_index':i,'layout_id':teacher['layout_id'],
                'goal_id':teacher['goal_id'],'step':step,'binding_hash':binding_hash,
                'teacher_action':action,'observation_hash':observations[(binding_hash,action)]})
    if len(states) != 40 or sum(s['step']==0 for s in states) != 24 or sum(s['step']==1 for s in states) != 16:
        raise ValueError('Fixed40/root24/second16 inventory changed')
    if {(s['binding_hash'],s['teacher_action']) for s in states} != set(observations):
        raise ValueError('IL receipts differ from the complete40 teacher-state set')
    return states


def summarize_logits(logits, mask, actions, modes, teacher_action):
    if not (len(logits)==len(mask)==len(actions)==len(modes)) or actions[0]!='STOP' or not mask[0]:
        raise ValueError('Malformed fixed candidate rows')
    chosen = actions.index(teacher_action)
    if not mask[chosen] or not all(math.isfinite(v) for v,m in zip(logits,mask) if m):
        raise ValueError('Illegal teacher action or nonfinite legal logits')
    if any(v != -math.inf for v,m in zip(logits,mask) if not m):
        raise ValueError('Masked logits must remain negative infinity')
    legal = [i for i,allowed in enumerate(mask) if allowed]
    maximum = max(logits[i] for i in legal)
    weights = [math.exp(v-maximum) if allowed else 0. for v,allowed in zip(logits,mask)]
    denominator = sum(weights); probabilities = [v/denominator for v in weights]
    greedy = max(legal,key=lambda i:logits[i])  # inventory-first tie, exactly the actor
    movement = [i for i in legal if modes[i]!='stop']
    return {'action_ids':list(actions),'action_modes':list(modes),'legal_mask':list(mask),
        'teacher_action_index':chosen,'logits':[v if allowed else None for v,allowed in zip(logits,mask)],
        'teacher_probability':probabilities[chosen],'STOP_probability':probabilities[0],
        'STOP_minus_best_movement_margin':None if not movement else logits[0]-max(logits[i] for i in movement),
        'greedy_action':actions[greedy],'greedy_mode':modes[greedy],'correct':greedy==chosen,
        'cross_entropy':maximum+math.log(denominator)-logits[chosen],
        'entropy':-sum(p*math.log(p) for p in probabilities if p>0),
        'masked_logit_encoding':'null means masked negative infinity; no available score altered'}


def aggregate_readouts(rows):
    result = {}
    for label, subset in [('all40',rows),('root24',[r for r in rows if r['step']==0]),
                           ('second16',[r for r in rows if r['step']==1])]:
        result[label] = {'count':len(subset), 'mean_cross_entropy':statistics.mean(r['cross_entropy'] for r in subset),
            'accuracy':statistics.mean(r['correct'] for r in subset),
            'mean_teacher_probability':statistics.mean(r['teacher_probability'] for r in subset),
            'mean_STOP_probability':statistics.mean(r['STOP_probability'] for r in subset),
            'greedy_STOP_count':sum(r['greedy_mode']=='stop' for r in subset)}
    return result
