"""Stdlib-only saved JSON/source audit; never reads weights or array payloads."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'build/frozen-gradient-diagnostic-owned-v1'
OUT = Path(__file__).resolve().parent
inputs = {}
checks = 0

def read(path, expected=None, parse=True):
    path = ROOT / path if not Path(path).is_absolute() else Path(path)
    assert path.suffix in ('.json', '.py'), path
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None:
        check(digest == expected.removeprefix('sha256:'), str(path))
    inputs[str(path.relative_to(ROOT))] = digest
    return json.loads(data) if parse else data

def check(value, message):
    global checks
    if not value:
        raise AssertionError(message)
    checks += 1

def close(a, b, message):
    check(math.isclose(a, b, rel_tol=2e-10, abs_tol=2e-12), message)

def semantic(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

r = read(BASE/'attempt-01/result.json', 'eef9082a81dfb56137c6f072184e699c233a9059bd24630ee1e45c93d681038c')
p = read(BASE/'attempt-01.supervision/receipt.json')
check(p['status'] == 'complete' and p['exit_code'] == 0 and not p['cleanup_errors']
      and not p['remaining_owned_pids'] and p['worker_termination_confirmed'], 'clean parent')
check(p['result_sha256'] == inputs[str((BASE/'attempt-01/result.json').relative_to(ROOT))], 'parent result pin')
release = read(BASE/'root-release.json', r['release_sha256'])
check(p['release_sha256'] == r['release_sha256'] == 'e702334009d41838d1ad13daaaf079a434c2da4f4e5eaca06742e51dbb3f0c3e', 'release')
index = read(ROOT/p['source_index']['path'], p['source_index']['sha256'])
check(index['head'] == release['expected_head'] == '4d1d0ff193275696c53cc907c8eb8aa79f6f3094', 'executed HEAD')
for name, pin in index['source_files'].items():
    read(name, pin, parse=False)
pins = read(BASE/'input-pins.json', release['input_pins_sha256'])
saved = {k: read(v['path'], v['sha256']) for k, v in pins.items()}
g = read(BASE/'attempt-01/gradient-alignment.json', r['gradient_report_sha256'])
c = read(BASE/'attempt-01/costs.json')
cache = read(BASE/'attempt-01/teacher-cache.json')
meta = read(BASE/'attempt-01/checkpoint-metadata.json')
corpus = read('build/public-motion-ranking-v1/public-score-corpus.json',
              index['files']['build/public-motion-ranking-v1/public-score-corpus.json'])
check(semantic(corpus) == g['corpus_hash'] == release['learning_protocol']['cohort_execution']['il_motion_supervision']['corpus_hash'], 'corpus')
check(semantic(release['learning_protocol']) == g['learning_protocol_hash'] == r['learning_protocol_hash'] == meta['learning_protocol_hash'] == cache['learning_protocol_hash'], 'protocol')
check({k:v for k,v in cache.items() if k not in ('cache_seal','traces')} ==
      {k:v for k,v in saved['teacher_cache'].items() if k not in ('cache_seal','traces')}, 'original cache scope')
for new, old in zip(cache['traces'], saved['teacher_cache']['traces']):
    check({k:v for k,v in new.items() if k!='independent_replay_hash'} ==
          {k:v for k,v in old.items() if k!='independent_replay_hash'}, 'exact original trace/plan/context/observation identity')
    subject = new['subject']
    replay = read(BASE/f'attempt-01/teachers/{subject}/native-replay.json')['independent_geometry']
    prior_replay = read(f'build/public-motion-ranking-il64-v1/IL64/attempt-01/teachers/{subject}/native-replay.json')['independent_geometry']
    check(semantic(replay) == new['independent_replay_hash'] and semantic(prior_replay) == old['independent_replay_hash'], 'fresh replay digest')
    check(replay['accepted'] and replay['complete_episode'] and replay['geometry']['feasible'], 'accepted replay')
    check({k:v for k,v in replay.items() if k!='evaluation_seconds'} ==
          {k:v for k,v in prior_replay.items() if k!='evaluation_seconds'}, 'replay differs only in measured evaluation duration')
check(semantic(cache['traces']) == cache['cache_seal'], 'new cache seal')
check(r['checkpoint'] == release['checkpoint'] and r['checkpoint']['parameter_hash'] == meta['parameter_hash'] == g['parameter_hash'], 'checkpoint metadata identity only; payload not read')
check(meta['completed_updates'] == 64 and meta['method'] == 'IL' and meta['architecture_hash'] == g['architecture_hash'], 'endpoint')
check(g['cache_seal'] == cache['cache_seal'] == r['cache_seal'] and cache['complete'], 'cache')
check(r['parameters_unchanged'] and r['gradient_buffers_unchanged'] and r['requires_grad_flags_restored'], 'worker frozen-state attestations')
for k, expected in {'policy_forwards':29, 'autograd_grad_calls':79, 'collector_replay_steps':58,
                    'completed_source_visits':4, 'checkpoint_loads':1, 'native_previews':5184,
                    'optimizer_calls':0, 'backward_calls':0, 'search_calls':0, 'forbidden_io_calls':0}.items():
    check(r[k] == expected, k)
check(g['counts'] == {'policy_forwards':29, 'autograd_grad_calls':79, 'optimizer_updates':0}, 'helper counts')
check(c['native_budget']['native_preview_entries'] == 5184 and sum(v.get('native_preview_entries',0) for v in c['costs'].values()) == 5184, 'preview accounting')
check(sum(v.get('native_transition_calls',0) for v in c['costs'].values()) == 58, 'transition accounting')
check(len(c['completed_patient_visits']) == 4 and all(v['source_released'] for v in c['completed_patient_visits']), 'source releases')
check(g['estimated_vector_payload_bound_bytes'] == 90*g['gradient_parameter_count']*8 <= 64*1024**2, 'vector bound')

labels = {(s['subject'], d['step']): d for s in corpus['subjects'] for d in s['decisions']}
check(len(g['rows']) == len(labels) == 29, 'denominator')
rows = []
for row in g['rows']:
    key = (row['subject'], row['step'])
    label, old = labels[key], saved[f'{key[0]}:{key[1]}']
    for field in ('observation_hash', 'teacher_action'):
        check(row[field] == label[field] == old[field], field)
    check(label['action_ids'] == old['action_ids'] and label['action_mask'] == old['action_mask'], 'exact actions and masks')
    moves = [i for i,m in enumerate(old['action_mask']) if m and i]
    check(row['legal_motion_count'] == len(moves), 'legal count')
    teacher = old['action_ids'].index(row['teacher_action'])
    if teacher:
        logits = old['scores']['logits']; chosen = max(moves, key=lambda i:logits[i])
        check(row['chosen_motion'] == old['action_ids'][chosen], 'chosen motion')
        close(row['teacher_minus_chosen_logit'], logits[teacher]-logits[chosen], 'margin')
        close(row['public_nominal_regret'], label['rewards'][teacher]-label['rewards'][chosen], 'regret')
        d = row['margin_directional_derivative_under_negative_raw_gradient']
        close(d['full_corpus'], sum(d[k] for k in ('ranking','gate','STOP')), 'derivative decomposition')
        rows.append({'subject':key[0], 'step':key[1], 'regret':row['public_nominal_regret'],
                     'margin':row['teacher_minus_chosen_logit'], **d})
    for summary in [row['whole_state_alignment_with_corpus_gradient'], *row['component_alignment_with_corpus_gradient'].values()]:
        close(summary['norm']**2, sum(v*v for v in summary['module_norms'].values()), 'module norm')
        if summary['norm']:
            close(summary['cosine'], summary['dot']/(summary['norm']*g['aggregate_total']['norm']), 'cosine')

cross = g['component_cross_dots']
for a in cross:
    close(g['aggregate_components'][a]['norm']**2, cross[a][a], 'diagonal norm')
    close(g['component_corpus_alignment'][a]['dot'], sum(cross[a].values()), 'corpus dot')
    close(sum(row['component_alignment_with_corpus_gradient'].get(a,{}).get('dot',0.) for row in g['rows']), sum(cross[a].values()), 'state component sum')
    for b in cross: close(cross[a][b], cross[b][a], 'symmetry')
close(g['aggregate_total']['norm']**2, sum(sum(v.values()) for v in cross.values()), 'total norm')
close(sum(row['whole_state_alignment_with_corpus_gradient']['dot'] for row in g['rows']), g['aggregate_total']['norm']**2, 'state sum')
counts = {}
for name, subset in [('all25', rows), ('positive_regret24', [r for r in rows if r['regret']>0]),
                     ('descriptive_regret_gt1', [r for r in rows if r['regret']>1])]:
    counts[name] = {'n':len(subset), **{k:{'positive':sum(r[k]>0 for r in subset),
        'negative':sum(r[k]<0 for r in subset),'zero':sum(r[k]==0 for r in subset)}
        for k in ('ranking','gate','STOP','full_corpus','same_state_ranking')}}
out = {'status':'PASS_saved_scalar_and_source_audit', 'checks':checks, 'inputs_verified':len(inputs),
    'source_files_verified':len(index['source_files']), 'saved_input_pins_verified':len(pins),
    'counts':counts, 'rows':rows,
    'gate_plus_STOP_norm':math.sqrt(cross['gate']['gate']+cross['STOP']['STOP']+2*cross['gate']['STOP']),
    'ranking_to_full_margin_sign_changes':sum((r['ranking']>0)-(r['ranking']<0) != (r['full_corpus']>0)-(r['full_corpus']<0) for r in rows),
    'same_state_help_full_harm':[[r['subject'],r['step']] for r in rows if r['same_state_ranking']>0 and r['full_corpus']<0],
    'scope':'Saved scalar identities and bindings only; no gradient recomputation, checkpoint payload, arrays, native/model execution or optimizer.'}
(OUT/'audit.json').write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')
(OUT/'inputs.json').write_text(json.dumps(inputs, indent=2, sort_keys=True)+'\n')
print(json.dumps({k:out[k] for k in ('status','checks','inputs_verified','counts','gate_plus_STOP_norm','ranking_to_full_margin_sign_changes')}, indent=2))
