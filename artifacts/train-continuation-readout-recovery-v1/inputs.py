"""Authenticate saved complete traces; no trace authority is minted here."""
from pathlib import Path
from common import ROOT, TRAIN, IL, RL, OLD_INDEX, PUBLIC, PUBLIC_SHA, COHORT, COHORT_SHA, CHECKPOINTS, read_json, read_bytes, semantic, sha, ORIGINAL_RUN,ORIGINAL_REPORT_SHA,ORIGINAL_TARGETS_SHA,ORIGINAL_AUDIT,ORIGINAL_AUDIT_SHA, CACHE_ATTEMPT,CACHE_SOURCE_INDEX,CACHE_SOURCE_SHA,CACHE_PINS

OLD_INDEX_SHA = 'b1e24a43731ba373fce2055784bfa81ce1e981177e6224cce6a80bd7ece827c8'
IL_INDEX = 'artifacts/post-exposure-il64-result-v1/input-index.json'
IL_INDEX_SHA = 'dca05801a36c979630200b08a486a72abbb61e8b9bcf6730d6c98f2921ba2ba7'
WORLD_KEYS = ('subject','source_hash','decision_model_hash','public_observation_binding','observation_track','max_steps')


def trace_rows():
    for subject in TRAIN:
        yield {'key': subject+'/teacher', 'kind': 'teacher', 'subject': subject, 'base': RL+'/teachers/'+subject}
        for update in range(1, 9):
            yield {'key': subject+'/RL%02d'%update, 'kind': 'RL', 'subject': subject,
                   'base': RL+'/RL/update-%02d/'%update+subject}


def metadata_pins():
    """Root freeze reads JSON only; existing arrays/weights stay unopened."""
    old = read_json(ROOT/OLD_INDEX, OLD_INDEX_SHA); prior_il = read_json(ROOT/IL_INDEX, IL_INDEX_SHA)
    pins = {OLD_INDEX: OLD_INDEX_SHA, IL_INDEX: IL_INDEX_SHA, PUBLIC: PUBLIC_SHA, COHORT: COHORT_SHA,
            ORIGINAL_RUN+'/report.json':ORIGINAL_REPORT_SHA,ORIGINAL_RUN+'/logged-targets.json':ORIGINAL_TARGETS_SHA,
            ORIGINAL_RUN+'/launcher.json':'7713f9fe9364a723a807dee7ebb09555fa1c3c67b6dde5989f907748d705e0dd',
            ORIGINAL_AUDIT:ORIGINAL_AUDIT_SHA}
    for row in trace_rows():
        for name in ('complete-trace.json','plan.json','native-replay.json'):
            path = row['base']+'/'+name; pins[path] = old[path]['sha256']
    for base in (IL, RL):
        pins[base+'/configuration.json'] = old[base+'/configuration.json']['sha256']
        pins[base+'/result.json'] = (old if base == RL else prior_il)[base+'/result.json']['sha256']
        for subject in TRAIN:
            path = base+'/'+subject+'-context.json'
            pins[path] = old[path]['sha256'] if path in old else sha(ROOT/path)
    for path in ('build/post-exposure-IL045-evaluation-recovery-v1/attempt-01/result.json',
                 'build/post-exposure-IL045-evaluation-recovery-v1/attempt-01.supervision/receipt.json',
                 RL+'.supervision/receipt.json'):
        pins[path] = old[path]['sha256']
    pins[CACHE_SOURCE_INDEX]=CACHE_SOURCE_SHA
    pins.update({CACHE_ATTEMPT+'/'+name:digest for name,digest in CACHE_PINS.items()})
    public = read_json(ROOT/PUBLIC, PUBLIC_SHA)
    if [r['patient_id'] for r in public['cases']] != list(TRAIN): raise ValueError('Original four TRAIN order')
    arrays = {}
    for row in public['cases']:
        path = Path(row['path']); name = str(path.relative_to(ROOT)); pins[name] = row['sha256']
        manifest = read_json(path, row['sha256'])
        if manifest['patient_id'] != row['patient_id'] or manifest['private_evaluation_files_included'] is not False:
            raise ValueError('Public-only original subject binding')
        for record in manifest['input_files'].values():
            arrays[str(Path(record['path']).relative_to(ROOT))] = {k:record[k] for k in ('sha256','bytes','dtype')}
    return pins, arrays


def pinned(index, path): return read_json(ROOT/path, index['input_metadata'][path])


def saved_trace(index, row):
    raw = pinned(index, row['base']+'/complete-trace.json')
    plan = pinned(index, row['base']+'/plan.json')
    replay = pinned(index, row['base']+'/native-replay.json')
    context = pinned(index, RL+'/'+row['subject']+'-context.json')
    history = raw['metrics']['history']; decisions = raw['decisions']
    behavior = decisions[0]['behavior_parameter_hash']
    if (not 1 <= len(decisions) <= 24 or len(history) != len(decisions)
            or raw['metrics']['terminated'] is not True or decisions[-1]['terminated'] is not True
            or any(d['terminated'] is not (i == len(decisions)-1) for i,d in enumerate(decisions))
            or any(d['behavior_parameter_hash'] != behavior for d in decisions)
            or (row['kind'] == 'teacher') != (behavior is None)
            or raw['context_hash'] != semantic(context) or context['role'] != 'TRAIN'
            or context['subject'] != row['subject'] or raw['metrics']['planning_estimator_only'] is not True
            or raw['metrics']['source_hash'] != context['source_hash']
            or raw['metrics']['decision_model_hash'] != context['decision_model_hash']
            or any(d['action_id'] != h['action_id'] or d['reward'] != h['reward'] for d,h in zip(decisions, history))):
        raise ValueError('Saved complete historical trace join differs')
    seal = semantic({'context': raw['context_hash'], 'observations': [d['observation_hash'] for d in decisions],
                     'history': history, 'behavior_parameter_hash': behavior})
    def normalized(rows): return [{k:v for k,v in r.items() if k != 'outcome_scope'} for r in rows]
    audit = replay['independent_geometry']
    if (raw['trace_seal'] != seal or semantic(plan['plan']) != plan['plan_seal']
            or plan['plan']['history'] != history or plan['plan']['actions'] != [d['action_id'] for d in decisions]
            or audit['accepted'] is not True or audit['complete_episode'] is not True
            or audit['committed_history_hash'] != semantic(replay['metrics']['history'])
            or normalized(history) != normalized(replay['metrics']['history'])):
        raise ValueError('Saved plan/certificate does not authenticate complete trace')
    descriptor = {'key':row['key'], 'kind':row['kind'], 'world':{k:context[k] for k in WORLD_KEYS},
        'behavior_parameter_hash':behavior, 'history_hash':semantic(history),
        'decisions':[{k:d[k] for k in ('observation_hash','action_id','reward','terminated')} for d in decisions]}
    return raw, descriptor, {'historical_trace_seal':seal, 'historical_plan_seal':plan['plan_seal'],
        'original_trace_sha256':index['input_metadata'][row['base']+'/complete-trace.json'],
        'original_replay_sha256':index['input_metadata'][row['base']+'/native-replay.json'],
        'behavior_parameter_hash_is_historical_only': True}


def manifest(index):
    rows = [saved_trace(index, row)[1] for row in trace_rows()]
    if (len(rows) != 36 or sum(len(r['decisions']) for r in rows) != 497
            or sum(len(r['decisions']) for r in rows if r['kind']=='teacher') != 29):
        raise ValueError('Exact36 complete traces/497 states/29teacher states required')
    return rows
