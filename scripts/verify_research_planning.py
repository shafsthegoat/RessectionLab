#!/usr/bin/env python3
"""Independently replay saved development continuations; never fit or search.

Content seals are checked, then observations, actions and endpoint claims are
recomputed. Seals alone cannot establish truthful endpoints or clinical validity.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from resectionlab.core import array_digest, semantic_digest, thaw_json
from resectionlab.workspace_bundle import load_workspace
from resectionlab.surgical_environment import load_environment
from resectionlab.surgical_engine import SurgicalEngine
from resectionlab.surgical_rl_protection import restore_protection_contract
from resectionlab.surgical_rl_rewards import RewardTracker, compute_reward
from resectionlab.surgical_rl_env import SurgicalRLEnv, RLConfig
from resectionlab.surgical_observed_policy import ACTION_IDS, candidates, observed_greedy

SCHEMA = 'resectionlab.surgical-planning.v1'
CHECKPOINT_SHA = '554c3c743dd8ede2b3aa9f630ca25e7977abf864d38402b7b50fa5959455ebba'
PARAMETER_SHA = '29382995eb3aee03af3be77a3eae10fa1aaed414b6b4834770c1c8cf010dc30f'
SOURCES = {
    'sha256:e292ba41b650b221c90fb582bae49cd069597b119f9635e3f881b951e1138c56': 'sha256:c90de8fa14c77f2d2623b4e2e0070cefde8ea29e28189ba18d41865287bd6025',
    'sha256:e32dca89c598eeb2ee4d5b3808ef00e9ed9bee35624d7ce2bcaf6b86edcfb225': 'sha256:2900f709c06ac759fe7666b817edd9133602067e7e6aecd61a75e359f3391b29',
}
INPUT_CODE = {
    'surgical_observed_policy.py': 'fc53cd09a3ca87392e4bc7daea74491f7185369e3042fb26fbf2dfb869a53842',
    'surgical_rl_env.py': '01ed51dfa6a4bb9a889904a6aa377dc0a4b86c2a0bbef8477a3482b0226d2251',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, name):
    """Compare numeric trees without accepting changed keys or null semantics."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), name + ': keys differ')
        for key in expected:
            same(actual[key], expected[key], name + '.' + key)
    elif isinstance(expected, (list, tuple)):
        require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), name + ': length differs')
        for index, value in enumerate(expected):
            same(actual[index], value, name + '.' + str(index))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(type(actual) in (int, float) and math.isfinite(actual) and
                math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10), name + ': numeric claim differs')
    else:
        require(type(actual) is type(expected) and actual == expected, name + ': claim differs')


def actor():
    """Load only independently pinned weights; no planning-module loader used."""
    import torch
    path = ROOT / 'outputs/surgical-pilot-v1/policy-seed-17.pt'
    require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == CHECKPOINT_SHA,
            'Pinned checkpoint unavailable or bytes differ')
    for name, digest in INPUT_CODE.items():
        require(hashlib.sha256((ROOT / 'src/resectionlab' / name).read_bytes()).hexdigest() == digest,
                'Pinned learner input code differs: ' + name)
    record = torch.load(path, map_location='cpu', weights_only=True)
    require(record['seed'] == 17 and record['final_sha256'] == PARAMETER_SHA, 'Checkpoint ancestry differs')
    digest = hashlib.sha256()
    for key, value in sorted(record['state_dict'].items()):
        digest.update(key.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    require(digest.hexdigest() == PARAMETER_SHA, 'Checkpoint parameter digest differs')
    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Sequential(torch.nn.Linear(20, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
    model.load_state_dict({key.removeprefix('net.'): value for key, value in record['state_dict'].items()}, strict=True)
    model.eval()
    def score(features):
        with torch.no_grad():
            return model(torch.as_tensor(features, dtype=torch.float32)).squeeze(-1).numpy()
    return score


def measurements(engine, tracker):
    # Native desktop snapshots do not serialize RL reward-only high-water ledgers.
    # Never infer them from a final displacement or a proximity-free end state.
    return tracker.measure(engine, {'protected_damage_mm3': engine.protection.damage_mm3(engine.state['removed'])})


def observation(engine, tracker):
    config = RLConfig(require_critical_coverage=engine.guard_config.require_critical_coverage,
        critical_max_displacement_mm=engine.guard_config.critical_max_displacement_mm, local_spacing_mm=1.)
    env = SurgicalRLEnv(engine.scene, protection=engine.protection, config=config)
    env.engine = engine; env.measurements = measurements(engine, tracker)
    env.attempts = []; env.invalid_actions = 0
    return env._observe()


def source_hashes(case, scene):
    return {'image': array_digest(case.mri), 'labels': array_digest(scene.labels),
            'affine': array_digest(case.affine),
            'compartments': {name: array_digest(value) for name, value in case.compartments.items()}}


def direct_volumes(engine):
    """Independently classify removed source cells, including overlaps/exclusions."""
    tumor = healthy = bone = 0
    for cell in engine.state['removed']:
        region = engine.by_label.get(int(engine.scene.labels.flat[cell]), {})
        target = (region.get('target') is True and region.get('material') == 'tumor'
                  and not region.get('protected') and not region.get('critical')
                  and int(cell) not in engine._protected_ids)
        tumor += bool(target)
        bone += region.get('material') == 'bone'
        healthy += bool(region) and not target and region.get('material') != 'bone'
    volume = engine.scene.voxel_mm3
    result = dict(tumor_removed_mm3=tumor*volume, healthy_removed_mm3=healthy*volume, bone_removed_mm3=bone*volume,
        protected_damage_mm3=len(set(map(int,engine.state['removed'])) & set(engine._protected_ids))*volume)
    def area(cells):
        a,b = cells
        delta = np.abs(np.subtract(np.unravel_index(a,engine.scene.shape),np.unravel_index(b,engine.scene.shape)))
        require(delta.sum() == 1,'Vascular face does not join neighboring source cells')
        axes = [axis for axis in range(3) if delta[axis] == 0]
        return float(np.linalg.norm(np.cross(engine.scene.affine[:3,axes[0]],engine.scene.affine[:3,axes[1]])))
    if 'vesselsV2' in engine.state:
        result['vessel_damage_mm2'] = sum(area(face['cells'])*float(face['damage'])
                                        for face in engine.state['vesselsV2'].get('faces',{}).values())
    else:
        cuts = {tuple(sorted(map(int,edge))) for field in ('cutEdges','transectedEdges') for edge in engine.state.get(field,[])}
        result['vessel_damage_mm2'] = sum(area(edge) for edge in cuts if any(
            engine.by_label.get(int(engine.scene.labels.flat[cell]),{}).get('material') == 'vessel' for cell in edge))
    irrigation = sum(float(event['volumeMm3']) for row in engine.history for event in row.get('events',[])
                     if event.get('kind') == 'irrigation')
    result['bleeding_generated_mm3'] = max(0.,engine.state['fluidAddedMm3']-irrigation)
    return result


def verify(path):
    started = time.perf_counter()
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    case, artifacts, _ = load_workspace(path)
    record = thaw_json(artifacts)['surgical_toolbox']
    scene = load_environment(path)
    original = source_hashes(case, scene)
    comparison = record['planning']; document = dict(comparison)
    seal = document.pop('comparisonSeal', None)
    require(semantic_digest(document) == seal and document.get('schema') == SCHEMA, 'Comparison seal/schema differs')
    require(scene.identity in SOURCES and comparison['sourceSceneId'] == scene.identity, 'Source not independently admitted')
    require(comparison['sourceLabelsHash'] == original['labels'], 'Source labels binding differs')
    require(comparison['guardContractId'] == SOURCES[scene.identity], 'Pinned source exclusion identity differs')
    same(comparison['limits']['decisions'], 6, 'decision limit')
    same(comparison['limits']['incrementalHealthyMm3'], 5., 'healthy limit')
    same(comparison['optimizerUpdates'], 0, 'optimizer updates')
    plans = comparison['alternatives']
    require(len(plans) == 2 and plans[0]['method'] == 'greedy' and
            plans[1]['method'] in ('stop', 'checkpoint') and len({p['branchId'] for p in plans}) == 2,
            'Expected two distinct declared alternatives')
    scorer = actor() if plans[1]['method'] == 'checkpoint' else None
    common_prefix = None; results = []; total_calls = 0
    unknown = ['near_contact_exposure_mm3_seconds', 'deformation_injury_mm3']
    for plan in plans:
        branch = record['branches'][plan['branchId']]
        require(branch['sceneId'] == scene.identity, 'Plan branch source differs')
        contract = branch['protectionContract']
        require(contract['contentHash'] == comparison['guardContractId'], 'Plan guard identity differs')
        protection, guard = restore_protection_contract(scene, contract)
        engine = SurgicalEngine(scene, protection=protection, guard_config=guard)
        require(not engine.guard_report()['missingRequired'] and bool(engine.guard_report()['structures']),
                'Source exclusion membership is missing')
        count = plan['prefixActions']
        require(type(count) is int and 0 <= count <= len(branch['actions']), 'Invalid prefix count')
        prefix = branch['actions'][:count]
        if common_prefix is None: common_prefix = prefix
        require(prefix == common_prefix, 'Alternative prefixes differ')
        parent = record.get('parents', {}).get(plan['branchId']) or {}
        require(parent.get('position') == count and parent.get('branchId') in record['branches'] and
                record['branches'][parent['branchId']]['actions'][:count] == prefix,
                'Saved plan parent ancestry differs')
        for action in prefix:
            require(engine.step(action)['status'] != 'rejected', 'Source prefix rejected on fresh engine')
            total_calls += 1
        require(engine.state_id == comparison['startStateId'] == plan['startStateId'], 'Fresh starting state differs')
        require(engine.state['slots']['left']['instrument'] == 'cusa' and engine.state['slots']['left']['inserted'],
                'Starting instrument/post-exposure contract differs')
        tracker = RewardTracker(engine, engine._protected_ids)
        initial = measurements(engine, tracker); travel = engine.state['travelMm']
        rows = plan['decisions']; require(1 <= len(rows) <= 6, 'Invalid decision count')
        require(plan['planSeal'] == semantic_digest({'sourceSceneId':scene.identity,
            'startStateId':engine.state_id, 'decisions':rows}), 'Plan ancestry seal differs')
        same(plan['checkpointSha256'], CHECKPOINT_SHA if plan['method'] == 'checkpoint' else None, 'checkpoint identity')
        reason = 'decision_limit'; reward_sum = 0.; invalid = 0
        for index, row in enumerate(rows):
            obs = observation(engine, tracker)
            same(row['index'], index, 'decision index')
            require(row['observationHash'] == semantic_digest({key:array_digest(value) for key,value in obs.items()}),
                    'Observation digest differs at decision ' + str(index))
            features, options = candidates(obs)
            if plan['method'] == 'greedy': selected, scores = observed_greedy(features)
            elif plan['method'] == 'stop': selected, scores = 8, [0.]*8+[1.]
            else:
                scores = np.asarray(scorer(features), float); selected = int(scores.argmax()); scores = scores.tolist()
            same(row['scores'], scores, 'actor scores')
            same(row['actionId'], ACTION_IDS[selected], 'selected action identity')
            same(row['action'], options[selected], 'selected public action')
            before = measurements(engine, tracker); outcome = engine.step(row['action']); total_calls += 1
            after = measurements(engine, tracker)
            same(row['status'], outcome['status'], 'execution status')
            same(row['reason'], outcome.get('reason'), 'execution reason')
            same(row['stateId'], engine.state_id, 'decision state')
            reward, terms = compute_reward(before, after, outcome)
            same(row['reward'], reward, 'decision reward'); same(row['rewardTerms'], terms, 'reward decomposition')
            reward_sum += reward; invalid += outcome['status'] == 'rejected'
            if (after['healthy_removed_mm3']-initial['healthy_removed_mm3'] > 5. or
                any(after[key] > initial[key] for key in ('vessel_damage_mm2','protected_damage_mm3','bleeding_generated_mm3'))):
                reason = 'constraint_budget_failure'
            elif selected == 8: reason = 'policy_stop'
            elif outcome['status'] == 'rejected': reason = 'exact_execution_rejected'
            if reason != 'decision_limit':
                require(index == len(rows)-1, 'Decisions continue after declared terminal condition')
        require(reason != 'decision_limit' or len(rows) == 6, 'Premature horizon truncation')
        final = measurements(engine, tracker)
        for key, value in direct_volumes(engine).items(): same(final[key], value, 'direct source accounting.'+key)
        same(plan['stopReason'], reason, 'stop reason'); same(plan['truncated'], reason == 'decision_limit', 'truncation')
        same(plan['invalidActions'], invalid, 'invalid actions')
        same(plan['earlyStop'], reason == 'policy_stop' and final['tumor_removed_mm3'] == initial['tumor_removed_mm3'], 'early STOP')
        same(plan['localReward'], reward_sum, 'total local reward')
        same(plan['unassessedEndpoints'], unknown, 'unassessed endpoint ledger')
        same(plan['cumulativeEndpoints'], {k:None if k in unknown else v for k,v in final.items()}, 'cumulative endpoints')
        same(plan['incrementalEndpoints'], {k:None if k in unknown else v-initial[k] for k,v in final.items()}, 'incremental endpoints')
        target_volume = sum(np.count_nonzero(scene.labels == r['id'])*scene.voxel_mm3 for r in scene.regions
                            if r.get('target') and r['material'] == 'tumor')
        same(plan['fullTargetFractionRemoved'],engine.metrics()['targetRemovedMm3']/target_volume,'full target fraction')
        same(plan['reachableTargetFractionRemoved'],None,'reachable target unknown')
        same(plan['neurologicalHarm'],'unknown; no calibrated outcome model','clinical outcome unknown')
        same(plan['travelMm'],engine.state['travelMm']-travel,'tool travel'); same(plan['instrumentChanges'],0,'tool changes')
        require(engine.actions == branch['actions'] and engine.state_id == branch['stateId'] == plan['finalStateId'],
                'Saved plan branch differs from fresh complete execution')
        checkpoint = branch['checkpoint']
        same(checkpoint['position'],len(engine.actions),'branch checkpoint position')
        same(checkpoint['stateId'],engine.state_id,'branch checkpoint identity')
        require(semantic_digest(checkpoint['state']) == semantic_digest(engine.state),'Saved branch anatomy differs')
        require(semantic_digest(record.get('timeline',{}).get(plan['branchId'])) == semantic_digest(engine.history),
                'Saved plan timeline differs from independently replayed history')
        cost = plan['cost']
        for key,value in {'executionCalls':len(rows),'freshReplayCalls':len(prefix)+len(rows),
                          'previewCalls':0,'optimizerUpdates':0}.items(): same(cost[key],value,'cost.'+key)
        for key in ('planningSeconds','freshReplaySeconds'):
            require(type(cost[key]) in (int,float) and math.isfinite(cost[key]) and cost[key] >= 0,'Invalid recorded timing')
        require(plan['verification']['exactStateMatch'] is True and plan['verification']['endpointVectorMatch'] is True,
                'Missing declared native verification')
        results.append({'branchId':plan['branchId'],'method':plan['method'],'decisions':len(rows),
            'stopReason':reason,'invalidActions':invalid,'finalStateId':engine.state_id,
            'decisionOutcomes':[{'actionId':row['actionId'],'status':row['status'],'reason':row['reason']} for row in rows],
            'localReward':reward_sum,
            'cumulativeEndpoints':plan['cumulativeEndpoints'],'incrementalEndpoints':plan['incrementalEndpoints']})
    require(source_hashes(case,scene) == original,'Source arrays mutated during verification')
    require(hashlib.sha256(path.read_bytes()).hexdigest() == file_sha,'Saved workspace changed during verification')
    return {'schema':'resectionlab.independent-research-planning-verification.v1','status':'verified',
        'workspace':str(path),'workspaceSha256':file_sha,'sourceSceneId':scene.identity,
        'sourceArraysUnchanged':True,'sourceArrayHashes':original,'alternatives':results,'freshEngineCalls':total_calls,
        'verificationWallSeconds':time.perf_counter()-started,'optimizerUpdates':0,'searchCalls':0,
        'timingScope':'Recorded elapsed times checked for finite nonnegative values; historical wall time cannot be reconstructed.',
        'scope':'Independent receipt/action/endpoint reconstruction in the shared simulator; no clinical or held-out validation.',
        'unassessedEndpoints':unknown}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output:
        output=args.output.resolve()
        if not output.is_relative_to((ROOT/'testcases').resolve()) or output.suffix != '.json':
            parser.error('--output must be a .json path under this repository testcases/')
    try:
        report=verify(args.workspace.resolve())
    except Exception as error:
        report={'schema':'resectionlab.independent-research-planning-verification.v1','status':'failed',
                'errorType':type(error).__name__,'error':str(error),'optimizerUpdates':0,'searchCalls':0}
    encoded=json.dumps(report,indent=2,allow_nan=False)+'\n'
    if args.output:
        output.parent.mkdir(parents=True,exist_ok=True); output.write_text(encoded)
    print(encoded,end='')
    return 0 if report['status']=='verified' else 1

if __name__ == '__main__':
    sys.exit(main())
