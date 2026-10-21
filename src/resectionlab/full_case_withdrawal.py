"""Read-only outward withdrawal proposals through an existing narrow passage.

Explicit close-contact waypoints use the unchanged public swept-envelope law.
No state/source data are changed, no transition executes, and every proposal
requires public preview/step. An incomplete outward prefix never claims that a
hand has been withdrawn or may exchange its instrument.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .surgical_motion import envelopes, swept_envelopes, pair_conflict
from .surgical_tools import TOOLS


def recorded_withdrawal_candidate(engine, slot='left', *, reference_position=None):
    """Retrace an accepted insertion route, checking current full-tool cover.

    Only fully accepted pose/path records since this hand's last placement are
    route evidence. Intended waypoints from partial actions are excluded. Freeze
    the reference with an explicit action boundary, or infer its end from the
    first small-aperture cleanup inspection after an insertion apply. The
    earliest nearby checked join avoids undoing a prior withdrawal prefix and
    permits repeated calls without a hidden route cursor or state mutation.
    """
    state = engine.state['slots'][slot]
    if (state['instrument'] != 'cusa' or not state['inserted'] or state['active']
            or state['grip'] or slot in engine.state['holds']
            or len(engine.history) != len(engine.actions)):
        return None
    placement = next((index for index in range(len(engine.actions) - 1, -1, -1)
                      if engine.actions[index]['slot'] == slot and engine.actions[index]['operation'] == 'place'), None)
    if placement is None:
        return None
    if reference_position is not None:
        if type(reference_position) is not int or not placement < reference_position <= len(engine.actions):
            raise ValueError('Recorded withdrawal reference boundary is unavailable')
        cutoff = reference_position
    else:
        cutoff = None
        inserted_apply = False
        for index in range(placement + 1, len(engine.actions)):
            action = engine.actions[index]
            if action['slot'] != slot or engine.history[index].get('status') != 'accepted':
                continue
            if action['operation'] == 'apply' and ('path' in action or 'pose' in action):
                inserted_apply = True
            if (inserted_apply and action['operation'] == 'inspect'
                    and action.get('parameters', {}).get('radiusMm', float('inf')) <= .5):
                cutoff = index
                break
        if cutoff is None:
            # Without a cleanup marker or explicit immutable boundary, a new
            # recovery apply is indistinguishable from another insertion fan.
            # Retain the original outward strategy rather than grow a route
            # that could reverse its own just-completed recovery prefix.
            return None
    route = []
    for action, receipt in zip(engine.actions[placement:cutoff], engine.history[placement:cutoff]):
        if action['slot'] != slot or receipt.get('status') != 'accepted':
            continue
        if receipt.get('action') != action:
            return None
        if action['operation'] in ('place', 'apply', 'move', 'withdraw'):
            route.extend(action.get('path', [action['pose']] if 'pose' in action else []))
    if not route or engine.history[placement].get('status') != 'accepted':
        return None
    # Sparse accepted pose actions still traversed linear/Slerp segments. Keep
    # join anchors along those segments so a 512-pose return prefix can resume
    # between the original explicit waypoints without reversing its new action.
    anchored = [route[0]]
    for endpoint in route[1:]:
        previous = anchored[-1]
        motion = (np.linalg.norm(np.subtract(previous['position'], endpoint['position']))
                  + 40. * (Rotation.from_quat(previous['quaternion']) *
                           Rotation.from_quat(endpoint['quaternion']).inv()).magnitude())
        count = max(1, int(np.ceil(motion / .25)))
        rotations = Slerp([0., 1.], Rotation.from_quat([previous['quaternion'], endpoint['quaternion']]))
        for index in range(1, count + 1):
            fraction = index / count
            anchored.append({'position': ((1. - fraction) * np.asarray(previous['position'])
                             + fraction * np.asarray(endpoint['position'])).tolist(),
                             'quaternion': rotations(fraction).as_quat().tolist()})
    route = anchored
    scene = engine.scene
    params = engine.parameters(state)
    tool = TOOLS[state['instrument']]
    owner = engine.occupancy()
    labels = scene.labels.ravel()
    targets = {region['id'] for region in scene.regions if region.get('target') and region['material'] == 'tumor'}
    other_parts = []
    for hand, other in engine.state['slots'].items():
        if hand != slot:
            other_parts.extend(envelopes(TOOLS[other['instrument']], engine.parameters(other),
                                          other['position'], other['quaternion']))
    current = {'position': list(state['position']), 'quaternion': list(state['quaternion'])}
    def distance(first, second):
        return float(np.linalg.norm(np.subtract(first['position'], second['position']))
                     + 40. * (Rotation.from_quat(first['quaternion']) *
                              Rotation.from_quat(second['quaternion']).inv()).magnitude())
    def checked(first, second, removed):
        parts = swept_envelopes(tool, params, first, second)
        if other_parts and pair_conflict(parts, other_parts):
            return None
        working, shaft = set(), set()
        for a, b, radius, part in parts:
            touched = set(owner.material_ids(engine.cells(a, b, radius))) - removed
            (shaft if part == 'shaft' else working).update(touched)
        if shaft:
            return None
        tip = set(owner.material_ids(engine.cells(first['position'], second['position'], params['radiusMm']))) - removed
        prior = set(owner.material_ids(engine.cells(first['position'], first['position'], params['radiusMm']))) - removed
        needs_active = bool(working or tip - prior)
        if needs_active and any(int(labels[cell]) not in targets for cell in working | tip):
            return None
        return needs_active, tip if needs_active else set()
    def segment(first, second, removed, budget):
        length = distance(first, second)
        if length < 1e-10:
            return [], set(removed), False, True
        best = ([], set(removed), False, False)
        for maximum in (.05, .025, .01, .005):
            count = max(1, int(np.ceil(length / min(maximum, float(scene.spacing.min()) / 3.))))
            rotations = Slerp([0., 1.], Rotation.from_quat([first['quaternion'], second['quaternion']]))
            path, local_removed, active = [], set(removed), False
            previous = first
            for index in range(1, min(count, budget) + 1):
                fraction = index / count
                point = ((1. - fraction) * np.asarray(first['position'])
                         + fraction * np.asarray(second['position'])).tolist()
                pose = {'position': point, 'quaternion': rotations(fraction).as_quat().tolist()}
                assessment = checked(previous, pose, local_removed)
                if assessment is None:
                    break
                needs_active, newly_removed = assessment
                active = active or needs_active
                local_removed.update(newly_removed)
                path.append(pose); previous = pose
            complete = len(path) == count
            proposal = (path, local_removed, active, complete)
            if complete or len(path) == budget:
                return proposal
            if len(path) > len(best[0]):
                best = proposal
        return best
    # Joining is always a checked motion, including its orientation. Choosing
    # the earliest feasible nearby record favors actual outward route progress
    # over reversing a newly appended withdrawal move back into the cavity.
    joins = [index for index, pose in enumerate(route) if distance(current, pose) <= .5]
    for join in joins:
        path, removed, active, connected = segment(current, route[join], set(), 512)
        if not connected:
            continue
        completed = True
        previous = path[-1] if path else current
        for endpoint in reversed(route[:join]):
            if len(path) == 512:
                completed = False; break
            following, removed, needed, reached = segment(previous, endpoint, removed, 512 - len(path))
            path.extend(following); active = active or needed
            if following: previous = following[-1]
            if not reached:
                completed = False; break
        if not path and not completed:
            continue
        operation = 'apply' if active else ('withdraw' if completed else 'move')
        if not path:
            return {'slot': slot, 'operation': operation, 'pose': current, 'duration': .1}
        travel = sum(distance(a, b) for a, b in zip([current, *path[:-1]], path))
        return {'slot': slot, 'operation': operation, 'path': path,
                'duration': max(.1, min(30., travel / 20.))}
    return None


def withdrawal_candidate(engine, slot='left', *, reference_position=None):
    """Keep complete outward exits fast; repair blocked prefixes by retracing."""
    outward = _outward_withdrawal_candidate(engine, slot=slot)
    if outward is not None and outward['operation'] == 'withdraw':
        return outward
    recorded = recorded_withdrawal_candidate(engine, slot=slot, reference_position=reference_position)
    return recorded if recorded is not None else outward


def _outward_withdrawal_candidate(engine, slot='left'):
    """Return a checked outward public action, or None when none is available.

    Root setup must deactivate/release first and may explicitly reduce a large
    CUSA aperture before planning. Physical shaft radius is unchanged. Step
    lengths adapt from .333 to .05/.025/.01 mm where the full swept shaft would
    otherwise touch retained material. At most 512 poses are returned. A move
    or target-only CUSA apply prefix remains inserted; only a complete outward
    route uses operation withdraw. Active pullback is a distinct apply proposal
    and must pass the caller's public-preview target-only removal admission.
    """
    state=engine.state['slots'][slot]
    if not state['inserted'] or state['active'] or state['grip'] or slot in engine.state['holds']:
        return None
    tool=TOOLS[state['instrument']]
    params=engine.parameters(state)
    q=list(state['quaternion'])
    start=np.asarray(state['position'],float)
    outward=-Rotation.from_quat(q).apply([0.,0.,1.])
    corners=np.asarray(list(itertools.product(*[(0,n-1) for n in engine.scene.shape])))
    world=corners @ engine.scene.affine[:3,:3].T+engine.scene.affine[:3,3]
    distance=max(0.,float(np.max(world @ outward)-start @ outward+8.))
    if distance <= 1e-8:
        # Finish the logical withdrawal even if a prior checked move prefix
        # already reached the exterior. An unchanged pose remains checked.
        return {'slot':slot,'operation':'withdraw','pose':{'position':start.tolist(),'quaternion':q},'duration':.1}
    owner=engine.occupancy()
    labels=engine.scene.labels.ravel()
    target_ids={r['id'] for r in engine.scene.regions if r.get('target')}
    other_parts=[]
    for other_slot,other in engine.state['slots'].items():
        if other_slot != slot:
            other_parts.extend(envelopes(TOOLS[other['instrument']],engine.parameters(other),other['position'],other['quaternion']))
    current={'position':start.tolist(),'quaternion':q}
    path=[]; travelled=0.; requires_active=False
    while travelled < distance-1e-8 and len(path)<512:
        selected=None
        for length in (.333,.05,.025,.01):
            step=min(length,distance-travelled)
            point=start+outward*(travelled+step)
            pose={'position':point.tolist(),'quaternion':q}
            parts=swept_envelopes(tool,params,current,pose)
            if other_parts and pair_conflict(parts,other_parts): continue
            working=set();shaft=set()
            for a,b,radius,part in parts:
                cells=set(owner.material_ids(engine.cells(a,b,radius)))
                (shaft if part=='shaft' else working).update(cells)
            if shaft: continue
            tip=set(owner.material_ids(engine.cells(current['position'],point,params['radiusMm'])))
            prior=set(owner.material_ids(engine.cells(current['position'],current['position'],params['radiusMm'])))
            needs_active=bool(working)
            if needs_active:
                # Only the remover's supplied owned target may be contacted
                # actively; normal tissue is never admitted by this helper.
                if state['instrument']!='cusa' or any(int(labels[c]) not in target_ids for c in working|tip): continue
            elif tip-prior:
                continue
            selected=(step,pose,needs_active)
            break
        if selected is None: break
        step,pose,needs_active=selected
        path.append(pose); current=pose; travelled+=step
        requires_active=requires_active or needs_active
    if not path:return None
    completed=travelled>=distance-1e-8
    operation='apply' if requires_active else ('withdraw' if completed else 'move')
    # An active pullback never marks the hand withdrawn, even on an outward
    # route: ordinary deactivate/withdraw must finish its instrument lifecycle.
    return {'slot':slot,'operation':operation,'path':path,
            'duration':max(.1,min(30.,travelled/20.))}
