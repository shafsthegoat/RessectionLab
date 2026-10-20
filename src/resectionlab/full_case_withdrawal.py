"""Read-only outward withdrawal proposals through an existing narrow passage.

Explicit close-contact waypoints use the unchanged public swept-envelope law.
No state/source data are changed, no transition executes, and every proposal
requires public preview/step. An incomplete outward prefix never claims that a
hand has been withdrawn or may exchange its instrument.
"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.spatial.transform import Rotation

from .surgical_motion import envelopes, swept_envelopes, pair_conflict
from .surgical_tools import TOOLS


def withdrawal_candidate(engine, slot='left'):
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
