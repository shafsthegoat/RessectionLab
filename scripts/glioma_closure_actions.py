"""Public return of a supported, folded generated scalp or dural sheet."""

import numpy as np
from scipy.spatial.transform import Rotation

from resectionlab.surgical_exposure_geometry import grip_trajectory

if __package__:
    from .glioma_bimanual_routes import _first_grip, _hand_geometry, retained_site
    from .glioma_walkthrough_actions import DOWN, body, edge_quaternion, pose
else:
    from glioma_bimanual_routes import _first_grip, _hand_geometry, retained_site
    from glioma_walkthrough_actions import DOWN, body, edge_quaternion, pose


def return_sheet(live, name, holding, assistant):
    """Engage the occupied source grip, remove the physical stay, return and release."""
    sheet = body(live, name)
    actual = np.asarray(sheet['gripMm'])
    ijk = np.rint(live.scene.inverse[:3, :3] @ actual + live.scene.inverse[:3, 3]).astype(int)
    grid_center = live.scene.points([np.ravel_multi_index(ijk, live.scene.shape)])[0]
    # The closed jaw meets the exterior face of the same occupied voxel that
    # native contact queries. Preserve true subvoxel XY in the proposed grip.
    contact = [actual[0], actual[1], grid_center[2] + 0.62]
    holding.prepare('membrane_forceps', contact, parameters={'jawOpeningMm': 0.0, 'forceN': 1.2})
    grasp = holding.contact('Regrasp supported ' + name, contact, operation='grasp')
    if not any(
        event['kind'] == 'exposure_body_grasped' and event['body'] == name for event in grasp['events']
    ):
        raise ValueError('Closure forceps did not register the ' + name + ' source grip')
    held = body(live, name)
    offset = np.asarray(held['gripOffsetMm'])
    stay = next(
        obj
        for obj in live.view['operativeExposureV1']['objects']
        if obj['kind'] == 'retaining-stay' and obj['body'] == name and obj['status'] == 'placed'
    )
    attitude = (
        (Rotation.from_euler('y', -22.0, degrees=True) * Rotation.from_quat(edge_quaternion([0, 0], [0, 1])))
        .as_quat()
        .tolist()
    )
    retained_site(
        assistant,
        'raney_clip_applier',
        'Retrieve the ' + name + ' retaining stay',
        stay['positionMm'],
        orientation=attitude,
        parameters={'jawOpeningMm': 0.0},
        retrieve=True,
    )
    angles = np.linspace(held['angleRad'], 0.0, 21)[1:]
    path = [pose(np.asarray(grip_trajectory(live.scene, name, angle)) + offset) for angle in angles]
    for index in range(0, len(path), 4):
        # During apposition the closed working jaws can contact the stationary
        # lip. Keep the grasp explicitly engaged just through that final stroke.
        operation = 'grasp' if index + 4 >= len(path) else 'move'
        holding.action(
            'Return ' + name + ' about its attachments', operation, 0.6, path=path[index : index + 4]
        )
    holding.action('Release reapproximated ' + name, 'release')
    holding.withdraw('Withdraw ' + name + ' forceps')


def retrieve_scalp_clip(live, hand):
    """Grasp the external backing and disengage the pressure jaw through the cut."""
    clip = next(
        obj
        for obj in live.view['operativeExposureV1']['objects']
        if obj['kind'] == 'raney-clip' and obj['status'] == 'placed'
    )
    if len(clip.get('clipSegments', ())) < 2 or 'incisionDirectionRas' not in clip:
        raise ValueError('Reopen the saved source with the registered scalp clip hardware before retrieval')
    center = np.asarray(clip['positionMm'])
    normal = np.asarray(live.view['operativeExposureV1']['exteriorNormalRas'])
    hand.prepare('raney_clip_applier', center, parameters={'jawOpeningMm': 0.0})
    tool, params, _ = _hand_geometry(live.view, hand.side)
    target = _first_grip(tool, params, DOWN, center, normal, clip['radiusMm'])
    grab = hand.contact('Regrasp temporary scalp hemostasis clip', target, operation='grasp')
    if not any(
        e['kind'] == 'exposure_object_grasped' and e['objectId'] == clip['id'] for e in grab['events']
    ):
        raise ValueError('The applier has no measured grip on the scalp clip')
    inside = target + 2.6 * np.asarray(clip['incisionDirectionRas'])
    hand.action('Disengage the clip jaws through the open scalp incision', 'grasp', 0.55, at=inside)
    hand.action(
        'Lift closed clip jaws clear of the scalp and carry to count',
        'grasp',
        0.8,
        at=[inside[0], inside[1], max(hand.clearance, inside[2] + 5.0)],
    )
    counted = hand.action('Count the removed scalp clip outside the head', 'release')
    if not any(
        e['kind'] == 'exposure_object_retrieved' and e['objectId'] == clip['id'] for e in counted['events']
    ):
        raise ValueError('The scalp clip has not reached the exterior count')
    hand.withdraw('Withdraw scalp clip applier')
