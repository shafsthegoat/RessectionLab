"""Source-bound bone opening and transport for the generated teaching case."""
import math

import numpy as np
from scipy.spatial.transform import Rotation


def open_bone(live, hand, *, pilot_open=False):
    """Run a physical perforation, then keep the side cutter's foot below the inner table."""
    if __package__:
        from .glioma_walkthrough_actions import body, edge_quaternion
    else:
        from glioma_walkthrough_actions import body, edge_quaternion

    cfg = live.scene.properties["operativeExposureV1"]["bodies"]["bone"]
    center = np.asarray(cfg["restCenterMm"])
    pilot = np.asarray(cfg["pilotCenterMm"])
    ring = live.scene.points(cfg["kerfCells"])
    depth = float(ring[:, 2].min())-.30
    if not pilot_open:
        hand.prepare("cranial_drill", pilot, parameters={"rpm": 60000., "forceN": 7., "duty": .8})
        for z in [float(ring[:, 2].max())+1.8, float(ring[:, 2].max()), float(ring[:, 2].min())+1.0]:
            station = [pilot[0], pilot[1], z]
            for _ in range(12):
                outcome = hand.action("Perforate local outer table, diploë and inner table", "apply", 1.,
                                      at=station, allowed_partial=("BONE_CUTTING_IN_PROGRESS",))
                if outcome["status"] == "accepted":
                    break
            else:
                raise ValueError("The burr did not clear the pilot contact with bounded work")
        hand.withdraw("Disengage and withdraw burr")
        live.save("bone-pilot")
    radial = np.linalg.norm(ring[:, :2]-center[:2], axis=1)
    radius = math.floor(radial.min()) + 1.25
    angle = math.atan2(pilot[1]-center[1], pilot[0]-center[0])
    lead = math.pi/3.
    travel = np.linspace(0., lead+2*math.pi, 113)
    blend = np.clip(travel/lead, 0., 1.)
    radii = radius+(np.linalg.norm(pilot[:2]-center[:2])-radius)*(1-blend)**2
    course = [[center[0]+r*math.cos(angle+a), center[1]+r*math.sin(angle+a), depth]
              for a,r in zip(travel,radii)]
    # The lateral starter blends into one complete orbit rather than turning
    # its guard against intact bone at the beginning of the kerf.
    # +X is the guarding post. Keep it behind the side cutter in the trough.
    def trailing(a, b):
        # Roll the guard behind the cut. On the posterior arc, modest anterior
        # inclination clears the reflected scalp above the calvarium.
        incline = np.clip(((a[1]+b[1])/2-center[1]-8.4)/1.6, 0., 1.)
        angle = 6.*incline
        tilt = Rotation.from_euler("x", angle, degrees=True)
        guard = Rotation.from_euler("z", 6.*incline, degrees=True)
        return (tilt*guard*Rotation.from_quat(edge_quaternion(b, a))).as_quat().tolist()
    orient = trailing(*course[:2])
    hand.prepare("craniotome", pilot, orientation=orient, parameters={"rpm": 60000., "forceN": 7., "duty": .8})
    hand.contact("Seat guarded foot in the actual perforation", [pilot[0],pilot[1],depth], orientation=orient)
    hand.trace("Cut the guarded circumferential kerf", course, .35, chunk=5, orientation=trailing)
    if body(live, "bone")["status"] != "free":
        raise ValueError("Residual native bony connections prevent flap mobilization")
    hand.withdraw("Return the guarded cutter through the completed kerf")


def move_bone(live, hand, *, closing=False):
    if __package__:
        from .glioma_walkthrough_actions import DOWN, body
    else:
        from glioma_walkthrough_actions import DOWN, body

    cfg = live.scene.properties["operativeExposureV1"]["bodies"]["bone"]
    other = "right" if hand.side == "left" else "left"
    parked = live.view["slots"][other]
    # A 12-degree rotation of the 165 mm holder sweeps its proximal handle
    # across the usual exterior staging point. Move the idle hand clear first.
    transfer_park = np.asarray(cfg["parkCenterMm"])+[42., -27., 21.]
    live.step("Retire the opposite instrument clear of the bone transfer", {
        "slot": other, "operation": "withdraw", "duration": .55,
        "pose": {"position": transfer_park.tolist(), "quaternion": parked["quaternion"]},
    })
    site = np.asarray(body(live, "bone")["gripMm"])
    hand.prepare("bone_flap_holder", site, parameters={"jawOpeningMm": 0., "forceN": 1.5})
    # The parked plate is already bearing on three pads. Approach only to
    # the first jaw contact so the completed grasp does not push it into them.
    grip_height = 1.02 if closing else .45
    hand.contact("Grasp "+("parked" if closing else "freed")+" bone flap",
                 site+[0., 0., grip_height], operation="grasp")
    if body(live, "bone")["heldBy"] != hand.side:
        raise ValueError("Bone holder has no measured material grip")
    rest = np.asarray(cfg["restCenterMm"])
    park = np.asarray(cfg["parkCenterMm"])
    start, finish = (park, rest) if closing else (rest, park)
    # First contact can precede the end of the grasp sweep. Use the *current*
    # held center to recover the measured rigid offset before prescribing the
    # final plate position on the pads or in the original seat.
    held_center = np.mean(body(live, "bone")["centersMm"], axis=0)
    offset = np.asarray(hand.slot["position"])-held_center
    lift_z = max(rest[2],park[2])+4.
    middle = (start+finish)/2
    positions = [[start[0],start[1],lift_z], [middle[0],middle[1],lift_z],
                 [finish[0],finish[1],lift_z], finish]
    for point, tilt in zip(positions, [0., 12., 0., 0.], strict=True):
        rotation = Rotation.from_euler("y", tilt, degrees=True)
        q = (rotation*Rotation.from_quat(DOWN)).as_quat().tolist()
        hand.action("Carry held flap to "+("its original seat" if closing else "its exterior support"),
                    "move", .7, at=np.asarray(point)+rotation.apply(offset), orientation=q)
    hand.action("Release "+("replaced" if closing else "parked")+" bone flap", "release")
    hand.withdraw("Withdraw bone holder")
