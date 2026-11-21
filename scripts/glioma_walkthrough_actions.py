"""Public operative motions on the generated source's registered physical landmarks.

Tool poses are RAS mm. Anatomical roll follows the incision tangent, while
the shaft remains on the exterior side of the surface. No action constructs
geometry, moves anatomy directly, or writes engine state.
"""
from __future__ import annotations

import math


DOWN = [1., 0., 0., 0.]


def pose(point, quaternion=DOWN):
    return {"position": list(map(float, point)), "quaternion": list(quaternion)}


def edge_quaternion(start, end):
    """Align the blade's local X/Z edge plane with an XY tangent and inward normal."""
    roll = math.atan2(end[1]-start[1], end[0]-start[0]) / 2.
    return [math.cos(roll), math.sin(roll), 0., 0.]


class Hand:
    def __init__(self, episode, side, parking):
        self.live = episode
        self.side = side
        self.parking = list(parking)
        self.clearance = parking[2]

    @property
    def slot(self):
        return self.live.view["slots"][self.side]

    def action(self, label, operation, duration=.1, *, at=None, orientation=DOWN,
               parameters=None, path=None, allowed_partial=()):
        request = {"slot": self.side, "operation": operation, "duration": duration}
        if at is not None:
            request["pose"] = pose(at, orientation)
        if path is not None:
            request["path"] = path
        if parameters:
            request["parameters"] = parameters
        return self.live.step(label, request, allowed_partial=allowed_partial)

    def prepare(self, instrument, point, *, orientation=DOWN, parameters=None):
        if self.slot["instrument"] != instrument:
            self.live.step(f"Select {instrument}", {"slot": self.side, "operation": "exchange",
                                                    "duration": .1, "instrument": instrument})
        outside = [point[0], point[1], max(self.clearance, point[2]+6.)]
        # Place at the *existing* exterior hand location, then sweep to the
        # staging point. Even an exchange outside the head has a continuous path.
        self.action(f"Mount {instrument} at the parked hand", "place", at=self.slot["position"],
                    orientation=self.slot["quaternion"], parameters=parameters)
        self.action(f"Align {instrument} over the operative site", "move", .7,
                    at=outside, orientation=orientation)

    def withdraw(self, label):
        tip = self.slot["position"]
        height = max(self.clearance, tip[2]+5.)
        q = self.slot["quaternion"]
        route = [pose([tip[0], tip[1], height], q), pose([*self.parking[:2], height], DOWN),
                 pose(self.parking, DOWN)]
        self.action(label, "withdraw", 1., path=route)

    def contact(self, label, point, *, operation="apply", duration=.35, orientation=DOWN, parameters=None):
        return self.action(label, operation, duration, at=point, orientation=orientation, parameters=parameters)

    def trace(self, label, path, speed, *, chunk=8, parameters=None, orientation=edge_quaternion,
              allowed_partial=()):
        for first in range(1, len(path), chunk):
            points = path[first-1:first+chunk]
            travel = sum(math.dist(a, b) for a, b in zip(points, points[1:]))
            frames = [pose(b, orientation(a, b)) for a, b in zip(points, points[1:])]
            self.action(label, "apply", max(.2, travel/speed), path=frames,
                        parameters=parameters, allowed_partial=allowed_partial)


def seam_course(scene, name):
    """Trace the prospective attachment boundary, starting beside the retained hinge."""
    import numpy as np
    body = scene.properties["operativeExposureV1"]["bodies"][name]
    edges = np.asarray(body["seamEdges"])
    midpoint = scene.points(edges.ravel()).reshape(-1, 2, 3).mean(axis=1)
    # The working length reaches the outer tissue stratum from its deep surface.
    # Each retained X/Y position appears once in contour order.
    projection = {}
    for point in midpoint:
        key = tuple(point[:2])
        if key not in projection or point[2] < projection[key][2]:
            projection[key] = point
    cx, cy, _ = body["restCenterMm"]
    hx, hy, _ = body["hingeOriginMm"]
    start_angle = math.atan2(hy-cy, hx-cx)
    return [(p + [0., 0., .33]).tolist() for p in sorted(projection.values(), key=lambda point:
        (math.atan2(point[1]-cy, point[0]-cx)-start_angle) % (2.*math.pi))]


def enter_incision(hand, name, course, instrument):
    angle = edge_quaternion(*course[:2])
    p = course[0]
    hand.prepare(instrument, p, orientation=angle,
                 parameters={"jawOpeningMm": 0.} if instrument == "dural_scissors" else None)
    hand.action("Approach intact "+name, "move", .6, at=[p[0], p[1], p[2]+3.], orientation=angle)
    return angle



class ScalpClearance:
    """Prospective blade clearance against this source's intact calvarium."""

    def __init__(self, hand):
        from resectionlab.geometry import GeometryScene
        import numpy as np
        scene = hand.live.scene
        self.grid = GeometryScene(np.zeros(scene.shape, dtype=bool), scene.affine)
        self.bone = np.isin(scene.labels, [r["id"] for r in scene.regions if r["material"] == "bone"])
        self.edges = [c for c in hand.slot["contactGeometry"] if c["part"] == "working"]

    def free(self, p, q, margin=.22):
        from resectionlab.geometry import capsule_voxel_indices_batch
        from scipy.spatial.transform import Rotation
        import numpy as np
        rotation = Rotation.from_quat(q)
        point = np.asarray(p)
        queries = [(point+rotation.apply(c["a"]), point+rotation.apply(c["b"]), c["radiusMm"]+margin)
                   for c in self.edges]
        return not any(self.bone[tuple(ids.T)].any() for ids in capsule_voxel_indices_batch(self.grid, queries))

    def shallow(self, p, q):
        import numpy as np
        for dz in (0., .15, .3, .5, .8, 1.2, 1.6):
            proposal = np.asarray(p)+[0.,0.,dz]
            if self.free(proposal, q):
                return proposal
        raise ValueError("No scalp blade clearance above the rigid surface")

    @staticmethod
    def fine_vertical(a, b, q):
        import numpy as np
        count = max(1, math.ceil(math.dist(a, b) / .04))
        start, delta = np.asarray(a), np.asarray(b) - a
        return [pose(start + delta * i / count, q) for i in range(1, count + 1)]

    def contour(self, hand, course, first_angle):
        import numpy as np
        previous = np.asarray(hand.slot["position"])
        old = first_angle
        path = []
        if not self.free(previous, old):
            raised = self.shallow(previous, old)
            path.extend(self.fine_vertical(previous, raised, old))
            previous = raised
        for a, b in zip(course, course[1:]):
            q = edge_quaternion(a, b)
            end = self.shallow(b, q)
            # The exact source rim can turn around a higher neighboring bone
            # voxel between safe tips. Cross that corner above the table before
            # re-engaging the lower skin stratum.
            if any(not self.free((1-t)*previous+t*end, q) for t in (0., .2, .4, .6, .8, 1.)):
                height = max(previous[2], end[2])+1.5
                path.extend([pose([*previous[:2], height], old), pose([*end[:2], height], q)])
            path.append(pose(end, q))
            previous, old = end, q
        previous = hand.slot["position"]
        for i in range(0, len(path), 8):
            points = path[i:i+8]
            positions = [previous] + [p["position"] for p in points]
            distance = sum(math.dist(x, y) for x, y in zip(positions, positions[1:]))
            hand.action("Incise scalp around the calvarial contour", "apply", max(.2, distance/6.), path=points)
            previous = positions[-1]


def open_seam(hand, name, course, angle):
    hand.contact("Engage "+name+" edge", course[0], orientation=angle)
    if name == "scalp":
        hand.live.save("first-engagement")
    clearance = ScalpClearance(hand) if name == "scalp" else None
    if clearance:
        clearance.contour(hand, course, angle)
    else:
        hand.trace("Incise "+name+" attachment", course, 6.)
    hand.withdraw("Disengage "+name+" cutting instrument")
    live = hand.live
    original = live.scene.properties["operativeExposureV1"]["bodies"][name]["seamEdges"]
    for edge in original:
        report = body(live, name)
        if report["status"] == "incised":
            break
        if edge in report["seamEdges"]:
            continue
        a, b = live.scene.points(edge)
        normal = b-a
        angle = edge_quaternion([0., 0.], [-normal[1], normal[0]])
        at = (a+b)/2. + [0.,0.,.3]
        if clearance:
            choices = [a*(1-t)+b*t+[0.,0.,h] for h in (.3,.38,.18,0.) for t in (.5,.25,.75,.05,.95)]
            at = next((point for point in choices if clearance.free(point,angle,margin=.025)), None)
            if at is None:
                raise ValueError(f"No physically accessible scalp cut on source edge {edge}")
        tool = hand.slot["instrument"]
        hand.prepare(tool, at, orientation=angle)
        if clearance:
            hover = at+[0.,0.,1.5]
            hand.contact("Reach the superficial side of the remaining scalp attachment", hover, orientation=angle)
            hand.action("Divide the remaining deep scalp attachment under local clearance", "apply", .45,
                        path=clearance.fine_vertical(hover,at,angle))
            hand.action("Back the blade away from the adjacent table", "apply", .35,
                        path=clearance.fine_vertical(at,hover,angle))
        else:
            hand.contact("Divide a remaining registered "+name+" attachment", at, orientation=angle)
        hand.withdraw("Withdraw from the completed local cut")
        if edge not in body(live, name)["seamEdges"]:
            if name == "dura" and report["cutCount"] == 0:
                finish_dural_course(hand, course)
                break
            raise ValueError(f"Instrument did not divide {name} attachment {edge}")
    if body(live, name)["status"] != "incised":
        raise ValueError(f"Cutting left {name} source attachments intact")


def finish_dural_course(hand, course):
    """Seat the cutting edges after a superficial pass leaves the seam attached."""
    live = hand.live
    # The blunt toe projects beyond the cutting edges. Verify edge engagement
    # on the tissue itself, then keep that depth through the registered course.
    engaged = [[x, y, z - .30] for x, y, z in course]
    first = enter_incision(hand, "dura", engaged, "dural_scissors")
    hand.contact("Engage both dural cutting edges", engaged[0], orientation=first)
    hand.trace("Divide the dural attachment under the closed scissor edges", engaged, 6.)
    hand.withdraw("Disengage the dural scissors from the divided course")
    source = live.scene.properties["operativeExposureV1"]["bodies"]["dura"]["seamEdges"]
    for edge in source:
        if edge in body(live, "dura")["seamEdges"]:
            continue
        a, b = live.scene.points(edge)
        normal = b - a
        angle = edge_quaternion([0., 0.], [-normal[1], normal[0]])
        contact = (a + b) / 2. + [0., 0., .03]
        hand.prepare("dural_scissors", contact, orientation=angle)
        hand.contact("Divide a remaining deep dural attachment", contact, orientation=angle)
        hand.withdraw("Clear the divided dural attachment")
        if edge not in body(live, "dura")["seamEdges"]:
            raise ValueError(f"The scissor edges did not divide source attachment {edge}")
    if body(live, "dura")["status"] != "incised":
        raise ValueError("Dural attachments remain intact")


def perform(live, landmarks, through):
    """Advance one intact registered source until the requested physical checkpoint."""
    if live.scene.identity != landmarks["sceneId"]:
        raise ValueError("Source geometry and policy landmarks differ")
    center = landmarks["entryCenterRasMm"]
    hands = []
    for side, sign in (("left", -1.), ("right", 1.)):
        rest = landmarks["outsideRasMm"][side]
        # The full ring handles must clear the bimanual working space while
        # parked, including the translated bone tray on the lateral side.
        hands.append(Hand(live, side, [center[0]+sign*52., rest[1]-6., rest[2]+7.]))
    left, right = hands
    if live.resume == "scalp-clipped" and math.dist(right.slot["position"], right.parking) > .01:
        right.withdraw("Clear parked clip applier from reflected flap and forceps")
    if live.resume is None:
        scalp_course = seam_course(live.scene, "scalp")
        live.phase = "approach"
        right.action("Inspect intact scalp and registered approach", "inspect")
        angle = enter_incision(right, "scalp", scalp_course, "scalpel")
        if through == "approach":
            return
        live.phase = "scalp"
        open_seam(right, "scalp", scalp_course, angle)
        live.save("scalp-incision")
    expose_and_treat(live, landmarks, left, right, through)


def body(live, name):
    return live.view["operativeExposureV1"]["bodies"][name]


def put(hand, tool, label, point, *, operation="apply", parameters=None, orientation=DOWN):
    hand.prepare(tool, point, parameters=parameters, orientation=orientation)
    effect = hand.contact(label, point, operation=operation, orientation=orientation)
    if not any(e["kind"] == "exposure_object_placed" for e in effect["events"]):
        raise ValueError(label+" did not place a physical object")
    hand.withdraw("Withdraw "+tool)


def retrieve(hand, tool, label, point, *, orientation=DOWN):
    """Pick the physical object, carry it out of the scalp plane, release and clear."""
    hand.prepare(tool, point, parameters={"jawOpeningMm": 0.}, orientation=orientation)
    effect = hand.contact(label, point, operation="grasp", orientation=orientation)
    if not any(e["kind"] == "exposure_object_grasped" for e in effect["events"]):
        raise ValueError(label+" did not grasp the physical object")
    hand.action("Carry object to the exterior count", "move", .8,
                at=[point[0], point[1], max(point[2]+5., hand.clearance)], orientation=orientation)
    released = hand.action("Release retrieved object at the exterior", "release")
    if not any(e["kind"] == "exposure_object_retrieved" for e in released["events"]):
        raise ValueError(label+" failed the exterior object count")
    hand.withdraw("Withdraw "+tool)


def sheet_handling(live, name, holding, assistant):
    """Propose each physical tip from the authored material grip and hinge kinematics."""
    from resectionlab.surgical_exposure_geometry import grip_trajectory, site_trajectory
    import numpy as np
    cfg = live.scene.properties["operativeExposureV1"]["bodies"][name]
    opened = math.radians(100.)
    source = np.asarray(grip_trajectory(live.scene, name, 0.))
    # The free-facing working ends touch the patch; their shafts stay outside.
    grip = source + [0., 0., .68]
    holding.prepare("membrane_forceps", grip, parameters={"jawOpeningMm": 0., "forceN": 1.2})
    holding.contact("Grasp "+name+" material", grip, operation="grasp")
    if body(live, name)["heldBy"] != holding.side:
        raise ValueError("No measured "+name+" grip")
    offset = np.asarray(body(live, name)["gripOffsetMm"])
    support = site_trajectory(live.scene, name, cfg["anchorCells"][0], opened)
    from scipy.spatial.transform import Rotation
    # The forceps remains vertical. The assistant approaches the reflected
    # lateral anchor from its own side, keeping the two handles separated.
    oblique = (Rotation.from_euler("y", -22., degrees=True)*Rotation.from_quat(edge_quaternion([0,0], [0,1]))).as_quat().tolist()
    from glioma_bimanual_routes import retained_site
    angles = np.linspace(0., opened, 21)[1:]
    path = [pose(np.asarray(grip_trajectory(live.scene, name, a))+offset) for a in angles]
    for first in range(0, len(path), 4):
        holding.action("Reflect "+name+" about its attachments",
                       "move", .6, path=path[first:first+4])
    live.save(name+"-held-reflection")
    retained_site(assistant, "raney_clip_applier", "Secure the reflected "+name, support,
                  parameters={"jawOpeningMm": 0.}, orientation=oblique)
    holding.action("Release supported "+name, "release")
    holding.withdraw("Withdraw "+name+" forceps")


def expose_and_treat(live, landmarks, left, right, through):
    stump = landmarks["vascularFacesRasMm"]["scalp"][0]
    scalp_wound = [stump[0], stump[1], stump[2]+1.]
    stages = ["scalp-incision", "scalp-clipped", "early-scalp", "bone-pilot", "bone-kerf",
              "early-bone", "dura-incision", "deep-access", "sample-exterior", "after-cusa",
              "after-treatment", "dura-closed", "bone-fixed", "raney-retrieved",
              "scalp-hemostasis", "scalp-seated"]
    start = stages.index(live.resume or "scalp-incision")
    live.phase = "scalp"
    if start <= stages.index("scalp-incision"):
        put(right, "raney_clip_applier", "Clip the incised scalp margin", scalp_wound,
            parameters={"jawOpeningMm": 0.})
        live.save("scalp-clipped")
    if start <= stages.index("scalp-clipped"):
        sheet_handling(live, "scalp", right, left)
    if through == "scalp":
        return
    if start < stages.index("early-scalp"):
        live.save("early-scalp")
    live.phase = "bone"
    from glioma_bone_actions import open_bone, move_bone
    if start <= stages.index("bone-pilot"):
        open_bone(live, right, pilot_open=live.resume=="bone-pilot")
        live.save("bone-kerf")
    if start <= stages.index("bone-kerf"):
        move_bone(live, left)
    if through == "bone":
        return
    if start < stages.index("early-bone"):
        live.save("early-bone")
    live.phase = "dura"
    if start <= stages.index("early-bone"):
        dura = seam_course(live.scene, "dura")
        angle = enter_incision(right, "dura", dura, "dural_scissors")
        open_seam(right, "dura", dura, angle)
        live.save("dura-incision")
    if start <= stages.index("dura-incision"):
        sheet_handling(live, "dura", right, left)
    if through == "dura":
        return
    if start < stages.index("deep-access"):
        live.save("deep-access")
    live.phase = "treatment"
    from glioma_intracranial_actions import treat
    if start <= stages.index("after-cusa"):
        treat(live, landmarks, left, right)
        live.save("after-treatment")
    if through == "treatment":
        return
    live.phase = "closure"
    if start <= stages.index("after-treatment"):
        from glioma_closure_actions import return_sheet
        return_sheet(live, "dura", right, left)
        close_sites(live, "dura", right)
        live.save("dura-closed")
    if start <= stages.index("dura-closed"):
        move_bone(live, left, closing=True)
        close_sites(live, "bone", right)
        live.save("bone-fixed")
    # Retrieve the actual object at its accepted tissue position. Once the
    # temporary clip leaves, verify the named stump's own hemostatic outcome.
    if start <= stages.index("bone-fixed"):
        from glioma_closure_actions import retrieve_scalp_clip
        retrieve_scalp_clip(live, right)
        live.save("raney-retrieved")
    if start <= stages.index("raney-retrieved"):
        finish_scalp_hemostasis(live, right)
        live.save("scalp-hemostasis")
    if start <= stages.index("scalp-hemostasis"):
        from glioma_closure_actions import return_sheet
        return_sheet(live, "scalp", right, left)
        live.save("scalp-seated")
    close_sites(live, "scalp", right)
    right.action("Review fixation, scalp approximation and final object count", "inspect")
    states = {name: body(live, name)["status"] for name in ("dura", "bone", "scalp")}
    if states != {"dura": "closed", "bone": "fixed", "scalp": "closed"}:
        raise ValueError(f"Closure remains incomplete: {states}")
    if not live.view["operativeExposureV1"]["complete"]:
        raise ValueError("The physical endpoint still has a temporary object in the field")
    for slot in live.view["slots"].values():
        if slot["active"] or slot["inserted"] or slot["grip"]:
            raise ValueError("Both instruments must be released, outside and inactive")


def close_sites(live, name, hand):
    tool = "fixation_driver" if name == "bone" else "needle_driver"
    anchors = body(live, name)["anchorsMm"]
    for number, site in enumerate(anchors, 1):
        contact = [site[0], site[1], site[2]+.45]
        put(hand, tool, f"{'Fix' if name=='bone' else 'Approximate'} {name} margins at anchor {number}",
            contact, parameters={"jawOpeningMm": 0.} if name!="bone" else None)


def finish_scalp_hemostasis(live, hand):
    port_name = "vascular:scalp:0"
    port = live.view["vascularGraphV1"]["ports"][port_name]
    site = live.scene.points([port["cell"]])[0].tolist()
    params = {"jawOpeningMm": .25, "power": 0., "forceN": .01}
    hand.prepare("bipolar_forceps", site, parameters=params)
    hand.contact("Approximate bipolar tips on the previously clipped scalp stump",
                 [site[0], site[1], site[2]+.3], parameters=params)
    hand.action("Seal the scalp stump after clip retrieval", "apply", 2.1,
                parameters={"jawOpeningMm": .25, "power": 1.})
    if live.view["vascularGraphV1"]["ports"][port_name]["sealFraction"] < .99:
        raise ValueError("The actual scalp source remains incompletely sealed")
    hand.action("Stop bipolar energy at the scalp", "deactivate")
    hand.withdraw("Withdraw scalp bipolar")
