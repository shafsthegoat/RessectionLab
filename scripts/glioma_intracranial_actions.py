"""Editable deep treatment on the exposed generated source, using desktop actions only."""

from math import cos, radians, sin

from glioma_walkthrough_actions import DOWN, put, retrieve


def _pitch(degrees):
    # Rotate the inferior-facing tool about world Y. With opposite pitches the
    # distal tips can meet over the aperture while the proximal grips diverge.
    angle = radians(degrees) / 2
    return [cos(angle), 0.0, sin(angle), 0.0]


def _above(point, height):
    return [point[0], point[1], point[2] + height]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _objects(live, kind, status):
    return [
        obj
        for obj in live.view['operativeExposureV1']['objects']
        if obj['kind'] == kind and obj['status'] == status
    ]


def _mapping(live, landmarks, hand):
    # The reflected dural sheet occupies the hinge side of the registered
    # landmarks. Approach the adjacent visible cortex from the free side.
    x, y, z = landmarks['mappingSiteRasMm']
    point = [x, y - 4.0, z + 1.65]
    hand.prepare('cortical_probe', point)
    hand.contact(
        'Map the exposed cortical site',
        point,
        operation='apply',
        parameters={'amplitudeMa': 1.0, 'frequencyHz': 50.0, 'pulseWidthUs': 200.0},
    )
    observations = live.view['observations']
    _require(
        any(r['probe'] == 'cortical' and r['cells'] and r['result'] == 'unknown' for r in observations),
        'Mapping failed to contact the declared unevaluated cortical site',
    )
    hand.withdraw('Withdraw the cortical mapping probe')


def _dissect_and_sample(live, landmarks, left, right):
    tip = landmarks['sampleTipRasMm']
    neck = landmarks['sampleNeckRasMm']
    retractor = landmarks['retractorSiteRasMm']
    lq, rq = _pitch(15.0), _pitch(-15.0)
    contact = [retractor[0] + 1.0, retractor[1] - 4.0, retractor[2] + 1.65]
    left.prepare('brain_retractor', contact, orientation=lq)
    left.contact('Seat the retractor blade on the exposed cortex', contact, operation='hold', orientation=lq)
    _require(left.side in live.view['holds'], 'Retractor failed to engage tissue')
    tip_pose = left.slot['position']
    left.action(
        'Apply controlled traction and maintain the working space',
        'move',
        0.3,
        at=[tip_pose[0] - 0.12, tip_pose[1], tip_pose[2] + 0.15],
        orientation=lq,
    )
    _require(live.view['metrics']['brainMaxDisplacementMm'] > 0.0, 'Retraction produced no deformation')
    right.prepare('dissector', tip, orientation=rq)
    right.contact(
        'Release the visible specimen attachment under retraction', _above(tip, 0.35), orientation=rq
    )
    _require(
        'surface-sampling-attachment' in live.view['separatedAttachments'],
        'Specimen attachment remained intact',
    )
    right.withdraw('Withdraw the blunt dissector')
    left.action('Release the cortex and allow viscoelastic recovery', 'release', 0.45)
    left.withdraw('Withdraw the retractor blade')

    # The cut is through the visible narrow neck. A probe direction or a
    # segmented tumor flag never selects which bonds the blade intersects.
    right.prepare('micro_scissors', neck, parameters={'jawOpeningMm': 0.0, 'depthMm': 1.0})
    right.contact(
        'Divide the exposed specimen neck with closed microscissors',
        _above(neck, 0.02),
        parameters={'jawOpeningMm': 0.0, 'depthMm': 1.0},
    )
    _require(
        'vascular:biopsy-bed:0' in live.view['vascularGraphV1']['ports'],
        'Scissors did not section the source-bound neck',
    )
    right.withdraw('Withdraw the microscissors clear of the specimen')
    right.prepare('tumor_forceps', tip, parameters={'jawOpeningMm': 0.0})
    right.contact('Grasp the detached local specimen', _above(tip, 0.25), operation='grasp')
    cells = live.view['holds'].get(right.side, {}).get('cells', [])
    _require(len(cells) == 2, 'The forceps did not isolate the two-cell local specimen')
    right.action(
        'Carry the specimen through the exposed opening', 'move', 0.9, at=[tip[0], tip[1], right.clearance]
    )
    outcome = right.action('Transfer the specimen at the exterior', 'extract')
    _require(
        any(e['kind'] == 'detached_extraction' and e['cells'] == len(cells) for e in outcome['events']),
        'Specimen failed external extraction verification',
    )
    right.withdraw('Withdraw the emptied specimen forceps')


def _coagulate_bed(live, right):
    graph = live.view['vascularGraphV1']
    port = graph['ports']['vascular:biopsy-bed:0']
    _require(port['generatedMm3'] > 0, 'The sectioned capillary bed has no modeled ooze')
    point = live.scene.points([port['cell']])[0].tolist()
    right.prepare('bipolar_forceps', point, parameters={'jawOpeningMm': 0.3, 'forceN': 0.01, 'power': 0.0})
    right.contact(
        'Approximate the bipolar tips on the oozing biopsy bed',
        _above(point, 0.15),
        parameters={'jawOpeningMm': 0.3, 'forceN': 0.01, 'power': 0.0},
    )
    right.action(
        'Deliver local bipolar energy to the retained stump',
        'apply',
        2.1,
        parameters={'power': 1.0, 'jawOpeningMm': 0.3},
    )
    port = live.view['vascularGraphV1']['ports']['vascular:biopsy-bed:0']
    _require(port['sealFraction'] >= 0.99, 'Bipolar contact did not seal the retained biopsy-bed port')
    right.action('Deactivate the bipolar forceps', 'deactivate')
    right.withdraw('Disengage the bipolar tips')
    return point


def _resection_entry(landmarks):
    tip = landmarks['sampleTipRasMm']
    # Reuse the visible specimen corridor, on the free side of the retained
    # stump. Its centerline follows the registered surface landmark without
    # consulting the tissue label or the hidden removal accounting.
    return [tip[0], tip[1] - 0.5, landmarks['resectionEntryRasMm'][2]]


def _debulk(live, landmarks, hand):
    entry = _resection_entry(landmarks)
    target = landmarks['targetRasMm']
    end = [entry[0], entry[1], max(target[2], entry[2] - 2.5)]
    hand.prepare('cusa', end, parameters={'power': 0.85})
    hand.action('Approach the planned superficial debulking corridor', 'move', 0.55, at=_above(entry, 4.0))
    before = live.view['cusaV3']['aspiratedSolidMm3']
    for i in range(20):
        label = (
            'Engage the ultrasonic tip at the planned entry'
            if i == 0
            else 'Debulk the reachable tissue under direct contact'
        )
        hand.action(
            label,
            'apply',
            1.5,
            at=end,
            parameters={'power': 0.85},
            allowed_partial=('TISSUE_ASPIRATION_IN_PROGRESS',),
        )
        if sum((x - y) ** 2 for x, y in zip(hand.slot['position'], end)) < 1e-8:
            break
    _require(live.view['cusaV3']['aspiratedSolidMm3'] > before, 'Ultrasonic action cleared no source tissue')
    hand.action('Deactivate ultrasonic aspiration', 'deactivate')
    hand.withdraw('Withdraw the aspirator through the treated corridor')
    return entry


def _clear_field(live, hand, entry):
    # The injected fluid enters the accepted cavity directly ahead of the nozzle.
    hand.prepare('irrigation', entry)
    hand.action('Approach the cavity with irrigation inactive', 'move', 0.5, at=_above(entry, 4.0))
    nozzle = _above(entry, 2.0)
    hand.contact(
        'Irrigate the operative bed',
        nozzle,
        duration=0.4,
        parameters={'flowMm3PerSecond': 5.0, 'depthMm': 2.0},
    )
    hand.withdraw('Withdraw the irrigation nozzle')
    removed = live.view['metrics']['fluidRemovedMm3']
    hand.prepare('frazier_suction', entry)
    # Select the visible retained fluid nearest the treated site. Distant scalp
    # effluent cannot satisfy the local field-clearance demonstration.
    wet = [(int(c), v) for c, v in live.view['fluid'].items() if v > 1e-9]
    reachable = []
    for c, amount in wet:
        xyz = live.scene.points([c])[0]
        if max(abs(xyz[:2] - entry[:2])) <= 3 and entry[2] - 3 <= xyz[2] <= entry[2] + 5:
            radial_distance_squared = sum((x - y) ** 2 for x, y in zip(xyz[:2], entry[:2]))
            reachable.append((radial_distance_squared, -amount, c, xyz))
    _require(reachable, 'No local retained irrigation fluid available to suction')
    _, _, _, pool = min(reachable, key=lambda r: r[:3])
    # Approach from the exterior, with suction enabled only at a free outlet.
    hand.contact(
        'Aspirate the visible irrigation pool',
        _above(pool, 1.65),
        duration=0.65,
        parameters={'flowMm3PerSecond': 12.0},
    )
    hand.action('Hold the suction lumen above the visible pool', 'apply', 0.45)
    _require(live.view['metrics']['fluidRemovedMm3'] > removed, 'Suction did not clear sensed field fluid')
    hand.withdraw('Withdraw the suction lumen from the cleared bed')


def treat(live, landmarks, left, right):
    _require(
        live.view['operativeExposureV1']['accessOpen'], 'Deep treatment requires supported open exposure'
    )
    if live.resume not in ('sample-exterior', 'after-cusa'):
        _mapping(live, landmarks, right)
        put(
            right,
            'dressing_forceps',
            'Place the counted protective cottonoid',
            _above(landmarks['cottonoidSiteRasMm'], 0.25),
            parameters={'jawOpeningMm': 0.0},
        )
        _require(_objects(live, 'cottonoid', 'placed'), 'Cottonoid placement failed')
        _dissect_and_sample(live, landmarks, left, right)
        live.save('sample-exterior')
    if live.resume != 'after-cusa':
        _coagulate_bed(live, right)
        _debulk(live, landmarks, right)
        live.save('after-cusa')
    port = live.view['vascularGraphV1']['ports']['vascular:biopsy-bed:0']
    _require(port['sealFraction'] >= 0.99, 'Biopsy-bed source remains unsealed')
    bed = live.scene.points([port['cell']])[0].tolist()
    _clear_field(live, right, _resection_entry(landmarks))
    put(
        right,
        'dressing_forceps',
        'Place the intended-retained hemostatic dressing on the biopsy bed',
        _above(bed, 0.2),
        parameters={'jawOpeningMm': 0.0},
    )
    _require(_objects(live, 'hemostat', 'intended-retained'), 'Hemostatic dressing placement failed')
    cottonoid_object = _objects(live, 'cottonoid', 'placed')[0]
    pickup = _above(cottonoid_object['positionMm'], cottonoid_object['radiusMm'] + 0.15)
    retrieve(right, 'dressing_forceps', 'Retrieve the counted cottonoid', pickup)
    _require(_objects(live, 'cottonoid', 'retrieved'), 'A temporary cottonoid remains in the field')
    _require(all(not s['inserted'] for s in live.view['slots'].values()), 'An instrument remains inserted')
    right.action('Review cavity, source-local hemostasis and field-material count', 'inspect')
