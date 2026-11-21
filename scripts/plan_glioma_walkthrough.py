"""Build optical chapters from the saved policy's accepted actions and source seal."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


# A cue changes the camera after the named recorded action starts. The hand is
# resolved from that action, so swapping hands in a new policy remains valid.
# Subsequent ordinary approach/withdrawal actions retain the last field.
PHASES = {
    'inspection': ('Inspect the operative field', 55, 30, 60, 'opening'),
    'approach': ('Establish the operative approach', 49, 26, 20, 'opening'),
    'scalp': ('Scalp exposure', 50, 37, 65, 'opening'),
    'bone': ('Prepare the bone opening', 47, 31, 70, 'opening'),
    'dura': ('Dural exposure', 36, 32, 65, 'opening'),
    'treatment': ('Inspect the exposed cortex', 31, 22, 75, 'opening'),
    'closure': ('Restore the exposed layers', 53, 33, 65, 'opening'),
}
CUES = [
    ('Engage scalp edge', 'Engage the scalp edge', 18, 45, 90, 'hand'),
    ('Incise scalp', 'Scalp incision', 49, 36, -65, 'opening'),
    ('Clip the incised scalp margin', 'Scalp hemostasis', 18, 42, 70, 'hand'),
    ('Grasp scalp material', 'Scalp grasp', 23, 43, -80, 'hand'),
    ('Reflect scalp about', 'Reflect the attached scalp', 64, 35, -80, 'opening'),
    ('Release supported scalp', 'Supported scalp', 64, 35, -80, 'opening'),
    ('Retrieve the scalp retaining stay', 'Scalp stay retrieval', 48, 34, -80, 'opening'),
    ('Secure the reflected scalp', 'Support the reflected scalp', 64, 35, -80, 'opening'),
    ('Perforate local', 'Open a pilot through the bone', 25, 37, 80, 'hand'),
    ('Seat guarded foot in the actual perforation', 'Guarded cutter entry', 27, 30, 70, 'hand'),
    ('Cut the guarded circumferential kerf', 'Guarded craniotomy', 48, 31, 65, 'opening'),
    ('Return the guarded cutter through the completed kerf', 'Withdraw the guarded cutter', 52, 28, -35, 'opening'),
    ('Retire the opposite instrument clear of the bone transfer', 'Completed bone opening', 56, 30, -35, 'opening'),
    ('Grasp freed bone flap', 'Grasp the freed bone flap', 32, 38, 70, 'hand'),
    ('Carry held flap to its exterior', 'Transfer the held flap to its support', 80, 32, -80, 'opening'),
    ('Release parked bone flap', 'Parked bone flap', 80, 32, -80, 'opening'),
    ('Release replaced bone flap', 'Restored bone seat', 55, 32, -75, 'opening'),
    ('Engage dura edge', 'Position the dural scissors', 28, 36, 70, 'hand'),
    ('Engage both dural cutting edges', 'Engage both dural cutting edges', 19, 37, 70, 'hand'),
    ('Divide the dural attachment under', 'Dural incision', 37, 29, 70, 'opening'),
    ('Grasp dura material', 'Grasp the dural edge', 21, 36, -80, 'hand'),
    ('Reflect dura about', 'Reflect the grasped dura', 52, 32, -80, 'opening'),
    ('Secure the reflected dura', 'Dural retention', 47, 32, -80, 'opening'),
    ('Release supported dura', 'Supported dura', 45, 30, -75, 'opening'),
    ('Retrieve the dura retaining stay', 'Dural stay retrieval', 43, 30, -75, 'opening'),
    ('Place the counted protective cottonoid', 'Protect the operative field', 28, 23, 65, 'hand'),
    ('Map the exposed', 'Inspect functional evidence', 24, 20, 70, 'hand'),
    ('Seat the retractor', 'Support the exposed cortex', 32, 24, 70, 'hand'),
    ('Release the visible specimen attachment', 'Dissect under supported retraction', 32, 24, 70, 'opening'),
    ('Divide the exposed specimen', 'Divide the specimen attachment', 23, 27, 75, 'hand'),
    ('Grasp the detached local specimen', 'Retrieve the local specimen', 29, 26, 75, 'hand'),
    ('Transfer the specimen at the exterior', 'Specimen transfer', 47, 29, -45, 'opening'),
    ('Approximate the bipolar tips on', 'Bipolar contact', 22, 27, 65, 'hand'),
    ('Deliver local bipolar', 'Coagulate the recorded wound source', 22, 27, 65, 'hand'),
    ('Approach the planned superficial debulking', 'Debulk through the exposed corridor', 30, 21, 70, 'opening'),
    ('Irrigate the operative bed', 'Irrigate the operative bed', 27, 23, 75, 'opening'),
    ('Aspirate the visible irrigation pool', 'Clear the free fluid and debris', 27, 23, 75, 'opening'),
    ('Place the intended-retained hemostatic dressing', 'Local hemostatic dressing', 26, 23, 70, 'hand'),
    ('Retrieve the counted cottonoid', 'Cottonoid retrieval', 29, 23, 70, 'hand'),
    ('Review cavity', 'Review the cavity and field materials', 33, 21, 70, 'opening'),
    ('Return dura about', 'Return the supported dura', 47, 34, 70, 'opening'),
    ('Approximate dura margins', 'Approximate the dural margins', 33, 32, 70, 'opening'),
    ('Grasp parked bone flap', 'Re-grasp the parked bone', 32, 36, 75, 'hand'),
    ('Carry held flap to its original seat', 'Replace the supported flap', 80, 32, -80, 'opening'),
    ('Fix bone margins', 'Fix the replaced bone', 44, 30, 75, 'opening'),
    ('Return scalp about', 'Return the reflected scalp', 64, 35, -80, 'opening'),
    ('Remove the scalp hemostasis clip', 'Scalp clip retrieval', 24, 38, 65, 'hand'),
    ('Seal the scalp stump after clip retrieval', 'Scalp hemostasis after clip retrieval', 23, 35, 65, 'hand'),
    ('Approximate scalp margins', 'Approximate the scalp layers', 43, 34, 75, 'opening'),
    ('Review fixation', 'Review closure and the final count', 55, 28, 65, 'opening'),
]


def plan(history, seconds):
    rows = history['actions']
    if not rows or len(rows) != history['position']:
        raise ValueError('A contiguous native accepted history is required.')
    chapters = []
    phase = current = None
    captions = []
    stills = set()
    holds = {}
    for zero, row in enumerate(rows):
        if zero + 1 != row['index'] or (zero and row['stateBefore'] != rows[zero - 1]['stateAfter']):
            raise ValueError('The action-state chain is discontinuous.')
        description = row['label'].replace('_', ' ')
        if row['phase'] == 'dura' and 'operativeBodies' in row:
            actual = row['operativeBodies']['dura']['cutCount']
            prior = rows[zero-1].get('operativeBodies', {}).get('dura', {}).get('cutCount', 0) if zero else 0
            if actual == prior == 0 and row['action']['operation'] == 'apply':
                description = 'Scissor contact · no dural attachment divided'
        captions.append({'stateAfter': row['stateAfter'], 'text': description})
        if row['phase'] != phase:
            phase = row['phase']
            current = PHASES[phase]
        cue = next((c[1:] for c in CUES if row['label'].startswith(c[0])), None)
        if cue:
            title, fov, tilt, azimuth, focus = cue
            current = (title, fov, tilt, azimuth, row['action']['slot'] if focus == 'hand' else focus)
        title, fov, tilt, azimuth, focus = current
        if not chapters or title != chapters[-1]['title'] or phase != chapters[-1]['phase']:
            if chapters:
                stills.add(zero - 1)
            chapter = {'first': zero, 'last': zero, 'title': title, 'fieldMm': fov, 'angleDeg': tilt,
                       'azimuthDeg': azimuth, 'phase': phase, 'clearParked': True}
            chapter['focus' if focus == 'opening' else 'focusHand'] = focus
            if focus in ('left', 'right') and row.get('acceptedPose'):
                chapter['focusPoint'] = row['acceptedPose']['position']
                chapter['focusSourceState'] = row['stateAfter']
            chapters.append(chapter)
            stills.add(zero)
            if cue:
                holds[str(zero + 1)] = .70 if focus in ('left', 'right') else .42
        chapters[-1]['last'] = zero
        if row.get('acceptedPose') and row['action']['operation'] == 'withdraw':
            chapters[-1].setdefault('withdrawnPoses', {})[row['action']['slot']] = row['acceptedPose']
        # Include the recorded, held tissue's path so a fixed lens still fits
        # the moving grip at the *end* of a fold or rigid flap transfer.
        if row.get('acceptedPose') and row['label'].startswith(('Reflect ', 'Return scalp about', 'Return dura about', 'Carry held flap')):
            chapters[-1].setdefault('workingPoints', []).append(row['acceptedPose']['position'])
            chapters[-1]['offsetMm'] = 7 if phase != 'bone' else 3
    stills.add(len(rows) - 1)
    return {'caseHash': history['caseHash'], 'branchId': history['branchId'], 'sourceFileSha256': history['sha256'],
            'anatomyLabel': 'Generated glioma research case  /  Assigned anatomy and mechanics',
            'overviewLabel': 'Generated head with intact scalp and an external three-pad bone-flap rest',
            'requireOpaqueOverview': True, 'requirePresentation': True, 'presentationSeconds': seconds,
            'approachZoom': {'fromMm': 108}, 'chapters': chapters, 'stills': sorted(stills),
            'holdSecondsAtAction': holds, 'actions': captions,
            'phaseWeights': {'approach': 5, 'scalp': 18, 'bone': 19, 'dura': 14, 'treatment': 26, 'closure': 17, 'inspection': 1}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('history', type=Path, help='The episode policy JSON stored beside the .ressectionlab')
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=float, default=96)
    parser.add_argument('--landmarks', type=Path, help='Optional source-specific physical camera landmarks')
    parser.add_argument('--match-before', type=Path, help='An earlier verified native review directory')
    parser.add_argument('--match-boundaries', type=int, nargs='+', default=[21,144])
    args = parser.parse_args()
    history = json.loads(args.history.read_text())
    result = plan(history, args.seconds)
    if args.landmarks:
        landmarks = json.loads(args.landmarks.read_text())
        if landmarks['caseHash'] != result['caseHash']:
            raise ValueError('The field and the external support must be registered to the same case.')
        entry, rest = landmarks['entryCenterRasMm'], landmarks['boneParkRasMm']
        for c in result['chapters']:
            if c['title'] in ('Transfer the held flap to its support', 'Parked bone flap', 'Replace the supported flap'):
                c['focusPoint'] = [(x+y)/2 for x,y in zip(entry, rest, strict=True)]
                c['offsetMm'] = 2
                c['workingPoints'] = [entry, rest, *c.get('workingPoints', [])]
                c['azimuthDeg'] = -35
                c['fieldMm'] = 66
                c['supportPair'] = [entry, rest]
    if args.match_before:
        previous = json.loads((args.match_before / 'capture.json').read_text())
        if not previous['passed'] or previous['sourceCaseHash'] != result['caseHash']:
            raise ValueError('The comparison requires an earlier verified capture from the same native source.')
        result['matchBefore'] = {}
        for boundary in args.match_boundaries:
            old = next(row for row in previous['review'] if row['boundary'] == boundary)
            if old['stateId'] != history['actions'][boundary-1]['stateAfter']:
                raise ValueError('The historical camera proof has a different committed state.')
            shot = next(c for c in previous['chapters'] if c['first'] <= boundary-1 <= c['last'])
            before = (args.match_before / f'action-{boundary:03}.png').resolve(strict=True)
            result['matchBefore'][boundary] = {'stateId': old['stateId'], 'caseHash': result['caseHash'],
                'shot': {**shot, 'clearParked': False}, 'fieldMm': old['fieldMm'], 'camera': old['camera'],
                'beforeFile': str(before), 'beforeSha256': hashlib.sha256(before.read_bytes()).hexdigest(),
                'beforeRenderer': previous['rendererAsset']}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
