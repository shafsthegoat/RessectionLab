#!/usr/bin/env python3
"""Merge reviewed display fields into a distinct native; never run patient physics.

Poses are camera bounds only, not assertions of actual observed tool motion.
Default minimum 3-D pane aspect is 0.8 (width/height); narrower panes may crop.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from zipfile import ZipFile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/resectionlab').is_dir())
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from resectionlab.surgical_tools import catalog
from resectionlab.workspace_bundle import load_workspace, encoded, session_digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''): h.update(chunk)
    return h.hexdigest()


def deny_physics(frame, event, arg):
    if (event == 'call' and frame.f_code.co_filename.endswith('/surgical_engine.py')
            and frame.f_code.co_name in ('__init__', 'step', 'restore', 'preview', '_execute', '_substep')):
        raise AssertionError('Display-only helper cannot construct or replay a patient engine')


def vector(value):
    v = np.asarray(value, dtype=float)
    if v.shape != (3,) or not np.all(np.isfinite(v)): raise ValueError('Invalid display-bound position')
    return v


def wide_camera(case, record, initial_slots, reference_camera, minimum_aspect, margin):
    renderer = ROOT / 'desktop/src/viewer/VolumeRenderer.ts'
    geometry = ROOT / 'desktop/src/viewer/surgicalGeometry.ts'
    match = re.search(r'new THREE\.PerspectiveCamera\((\d+(?:\.\d+)?),', renderer.read_text())
    if match is None: raise ValueError('Cannot read actual renderer perspective FOV')
    fov = float(match[1]); tools = {t['id']: t for t in catalog()}
    # Whole source-grid support, including voxel half extents; no segmentation
    # completeness or actual-path claim is inferred from this framing box.
    corners = np.array([case.affine @ np.array([*p, 1.]) for p in
                       itertools.product(*[(-.5, n-.5) for n in case.mri.shape])])[:, :3]
    if case.frame == 'LPS+': corners *= [-1, -1, 1]
    low, high = corners.min(axis=0), corners.max(axis=0)
    source_bounds = [low.tolist(), high.tolist()]
    counts = {}; largest_radius = 0.; implicit = 0

    def include(a, b, slot):
        nonlocal low, high, largest_radius
        tool = tools[slot['instrument']]; params = slot['parameters']
        length = tool['workingLengthMm']; shaft = tool['shaftRadiusMm']
        radius = float(params.get('radiusMm', tool['radiusMm']))
        jaw = float(params.get('jawOpeningMm', 2))
        if not all(math.isfinite(x) and x >= 0 for x in (length, shaft, radius, jaw)):
            raise ValueError('Invalid recorded tool geometry parameter')
        # Encloses the renderer's shaft (-length-3), rear collar
        # (-length-3.5), and all catalog tip/jaw geometries, any orientation.
        envelope = max(math.hypot(length+3, shaft), math.hypot(length+3.5, 2.1*shaft),
                       3*radius+12, jaw+12) + 2
        largest_radius = max(largest_radius, envelope)
        low = np.minimum(low, a-envelope); high = np.maximum(high, b+envelope)

    for branch, snapshot in sorted(record['branches'].items()):
        slots = copy.deepcopy(initial_slots)
        possible = {key: (vector(slot['position']), vector(slot['position'])) for key, slot in slots.items()}
        for key, slot in slots.items(): include(*possible[key], slot)
        pose_count = 0
        for action in snapshot['actions']:
            key = action['slot']; slot = slots[key]
            if action['operation'] == 'exchange':
                slot['instrument'] = action['instrument']; slot['parameters'] = {}
            slot['parameters'].update(action.get('parameters', {}))
            a, b = possible[key]
            poses = action.get('path') or ([action['pose']] if action.get('pose') else [])
            if poses:
                points = np.asarray([vector(p['position']) for p in poses])
                pose_count += len(poses)
                if action['operation'] == 'place':
                    if len(points) != 1: raise ValueError('Recorded placement must have one pose')
                    a = b = points[0]  # Public place sets this exact free-space pose.
                else:
                    # A partial stroke can stop anywhere along a requested leg.
                    a = np.minimum(a, points.min(axis=0)); b = np.maximum(b, points.max(axis=0))
            elif action['operation'] == 'withdraw':
                # Public implicit withdrawal requests 50mm along tool -Z.
                # Unknown partial orientation is enclosed without simulating it.
                a = a-50; b = b+50; implicit += 1
            possible[key] = (a, b)
            include(a, b, slot)
        # Include the actually saved endpoint bodies too, directly from records.
        for slot in snapshot.get('checkpoint', {}).get('state', {}).get('slots', {}).values():
            p = vector(slot['position']); include(p, p, slot)
        counts[branch] = {'actions': len(snapshot['actions']), 'requestedPosesForBoundsOnly': pose_count}

    low -= 8; high += 8  # Same complete-envelope padding used by the renderer.
    centre = (low+high)/2
    direction = vector(reference_camera['position'])-vector(reference_camera['target'])
    direction /= np.linalg.norm(direction)
    up = vector(reference_camera['up']); right = np.cross(up, direction)
    if np.linalg.norm(right) < 1e-8: raise ValueError('Reference camera up is parallel to its direction')
    right /= np.linalg.norm(right); up = np.cross(direction, right); up /= np.linalg.norm(up)
    vertical = math.radians(fov); horizontal = 2*math.atan(math.tan(vertical/2)*minimum_aspect)
    box = np.array(list(itertools.product(*zip(low, high))))
    distance = 40.
    for p in box-centre:
        distance = max(distance, float(p@direction + abs(p@right)/math.tan(horizontal/2)),
                       float(p@direction + abs(p@up)/math.tan(vertical/2)))
    distance *= margin
    max_distance = re.search(r'this\.controls\.maxDistance = (\d+(?:\.\d+)?);', renderer.read_text())
    if max_distance is None or distance > float(max_distance[1]):
        raise ValueError('Computed camera exceeds actual renderer orbit distance limit')
    for p in box-centre:
        depth = distance-p@direction
        assert depth > 0 and abs(p@right) <= depth*math.tan(horizontal/2)+1e-7
        assert abs(p@up) <= depth*math.tan(vertical/2)+1e-7
    camera = {'position': (centre+distance*direction).tolist(), 'target': centre.tolist(), 'up': up.tolist()}
    return camera, {'sourceGridBoundsRasMm': source_bounds, 'completePaddedBoundsRasMm': [low.tolist(), high.tolist()],
        'fovDegrees': fov, 'minimumPaneAspectWidthOverHeight': minimum_aspect, 'distanceMargin': margin,
        'distanceMm': distance, 'largestOrientationIndependentBodyRadiusMm': largest_radius,
        'implicitWithdrawBoundsExpanded': implicit, 'branches': counts, 'allBoxCornersInsideBothFovAxes': True,
        'initialSlots': initial_slots, 'requestedPosesMeaning': 'Conservative camera bounds only; not actual-motion observations.',
        'rendererSha256': sha(renderer), 'toolGeometrySha256': sha(geometry),
        'catalogSourceSha256': sha(ROOT/'src/resectionlab/surgical_tools.py')}


def apply(native, reference, output, *, minimum_aspect=.8, margin=1.08):
    native, reference, output = (Path(p).resolve() for p in (native, reference, output))
    provenance = output.with_suffix(output.suffix+'.display-provenance.json')
    if output in (native, reference) or output.exists() or provenance.exists():
        raise ValueError('Output and provenance must be distinct new paths')
    if not .2 <= minimum_aspect <= 4 or not 1.01 <= margin <= 2:
        raise ValueError('Require minimum pane aspect0.2..4 and margin1.01..2')
    output.parent.mkdir(parents=True, exist_ok=True)
    old_profile = sys.getprofile(); sys.setprofile(deny_physics)
    try:
        before_sha, reference_sha = sha(native), sha(reference)
        case, artifacts, workspace = load_workspace(native)
        ref_case, ref_artifacts, ref_workspace = load_workspace(reference)
        if workspace is None or ref_workspace is None: raise ValueError('Workspace-native inputs required')
        if case.semantic_hash != ref_case.semantic_hash: raise ValueError('Display reference is a different source case')
        record = artifacts['surgical_toolbox']; ref_record = ref_artifacts['surgical_toolbox']
        scene_ids = {s['sceneId'] for s in record['branches'].values()}
        candidates = [(name, s) for name, s in ref_record['branches'].items()
                      if not s['actions'] and s.get('checkpoint', {}).get('position') == 0 and s['sceneId'] in scene_ids]
        if len(scene_ids) != 1 or not candidates: raise ValueError('Matching recorded zero-action initial slots required')
        initial_branch, initial = candidates[0]; initial_slots = initial['checkpoint']['state']['slots']
        if set(initial_slots) != {'left', 'right'}: raise ValueError('Exact both-hand initial slots required')
        ref = ref_workspace['imagingState']['states']['primary']
        camera, fit = wide_camera(case, record, initial_slots, ref['viewer']['camera'], minimum_aspect, margin)
        imaging = copy.deepcopy(workspace['imagingState']); primary = imaging['states']['primary']
        primary['cursor'] = copy.deepcopy(ref['cursor']); primary['visibleLayers'] = copy.deepcopy(ref['visibleLayers'])
        merged_fields = ['primary.cursor', 'primary.visibleLayers']
        for field in ('layerColor', 'layerOpacity', 'smoothSurfaces'):
            if field in ref['viewer']:
                if field == 'smoothSurfaces' and type(ref['viewer'][field]) is not bool:
                    raise ValueError('Reference smoothSurfaces must be an explicit boolean')
                primary['viewer'][field] = copy.deepcopy(ref['viewer'][field])
                merged_fields.append('primary.viewer.'+field)
        primary['viewer']['camera'] = camera
        merged_fields.append('primary.viewer.camera')
        display_artifact = copy.deepcopy(artifacts.get('workspace', {}))
        for field in ('cursor', 'visibleLayers', 'overlayOpacity'):
            if field in ref_artifacts.get('workspace', {}):
                display_artifact[field] = copy.deepcopy(ref_artifacts['workspace'][field])
        # ImagingState and legacy workspace cursor/visibility must agree.
        display_artifact['cursor'] = copy.deepcopy(primary['cursor'])
        display_artifact['visibleLayers'] = copy.deepcopy(primary['visibleLayers'])
        merged_fields.extend(['artifacts.workspace.cursor', 'artifacts.workspace.visibleLayers'])
        if 'overlayOpacity' in ref_artifacts.get('workspace', {}):
            merged_fields.append('artifacts.workspace.overlayOpacity')
        with tempfile.TemporaryDirectory(prefix='.rehearsal-display-', dir=output.parent) as temp:
            staged = Path(temp)/'display.ressectionlab'
            with ZipFile(native) as src:
                manifest = json.loads(src.read('manifest.json')); document = json.loads(src.read('workspace.json'))
                original_manifest = copy.deepcopy(manifest); original_document = copy.deepcopy(document)
                manifest['artifacts']['workspace'] = display_artifact
                document['imagingState'] = imaging; document['sessionHash'] = session_digest(document)
                workspace_bytes = encoded(document); manifest['workspace_sha256'] = hashlib.sha256(workspace_bytes).hexdigest()
                with ZipFile(staged, 'x') as dst:
                    for member in src.infolist():
                        data = encoded(manifest) if member.filename == 'manifest.json' else workspace_bytes if member.filename == 'workspace.json' else src.read(member.filename)
                        dst.writestr(member, data)
            reopened, after_artifacts, after_workspace = load_workspace(staged)
            assert reopened.to_manifest() == case.to_manifest()
            expected_artifacts = copy.deepcopy(artifacts); expected_artifacts['workspace'] = display_artifact
            assert after_artifacts == expected_artifacts and after_workspace['imagingState'] == imaging
            assert after_artifacts['surgical_toolbox'] == record
            # No source array, auxiliary native, original file or other ZIP payload is re-encoded.
            unchanged_members = {}
            with ZipFile(native) as before, ZipFile(staged) as after:
                assert before.namelist() == after.namelist()
                for name in before.namelist():
                    if name not in ('manifest.json', 'workspace.json'):
                        data = before.read(name); assert data == after.read(name)
                        unchanged_members[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
                assert original_manifest['artifacts']['surgical_toolbox'] == manifest['artifacts']['surgical_toolbox']
            assert sha(native) == before_sha and sha(reference) == reference_sha
            # Hard-link publication refuses an existing output, including a race.
            os.link(staged, output)
        report = {'schema': 'resectionlab.rehearsal-display-merge.v1', 'status': 'passed',
            'native': str(native), 'nativeSha256': before_sha, 'reference': str(reference), 'referenceSha256': reference_sha,
            'output': str(output), 'outputSha256': sha(output), 'sourceCaseHash': case.semantic_hash,
            'initialSlotsReferenceBranch': initial_branch, 'initialSnapshotStateId': initial['stateId'],
            'surgicalToolboxCanonicalJsonSha256': hashlib.sha256(encoded(record)).hexdigest(), 'surgicalToolboxEncodedArtifactUnchanged': True,
            'allOtherArtifactsUnchanged': True, 'decodedCaseManifestUnchanged': True,
            'nonMetadataZipMemberBytesUnchanged': unchanged_members,
            'displayFieldsMerged': merged_fields,
            'otherViewerSettingsPreserved': True, 'wideCameraFit': fit,
            'physicsCalls': 0, 'guiVerified': False, 'helperSha256': sha(__file__)}
        with provenance.open('x') as stream: stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
        return report
    finally: sys.setprofile(old_profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path); parser.add_argument('reference', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('--minimum-pane-aspect', type=float, default=.8, help='Minimum supported 3-D pane width/height (default0.8; use0.5 for narrower panes)')
    parser.add_argument('--margin', type=float, default=1.08, help='Camera distance margin, default1.08')
    args = parser.parse_args(); report = apply(args.native, args.reference, args.output,
                                             minimum_aspect=args.minimum_pane_aspect, margin=args.margin)
    print(json.dumps({k: report[k] for k in ('status', 'output', 'outputSha256', 'surgicalToolboxEncodedArtifactUnchanged', 'physicsCalls', 'guiVerified')}))


if __name__ == '__main__': main()
