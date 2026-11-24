"""Memory-bounded research derivative of explicitly selected patient annotations.

The source collection is read-only. Unreviewed registration requires an explicit
opt-in; its status and evidence are copied verbatim, never promoted to reviewed.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from urllib.parse import quote
import numpy as np
from scipy.ndimage import affine_transform
from .core import CaseData, SourceRef, array_digest
from .imaging import save_case
from .patient_anatomy import PatientImagingCollection

ORDER = ('brain', 'tumor', 'bone', 'vessels')
MATERIAL = {'brain': 'soft_tissue', 'tumor': 'tumor', 'bone': 'bone', 'vessels': 'vessel'}
COLORS = {'brain': '#c799b0', 'tumor': '#db6b95', 'bone': '#e5d6b8', 'vessels': '#e7645c'}
ANATOMY_KINDS = {'brain': 'brain', 'bone': 'skull', 'tumor': 'tumor', 'vessels': 'vessels'}
COVERAGE = {
    'brain': 'Brain extraction estimate; boundary and tumor-distorted regions unreviewed.',
    'bone': 'CT bone-density estimate includes teeth and other bone; skull boundary and registration unreviewed.',
    'tumor': 'Tumor proposal without supplied ground-truth annotation; extent and boundaries unreviewed.',
    'vessels': 'Experimental structural-MRI vessel estimate; vascular completeness and small branches unvalidated.',
}


def _anatomy_metadata(name, row):
    provenance = row.get('provenance', {})
    inference = provenance.get('inference', {})
    model = inference.get('model', {})
    model_name = model.get('name') if isinstance(model, dict) else model
    if not model_name:
        declared = provenance.get('model')
        model_name = declared.get('name') if isinstance(declared, dict) else declared
    if not model_name:
        model_name = provenance.get('method') or 'Unreviewed source segmentation estimate'
    source_annotation = provenance.get('kind') == 'observed_source_annotation'
    coverage = COVERAGE[name]
    if name == 'vessels' and provenance.get('source_modality') in {'MRA', 'TOF', 'TOF-MRA', 'CTA'}:
        coverage = 'Angiography-derived vessel annotation or estimate; vascular completeness and small branches unvalidated.'
    return {'kind': ANATOMY_KINDS[name], 'color': COLORS[name],
            'provenance': 'observed' if source_annotation else 'estimated',
            'review_status': row.get('review_status', 'unreviewed'), 'model': {'name': model_name},
            'coverage': (row.get('coverage') or 'Source annotation; completeness unassessed.') if source_annotation
                        else coverage + ' ' + row.get('coverage', ''),
            'source_provenance': copy.deepcopy(provenance)}

DEFAULT_PHYSICS = {'schema': 'resectionlab.tissue-physics.v1',
    'provenance': {'kind': 'assumed-research', 'description': 'Explicit research wound-aperture assumptions; radius is NOT inferred vessel diameter. No calibrated patient mechanics.'},
    'bone': {'model': 'rigid-cell-removal-v1'},
    'vessel': {'model': 'local-poiseuille-v1', 'radiusMm': .15, 'pressurePa': 1000,
               'viscosityPaS': .0035, 'pathLengthMm': 5, 'maxFlowMm3PerSecond': 50, 'coagulationExposure': 2}}


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _resample(array, source_affine, target_affine, shape, *, label=None, conservative=False):
    """Chunked linear intensities or categorical sampling at <=0.5 mm spacing."""
    matrix = np.linalg.inv(source_affine) @ target_affine
    result = np.zeros(shape, np.float32 if label is None else bool)
    # 2 samples per mm per axis; finer requested grids use a single center.
    subdivisions = np.maximum(1, np.ceil(np.linalg.norm(target_affine[:3, :3], axis=0) / .5).astype(int)) if conservative else np.ones(3, int)
    offsets = list(itertools.product(*[(np.arange(n) + .5) / n - .5 for n in subdivisions]))
    for z in range(0, shape[2], 8):
        block_shape = (shape[0], shape[1], min(8, shape[2] - z))
        target = result[:, :, z:z + block_shape[2]]
        for offset in offsets:
            origin = matrix[:3, 3] + matrix[:3, :3] @ (np.array([0, 0, z]) + offset)
            sampled = affine_transform(array, matrix[:3, :3], origin, output_shape=block_shape,
                                       output=np.float32 if label is None else array.dtype,
                                       order=1 if label is None else 0, mode='constant', cval=0, prefilter=False)
            if label is None:
                target[:] = sampled
            else:
                target |= sampled == label
    return result


ENVIRONMENT_KINDS = {
    'brain': {'brain'}, 'tumor': {'tumor'}, 'bone': {'bone', 'skull'},
    'vessels': {'vessel', 'vessels', 'artery', 'vein'},
}
REVIEWED_REGISTRATION = {'reference', 'reviewed', 'source-provided'}


def _check_layer_export(collection, role, row):
    from .anatomy_evidence import vascular_evidence_supported

    provenance = row.get('provenance', {})
    if (row['kind'] == 'dura' or provenance.get('model') == 'dural-envelope'
            or not vascular_evidence_supported(collection, row)):
        raise ValueError(f"{row['label']} supplies display context only")
    if role == 'bone':
        if row['kind'] not in ENVIRONMENT_KINDS['bone']:
            raise ValueError("Bone material requires a source-associated bone or skull layer")
        modality = collection.series(row['source_series'])['modality']
        if modality not in {'CT', 'CTA'} and (
                provenance.get('model') or provenance.get('evidence_use') == 'context_only'
                or row['id'] in {'cortical-bone', 'cancellous-bone'}):
            raise ValueError("MRI skull estimates cannot supply measured bone")
    if row.get('units_status', 'mm') != 'mm':
        raise ValueError('Selected anatomy layer requires declared millimeter units: ' + role)
    collection.require_native_geometry(row, allow_legacy_annotation_grid=True)
    if row.get('availability') == 'empty_output':
        raise ValueError('Selected anatomy layer is empty')


def _wall_layer_issue(role, row):
    provenance = row.get('provenance', {})
    if row['kind'] not in ENVIRONMENT_KINDS[role]:
        return f"Local native-wall rehearsal cannot use {row['label']} for the {role} material"
    if (provenance.get('kind') in {'synthetic', 'simulated', 'generated'}
            or provenance.get('evidence_use') not in (None, 'positive_labels_only')):
        return f"Local native-wall rehearsal cannot use {row['label']} as mechanical anatomy"
    if role == 'vessels' and (not row.get('probability') or row.get('edits') or row.get('revision') != 0):
        return 'Local native-wall rehearsal requires an intact native angiography score layer'
    return None


def _source_registration(collection, series):
    sid = series['id']
    if series.get('units_status') != 'mm':
        raise ValueError('Every source series must declare millimeter units')
    transforms = collection.document['transforms']
    if sid not in transforms:
        raise ValueError('Missing registration for ' + sid)
    matrix = collection.registration_matrix(sid, require_geometry=True)
    unreviewed, current = [], sid
    while current:
        record = transforms[current]
        if record['status'] not in REVIEWED_REGISTRATION:
            unreviewed.append(current)
        current = record.get('parent_series')
    return matrix, unreviewed


def _research_grid(collection, reference, resolution_mm):
    reference_affine = collection.patient_affine(reference)
    spacing = np.linalg.norm(reference_affine[:3, :3], axis=0)
    # Preserve original voxel axes, including handedness and obliquity.
    target_affine = reference_affine.copy()
    target_affine[:3, :3] = reference_affine[:3, :3] / spacing * resolution_mm
    ratio = resolution_mm / spacing
    target_affine[:3, 3] = (reference_affine @ np.r_[((ratio - 1) / 2), 1])[:3]
    shape = tuple(np.ceil(np.asarray(reference['data']['shape']) / ratio).astype(int))
    if np.prod(shape) > 80_000_000:
        raise ValueError('Requested derivative exceeds 80 million cells')
    return reference_affine, target_affine, shape


def surgical_environment_options(collection, *, check=lambda: None):
    """Source and role preflight for the separate limited rehearsal derivative."""
    from .anatomy_local_wall_review import PROPERTY, native_wall_alignment_review
    from .surgical_patient_wall import native_wall_lattice
    from .surgical_selected_decision import selected_route_forecast
    native_wall = {'referenceIssue': None, 'pairs': []}
    result = {'referenceIssue': None, 'layers': [], 'selectedRouteForecast': selected_route_forecast(),
              'nativeWall': native_wall}
    try:
        reference = collection.series(collection.document['reference_series'])
        reference_registration = _source_registration(collection, reference)
    except (ValueError, StopIteration) as error:
        issue = str(error) or 'An acquired reference scan is required'
        result['referenceIssue'] = native_wall['referenceIssue'] = issue
        return result
    if reference_registration[1]:
        native_wall['referenceIssue'] = 'Local native-wall rehearsal requires reviewed source alignment: ' + ', '.join(reference_registration[1])
    try:
        _research_grid(collection, reference, 1.)
    except ValueError as error:
        result['referenceIssue'] = str(error)
    registration = {reference['id']: reference_registration}
    native_brains, native_vessels = {}, []
    for role, kinds in ENVIRONMENT_KINDS.items():
        for row in collection.document['layers']:
            if row['kind'] not in kinds:
                continue
            check()
            option = {'role': role, 'layerId': row['id'], 'status': 'unavailable', 'reason': None}
            native_issue = _wall_layer_issue(role, row)
            if native_issue:
                option['nativeWallIssue'] = native_issue
            try:
                _check_layer_export(collection, role, row)
                sid = row['source_series']
                if sid not in registration:
                    registration[sid] = _source_registration(collection, collection.series(sid))
                matrix, unreviewed = registration[sid]
                data = collection.array(row['data'])
                slab = max(1, 262_144 // max(1, np.prod(data.shape[1:])))
                for offset in range(0, len(data), slab):
                    check()
                    if np.any(data[offset:offset + slab] == row['label_value']):
                        break
                else:
                    raise ValueError('Selected anatomy layer is empty')
                option.update(status='estimated_alignment' if unreviewed else 'ready')
                if unreviewed:
                    option['reason'] = 'Estimated source alignment requires an explicit research choice'
                if not native_issue and role == 'brain' and not unreviewed:
                    native_brains[row['id']] = row
                if not native_issue and role == 'vessels':
                    try:
                        native_wall_lattice(collection.series(sid), row, matrix)
                    except ValueError as error:
                        option['nativeWallIssue'] = str(error)
                    else:
                        native_vessels.append((row, option, unreviewed))
            except (ValueError, OSError, StopIteration) as error:
                option['reason'] = str(error) or 'Source series unavailable'
            result['layers'].append(option)

    if native_wall['referenceIssue']:
        return result
    for vessel, option, unreviewed in native_vessels:
        check()
        if not unreviewed:
            native_wall['pairs'].extend({'brainLayerId': brain_id, 'vesselLayerId': vessel['id'], 'scope': 'cursor'}
                                        for brain_id in native_brains)
            continue
        transform = collection.document['transforms'][vessel['source_series']]
        local = transform.get(PROPERTY)
        binding = local.get('binding') if isinstance(local, dict) else None
        brain_id = binding.get('brainLayerId') if isinstance(binding, dict) else None
        brain = native_brains.get(brain_id) if isinstance(brain_id, str) else None
        if brain is not None and (review := native_wall_alignment_review(collection, brain, vessel)):
            alignment = {'brainLayerId': brain['id'], 'centerRasMm': review['centerRasMm']}
            option['nativeWallAlignment'] = alignment
            native_wall['pairs'].append({**alignment, 'vesselLayerId': vessel['id'], 'scope': 'reviewed-crop'})
    return result


def prepare_surgical_patient(collection_path, output_dir, *, layer_ids,
                             resolution_mm=1., allow_unreviewed_registration=False,
                             tissue_physics=None, require_complete=True, approach=None,
                             native_wall_center=None, check=lambda: None):
    """Save patient.ressectionlab and report.json; return the auditable report.

    layer_ids maps exactly brain/tumor/bone/vessels to existing collection IDs.
    MRI uses linear interpolation; brain/tumor use nearest-neighbor centers.
    Bone/vessels use any-hit samples spaced at most 0.5 mm within each output
    cell. This deliberately expands coarse-cell occupancy; it neither fills
    holes nor guarantees capture of structures below the sampling spacing.
    All overlapping sampled masks remain annotations; ownership priority is
    brain < tumor < bone < vessels. No attachments are fabricated.
    """
    if require_complete and (set(layer_ids) != set(ORDER) or len(set(layer_ids.values())) != 4):
        raise ValueError('Select exactly four distinct brain/tumor/bone/vessels layers')
    if not layer_ids or not set(layer_ids) <= set(ORDER) or len(set(layer_ids.values())) != len(layer_ids):
        raise ValueError('Select distinct existing anatomy layers for the declared roles')
    selected_order = tuple(name for name in ORDER if name in layer_ids)
    if native_wall_center is not None and (approach is not None or not {'brain', 'vessels'} <= layer_ids.keys()):
        raise ValueError('Local native-wall rehearsal requires a brain layer and a vascular layer, with no selected procedure')
    if not np.isfinite(resolution_mm) or not .5 <= resolution_mm <= 2:
        raise ValueError('Research derivative resolution must be between 0.5 and 2 mm')
    root = Path(collection_path)
    if root.is_file(): root = root.parent
    collection = PatientImagingCollection(root)
    manifest_path = root / 'patient-anatomy.json'
    def selected_approach():
        from .anatomy_operative_field import _saved_route
        required = {'approachId', 'parameterHash', 'targetLayerId', 'snapshotId'}
        if type(approach) is not dict or not required <= set(approach) <= required | {'measurementHash'}:
            raise ValueError('Select a current saved approach and target')
        if layer_ids.get('tumor') != approach['targetLayerId']:
            raise ValueError('Environment tumor differs from the selected approach target')
        route, binding, revision = _saved_route(collection, approach)
        if 'measurementHash' in approach and approach['measurementHash'] != route['measurementHash']:
            raise ValueError('The selected route measurements changed; review the forecast again')
        from .surgical_selected_decision import selected_route_forecast
        measured = route['measurements']
        geometry = {'lengthMm': measured['lengthMm'],
                    'traversedNonTargetMm3': measured['normalTraversal'].get('supportedNonTargetVoxelCenterVolumeMm3'),
                    'unknowns': list(measured['unknowns'])}
        return {'schema': 'resectionlab.native-environment-approach.v1', 'planRevision': revision,
            'geometry': geometry, 'forecast': selected_route_forecast(),
            'targetBinding': binding, **{key: copy.deepcopy(route[key]) for key in
                ('id', 'name', 'sourceSnapshotId', 'targetLayerId', 'parameterHash', 'measurementHash',
                 'entry', 'target', 'radiusMm', 'direction')}}
    planned = selected_approach() if approach is not None else None
    manifest_hash = _sha(manifest_path)
    manifest = copy.deepcopy(collection.document)
    reference = collection.series(manifest['reference_series'])
    if native_wall_center is None:
        reference_affine, target_affine, shape = _research_grid(collection, reference, resolution_mm)
    else:
        reference_affine = collection.patient_affine(reference)
    rows = {name: collection.layer(layer_ids[name]) for name in selected_order}
    for name, row in rows.items():
        _check_layer_export(collection, name, row)
        if native_wall_center is not None:
            native_issue = _wall_layer_issue(name, row)
            if native_issue:
                raise ValueError(native_issue)
    warnings = ['Research derivative of source annotations and segmentation estimates; provenance and review status are retained, not independently validated.',
                'Bone/vessel any-hit coarse occupancy can enlarge boundaries and still miss structures smaller than 0.5 mm.',
                'Wound aperture and flow parameters are explicit assumptions, not measured vessel diameter or patient physiology.']
    affines = {}
    source_refs = []
    native_alignment = None
    used_series = {reference['id'], *(r['source_series'] for r in rows.values())}
    for sid in sorted(used_series):
        check()
        series = collection.series(sid)
        # Validate source hashes, transform digest and parent graph regardless of status.
        matrix, unreviewed = _source_registration(collection, series)
        transform = manifest['transforms'][sid]
        if unreviewed:
            if native_wall_center is not None:
                from .anatomy_local_wall_review import native_wall_alignment_review

                if sid == rows['vessels']['source_series']:
                    native_alignment = native_wall_alignment_review(collection, rows['brain'], rows['vessels'])
                if sid != rows['vessels']['source_series'] or native_alignment is None:
                    raise ValueError('Local native-wall rehearsal requires reviewed source alignment: ' + ', '.join(unreviewed))
                warnings.append('The vascular alignment has a landmark review for this local native-wall crop. '
                                'The source transform retains its general review status.')
            else:
                if not allow_unreviewed_registration:
                    raise ValueError('Unreviewed registration requires explicit research opt-in: ' + ', '.join(unreviewed))
                warnings.append('Research opt-in uses unreviewed registration: ' + ', '.join(unreviewed) + '; source review status unchanged.')
        warnings.extend(transform.get('quality', {}).get('warnings', []))
        affines[sid] = matrix
        original = series['original']
        original_path = collection.path(original['path'])
        if _sha(original_path) != original['sha256']:
            raise ValueError('Source original checksum differs')
        # The native package may be extracted under a new directory on each open.
        # Relative references resolve within the hash-bound source manifest.
        uri = quote(original['path']) if planned else original_path.as_uri()
        source_refs.append(SourceRef(sid, uri, original['sha256'], native_frame='RAS+'))
    manifest_uri = 'patient-anatomy.json' if planned else manifest_path.resolve().as_uri()
    source_refs.append(SourceRef('source-manifest', manifest_uri, manifest_hash, native_frame='RAS+'))
    local_wall = None
    if native_wall_center is not None:
        from .anatomy_sampling import sample_layer
        from .surgical_patient_wall import native_wall_grid

        local_wall = native_wall_grid(collection, rows['vessels'], affines[rows['vessels']['source_series']], native_wall_center)
        if native_alignment is not None:
            native_wall_alignment_review(collection, rows['brain'], rows['vessels'], local_wall)
        target_affine, shape = local_wall.affine, local_wall.mask.shape
        warnings[1] = 'Vessel score fields and masks retain native samples. The local domain clips coverage; field edges and wall caps mark unassessed continuation.'
        warnings[2] = 'Tissue attachments and vessel mechanics are assigned research assumptions. There is no measured patient wall material, attachment map, or circulation model.'
    mri = _resample(collection.array(reference['data']), reference_affine, target_affine, shape)
    masks, layer_report = {}, {}
    sampling_cache = {}
    for name in selected_order:
        check()
        row = rows[name]
        raw = collection.array(row['data'])
        registered = affines[row['source_series']] @ np.asarray(row['affine'])
        support_cells = None
        if local_wall is not None and name == 'vessels':
            sampled = local_wall.mask
            sampling = 'exact native crop'
            support_cells = sampled.size
        elif local_wall is not None:
            source = collection.series(row['source_series'])
            frames = (registered, affines[row['source_series']] @ np.asarray(source['affine']))
            sampled, covered = sample_layer(collection, row, target_affine, shape, check, sampling_cache, frames=frames)
            sampling = 'nearest-neighbor center inside acquired and declared label coverage'
            support_cells = int(covered.sum())
            if support_cells < sampled.size:
                warnings.append(f'{name.capitalize()} source coverage is partial in the local domain. '
                                'Uncovered cells supply no mechanical material.')
        else:
            conservative = name in ('bone', 'vessels')
            sampled = _resample(raw, registered, target_affine, shape, label=row['label_value'], conservative=conservative)
            sampling = 'any-hit <=0.5mm samples' if conservative else 'nearest-neighbor center'
        if not sampled.any():
            if local_wall is not None and name not in ('brain', 'vessels'):
                warnings.append(f'The selected {name} layer has no occupied samples in the local domain.')
                continue
            raise ValueError('Selected layer has no occupied cells in the derivative: ' + name)
        masks[name] = sampled
        original_count = sum(int(np.count_nonzero(raw[:, :, z] == row['label_value'])) for z in range(raw.shape[2]))
        layer_report[name] = {'layerId': row['id'], 'sourceOccupiedCells': original_count,
            'sampledOccupiedCells': int(sampled.sum()), 'sampledMaskHash': array_digest(sampled),
            'sourceArray': copy.deepcopy(row['data']), 'sourceAffine': row['affine'],
            'revision': row.get('revision', 0), 'originalData': copy.deepcopy(row.get('original_data')),
            'edits': copy.deepcopy(row.get('edits', [])),
            'registeredAffine': registered.tolist(), 'provenance': copy.deepcopy(row.get('provenance', {})),
            'reviewStatus': row.get('review_status'),
            'uncertainty': copy.deepcopy(row.get('uncertainty')), 'scoreMeaning': row.get('score_meaning'),
            'probability': copy.deepcopy(row.get('probability')),
            'sampling': sampling}
        if support_cells is not None:
            layer_report[name]['sampledCoverageCells'] = support_cells
    selected_order = tuple(masks)
    overlaps = {a + ':' + b: int(np.count_nonzero(masks[a] & masks[b])) for a, b in itertools.combinations(selected_order, 2)}
    ownership = np.zeros(shape, np.uint8)
    for i, name in enumerate(selected_order, 1): ownership[masks[name]] = i
    ownership_counts = {name: int(np.count_nonzero(ownership == i)) for i, name in enumerate(selected_order, 1)}
    if not all(ownership_counts.values()):
        raise ValueError('Ownership precedence eliminated an entire requested material')
    if local_wall is not None and tissue_physics is None:
        from .surgical_physics_v2 import default_profile
        tissue_physics = default_profile()
    physics = copy.deepcopy(DEFAULT_PHYSICS if tissue_physics is None else tissue_physics)
    adapter = {'type': 'native-patient-research-derivative', 'warnings': list(dict.fromkeys(warnings)),
        'representations': {'acquiredImages': 'immutable source original checksums and source reference array',
            'anatomicalEstimates': 'selected revisioned segmentation masks; review status and uncertainty retained',
            'displaySurfaces': 'excluded; display smoothing never changes these simulation masks'},
        'sourcePatient': manifest['patient_id'], 'sourceCollectionHash': collection.identity(),
        'sourceManifestSha256': manifest_hash, 'sourceManifest': manifest,
        'resolutionMm': resolution_mm, 'targetAffine': target_affine.tolist(),
        'intensityResampling': 'linear interpolation of original reference image; outside-source samples zero; no modeled intensity changes',
        'sourceLayers': layer_report, 'overlapCells': overlaps, 'ownershipPrecedence': list(selected_order),
        'ownershipCells': ownership_counts, 'allowUnreviewedRegistration': bool(allow_unreviewed_registration)}
    physics_key = 'tissuePhysicsV2' if physics.get('schema') == 'resectionlab.tissue-physics.v2' else 'tissuePhysics'
    properties = {physics_key: physics}
    wall_metadata = {}
    if local_wall is not None:
        from .surgical_motion import AXIAL_SWEEP_PROPERTY, AXIAL_SWEEP_POLICY
        from .surgical_patient_wall import qualifying_properties
        from .surgical_vessel_rods import TOOL_CONTACT_CLOCK_PROPERTY, TOOL_CONTACT_CLOCK_POLICY
        from .surgical_vessel_surface import GEOMETRY_PROPERTY

        check()
        wall_properties, binding = qualifying_properties(local_wall, masks, rows['vessels'], manifest['patient_id'])
        adapter['nativeWall'] = binding
        if native_alignment is not None:
            adapter['nativeWallAlignmentReview'] = copy.deepcopy(native_alignment)
        wall_metadata[GEOMETRY_PROPERTY] = binding['geometryPolicy']
        adapter['resolutionMm'] = np.linalg.norm(target_affine[:3, :3], axis=0).tolist()
        adapter['representations']['nativeVesselWall'] = 'bounded conditioning of 0.5 score crossings on the unchanged native crop; accepted vertices shared by contact and rendering'
        properties.update(wall_properties)
        if not planned:
            properties[AXIAL_SWEEP_PROPERTY] = AXIAL_SWEEP_POLICY
            properties[TOOL_CONTACT_CLOCK_PROPERTY] = binding[TOOL_CONTACT_CLOCK_PROPERTY] = TOOL_CONTACT_CLOCK_POLICY
    anatomy_layers = {name: _anatomy_metadata(name, rows[name]) for name in selected_order}
    if local_wall is not None:
        anatomy_layers['vessels'].update(kind=rows['vessels']['kind'], color=rows['vessels']['color'])
    if planned:
        from .surgical_source_inspection import PROPERTY
        adapter['selectedApproach'] = planned
        # The native guard uses presence of PROPERTY for every inspection variant.
        # This derivative has sampled materials, unlike the exact-grid annotation export.
        properties[PROPERTY] = {'schema': 'resectionlab.planned-derivative-inspection.v1',
            'status': 'inspection-only', 'patientTaskAdmission': False,
            'sourceSnapshotId': planned['sourceSnapshotId'], 'targetBinding': planned['targetBinding'],
            'approachId': planned['id'], 'planRevision': planned['planRevision'], 'blockers': [
            {'code': 'PROCEDURE_ACQUIRED_SOURCE_QUALIFICATION_UNAVAILABLE',
             'message': 'The saved approach supplies proposed geometry. Executing it requires qualified source coverage, entry, compartments, operative stages and swept instrument workspace.'}]}
    case = CaseData(case_id=manifest['patient_id'] + '-surgical-research', mri=mri,
        compartments=masks, affine=target_affine, frame='RAS+', source_refs=tuple(source_refs),
        vascular_probabilities={} if local_wall is None else {'vessels': local_wall.scores},
        metadata={'surgical_adapter': adapter, 'surgical_properties': properties, **wall_metadata,
            **({'anatomy_workbench_only': True} if planned else {}),
            'anatomy_layers': anatomy_layers,
            'missing_anatomy': [name for name in ORDER if name not in masks],
            'description': 'Acquired reference image with explicitly selected source annotations and research estimates; missing anatomy remains unknown.'})
    if local_wall is not None:
        from .surgical_engine import SurgicalEngine
        from .surgical_rl_protection import GuardConfig
        from .surgical_scene import from_case

        # The native sweep certifies the intact wall before any report or case is written.
        check()
        SurgicalEngine(from_case(case), guard_config=GuardConfig(require_critical_coverage=False))
        if _sha(manifest_path) != manifest_hash:
            raise ValueError('Source manifest changed during export')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = save_case(case, output / 'patient.ressectionlab')
    if planned is not None and selected_approach() != planned:
        raise ValueError('Selected approach changed during environment export')
    if _sha(manifest_path) != manifest_hash:
        raise ValueError('Source manifest changed during export')
    report = {'schema': 'resectionlab.surgical-patient-export.v1', 'casePath': str(path.resolve()),
              'caseHash': case.semantic_hash, 'shape': list(map(int, shape)), 'voxelMm3': abs(float(np.linalg.det(target_affine[:3, :3]))),
              'ownershipVolumesMm3': {k: v * abs(float(np.linalg.det(target_affine[:3, :3]))) for k, v in ownership_counts.items()},
              'tissuePhysics': physics, **adapter}
    (output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def prepare_surgical_environment(collection_path, output_dir, *, layer_ids,
                                 resolution_mm=1., allow_unreviewed_registration=False, approach=None,
                                 native_wall_center=None, check=lambda: None):
    """One interoperable export path for desktop preparation, CLI, and RL.

    Missing roles are explicit. Source manifests are never rewritten. The saved
    scene archive pins the complete physics profile, mapping, and source identity.
    """
    from .imaging import load_case
    from .surgical_scene import from_case
    from .surgical_engine import SurgicalEngine
    from .surgical_environment import SCHEMA, standardize
    from .surgical_physics_v2 import default_profile
    report = prepare_surgical_patient(collection_path, output_dir, layer_ids=layer_ids,
        resolution_mm=resolution_mm, allow_unreviewed_registration=allow_unreviewed_registration,
        tissue_physics=default_profile(), require_complete=False, approach=approach,
        native_wall_center=native_wall_center, check=check)
    case = load_case(report['casePath'])
    scene = standardize(from_case(case))
    from .surgical_rl_protection import GuardConfig
    engine = (SurgicalEngine(scene) if approach is not None else
              SurgicalEngine(scene, guard_config=GuardConfig(require_critical_coverage=False)))
    record = {'environmentSchema': SCHEMA, 'physicsProfile': copy.deepcopy(scene.properties['tissuePhysicsV2']),
              'mapping': scene.regions, 'branches': {'main': engine.snapshot()}, 'branchId': 'main',
              'position': 0, 'parents': {'main': None}, 'timeline': {'main': []}}
    save_case(case, report['casePath'], artifacts={'surgical_toolbox': record})
    environment = {**scene.descriptor()['environment'], 'caseHash': case.semantic_hash,
                   'sceneId': scene.identity, 'regions': scene.regions,
                   'physicsProfile': record['physicsProfile'],
                   'provenanceReport': 'report.json'}
    destination = Path(output_dir) / 'environment.json'
    destination.write_text(json.dumps(environment, indent=2, allow_nan=False) + '\n')
    return {**report, 'environmentPath': str(destination.resolve()), 'environment': environment}
