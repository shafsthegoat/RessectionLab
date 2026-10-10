"""Session display attachments to an existing CaseData, never actor inputs.

Native grids remain independent until a separately reviewed registration exists.
An explicit user association is not a verified same-person or same-time claim.
"""
from __future__ import annotations
from dataclasses import replace
import math
import numpy as np
from .core import semantic_digest, thaw_json
from .imaging import inspect_nifti, load_nifti_case
from .structural_evidence import structural_frame_hash

MODALITIES = frozenset({'T1','T1CE','T2','FLAIR','CT','CTA','TOF-MRA','MRA','SWI','other-3D-scalar'})
ANNOTATION_KINDS = frozenset({'none','source-provided','estimated'})
MAX_SERIES = 4
MAX_IMAGE_BYTES = 128 * 1024**2
MAX_ATTACHMENT_BYTES = 384 * 1024**2
MAX_LABELS = 8


def retained_paths(entries):
    for series in entries.values():
        case = series['volume']
        yield case['mri']['path']
        for layer in case['compartments']:
            yield layer['array']['path']
            yield layer['sourceArray']['path']


def import_display_series(session, args, request, progress):
    # Import bridge primitives lazily: the bridge calls this module, not vice versa.
    from .desktop_bridge import BridgeError, _keys, _path
    _keys(args, {'caseHash','imagePath','annotationPath','modality','annotationKind'})
    parent = session._get_case(args.get('caseHash'))
    modality = args.get('modality');kind = args.get('annotationKind','none')
    if modality not in MODALITIES or kind not in ANNOTATION_KINDS:
        raise BridgeError('INVALID_ARGUMENT','Choose a supported 3D modality and annotation provenance.')
    if len(parent.display_series) >= MAX_SERIES:
        raise BridgeError('DISPLAY_SERIES_LIMIT','This workspace supports four additional native-grid display series.')
    image = _path(args.get('imagePath'),kind='NIfTI')
    annotation = None if args.get('annotationPath') is None else _path(args['annotationPath'],kind='NIfTI')
    if (annotation is None) != (kind=='none'):
        raise BridgeError('ANNOTATION_PROVENANCE_REQUIRED','A supplied annotation needs an explicit source-provided or estimated status.')
    qc = inspect_nifti(image);count=math.prod(qc['shape'])
    if count*4 > MAX_IMAGE_BYTES:
        raise BridgeError('ARRAY_SIZE_LIMIT','This display series exceeds the bounded 3D image limit.')
    labels=[]
    if annotation is not None:
        aqc=inspect_nifti(annotation)
        if math.prod(aqc['shape'])*4 > MAX_IMAGE_BYTES:
            raise BridgeError('ARRAY_SIZE_LIMIT','The annotation exceeds the display limit.')
        import nibabel as nib
        source=nib.load(annotation);unique=set()
        for z in range(0,source.shape[2],8):
            request.check();block=np.asarray(source.dataobj[:,:,z:z+8])
            if not np.isfinite(block).all() or (block<0).any() or (block!=np.floor(block)).any():
                raise BridgeError('INVALID_ANNOTATION','Display annotations require finite nonnegative integer labels.')
            unique.update(np.unique(block).tolist())
            if sum(v>0 for v in unique)>MAX_LABELS:
                raise BridgeError('LABEL_COUNT_LIMIT','At most eight annotation labels are supported in a display attachment.')
        labels=[int(v) for v in sorted(unique) if v>0]
    expanded=count*(4+2*len(labels))
    retained=sum(s['expandedBytes'] for s in parent.display_series.values())
    if expanded+retained > MAX_ATTACHMENT_BYTES:
        raise BridgeError('CASE_SIZE_LIMIT','Display attachments exceed the session cache budget.')
    progress(.1,'Checking a separate native-grid display series')
    display=load_nifti_case(image,annotation,case_id=parent.case.case_id,
        label_map={label:f'source_label_{label}' for label in labels} if annotation else None)
    if kind=='estimated':
        display=display.revised(source_refs=tuple(replace(source,provenance='estimated')
            if source.source_id=='supplied_target_annotation' else source for source in display.source_refs))
    request.check()
    source_refs=[s.to_dict() for s in display.source_refs]
    identity={'referenceCaseHash':parent.case.semantic_hash,'sourceCaseHash':display.semantic_hash,
              'modality':modality,'annotationKind':kind}
    series_id=semantic_digest(identity)
    if series_id in parent.display_series:return parent.display_series[series_id]
    volume={'caseId':display.case_id,'caseHash':display.semantic_hash,'frame':display.frame,
        'affine':display.affine.tolist(),'shape':list(display.mri.shape),'spacingMm':list(display.spacing_mm),
        'mri':session.transfers.array(display.mri,'float32'),
        'intensityRange':[float(display.mri.min()),float(display.mri.max())],
        'compartments':[{'name':name,'volumeMm3':float(mask.sum()*display.voxel_volume_mm3),
            'array':session.transfers.array(mask,'uint8'),'sourceArray':session.transfers.array(mask,'uint8')}
            for name,mask in display.compartments.items()],
        'brainMask':None,'structuralEvidence':[],'priorProposals':[],
        'unknowns':['display_only_attachment','same_patient_association_unverified','registration_to_reference_unreviewed',
                    'annotation_anatomy_and_coverage_unreviewed','acquisition_time_unknown'],
        'metadata':{**thaw_json(display.metadata),'workspace_display_only':True,'selected_modality':modality,
            'modality_origin':'user_declared','annotation_kind':kind,'annotation_use':'display_only',
            'model_identity':None,'model_lineage':'not_supplied' if kind=='estimated' else 'not_applicable'},
        'sourceRefs':source_refs}
    result={'schema':'workspace-display-series-v1','seriesId':series_id,'referenceCaseHash':parent.case.semantic_hash,
        'referenceFrameHash':structural_frame_hash(parent.case),
        'sourceFrameHash':structural_frame_hash(display),
        'association':{'kind':'explicit_user_selected','samePersonVerified':False,'sameTimeVerified':False},
        'registration':{'status':'unreviewed','reason':'No accepted source-to-reference registration has been supplied.',
            'sourceToReferenceRasMm':None,'registrationHash':None,'overlayPermitted':False},
        'scope':'display-only-native-grid','planningEligible':False,'evaluationEligible':False,
        'modality':modality,'modalityOrigin':'user_declared','annotationKind':kind,
        'annotationCoverage':'unreviewed; unlabelled voxels are unknown','acquisitionDatetime':None,
        'expandedBytes':expanded,'volume':volume}
    request.begin_commit();parent.display_series[series_id]=result
    session.prune_transfers()
    return result
