"""Repeatable source reconstruction for four fixed TRAIN subjects, no cache.

Only pinned public manifests/arrays are used by the existing factory. Building
this map reads metadata only. Each call writes into a fresh caller-owned visit
directory and returns one task/context; no previous source object is retained.
"""
import hashlib
import json
from pathlib import Path

from .core import freeze_json, semantic_digest, thaw_json
from .native_proposals import NominalCavityProposalConfig
from .patient_planning_admission import COHORT_SHA256
from .patient_planning_cohort_spec import TRAIN, validate_limits, validate_factories


def make_train_visit_factories(*, manifest_index_path, manifest_index_sha256,
        cohort_bytes, learning_protocol, limits, released_record, released_sha256, progress):
    validate_limits(learning_protocol, limits)
    raw = Path(manifest_index_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest_index_sha256:
        raise ValueError('Public manifest index changed')
    index = json.loads(raw); rows = index['cases']
    if (len(rows) != 4 or {r['patient_id'] for r in rows} != set(TRAIN)
            or any(r['role'] != 'TRAIN' for r in rows)):
        raise ValueError('Exactly fixed four TRAIN manifests required; held-out files closed')
    if (type(cohort_bytes) is not bytes or hashlib.sha256(cohort_bytes).hexdigest() != COHORT_SHA256
            or index['cohort_sha256'] != COHORT_SHA256):
        raise ValueError('Original frozen cohort identity required')
    if (type(released_sha256) is not str or len(released_sha256) != 64
            or any(c not in '0123456789abcdef' for c in released_sha256)):
        raise ValueError('Exact owned release SHA required')
    admission_limits = {k:v for k,v in limits.items() if k not in ('output_bytes', 'checkpoint_bytes')}
    if (released_record['limits'] != thaw_json(freeze_json(admission_limits))
            or released_record.get('learning_protocol_hash') != semantic_digest(learning_protocol)):
        raise ValueError('Owned release must bind the identical admission limits and learning protocol')
    release = freeze_json(released_record); protocol = freeze_json(learning_protocol)
    selected = {r['patient_id']: freeze_json(r) for r in rows}
    config = NominalCavityProposalConfig(**thaw_json(protocol['cohort_execution']['proposal_config']))
    def closure(subject):
        def construct(*, output):
            from .public_patient_factory import prepare_public_source
            from .patient_planning_admission import make_patient_planning_task
            row = selected[subject]; target = Path(output)
            # The runner reserves this directory once. A cached source is never
            # captured here; every visit re-verifies public input byte hashes.
            if not target.is_dir() or any(target.iterdir()):
                raise ValueError('Fresh empty owned source visit directory required')
            source, binding, qc, admission = prepare_public_source(target,
                thaw_json(release), released_sha256, None, progress,
                public_manifest_path=Path(row['path']), public_manifest_sha256=row['sha256'],
                cohort_bytes=cohort_bytes, learning_protocol_hash=semantic_digest(protocol),
                proposal_config=config, public_target_context_variant=protocol['public_target_context_variant'])
            return make_patient_planning_task(source, cohort_bytes=cohort_bytes,
                source_binding=binding, qc_receipt=qc, protocol=admission)
        return construct
    factories = {subject: closure(subject) for subject in TRAIN}
    validate_factories(factories)
    return factories
