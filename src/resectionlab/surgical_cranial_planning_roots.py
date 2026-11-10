"""Bounded persistence of immutable opening variants over one exact source.

This registry resolves native scenes; it grants no stage, search, coverage or
patient permission. Ordinary SurgicalEngine.restore still validates protection,
stage declarations, every accepted action and the saved endpoint.
"""
from __future__ import annotations

import copy
import json
import re
from types import MappingProxyType

import numpy as np

from .core import array_digest, semantic_digest, thaw_json
from .surgical_scene import (
    SurgicalScene, original_target_memberships, propagate_original_target_memberships)

SCHEMA = 'resectionlab.cranial-planning-roots.v1'
FIELD = 'cranialPlanningRoots'
MAX_ROOTS = 3
_OPENINGS = ('cranialAccessV1', 'cranialMillingV1')
_ROOT_FIELDS = {'id', 'label', 'openingProperty', 'openingContract', 'sourceSnapshot'}
_PLANNING_FIELDS = {'limits', 'openingSkill', 'preparationActions', 'workRoutes', 'outsidePoses', 'qualification'}


def _planning(value):
    _require(type(value) is dict and set(value) == _PLANNING_FIELDS,
             'EXPLICIT_PLANNING_FIELDS_REQUIRED')
    try:
        size = len(json.dumps(value, allow_nan=False, separators=(',', ':')).encode())
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError('CRANIAL_PLANNING_ROOTS_PLANNING_JSON_REQUIRED') from error
    _require(size <= 256*1024, 'PLANNING_METADATA_TOO_LARGE')
    _require(type(value['limits']) is dict and type(value['openingSkill']) is dict and
        type(value['qualification']) is dict and type(value['outsidePoses']) is dict and
        set(value['outsidePoses']) <= {'left', 'right'} and
        type(value['preparationActions']) is list and len(value['preparationActions']) <= 512 and
        all(type(a) is dict for a in value['preparationActions']) and
        type(value['workRoutes']) is list and len(value['workRoutes']) <= 48 and
        all(type(r) is dict for r in value['workRoutes']), 'BOUNDED_PLANNING_METADATA_REQUIRED')


def _require(condition, reason):
    if not condition:
        raise ValueError('CRANIAL_PLANNING_ROOTS_'+reason)


def _seal(document, field):
    return {**document, field: semantic_digest(document)}


def _check_seal(document, field):
    _require(type(document) is dict and document.get(field) == semantic_digest(
        {k: v for k, v in document.items() if k != field}), 'CONTENT_CHANGED')


def source_binding(scene):
    """Opening-independent source identity, including full annotation membership.

    Only the two opening properties may vary. The source identity itself,
    attachments, all other anatomy/model/domain properties and tools stay bound.
    Missing original target evidence cannot silently become an owned-label mask.
    """
    from .surgical_tools import catalog
    scene.require_cache_source()
    original = original_target_memberships(scene)
    _require(original is not None and bool(original), 'ORIGINAL_TARGET_EVIDENCE_REQUIRED')
    union = np.unique(np.concatenate(list(original.values())))
    _require(bool(len(union)), 'NONEMPTY_ORIGINAL_TARGET_REQUIRED')
    return {'baseSceneId': scene.identity, 'labelsHash': array_digest(scene.labels),
        'affineHash': array_digest(scene.affine), 'regionsHash': semantic_digest(scene.regions),
        'sourceHash': semantic_digest(scene.source),
        'propertiesHash': semantic_digest({k: v for k, v in scene.properties.items() if k not in _OPENINGS}),
        'originalTargetHash': array_digest(union),
        'originalMembershipHashes': {k: array_digest(v) for k, v in original.items()},
        'toolCatalogHash': semantic_digest(catalog())}


def same_source_anatomy(first_scene, second_scene):
    """Compare opening variants without confusing their distinct scene IDs."""
    first, second = source_binding(first_scene), source_binding(second_scene)
    return {k: v for k, v in first.items() if k != 'baseSceneId'} == {
        k: v for k, v in second.items() if k != 'baseSceneId'}


def candidate_scene(base_scene, opening_property, opening_contract):
    """Construct a new scene; never edit the base, arrays or accepted history."""
    _require(opening_property in _OPENINGS and type(opening_contract) is dict,
             'EXPLICIT_OPENING_DECLARATION_REQUIRED')
    base_scene.require_cache_source()
    properties = copy.deepcopy({k: v for k, v in base_scene.properties.items() if k not in _OPENINGS})
    properties[opening_property] = copy.deepcopy(opening_contract)
    # SurgicalScene validates the full source/target/attachment opening binding.
    return propagate_original_target_memberships(base_scene, SurgicalScene(
        base_scene.labels, base_scene.affine, base_scene.regions, base_scene.source, properties))


def _guard_content(snapshot):
    raw = snapshot.get('protectionContract')
    if raw is None:
        return None
    _check_seal(raw, 'contentHash')
    return {k: v for k, v in raw.items() if k not in ('sceneId', 'contentHash')}


def _root(base_scene, value):
    _require(type(value) is dict and _ROOT_FIELDS <= set(value) <= _ROOT_FIELDS | {'planning'},
             'EXACT_ROOT_FIELDS_REQUIRED')
    if 'planning' in value:
        _planning(value['planning'])
    _require(type(value['id']) is str and re.fullmatch(r'[a-z][a-z0-9_-]{0,47}', value['id']) is not None,
             'INVALID_ROOT_ID')
    _require(type(value['label']) is str and 0 < len(value['label']) <= 160, 'INVALID_ROOT_LABEL')
    scene = candidate_scene(base_scene, value['openingProperty'], value['openingContract'])
    snapshot = value['sourceSnapshot']
    _require(type(snapshot) is dict and snapshot.get('sceneId') == scene.identity and
             snapshot.get('actions') == [] and type(snapshot.get('checkpoint')) is dict and
             snapshot['checkpoint'].get('position') == 0 and
             snapshot['checkpoint'].get('stateId') == snapshot.get('stateId'), 'CANONICAL_ROOT_SNAPSHOT_REQUIRED')
    stage = snapshot.get('cranialStagesContract')
    _require(type(stage) is dict and stage.get('schema') == 'resectionlab.cranial-stages.v1',
             'SOURCE_STAGE_CONTRACT_REQUIRED')
    _check_seal(stage, 'contentHash')
    target = original_target_memberships(scene)
    _require(target is not None, 'ORIGINAL_TARGET_EVIDENCE_REQUIRED')
    original = sorted(set(int(c) for values in target.values() for c in values))
    _require(stage.get('sourceSceneId') == scene.identity and
             stage.get('boneContractHash') == value['openingContract'].get('contentHash') and
             stage.get('originalTargetCells') == original, 'STAGE_SOURCE_OR_FULL_TARGET_CHANGED')
    guard = snapshot.get('protectionContract')
    _require(stage.get('protectionBinding') == (guard.get('contentHash') if guard is not None else None),
             'STAGE_PROTECTION_CHANGED')
    if guard is not None:
        _require(guard.get('sceneId') == scene.identity, 'PROTECTION_SCENE_CHANGED')
    _guard_content(snapshot)
    # A seal plus an empty action list cannot authenticate an invented physical
    # checkpoint. Zero-action native restore validates the stage, guard and
    # exact intact state without executing any transition or granting admission.
    from .surgical_engine import SurgicalEngine
    restored = SurgicalEngine.restore(scene, snapshot, use_cache=False)
    _require(restored.snapshot() == snapshot, 'CANONICAL_ROOT_SNAPSHOT_REQUIRED')
    return scene


def create_registry(base_scene, roots, branch_roots=None):
    """Seal up to three already-declared roots; this performs no native action."""
    _require(type(roots) is list and 1 <= len(roots) <= MAX_ROOTS, 'ROOT_COUNT_EXCEEDED')
    binding = source_binding(base_scene)
    saved, guard = [], None
    for index, value in enumerate(roots):
        scene = _root(base_scene, value)
        content = _guard_content(value['sourceSnapshot'])
        if index == 0:
            guard = content
        _require(content == guard, 'COMMON_PROTECTION_REQUIRED')
        saved.append(_seal({**copy.deepcopy(value), 'sceneId': scene.identity}, 'rootSeal'))
    _require(len({r['id'] for r in saved}) == len(saved), 'DUPLICATE_ROOT_ID')
    _require(len({r['sceneId'] for r in saved}) == len(saved), 'DUPLICATE_OPENING_ROOT')
    document = {'schema': SCHEMA, 'sourceBinding': binding,
        'commonProtectionHash': semantic_digest(guard), 'roots': saved,
        'branchRoots': copy.deepcopy(branch_roots if branch_roots is not None else {}),
        'admittedPatientTask': False}
    result = _seal(document, 'registrySeal')
    validate_registry(base_scene, result)
    return result


def validate_registry(base_scene, registry):
    """Rebuild contract variants against the current exact base; no seal grants permission."""
    if type(registry) is MappingProxyType:
        registry = thaw_json(registry)
    _check_seal(registry, 'registrySeal')
    _require(set(registry) == {'schema', 'sourceBinding', 'commonProtectionHash', 'roots',
        'branchRoots', 'admittedPatientTask', 'registrySeal'} and registry['schema'] == SCHEMA and
        registry['admittedPatientTask'] is False, 'REGISTRY_SCHEMA_CHANGED')
    _require(registry['sourceBinding'] == source_binding(base_scene), 'SOURCE_IDENTITY_CHANGED')
    roots = registry['roots']
    _require(type(roots) is list and 1 <= len(roots) <= MAX_ROOTS, 'ROOT_COUNT_EXCEEDED')
    resolved = {}
    for row in roots:
        _check_seal(row, 'rootSeal')
        _require(_ROOT_FIELDS | {'sceneId', 'rootSeal'} <= set(row) <=
            _ROOT_FIELDS | {'planning', 'sceneId', 'rootSeal'}, 'EXACT_ROOT_FIELDS_REQUIRED')
        scene = _root(base_scene, {k: row[k] for k in _ROOT_FIELDS | {'planning'} if k in row})
        _require(row['sceneId'] == scene.identity, 'ROOT_SCENE_CHANGED')
        _require(row['id'] not in resolved and all(s.identity != scene.identity for s in resolved.values()),
                 'DUPLICATE_OPENING_ROOT')
        _require(semantic_digest(_guard_content(row['sourceSnapshot'])) == registry['commonProtectionHash'],
                 'COMMON_PROTECTION_REQUIRED')
        resolved[row['id']] = scene
    mapping = registry['branchRoots']
    _require(type(mapping) is dict and len(mapping) <= 64 and all(
        type(branch) is str and 0 < len(branch) <= 160 and type(root) is str and root in resolved
        for branch, root in mapping.items()), 'INVALID_BRANCH_ROOT_BINDING')
    return resolved


def scene_for_branch(base_scene, record, branch_id):
    """One resolver for editable/cold/prefix/frame owners; legacy stays single-scene."""
    branches = record.get('branches')
    _require(type(branches) in (dict, MappingProxyType) and branch_id in branches, 'UNKNOWN_BRANCH')
    registry = record.get(FIELD)
    if registry is None:
        _require(all(s.get('sceneId') == base_scene.identity for s in branches.values()),
                 'UNREGISTERED_SCENE')
        return base_scene
    scenes = validate_registry(base_scene, registry)
    mapping = registry['branchRoots']
    _require(set(mapping) <= set(branches), 'UNKNOWN_MAPPED_BRANCH')
    roots = {r['id']: r for r in registry['roots']}
    for name, snapshot in branches.items():
        root_id = mapping.get(name)
        expected = scenes[root_id] if root_id is not None else base_scene
        _require(type(snapshot) in (dict, MappingProxyType) and snapshot.get('sceneId') == expected.identity,
                 'BRANCH_SCENE_CHANGED')
        if root_id is not None:
            original = roots[root_id]['sourceSnapshot']
            _require(semantic_digest(thaw_json(snapshot.get('cranialStagesContract'))) ==
                semantic_digest(thaw_json(original.get('cranialStagesContract'))) and
                semantic_digest(thaw_json(snapshot.get('protectionContract'))) ==
                semantic_digest(thaw_json(original.get('protectionContract'))),
                'BRANCH_STAGE_OR_PROTECTION_CHANGED')
    root_id = mapping.get(branch_id)
    return scenes[root_id] if root_id is not None else base_scene


def bind_branch_root(base_scene, record, branch_id, root_id):
    """Bind one newly appended native branch, publishing only after validation.

    Add and bind generated/edited branches one at a time. Existing branches may
    not switch roots. The snapshot, stage and guard must already be exact native
    output; this helper does not manufacture a checkpoint or alter its actions.
    """
    _require(type(record) is dict and FIELD in record, 'MUTABLE_REGISTERED_RECORD_REQUIRED')
    registry = thaw_json(record[FIELD])
    scenes = validate_registry(base_scene, registry)
    _require(type(root_id) is str and root_id in scenes, 'UNKNOWN_ROOT')
    prior = registry['branchRoots'].get(branch_id)
    _require(prior is None or prior == root_id, 'ACCEPTED_BRANCH_ROOT_CHANGED')
    registry['branchRoots'][branch_id] = root_id
    registry = _seal({k: v for k, v in registry.items() if k != 'registrySeal'}, 'registrySeal')
    staged = {**record, FIELD: registry}
    scene_for_branch(base_scene, staged, branch_id)
    record[FIELD] = registry
    return registry
