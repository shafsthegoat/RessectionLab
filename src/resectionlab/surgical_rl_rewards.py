"""Deterministic engineering reward proxies, not validated clinical utilities.

Only newly removed, explicitly designated tumor earns credit. All injury
ledgers are monotone: releasing tissue, suction and coagulation cannot refund
injury. A tracker belongs to one episode and must be recreated on reset. Reads
are restricted to sparse changed source cells, bonds and action histories.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import copy
import math
import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Nonnegative weights and positive physical normalization scales.

    Defaults are illustrative engineering choices, not patient calibration.
    Strain injury is volume times squared neighbor displacement gradient.
    """
    tumor_removal_weight: float = 1.
    healthy_removal_weight: float = 2.
    bone_removal_weight: float = .05
    healthy_thermal_weight: float = .1
    healthy_cut_weight: float = 1.
    deformation_weight: float = 1.
    vessel_damage_weight: float = 4.
    bleeding_weight: float = 1.
    protected_damage_weight: float = 5.
    near_contact_weight: float = 1.
    action_cost: float = .001
    time_cost: float = .001
    invalid_action_cost: float = .1
    protected_blocked_weight: float = 1.
    volume_scale_mm3: float = 1000.
    area_scale_mm2: float = 100.
    thermal_exposure_scale: float = 100.
    near_contact_scale_mm3_seconds: float = 1000.
    time_scale_seconds: float = 1.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.type in (bool, 'bool'):
                if type(value) is not bool:
                    raise ValueError(field.name + ' must be a boolean')
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(field.name + ' must be finite')
            if value < 0 or ('scale' in field.name and value == 0):
                raise ValueError(field.name + ' must be nonnegative (scales positive)')


@dataclass(frozen=True)
class GeneratedGraphRewardConfig(RewardConfig):
    """Opt-in numerical control. Weights are engineering costs per modeled deficit second."""

    supply_deficit_weight: float = 1.
    drainage_deficit_weight: float = 1.

    def __post_init__(self):
        super().__post_init__()
        if min(self.supply_deficit_weight, self.drainage_deficit_weight) <= 0:
            raise ValueError('Both declared vascular mechanisms require positive costs')


@dataclass(frozen=True)
class InjuryRewardConfig(RewardConfig):
    """Explicit injury-mode costs (scenes declaring ``surgicalInjuryV1``).

    All weights are assumed research values, not calibrated harm. Contused and
    lacerated volume use ``volume_scale_mm3``; vessel injury is per source cell
    and avulsion per torn attachment. A partial action stopped by a remaining
    physical refusal costs ``invalid_action_cost`` like a rejection.
    """
    injury_contused_weight: float = 1.
    injury_lacerated_weight: float = 4.
    injury_vessel_cell_weight: float = .05
    injury_avulsion_weight: float = .5
    penalize_physical_stops: bool = True


@dataclass(frozen=True)
class StructureRewardConfig(RewardConfig):
    """Opt-in structure-aware costs (scenes declaring ``functionalMappingV1``).

    Replaces the flat healthy-removal weight: declared tract cells are costed by
    function family, vessel cells by kind and the remaining healthy tissue by
    ``healthy_removal_weight``. Retained tract volume within 0-2 mm and 2-5 mm of
    the resection cavity costs a fraction of its family weight (a high-water
    ledger, so it is never refunded). All defaults are assumed-research values;
    tracts are estimates from a published baseline tract, not patient function,
    and no weight is a clinical deficit probability. Volumes use
    ``volume_scale_mm3``.
    """
    tract_motor_weight: float = 10.
    tract_language_weight: float = 8.
    tract_vision_weight: float = 6.
    tract_other_weight: float = 3.
    tract_unclassified_weight: float = 3.
    band_0_2mm_fraction: float = .25
    band_2_5mm_fraction: float = .05
    vessel_artery_weight: float = 8.
    vessel_vein_weight: float = 6.
    vessel_unclassified_weight: float = 6.


INJURY_METRICS = ('injury_contused_mm3', 'injury_lacerated_mm3', 'injury_vessel_cells', 'injury_avulsions')
# Progress stops are expected dwell at a hard contact, not a physical refusal.
_PROGRESS_STOPS = ('TISSUE_ASPIRATION_IN_PROGRESS', 'BONE_CUTTING_IN_PROGRESS')


def reward_config_for(engine):
    """Injury scenes get the explicit injury config; every other scene keeps RewardConfig."""
    return InjuryRewardConfig() if 'injuryV1' in engine.state else RewardConfig()


class RewardTracker:
    """Source-identity based ledger with per-source deformation high-water marks.

    protected_source_ids must include original annotation overlaps, not just
    categorical owners. protection can supply cumulative protected_damage_mm3
    and near_contact_exposure_mm3_seconds from a separate anatomical evaluator.
    deformation_cell_peaks accepts cumulative microstep maxima by source ID.
    """
    def __init__(self, engine, protected_source_ids=(), *, procedure_evaluation=False, graph_evaluation=False):
        if 'brainContinuumV1' in engine.scene.properties:
            raise ValueError('CONTINUUM_RL_HOLD: material/contact, artificial boundaries and cumulative exposure reward remain unqualified')
        if 'bipolarThermalV1' in engine.scene.properties:
            raise ValueError('BIPOLAR_THERMAL_RL_HOLD: device response and heat-aware reward remain unqualified')
        graph = engine.scene.properties.get('vascularGraphV1')
        self.graph_evaluation = graph_evaluation
        if graph_evaluation:
            # Any source class with a declared graph and supply/drainage regions;
            # the graph's own coverage status (generated-control or
            # partial-acquired) stays its provenance label.
            if not graph or not graph['regions']:
                raise ValueError('VASCULAR_GRAPH_RL_HOLD: a declared vascular graph with supply/drainage interfaces is required')
        elif graph is not None:
            raise ValueError('VASCULAR_GRAPH_RL_HOLD: this consumer requires the generated graph reward contract')
        if 'surgicalProcedurePolicyV1' in engine.scene.properties and procedure_evaluation is not True:
            raise ValueError('PROCEDURE_POLICY_INCOMPATIBLE: historical reward consumers cannot evaluate the supported procedure; use the versioned evaluation-only SurgicalRLEnv contract')
        self.scene = engine.scene
        self.protected = frozenset(map(int, protected_source_ids))
        if any(c < 0 or c >= self.scene.labels.size for c in self.protected):
            raise ValueError('Protected source cell outside scene')
        self._strain = {}
        self._thermal = {}
        self._ledger = {}
        self._history_position = 0
        self._irrigation = 0.

    def clone(self):
        """Detach a preview ledger while retaining the immutable source owner."""
        result = copy.copy(self)
        for name in ('_strain', '_thermal', '_ledger'):
            setattr(result, name, getattr(self, name).copy())
        return result

    def _target(self, engine, cell):
        region = engine.by_label.get(int(self.scene.labels.flat[cell]), {})
        return (region.get('target') is True and region.get('material') == 'tumor'
                and not region.get('protected', False) and not region.get('critical', False)
                and cell not in self.protected)

    def _occupied(self, cell):
        return bool(self.scene.labels.flat[cell])

    def _area(self, a, b):
        delta = np.abs(np.subtract(np.unravel_index(a, self.scene.shape),
                                   np.unravel_index(b, self.scene.shape)))
        if delta.sum() != 1:
            raise ValueError('Cut reward requires neighboring source-cell faces')
        axis = int(np.argmax(delta))
        others = [i for i in range(3) if i != axis]
        basis = self.scene.affine[:3, :3]
        return float(np.linalg.norm(np.cross(basis[:, others[0]], basis[:, others[1]])))

    def measure(self, engine, protection=None):
        if engine.scene is not self.scene:
            raise ValueError('RewardTracker cannot be reused across scenes or episodes')
        protection = protection or {}
        state = engine.state
        volume = self.scene.voxel_mm3
        removed = set(map(int, state['removed']))
        injury = state.get('injuryV1')
        if injury is not None:
            # Tissue torn by an unsafe move is injury, not resection: it earns no
            # tumour credit and is costed by the laceration term instead.
            removed -= set(map(int, injury['lacerated']))
        tumor = sum(self._target(engine, c) for c in removed) * volume
        bone = sum(engine.by_label.get(int(self.scene.labels.flat[c]), {}).get('material') == 'bone'
                   for c in removed) * volume
        healthy = sum(self._occupied(c) and not self._target(engine, c)
                      and engine.by_label[int(self.scene.labels.flat[c])]['material'] != 'bone'
                      for c in removed) * volume
        cuts = {tuple(sorted(map(int, edge))) for name in ('cutEdges', 'transectedEdges')
                for edge in state.get(name, [])}
        cut_area = sum(self._area(a, b) for a, b in cuts
                       if any(self._occupied(c) and not self._target(engine, c) for c in (a, b)))
        scores = deformation_scores(engine)
        for key, value in protection.get('deformation_cell_peaks', {}).items():
            cell = int(key); score = float(value)
            if not 0 <= cell < self.scene.labels.size or not math.isfinite(score) or score < 0:
                raise ValueError('Invalid deformation source-cell peak')
            scores[cell] = max(scores.get(cell, 0.), score)
        for cell, score in scores.items():
            if self._occupied(cell) and not self._target(engine, cell):
                self._strain[cell] = max(self._strain.get(cell, 0.), score)
        # The engine integrates declared tool power * seconds per source cell.
        # Weight by affine voxel volume for a physical affected-volume proxy;
        # this is neither temperature nor a calibrated tissue injury dose.
        for key, value in state.get('thermal', {}).items():
            cell = int(key); dose = float(value)
            if not 0 <= cell < self.scene.labels.size or not math.isfinite(dose) or dose < 0:
                raise ValueError('Invalid thermal source-cell exposure')
            if (self._occupied(cell) and not self._target(engine, cell)
                    and engine.by_label[int(self.scene.labels.flat[cell])]['material'] != 'bone'):
                self._thermal[cell] = max(self._thermal.get(cell, 0.), dose * volume)
        faces = state.get('vesselsV2', {}).get('faces', {})
        vessel = sum(self._area(*face['cells']) * float(face['damage']) for face in faces.values())
        if 'vesselsV2' not in state:
            vessel = sum(self._area(a, b) for a, b in cuts if any(
                engine.by_label.get(int(self.scene.labels.flat[c]), {}).get('material') == 'vessel' for c in (a, b)))
        # fluidAdded is a cumulative gross inflow ledger, independent of suction
        # and escape. Subtract only committed irrigation, never standing fluid.
        for item in engine.history[self._history_position:]:
            self._irrigation += sum(float(e['volumeMm3']) for e in item.get('events', []) if e.get('kind') == 'irrigation')
        self._history_position = len(engine.history)
        current = dict(tumor_removed_mm3=tumor, healthy_removed_mm3=healthy, bone_removed_mm3=bone,
                       healthy_cut_mm2=cut_area, deformation_injury_mm3=sum(self._strain.values()),
                       healthy_thermal_exposure=sum(self._thermal.values()),
                       vessel_damage_mm2=vessel, bleeding_generated_mm3=max(0., state['fluidAddedMm3'] - self._irrigation),
                       protected_damage_mm3=float(protection.get('protected_damage_mm3', 0.)),
                       near_contact_exposure_mm3_seconds=float(protection.get('near_contact_exposure_mm3_seconds', 0.)),
                       elapsed_seconds=float(state['elapsedSeconds']))
        if injury is not None:
            current.update(injury_contused_mm3=len(injury['contused']) * volume,
                           injury_lacerated_mm3=len(injury['lacerated']) * volume,
                           injury_vessel_cells=float(len(injury['vesselInjured'])),
                           injury_avulsions=float(len(injury['avulsions'])))
        if 'functionalMappingV1' in self.scene.properties:
            current.update(self._structure(engine, removed, volume))
        if self.graph_evaluation:
            from .surgical_vascular_graph import maximum_interface_deficits
            if state.get('toolMechanicsRevision') != 2 or 'vascularGraphV1' not in state:
                raise ValueError('VASCULAR_GRAPH_RL_HOLD: recorded revision-2 graph setup required')
            graph = state['vascularGraphV1']
            current.update(maximum_interface_deficits(graph['regions']))
            current['bleeding_generated_mm3'] = graph['generatedMm3']
        for key, value in current.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid reward measurement: ' + key)
            self._ledger[key] = max(self._ledger.get(key, 0.), value)
        return dict(self._ledger)


    def _structure(self, engine, removed, volume):
        """Split healthy removal by declared tract family and vessel kind, plus band volumes."""
        from .surgical_functional_mapping import STRUCTURE_METRICS, band_volumes, label_classes
        classes = label_classes(self.scene)
        result = dict.fromkeys(STRUCTURE_METRICS, 0.)
        for cell in removed:
            if not self._occupied(cell) or self._target(engine, cell):
                continue
            label = int(self.scene.labels.flat[cell])
            if engine.by_label[label]['material'] == 'bone':
                continue
            kind = classes.get(label)
            key = ('structure_other_healthy_removed_mm3' if kind is None else
                   f'structure_tract_{kind[1]}_removed_mm3' if kind[0] == 'tract' else
                   f'structure_vessel_{kind[1]}_removed_mm3')
            result[key] += volume
        for family, (near, far) in band_volumes(engine).items():
            result[f'structure_tract_{family}_band_0_2mm_mm3'] = near
            result[f'structure_tract_{family}_band_2_5mm_mm3'] = far
        return result


def _neighbors(scene, cell):
    index = list(np.unravel_index(cell, scene.shape))
    for axis in range(3):
        for step in (-1, 1):
            other = index.copy(); other[axis] += step
            if 0 <= other[axis] < scene.shape[axis]:
                yield int(np.ravel_multi_index(tuple(other), scene.shape)), float(scene.spacing[axis])


def deformation_scores(engine):
    """Current sparse source-cell strain squared times physical voxel volume.

    Includes all occupied material, including tumor. RewardTracker applies its
    target/protection designation when consuming scores. Call after committed
    microsteps and retain per-cell maxima to capture out-and-back path injury.
    This function does not mutate the engine or maintain episode history.
    """
    scene, state = engine.scene, engine.state
    removed = set(map(int, state['removed']))
    deformation = {}
    for hold in state.get('holds', {}).values():
        if 'target' in hold:
            continue
        offset = np.asarray(hold.get('offset', [0., 0., 0.]), float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError('Invalid lattice hold offset')
        displacement = scene.affine[:3, :3] @ offset
        if np.any(displacement):
            for cell in hold.get('cells', []):
                if int(cell) not in removed:
                    deformation[int(cell)] = displacement
    # Continuous v2 deformation owns overlapping cells in engine occupancy.
    deformation.update({int(c): np.asarray(u, float) for c, u in state.get('deformation', {}).items()
                        if int(c) not in removed})
    if not deformation:
        return {}
    for displacement in deformation.values():
        if displacement.shape != (3,) or not np.isfinite(displacement).all():
            raise ValueError('Invalid deformation displacement')
    removed = set(map(int, state['removed']))
    cuts = {tuple(sorted(map(int, edge))) for name in ('cutEdges', 'transectedEdges')
            for edge in state.get(name, [])}
    candidates = set(deformation)
    for cell in deformation:
        candidates.update(n for n, _ in _neighbors(scene, cell))
    scores = {}
    zero = np.zeros(3)
    for cell in candidates - removed:
        if not scene.labels.flat[cell]:
            continue
        u = deformation.get(cell, zero)
        strain = max((float(np.linalg.norm(u - deformation.get(n, zero)) / length)
                      for n, length in _neighbors(scene, cell)
                      if n not in removed and scene.labels.flat[n] and tuple(sorted((cell, n))) not in cuts), default=0.)
        score = strain * strain * scene.voxel_mm3
        if not math.isfinite(score):
            raise ValueError('Deformation score overflow')
        if score:
            scores[cell] = score
    return scores


def compute_reward(before, after, outcome, config=None):
    """Return scalar reward and weighted terms for exactly one attempted action.

    Rejected actions still incur action and invalid-action costs. Improvements
    in an injury metric never create positive rewards. Input metrics must be
    cumulative tracker snapshots; costs use actual committed elapsed time.
    """
    config = config or RewardConfig()
    def delta(key):
        a, b = float(before[key]), float(after[key])
        if not math.isfinite(a) or not math.isfinite(b) or min(a, b) < 0:
            raise ValueError('Invalid reward metric: ' + key)
        return max(0., b - a)
    specifications = (
        ('tumor_removal', 'tumor_removed_mm3', config.tumor_removal_weight, config.volume_scale_mm3),
        ('healthy_removal', 'healthy_removed_mm3', -config.healthy_removal_weight, config.volume_scale_mm3),
        ('bone_removal', 'bone_removed_mm3', -config.bone_removal_weight, config.volume_scale_mm3),
        ('healthy_thermal', 'healthy_thermal_exposure', -config.healthy_thermal_weight, config.thermal_exposure_scale),
        ('healthy_cut', 'healthy_cut_mm2', -config.healthy_cut_weight, config.area_scale_mm2),
        ('deformation', 'deformation_injury_mm3', -config.deformation_weight, config.volume_scale_mm3),
        ('vessel_damage', 'vessel_damage_mm2', -config.vessel_damage_weight, config.area_scale_mm2),
        ('bleeding', 'bleeding_generated_mm3', -config.bleeding_weight, config.volume_scale_mm3),
        ('protected_damage', 'protected_damage_mm3', -config.protected_damage_weight, config.volume_scale_mm3),
        ('near_contact', 'near_contact_exposure_mm3_seconds', -config.near_contact_weight, config.near_contact_scale_mm3_seconds),
        ('time', 'elapsed_seconds', -config.time_cost, config.time_scale_seconds))
    if isinstance(config, StructureRewardConfig):
        from .surgical_functional_mapping import FAMILIES, STRUCTURE_METRICS, VESSEL_KINDS
        if not all(key in before and key in after for key in STRUCTURE_METRICS):
            raise ValueError('STRUCTURE_REWARD_REQUIRES_FUNCTIONAL_MAPPING: declare functionalMappingV1 on the scene')
        # Remaining healthy tissue keeps the flat weight; declared structures replace it.
        specifications = tuple(
            ('healthy_removal', 'structure_other_healthy_removed_mm3', -config.healthy_removal_weight, config.volume_scale_mm3)
            if name == 'healthy_removal' else (name, key, weight, scale) for name, key, weight, scale in specifications)
        for family in FAMILIES + ('unclassified',):
            weight = getattr(config, f'tract_{family}_weight')
            specifications += (
                (f'tract_removal:{family}', f'structure_tract_{family}_removed_mm3', -weight, config.volume_scale_mm3),
                (f'tract_band_0_2mm:{family}', f'structure_tract_{family}_band_0_2mm_mm3',
                 -weight * config.band_0_2mm_fraction, config.volume_scale_mm3),
                (f'tract_band_2_5mm:{family}', f'structure_tract_{family}_band_2_5mm_mm3',
                 -weight * config.band_2_5mm_fraction, config.volume_scale_mm3))
        specifications += tuple((f'vessel_removal:{kind}', f'structure_vessel_{kind}_removed_mm3',
                                 -getattr(config, f'vessel_{kind}_weight'), config.volume_scale_mm3)
                                for kind in VESSEL_KINDS)
    from .surgical_vascular_graph import DEFICIT_METRICS
    if isinstance(config, GeneratedGraphRewardConfig):
        if not all(key in before and key in after for key in DEFICIT_METRICS):
            raise ValueError('Generated vascular graph measurements required')
        specifications += (
            ('supply_deficit', 'supplyDeficitSeconds', -config.supply_deficit_weight, config.time_scale_seconds),
            ('drainage_deficit', 'drainageDeficitSeconds', -config.drainage_deficit_weight, config.time_scale_seconds))
    elif any(key in before or key in after for key in DEFICIT_METRICS):
        raise ValueError('VASCULAR_GRAPH_RL_HOLD: graph reward weights required')
    # Injury terms exist only for injury-mode ledgers. A plain RewardConfig on
    # such a scene uses the documented InjuryRewardConfig defaults.
    injured = any(key in before or key in after for key in INJURY_METRICS)
    if injured:
        if not all(key in before and key in after for key in INJURY_METRICS):
            raise ValueError('Injury-mode measurements required before and after')
        weights = config if isinstance(config, InjuryRewardConfig) else InjuryRewardConfig()
        specifications += (
            ('injury_contused', 'injury_contused_mm3', -weights.injury_contused_weight, config.volume_scale_mm3),
            ('injury_lacerated', 'injury_lacerated_mm3', -weights.injury_lacerated_weight, config.volume_scale_mm3),
            ('injury_vessel', 'injury_vessel_cells', -weights.injury_vessel_cell_weight, 1.),
            ('injury_avulsion', 'injury_avulsions', -weights.injury_avulsion_weight, 1.))
    penalize_stops = (config.penalize_physical_stops if isinstance(config, InjuryRewardConfig) else injured)
    terms = {name: weight * delta(key) / scale for name, key, weight, scale in specifications}
    # Feasibility is authoritative. A rejected proposal never buys treatment
    # credit, even if a caller supplies a speculative endpoint as ``after``.
    if outcome.get('status') == 'rejected':
        terms['tumor_removal'] = 0.
    terms['action'] = -config.action_cost
    terms['invalid_action'] = -config.invalid_action_cost if outcome.get('status') == 'rejected' else 0.
    blocked = str(outcome.get('reasonCode') or outcome.get('reason') or '').startswith('PROTECTED_STRUCTURE')
    terms['protected_blocked'] = -config.protected_blocked_weight if blocked else 0.
    if penalize_stops:
        # A partial cut short by a physical refusal is no longer free.
        reason = str(outcome.get('reasonCode') or outcome.get('reason') or '').split(':')[0]
        stopped = outcome.get('status') == 'partial' and bool(reason) and reason not in _PROGRESS_STOPS
        terms['physical_stop'] = -config.invalid_action_cost if stopped else 0.
    reward = math.fsum(terms.values())
    if not math.isfinite(reward):
        raise ValueError('Reward overflow; choose representable weights and scales')
    return reward, terms
