"""Deterministic engineering reward proxies, not validated clinical utilities.

Only newly removed, explicitly designated tumor earns credit. All injury
ledgers are monotone: releasing tissue, suction and coagulation cannot refund
injury. A tracker belongs to one episode and must be recreated on reset. Reads
are restricted to sparse changed source cells, bonds and action histories.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
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
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(field.name + ' must be finite')
            if value < 0 or ('scale' in field.name and value == 0):
                raise ValueError(field.name + ' must be nonnegative (scales positive)')


class RewardTracker:
    """Source-identity based ledger with per-source deformation high-water marks.

    protected_source_ids must include original annotation overlaps, not just
    categorical owners. protection can supply cumulative protected_damage_mm3
    and near_contact_exposure_mm3_seconds from a separate anatomical evaluator.
    deformation_cell_peaks accepts cumulative microstep maxima by source ID.
    """
    def __init__(self, engine, protected_source_ids=()):
        if 'bipolarThermalV1' in engine.scene.properties:
            raise ValueError('BIPOLAR_THERMAL_RL_HOLD: device response and heat-aware reward remain unqualified')
        if 'vascularGraphV1' in engine.scene.properties:
            raise ValueError('VASCULAR_GRAPH_RL_HOLD: existing reward omits declared supply/drainage consequences')
        self.scene = engine.scene
        self.protected = frozenset(map(int, protected_source_ids))
        if any(c < 0 or c >= self.scene.labels.size for c in self.protected):
            raise ValueError('Protected source cell outside scene')
        self._strain = {}
        self._thermal = {}
        self._ledger = {}
        self._history_position = 0
        self._irrigation = 0.

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
        for key, value in current.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid reward measurement: ' + key)
            self._ledger[key] = max(self._ledger.get(key, 0.), value)
        return dict(self._ledger)


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
    terms = {name: weight * delta(key) / scale for name, key, weight, scale in specifications}
    terms['action'] = -config.action_cost
    terms['invalid_action'] = -config.invalid_action_cost if outcome.get('status') == 'rejected' else 0.
    blocked = str(outcome.get('reasonCode') or outcome.get('reason') or '').startswith('PROTECTED_STRUCTURE')
    terms['protected_blocked'] = -config.protected_blocked_weight if blocked else 0.
    reward = math.fsum(terms.values())
    if not math.isfinite(reward):
        raise ValueError('Reward overflow; choose representable weights and scales')
    return reward, terms
