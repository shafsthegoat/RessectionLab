"""Generated-only sparse verified proposal pilot; no patient/model admission.

One ridge ordering, one deterministic public-sample ordering, exhaustive control.
Acquired use requires the separate fixed-TRAIN capability in patient_sparse.py.
All motions still use the unchanged native certificate and commit machinery.
Scores are proposal priorities, not predicted safety or long-horizon values.
"""
from dataclasses import asdict, dataclass, field
from contextlib import contextmanager
import copy
import time

import numpy as np

from resectionlab.core import array_digest, immutable_array, semantic_digest
from resectionlab.native_spatial_task import NativeSpatialCase, NativeSpatialTask
from resectionlab.observed_search import ObservedSearchLimit

VERSION = "generated-public-local-ridge-shortlist-v1"
K = 8
FEATURES = 681  # 81 points * (4 values + 4 coverage) + geometry16/state10/spacing3/available4
RIDGE = 1e-3


@dataclass
class Costs:
    counts: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    inventory_callback: object = None

    def count(self, key, n=1):
        self.counts[key] = self.counts.get(key, 0) + n

    @contextmanager
    def measure(self, key):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[key] = self.seconds.get(key, 0.) + time.perf_counter() - start

    def record(self):
        return {"counts": dict(self.counts), "seconds": dict(self.seconds),
                "nesting": "constructor and inventory are inclusive; proposal/features/ranking/verification are disjoint within inventory"}


@dataclass(frozen=True)
class PublicRows:
    ids: tuple
    features: np.ndarray
    sampled_target_proxy: np.ndarray
    public_binding: str

    def __post_init__(self):
        x, p = np.asarray(self.features), np.asarray(self.sampled_target_proxy)
        if (len(self.ids) > 120 or len(set(self.ids)) != len(self.ids)
                or x.shape != (len(self.ids), FEATURES) or p.shape != (len(self.ids),)
                or not np.isfinite(x).all() or not np.isfinite(p).all()):
            raise ValueError("Malformed bounded public proposal rows")
        object.__setattr__(self, "features", immutable_array(x, np.float64))
        object.__setattr__(self, "sampled_target_proxy", immutable_array(p, np.float64))


def public_rows(case, cavity, rays, *, steps, horizon, current_tool):
    """Whitelisted public arrays, not a task/engine/preview or reward callback.

    Nearest native-cell samples use half-open cells floor(index+.5). Unknown
    samples are zero WITH false per-channel coverage, never clipped to anatomy.
    Nine longitudinal planes and a 3x3 transverse stencil cover the larger of
    shaft/tip radii. This is a lossy feature stencil, not collision geometry.
    """
    if (type(case) is not NativeSpatialCase or case.nominal_target is None
            or not 1 <= horizon <= 24 or not 0 <= steps <= horizon
            or len(rays) > 120):
        raise ValueError("Declared public task and bounded proposals required")
    cavity = np.asarray(cavity)
    if cavity.dtype != np.bool_ or cavity.shape != case.observed_support.shape:
        raise ValueError("Observed cavity shape/type differs")
    affine = case._native_affine_ras_mm
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    directions = basis / spacing
    align = directions.T @ case.access.normal_inward
    axis = int(np.argmax(np.abs(align)))
    if abs(abs(align[axis])-1.) > 1e-10:
        raise ValueError("Source-normal stencil required")
    transverse = [i for i in range(3) if i != axis]
    inverse = np.linalg.inv(basis)
    shape = np.asarray(cavity.shape)
    tools = {tool.tool_id: tool for tool in case.tools}
    arrays = (case.structural_intensity, case.observed_support, case.nominal_target, cavity)
    domains = (None, case.support_domain, case.public_target_domain, case._interaction_domain)
    geometry, samples, proxies = [], [], []
    for ray in rays:
        tool = tools[ray.tool_id]
        entry, tip = np.asarray(ray.entry_mm), np.asarray(ray.tip_mm)
        delta = tip-entry
        length = float(np.linalg.norm(delta))
        direction = delta/length if length else np.zeros(3)
        radius = max(tool.tip_radius_mm, tool.shaft_radius_mm)
        center = entry + np.linspace(0., 1., 9)[:, None]*delta
        offsets = np.asarray([a*directions[:, transverse[0]] + b*directions[:, transverse[1]]
                              for a in (-radius, 0., radius) for b in (-radius, 0., radius)])
        points = (center[:, None, :] + offsets[None, :, :]).reshape(81, 3)
        indices = np.floor((points-affine[:3, 3]) @ inverse.T + .5).astype(np.int64)
        inside = np.all((indices >= 0) & (indices < shape), axis=1)
        values = np.zeros((81, 4)); covered = np.zeros((81, 4), dtype=bool)
        for c, (array, domain) in enumerate(zip(arrays, domains)):
            valid = inside.copy()
            if domain is not None:
                valid[inside] &= domain[tuple(indices[inside].T)]
            covered[:, c] = valid
            values[valid, c] = array[tuple(indices[valid].T)]
        if case.intensity_normalization != "raw":
            low, high = case._normalization_record["lower"], case._normalization_record["upper"]
            values[:, 0] = np.where(covered[:, 0], np.clip((values[:, 0]-low)/(high-low), 0., 1.), 0.)
        local_entry = (entry-case.access.center_mm) @ directions / 10.
        local_tip = (tip-case.access.center_mm) @ directions / 10.
        g = [0., *local_entry, *local_tip, *(direction @ directions),
             tool.tip_radius_mm/10., tool.shaft_radius_mm/10., tool.working_length_mm/10.,
             tool.tip_length_mm/10., tool.max_access_angle_deg/90.,
             float(current_tool is not None and current_tool != ray.tool_id)]
        geometry.append(g)
        samples.append(np.concatenate((values, covered), axis=1).reshape(-1))
        # Declared cheap proxy only: observed target along sampled path, not
        # exact contained removal, reachability, exposure or clinical utility.
        target = values[:, 2] * covered[:, 2]
        remaining_proxy = target * (1.-values[:, 3]*covered[:, 3])
        proxies.append(float(remaining_proxy.mean()*np.pi*tool.tip_radius_mm**2*length))
    state = [steps/horizon, horizon/10., float(current_tool is not None), 0., 0., 0.,
             *(case.access.normal_inward @ directions), case.access.radius_mm/10.]
    x = np.asarray([np.concatenate((s, g, state, spacing/10., np.ones(4)))
                    for s, g in zip(samples, geometry)], dtype=np.float64).reshape(len(rays), FEATURES)
    # Public binding is integrity metadata only; it never enters x or tie order.
    binding = public_binding(case, cavity, rays, steps=steps, horizon=horizon, current_tool=current_tool)
    return PublicRows(tuple(ray.proposal_id for ray in rays), x, np.asarray(proxies), binding)


def public_binding(case, cavity, rays, *, steps, horizon, current_tool):
    """Integrity-only metadata, also available without constructing any features."""
    return semantic_digest({"version": VERSION, "source": case.source_hash,
        "cavity": array_digest(cavity), "steps": steps, "horizon": horizon,
        "current_tool": current_tool,
        "rays": [(ray.tool_id, ray.entry_mm, ray.tip_mm) for ray in rays]})


def regression_weights(y, state_ids, *, positive_zero_balanced=False):
    """Each represented state has mass 1/S; single-class states keep uniform mass."""
    y, states = np.asarray(y), np.asarray(state_ids)
    if type(positive_zero_balanced) is not bool or y.ndim != 1 or states.shape != y.shape:
        raise ValueError("Aligned regression groups and exact bool required")
    unique, inverse, count = np.unique(states, return_inverse=True, return_counts=True)
    w = 1./(len(unique)*count[inverse])
    if positive_zero_balanced:
        for state in unique:
            mask = states == state
            positive, zero = mask & (y > 0), mask & (y == 0)
            if positive.any() and zero.any():
                w[positive] = .5 / (len(unique)*int(positive.sum()))
                w[zero] = .5 / (len(unique)*int(zero.sum()))
    return w


@dataclass(frozen=True)
class RidgePriority:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: float

    def __post_init__(self):
        for name in ("mean", "scale", "weights"):
            value = np.asarray(getattr(self, name))
            if value.shape != (FEATURES,) or not np.isfinite(value).all():
                raise ValueError("Malformed ridge parameters")
            object.__setattr__(self, name, immutable_array(value, np.float64))
        if np.any(self.scale <= 0) or not np.isfinite(self.intercept):
            raise ValueError("Malformed ridge scaling/intercept")

    @property
    def fingerprint(self):
        return semantic_digest({"version": VERSION, "ridge": RIDGE,
            **{k: array_digest(getattr(self, k)) for k in ("mean", "scale", "weights")},
            "intercept": self.intercept})

    @classmethod
    def fit(cls, x, y, state_ids, *, positive_zero_balanced=False):
        """One fixed ridge solve; source/split admission belongs to the caller.

        y=max(0, nominal immediate reward) if certified legal, otherwise zero.
        Rejected=zero is a priority label, NEVER an inference safety assertion.
        Equal state weights prevent larger inventories dominating the solve.
        """
        x, y, states = np.asarray(x, float), np.asarray(y, float), np.asarray(state_ids)
        if (x.ndim != 2 or x.shape[1] != FEATURES or not 1 <= len(x) <= 29*120
                or y.shape != (len(x),) or states.shape != y.shape
                or not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(y < 0)
                or len(np.unique(states)) > 29):
            raise ValueError("Bounded fit rows required")
        if type(positive_zero_balanced) is not bool:
            raise ValueError("Exact weighting bool required")
        unique, inverse, count = np.unique(states, return_inverse=True, return_counts=True)
        w = 1./(len(unique)*count[inverse])
        mean = (x*w[:, None]).sum(axis=0)
        scale = np.maximum(np.sqrt(((x-mean)**2*w[:, None]).sum(axis=0)), 1e-6)
        if positive_zero_balanced:
            # Normalization remains exactly the original equal-state fit.
            # Center normalized X with the new objective weights so the
            # intercept stays unpenalized even though the original mean is fixed.
            w = regression_weights(y, states, positive_zero_balanced=True)
            normalized = (x-mean)/scale
            center_x, center_y = (normalized*w[:, None]).sum(axis=0), float(y @ w)
            z = (normalized-center_x)*np.sqrt(w[:, None])
            centered = (y-center_y)*np.sqrt(w)
            if len(x) < FEATURES:
                weights = z.T @ np.linalg.solve(z @ z.T + RIDGE*np.eye(len(x)), centered)
            else:
                weights = np.linalg.solve(z.T @ z + RIDGE*np.eye(FEATURES), z.T @ centered)
            return cls(mean, scale, weights, float(center_y-center_x @ weights))
        intercept = float(y @ w)
        z = (x-mean)/scale*np.sqrt(w[:, None])
        centered = (y-intercept)*np.sqrt(w)
        if len(x) < FEATURES:
            weights = z.T @ np.linalg.solve(z @ z.T + RIDGE*np.eye(len(x)), centered)
        else:
            weights = np.linalg.solve(z.T @ z + RIDGE*np.eye(FEATURES), z.T @ centered)
        return cls(mean, scale, weights, intercept)

    fit_generated = fit  # Kept for the original generated-only control API.

    def score(self, rows):
        if type(rows) is not PublicRows:
            raise TypeError("Only bounded public rows enter the ranker")
        return (rows.features-self.mean)/self.scale @ self.weights + self.intercept


class GeneratedSparseTask(NativeSpatialTask):
    """Temporary generated-only adapter. No patient/frozen-policy admission.

    Subclassing dispatches reset/observation into this sparse inventory before
    the base constructor can eagerly certify the full proposal catalog.
    Native engine, commit, source arrays, proposal provider and rewards stay old.
    """
    def __init__(self, case, *, mode, ranker=None, max_steps=24, cancelled=None, costs=None):
        self._require_source(case)
        if (case.proposal_mode != "nominal_cavity_v1" or case.proposal_config.obstruction_opening
                or mode not in {"ridge_top8", "sampled_target_top8", "exhaustive"}
                or (mode == "ridge_top8") != (type(ranker) is RidgePriority)):
            raise ValueError("Generated nominal proposal condition and explicit selector required")
        self._selection = mode
        self._ranker = ranker
        self.costs = costs if costs is not None else Costs()
        self._raw_rows = None
        self._shortlist_record = None
        with self.costs.measure("constructor_inclusive"):
            super().__init__(case, max_steps=max_steps, cancelled=cancelled)

    def _require_source(self, case):
        if type(case) is not NativeSpatialCase or case.track != "synthetic_scan":
            raise ValueError("Generated public source required")

    def _contract_record(self):
        return {**super()._contract_record(), "sparse_proposal": {"version": VERSION,
            "mode": self._selection, "query_limit_per_state": None if self._selection == "exhaustive" else K,
            "ranker": None if self._ranker is None else self._ranker.fingerprint,
            "ties": "original_raw_order", "STOP": "best_verified_reward_le_zero; not_global_exhaustion"}}

    def _prepare_inventory(self):
        self._assert_frozen(); self._check_cancelled()
        if self._inventory is not None:
            return self._inventory
        if self._terminated:
            self._inventory, self._ledger, self._proposal_batch = {}, (), None
            self._raw_rows, self._shortlist_record = None, None
            return self._inventory
        with self.costs.measure("inventory_inclusive"):
            with self.costs.measure("proposal_enumeration"):
                self.costs.count("proposal_calls")
                batch = self.case._nominal_proposer.propose(self._engine, cancelled=self._cancelled)
            cavity = self._engine.removed_mask if self.case.post_exposure is None else self._engine.connected_free_mask
            ids = tuple(ray.proposal_id for ray in batch.proposals)
            rows, scores = None, None
            if self._selection == "exhaustive":
                with self.costs.measure("input_binding"):
                    binding = public_binding(self.case, cavity, batch.proposals, steps=self._steps,
                        horizon=self.max_steps, current_tool=self._current_tool)
                selected = list(range(len(ids)))
            else:
                with self.costs.measure("public_features"):
                    rows = public_rows(self.case, cavity, batch.proposals, steps=self._steps,
                        horizon=self.max_steps, current_tool=self._current_tool)
                    self.costs.count("feature_rows", len(rows.ids))
                binding = rows.public_binding
                with self.costs.measure("ranking"):
                    scores = (self._ranker.score(rows) if self._selection == "ridge_top8" else rows.sampled_target_proxy)
                    if self._selection == "ridge_top8": self.costs.count("ridge_calls")
                    if np.asarray(scores).shape != (len(ids),) or not np.isfinite(scores).all():
                        raise ValueError("Finite aligned proposal priorities required")
                    selected = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i))[:K]
            inventory, outcomes = {}, {}
            for i in selected:
                self._check_cancelled()
                ray = batch.proposals[i]
                with self.costs.measure("exact_verification"):
                    self.costs.count("native_previews")
                    result = self._engine.preview_stroke(ray.tool_id, ray.tip_mm, entry_mm=ray.entry_mm)
                outcomes[ray.proposal_id] = result
                if result.feasible: inventory[ray.proposal_id] = result
            # Preserve original proposal order for exact reward ties after
            # ranking selected the set; STOP stays first in native greedy.
            inventory = {ray.proposal_id: inventory[ray.proposal_id] for ray in batch.proposals
                         if ray.proposal_id in inventory}
            ledger = []
            for slot in batch.ledger:
                record = asdict(slot)
                result = outcomes.get(slot.proposal_id)
                record.update(action_id=slot.proposal_id,
                    query_disposition=("NOT_QUERIED" if result is None else
                                       "CERTIFIED_LEGAL" if result.feasible else "CERTIFIED_REJECTED"))
                if result is not None:
                    record.update(feasible=bool(result.feasible), native_reason=result.reason)
                ledger.append(record)
            self._check_cancelled()
            self._inventory, self._ledger, self._proposal_batch = inventory, tuple(ledger), batch
            self._raw_rows = rows
            self._shortlist_record = {"raw_count": len(ids), "queried_count": len(selected),
                "not_queried_count": len(ids)-len(selected), "legal_count": len(inventory),
                "input_binding": binding, "selected_ids": [ids[i] for i in selected],
                "raw_ids": list(ids), "priority_scores": None if scores is None else np.asarray(scores).tolist(),
                "exhaustive_over_emitted_catalog": len(selected) == len(ids),
                "priority_is_not_legality": True}
            if self.costs.inventory_callback is not None:
                with self.costs.measure("inventory_logging"):
                    keys = ("action_id", "proposal_id", "tool_id", "column_index", "family", "voxel",
                            "reason", "query_disposition", "feasible", "native_reason")
                    self.costs.inventory_callback({"step": self._steps, "source_state_hash": self._engine.state_hash,
                        "shortlist": copy.deepcopy(self._shortlist_record),
                        "ledger": [{k:r[k] for k in keys if k in r} for r in self._ledger]})
        return self._inventory

    def candidate_inventory(self):
        self._prepare_inventory()
        return {"scope": VERSION, "terminal": self.terminated,
                "shortlist": copy.deepcopy(self._shortlist_record), "ledger": copy.deepcopy(self._ledger)}

    def _qualify_accounting(self, accounting):
        accounting["method"] = self._selection
        accounting["verification_scope"] = "full_emitted_catalog" if self._selection == "exhaustive" else "fixed_top8_raw_proposals"
        accounting["STOP_meaning"] = "chosen_against_queried_actions_only; no_global_unreachability_claim"
        for row in accounting["decisions"]:
            row["all_queried_legal_actions_scored"] = row.pop("all_current_legal_actions_scored")
        return accounting

    def _observed_greedy_search(self, *, seconds, one_step):
        try:
            actions, accounting = super()._observed_greedy_search(seconds=seconds, one_step=one_step)
        except ObservedSearchLimit as exc:
            self._qualify_accounting(exc.accounting)
            raise
        return actions, self._qualify_accounting(accounting)

    def fresh(self):
        self._assert_frozen()
        return type(self)(self.case, mode=self._selection, ranker=self._ranker,
                          max_steps=self.max_steps, cancelled=self._cancelled, costs=self.costs)


def generated_training_rows(task):
    """Label one generated planning state's raw rows; no private reward path."""
    if type(task) is not GeneratedSparseTask or not task._planning or task._selection != "exhaustive":
        raise ValueError("Only exhaustive generated nominal planning states may label proposals")
    inventory = task._prepare_inventory()
    if task._raw_rows is None:
        cavity = task._engine.removed_mask if task.case.post_exposure is None else task._engine.connected_free_mask
        task._raw_rows = public_rows(task.case, cavity, task._proposal_batch.proposals,
            steps=task._steps, horizon=task.max_steps, current_tool=task._current_tool)
    labels = []
    for identifier in task._raw_rows.ids:
        preview = inventory.get(identifier)
        reward = 0. if preview is None else task._score_record(
            {**preview.to_history_record(), "action_id": identifier}, task._current_tool)["reward"]
        labels.append(max(0., reward))
    return task._raw_rows, np.asarray(labels)
