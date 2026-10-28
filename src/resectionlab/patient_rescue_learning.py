"""Observed-state policy contract for the prepared wound-rescue experiment.

The publisher may read the current modeled scene to produce declared sensors.
The frozen policy receives only the returned numeric vector. It never receives
face coagulation dose, a native snapshot, a candidate outcome, or a step index.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = "resectionlab.prepared-rescue-observation.v2"
ROUTES = ("contact", "coverage_lower", "coverage_forward", "suction_pool")
OPTIONS = ("coagulate_center", "coagulate_lower", "coagulate_forward",
           "aspirate_pool", "wait", "finish_verify")
FEATURE_NAMES = ("retained_fluid_div3", "active_outlets", "outlet_rate_div0_5",
                 "fluid_near_center_div3", "fluid_near_lower_div3",
                 "fluid_near_forward_div3", "fluid_near_suction_div3",
                 "left_controls_clear", "right_controls_clear")


def features_from_sensors(field, route_fluid, slots):
    """Encode current public model sensors without history or phase counters."""
    field = np.asarray(field, dtype=float)
    route_fluid = np.asarray(route_fluid, dtype=float)
    slots = np.asarray(slots, dtype=float)
    if field.shape != (4,) or route_fluid.shape != (4,) or slots.shape != (2, 12):
        raise ValueError("Prepared rescue sensor shape differs")
    clear = [float(not np.any(row[8:12])) for row in slots]
    result = np.asarray([field[0] / 3., field[1], field[2] / .5,
                         *(route_fluid / 3.), *clear], dtype=np.float32)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("Prepared rescue sensors are nonfinite")
    return result


def publish_observation(task):
    """Publish current route-local fluid and controls from the same live engine.

    Route centers are fixed, source-bound task inputs. Each local value is the
    modeled fluid amount within a 5^3 voxel window around that named route.
    This is spatially indexed fluid, not hidden wound-face dose or a preview.
    """
    scene = task.raw.scene
    fluid = task.raw.engine.state["fluid"]
    route_fluid = []
    for route in ROUTES:
        position = np.asarray(task.waypoints[route]["position"], dtype=float)
        center = np.rint(scene.inverse[:3, :3] @ position + scene.inverse[:3, 3]).astype(int)
        total = 0.
        for i in range(-2, 3):
            for j in range(-2, 3):
                for k in range(-2, 3):
                    point = center + (i, j, k)
                    if np.all((point >= 0) & (point < scene.shape)):
                        index = int(np.ravel_multi_index(tuple(point), scene.shape))
                        total += float(fluid.get(str(index), 0.))
        route_fluid.append(total)
    field = task.raw.field()
    slots = task.raw._observe()["slots"]
    vector = features_from_sensors(field, route_fluid, slots)
    return vector, {"schema": SCHEMA, "featuresSha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                    "field": field.tolist(), "routeFluidMm3": route_fluid,
                    "routeOrder": list(ROUTES), "controlsClear": vector[-2:].tolist()}


@dataclass(frozen=True)
class FrozenRescueBCPolicy:
    """Small learned classifier over the six fixed, meaningful skill options."""

    weight: np.ndarray
    bias: np.ndarray

    def __post_init__(self):
        if self.weight.shape != (len(OPTIONS), len(FEATURE_NAMES)) or self.bias.shape != (len(OPTIONS),):
            raise ValueError("Prepared rescue checkpoint shape differs")
        if not np.isfinite(self.weight).all() or not np.isfinite(self.bias).all():
            raise ValueError("Prepared rescue checkpoint is nonfinite")

    def predict(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (len(FEATURE_NAMES),) or not np.isfinite(observation).all():
            raise ValueError("Prepared rescue policy input differs")
        scores = self.weight @ observation + self.bias
        return int(np.argmax(scores)), scores.tolist()


def load_bc_checkpoint(path, *, manifest_sha256):
    """Bind a frozen BC actor to its original task and observation contract."""
    raw = Path(path).read_bytes()
    record = json.loads(raw)
    if (record.get("schema") != "resectionlab.patient-rescue-bc-checkpoint.v1"
            or record.get("observationSchema") != SCHEMA
            or record.get("taskManifestSha256") != manifest_sha256
            or record.get("options") != list(OPTIONS)
            or record.get("featureNames") != list(FEATURE_NAMES)):
        raise ValueError("BC checkpoint task/observation/action identity differs")
    policy = FrozenRescueBCPolicy(np.asarray(record["weight"], dtype=np.float32),
                                  np.asarray(record["bias"], dtype=np.float32))
    return policy, hashlib.sha256(raw).hexdigest()
