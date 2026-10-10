"""Versioned mixed-action observation over the existing permitted spatial DTO.

This is a generated-development candidate. NativeSpatialTask must construct the
contact grid from committed *observed probe* events; this DTO cannot prove that
upstream provenance. Legacy SpatialPolicy deliberately does not accept it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from resectionlab.core import array_digest, semantic_digest
from resectionlab.spatial_observations import SpatialObservation


VERSION = "sequential-spatial-observation-v1"
ACTION_MODES = frozenset({"stop", "aspirate", "probe"})


@dataclass(frozen=True)
class SequentialSpatialObservation:
    base: SpatialObservation
    action_modes: tuple[str, ...]
    observed_probe_contact_grid: np.ndarray
    _contact_bytes: bytes = field(init=False, repr=False)
    _identity: tuple = field(init=False, repr=False)
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.base) is not SpatialObservation:
            raise TypeError("Mixed-action observation requires the exact permitted spatial DTO")
        self.base.assert_intact()
        modes = tuple(self.action_modes)
        if (len(modes) != len(self.base.action_ids) or not modes or modes[0] != "stop"
                or any(mode not in ACTION_MODES for mode in modes)
                or any(mode == "stop" for mode in modes[1:])):
            raise ValueError("Action modes must align with STOP-first certified action IDs")
        contact = np.asarray(self.observed_probe_contact_grid)
        if (contact.dtype != np.bool_ or contact.shape != self.base.image_channels.shape[1:]
                or np.any(contact & ~self.base.coverage[0])):
            raise ValueError("Observed probe contact must be a covered boolean source-grid crop")
        payload = contact.tobytes(order="C")
        frozen = np.frombuffer(payload, dtype=np.bool_).reshape(contact.shape)
        object.__setattr__(self, "action_modes", modes)
        object.__setattr__(self, "observed_probe_contact_grid", frozen)
        object.__setattr__(self, "_contact_bytes", payload)
        identity = (id(self.base), self.base.fingerprint, modes, contact.shape, payload)
        object.__setattr__(self, "_identity", identity)
        object.__setattr__(self, "_fingerprint", semantic_digest({
            "version": VERSION, "base": self.base.fingerprint,
            "action_ids": self.base.action_ids, "action_modes": modes,
            "observed_probe_contact": array_digest(frozen),
        }))

    def assert_intact(self) -> None:
        self.base.assert_intact()
        current = (id(self.base), self.base.fingerprint, self.action_modes,
                   self.observed_probe_contact_grid.shape,
                   self.observed_probe_contact_grid.tobytes(order="C"))
        if current != self._identity or not np.shares_memory(self.observed_probe_contact_grid,
                                                              np.frombuffer(self._contact_bytes, bool)):
            raise ValueError("Sequential spatial observation was changed after construction")

    @property
    def fingerprint(self) -> str:
        self.assert_intact()
        return self._fingerprint

    @property
    def action_ids(self) -> tuple[str, ...]:
        return self.base.action_ids

    @property
    def action_mask(self) -> np.ndarray:
        return self.base.action_mask

    @property
    def action_tool_ids(self) -> tuple[str | None, ...]:
        return self.base.action_tool_ids

    @property
    def source_id(self) -> str:
        return self.base.source_id

    @property
    def track(self) -> str:
        return self.base.track
