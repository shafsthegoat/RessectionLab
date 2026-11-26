"""Soft vessel ensembles of named multi-class models, with connected single-class additions.

Every member is a softmax over the same named labels on one source-image grid
and acquired with one modality. Members are pooled by a declared combiner,
never relabelled, and a model from one modality is never combined with an
image or reference of the other. A single-class vessel score may only extend
the named tree: a scored voxel is added when it is 26-connected to the tree
through voxels at or above the low threshold and its component holds at least
one voxel at the high threshold. Added voxels keep their score. Bridging gaps
to the tree is left to the cleanup phase.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

MODALITIES = ('ct', 'mr')
FULL = np.ones((3, 3, 3), bool)


def _modality(value):
    if value not in MODALITIES:
        raise ValueError(f'Modality must be one of {MODALITIES}, got {value!r}')
    return value


def _affine(value):
    value = np.asarray(value, float)
    if value.shape != (4, 4) or not np.isfinite(value).all() or abs(np.linalg.det(value[:3, :3])) < 1e-9:
        raise ValueError('A finite invertible 4 x 4 affine is required')
    return value


def require_same_grid(shape, affine, other_shape, other_affine, atol=1e-3):
    if tuple(shape) != tuple(other_shape):
        raise ValueError(f'Grids differ in shape: {tuple(shape)} and {tuple(other_shape)}')
    if not np.allclose(_affine(affine), _affine(other_affine), atol=atol):
        raise ValueError('Grids differ in their affine')


def _probability(values, name):
    """Check floating probabilities one leading slice at a time, so volumes read channel by channel from disk
    (memory maps or any array-like with shape, dtype and per-channel indexing) never load whole."""
    values = values if all(hasattr(values, key) for key in ('shape', 'dtype', 'ndim')) else np.asarray(values)
    if not np.issubdtype(values.dtype, np.floating):
        raise ValueError(f'{name} must be floating point probabilities')
    for part in (values if values.ndim == 4 else [values]):
        if part.size and (not np.isfinite(part).all() or part.min() < 0 or part.max() > 1 + 1e-3):
            raise ValueError(f'{name} must lie in [0, 1]')
    return values


@dataclass(frozen=True)
class SoftSegmentation:
    """Class probabilities with channel 0 for background, in source-image voxel order."""

    modality: str
    labels: tuple
    probabilities: np.ndarray
    affine: np.ndarray

    def __post_init__(self):
        _modality(self.modality)
        labels = tuple(int(value) for value in self.labels)
        probabilities = _probability(self.probabilities, 'Class probabilities')
        if probabilities.ndim != 4:
            raise ValueError('Class probabilities must be channels by three spatial axes')
        if len(labels) != probabilities.shape[0] or labels[0] != 0 or len(set(labels)) != len(labels):
            raise ValueError('Labels must name every channel once, starting with background 0')
        object.__setattr__(self, 'labels', labels)
        object.__setattr__(self, 'probabilities', probabilities)
        object.__setattr__(self, 'affine', _affine(self.affine))

    @property
    def shape(self):
        return self.probabilities.shape[1:]


COMBINERS = ('mean', 'weighted', 'maximum', 'median', 'base')


def _check_members(members):
    members = list(members)
    if not members:
        raise ValueError('At least one member is required')
    first = members[0]
    for member in members[1:]:
        if member.modality != first.modality:
            raise ValueError(f'Modality mismatch: {first.modality} and {member.modality} models cannot be combined')
        if member.labels != first.labels:
            raise ValueError('Members must share one ordered label set')
        require_same_grid(first.shape, first.affine, member.shape, member.affine)
    return members


def _weights(members, method, weights, base):
    if method not in COMBINERS:
        raise ValueError(f'Combiner must be one of {COMBINERS}')
    if method == 'weighted':
        weights = np.asarray(weights, float)
        if weights.shape != (len(members),) or not (weights > 0).all():
            raise ValueError('One positive weight per member is required')
        return weights / weights.sum()
    if method == 'base' and not (isinstance(base, int) and 0 <= base < len(members) and len(members) > 1):
        raise ValueError('The base member must index one of at least two members')
    return np.full(len(members), 1 / len(members))


def vessel_probability(members, method='mean', weights=None, base=None):
    """Combined vessel probability (1 - background) from members' background channels.

    mean and weighted average the members; maximum takes the most confident member (a union at 0.5); median is two
    of three; base keeps one member and adds the others' strongest vessel probability only where it is 26-connected
    at 0.5 to the base member's vessels (ConnectedAddition low = high = 0.5).
    """
    members = _check_members(members)
    weights = _weights(members, method, weights, base)
    if method in ('mean', 'weighted'):
        vessel = np.zeros(members[0].shape, np.float32)
        for weight, member in zip(weights, members):
            vessel += weight * (1 - np.asarray(member.probabilities[0], np.float32))
        return np.clip(vessel, 0, 1)
    vessels = [1 - np.asarray(member.probabilities[0], np.float32) for member in members]
    if method == 'maximum':
        return np.clip(np.max(vessels, axis=0), 0, 1)
    if method == 'median':
        return np.clip(np.median(vessels, axis=0), 0, 1).astype(np.float32)
    kept = vessels.pop(base)
    others = np.clip(np.max(vessels, axis=0), 0, 1)
    first = members[0]
    added, _ = ConnectedAddition(low=0.5, high=0.5).apply(
        kept >= 0.5, first.modality, first.affine, others, first.modality, first.affine)
    return np.clip(np.maximum(kept, added), 0, 1)


def combine(members, method='mean', weights=None, base=None, out=None):
    """Class probabilities under a combiner: P(background) = 1 - V for the combined vessel probability V, and each
    named label takes V times its share of the members' named probability.

    The share is the (weighted) mean of the members' class probabilities, renormalised over named labels; for base,
    the base member's share where it calls vessel and the other members' mean share elsewhere. Where only one member
    supports a voxel, its labels therefore decide the name. Channels are written in order into `out` as in average.
    """
    members = _check_members(members)
    weights = _weights(members, method, weights, base)
    vessel = vessel_probability(members, method, weights if method == 'weighted' else None, base)
    first = members[0]
    out = np.zeros(first.probabilities.shape, np.float32) if out is None else out
    if out.shape != first.probabilities.shape:
        raise ValueError('The output must match the members\' channels and grid')
    if method == 'base':
        groups = [[base], [index for index in range(len(members)) if index != base]]
        inside = (1 - np.asarray(members[base].probabilities[0], np.float32)) >= 0.5
    else:
        groups, inside = [list(range(len(members)))], None
    coefficients = [{index: (weights[index] if method == 'weighted' else 1 / len(group)) for index in group}
                    for group in groups]
    mass = [sum(c * (1 - np.asarray(members[i].probabilities[0], np.float32)) for i, c in group.items())
            for group in coefficients]
    out[0] = 1 - vessel
    for channel in range(1, out.shape[0]):
        shares = []
        for group, total in zip(coefficients, mass):
            share = sum(c * np.asarray(members[i].probabilities[channel], np.float32) for i, c in group.items())
            shares.append(np.divide(share, total, out=np.zeros_like(total), where=total > 1e-6))
        # Base: the base member names its own vessels, and any voxel the other members do not support at all.
        share = shares[0] if inside is None else np.where(inside | (mass[1] <= 1e-6), shares[0], shares[1])
        out[channel] = vessel * share
    if hasattr(out, 'close'):
        out = out.close()
    return SoftSegmentation(first.modality, first.labels, out, first.affine)


def average(members, out=None):
    """Unweighted mean of members that share modality, labels and grid.

    Channels are averaged one at a time, in order, into `out` (float32 by default). `out` may be any array-like
    of the members' shape that accepts channel assignment; if it has `close()`, that is called after the last
    channel and must return the readable result (a sequential writer).
    """
    members = _check_members(members)
    first = members[0]
    out = np.zeros(first.probabilities.shape, np.float32) if out is None else out
    if out.shape != first.probabilities.shape:
        raise ValueError('The output must match the members\' channels and grid')
    for channel in range(out.shape[0]):
        total = np.zeros(first.shape, np.float32)
        for member in members:
            total += member.probabilities[channel]
        out[channel] = total / len(members)
    if hasattr(out, 'close'):
        out = out.close()
    return SoftSegmentation(first.modality, first.labels, out, first.affine)


def vessel_groups(segmentation, vein_labels):
    """Soft artery, vein-and-sinus and all-vessel probabilities."""
    unknown = set(vein_labels) - set(segmentation.labels[1:])
    if unknown:
        raise ValueError(f'Vein labels absent from the model: {sorted(unknown)}')
    probabilities = segmentation.probabilities
    vessel = np.clip(1 - probabilities[0].astype(np.float32), 0, 1)
    vein = np.zeros_like(vessel)
    for index, label in enumerate(segmentation.labels):
        if label in set(vein_labels):
            vein += probabilities[index]
    vein = np.clip(vein, 0, 1)
    return {'artery': np.clip(vessel - vein, 0, 1), 'vein': np.minimum(vein, vessel), 'vessel': vessel}


def named_label_map(segmentation, threshold=0.5):
    """Most probable named label wherever the vessel probability reaches the threshold."""
    probabilities = segmentation.probabilities
    named = np.zeros(segmentation.shape, np.uint8 if max(segmentation.labels) < 256 else np.uint16)
    best = np.full(segmentation.shape, -1, np.float32)
    for index, label in enumerate(segmentation.labels[1:], start=1):
        channel = np.asarray(probabilities[index], np.float32)
        higher = channel > best
        named[higher], best[higher] = label, channel[higher]
    named[1 - np.asarray(probabilities[0], np.float32) < threshold] = 0
    return named


@dataclass(frozen=True)
class ConnectedAddition:
    low: float = 0.5
    high: float = 0.5

    def __post_init__(self):
        if not 0 < self.low <= self.high <= 1:
            raise ValueError('Thresholds must satisfy 0 < low <= high <= 1')

    def apply(self, tree, tree_modality, tree_affine, score, score_modality, score_affine):
        """Return the added soft score (zero elsewhere) and the added mask.

        Components of voxels outside the tree with score at or above `low`
        (26-connected) are added when they touch the tree (26-neighbourhood)
        and contain a voxel at or above `high`. The tree plus the addition
        therefore has no more components than the tree. Islands are rejected.
        """
        if _modality(tree_modality) != _modality(score_modality):
            raise ValueError(f'Modality mismatch: a {score_modality} score cannot extend a {tree_modality} tree')
        tree = np.asarray(tree).astype(bool)
        score = _probability(score, 'The added score')
        if tree.ndim != 3 or not tree.any():
            raise ValueError('A non-empty three-dimensional named tree is required')
        require_same_grid(tree.shape, tree_affine, score.shape, score_affine)
        candidates = (score >= self.low) & ~tree
        components, count = ndi.label(candidates, FULL)
        if not count:
            return np.zeros(score.shape, np.float32), np.zeros(score.shape, bool)
        touching = np.unique(components[ndi.binary_dilation(tree, FULL) & candidates])
        strong = np.unique(components[candidates & (score >= self.high)])
        kept = np.intersect1d(touching[touching > 0], strong[strong > 0])
        added = np.isin(components, kept)
        return np.where(added, score, 0).astype(np.float32), added
