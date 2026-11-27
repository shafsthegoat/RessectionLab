"""Conservative, reversible review of native angiographic vessel predictions."""

from base64 import b64encode
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import json

import numpy as np
from scipy import ndimage as ndi, sparse
from scipy.spatial import cKDTree

from .anatomy_siam_display import _head_support
from .data_root import atomic_json
from .patient_anatomy import affine, digest, file_hash, resample, world


RETAINED, UNCERTAIN, HIDDEN = 1, 2, 3
CONNECTIVITY = np.ones((3, 3, 3), dtype=bool)
EXTERIOR_DISPLAY = "native-tof-exterior-display-v1"
_AIR_CLEARANCE_MM = 2.0
_MAX_LABEL_CELLS = 128_000_000
_MAX_NEIGHBORHOOD_CELLS = 4_000_000
_MAX_EXTERIOR_FOREGROUND = 4_000_000
ATTACHED_REVIEW = "native-tof-attached-uncertainty-v1"
ATTACHED_ERASE = "erase-attached-tof"
ATTACHED_SCORE_SURFACE = "subtractive-native-tof-scores-v1"
_ATTACHED_DARK_PERCENTILE = 10.0
_ATTACHED_CORE_RADIUS_MM = 1.0
_ATTACHED_MINIMUM_VOLUME_MM3 = 2.5
_ATTACHED_CONTEXT_MM = 2.0
_ATTACHED_ANCHOR_CONTRAST_IQR = 0.15
_MAX_ATTACHED_WEAK_CELLS = 500_000
_MAX_ATTACHED_FINDINGS = 24


@dataclass(frozen=True)
class ComponentPolicy:
    version: str = "native-tof-detached-signal-v1"
    compact_volume_mm3: float = 0.32
    compact_axis_ratio: float = 2.0
    small_volume_mm3: float = 0.85
    isolation_mm: float = 1.5
    outer_ring_mm: float = 2.0
    inner_ring_mm: float = 0.55
    maximum_peak_contrast: float = 0.75
    score_median_ceiling: float = 0.8
    baseline_signal_percentile: float = 10.0
    very_low_signal_percentile: float = 5.0


def _bounds(points, spacing, shape, radius):
    pad = np.ceil(radius / spacing).astype(int)
    start = np.maximum(points.min(axis=0) - pad, 0)
    stop = np.minimum(points.max(axis=0) + pad + 1, shape)
    return start, tuple(slice(a, b) for a, b in zip(start, stop))


def _source_contrast(points, signal, mask, spacing, policy):
    start, slices = _bounds(points, spacing, signal.shape, policy.outer_ring_mm)
    local_signal = signal[slices]
    local = np.ones(local_signal.shape, dtype=bool)
    local[tuple((points - start).T)] = False
    distance = ndi.distance_transform_edt(local, sampling=spacing)
    ring = (
        (distance >= policy.inner_ring_mm)
        & (distance <= policy.outer_ring_mm)
        & ~mask[slices]
        & np.isfinite(local_signal)
    )
    nearby = local_signal[ring]
    foreground = signal[tuple(points.T)]
    if len(nearby) < 20 or not np.isfinite(foreground).all():
        return None
    lo, middle, hi = np.percentile(nearby, [25, 50, 75])
    peak = float(foreground.max())
    scale = max(float(hi - lo), np.finfo(np.float32).eps * max(abs(float(middle)), 1.0))
    return {
        "foreground_peak": peak,
        "foreground_median": float(np.median(foreground)),
        "ring_median": float(middle),
        "ring_iqr": float(hi - lo),
        "peak_contrast_iqr": (peak - float(middle)) / scale,
    }


def exterior_air_components(mask, signal, native_affine, *, check=lambda: None):
    """Propose whole-component presentation exclusions in sampled exterior air.

    The head envelope nominates a region for review. Both the component and
    its native-resolution neighborhood must be dark. An attached candidate,
    a candidate at an acquisition face, or nearby measured signal stays
    visible. The original foreground and source field are never written.
    """
    mask = np.asarray(mask)
    signal = np.asarray(signal)
    if (mask.dtype != np.bool_ or mask.ndim != 3 or mask.shape != signal.shape
            or signal.dtype.kind not in "fui"):
        raise ValueError("Expected a binary prediction and matched native TOF signal")
    frame = affine(native_affine)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    axes = frame[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        return [], {"status": "unavailable", "reason": "Native TOF lattice is sheared"}
    foreground = int(np.count_nonzero(mask))
    if not foreground:
        return [], {"status": "unchanged", "reason": "The source prediction is empty"}
    if foreground > _MAX_EXTERIOR_FOREGROUND:
        return [], {"status": "unavailable", "reason": "Native TOF foreground exceeds review workspace"}
    points = np.column_stack(np.nonzero(mask))
    start, slices = _bounds(points, spacing, mask.shape, 0)
    if np.prod([axis.stop - axis.start for axis in slices]) > _MAX_LABEL_CELLS:
        return [], {"status": "unavailable", "reason": "Native TOF prediction exceeds labeling workspace"}
    check()
    head, evidence = _head_support(signal, frame, check)
    if head is None or not np.isfinite(evidence["percentileWindow"]).all():
        return [], {"status": "unavailable", "reason": "No reliable native TOF head envelope"}
    support, _, stride = head
    sample = np.minimum((points + stride // 2) // stride, np.array(support.shape) - 1)
    outside = ~support[tuple(sample.T)]
    # Tie the low component ceiling to the observed signal floor. Use the
    # more permissive head threshold solely to protect neighboring signal.
    low, high = evidence["percentileWindow"]
    ceiling = low + 0.02 * (high - low)
    foreground_signal = signal[tuple(points.T)]
    distance_to_face = np.minimum(points + .5, np.array(mask.shape) - .5 - points) * spacing
    eligible = outside & np.isfinite(foreground_signal) & (foreground_signal <= ceiling)
    eligible &= (distance_to_face > _AIR_CLEARANCE_MM).all(axis=1)
    report = {
        "status": "unchanged",
        "sourceVoxels": foreground,
        "foregroundSignalCeiling": float(ceiling),
        **evidence,
    }
    if not eligible.any():
        return [], report

    check()
    labels, count = ndi.label(mask[slices], structure=CONNECTIVITY)
    memberships = labels[tuple((points - start).T)]
    del labels
    reject = np.zeros(count + 1, dtype=bool)
    reject[memberships[~eligible]] = True
    candidates = np.flatnonzero(~reject[memberships])
    candidates = candidates[np.argsort(memberships[candidates], kind="stable")]
    breaks = np.flatnonzero(np.diff(memberships[candidates])) + 1
    excluded = []
    for group in np.split(candidates, breaks):
        if not len(group):
            break
        check()
        cells = points[group]
        origin, bounds = _bounds(cells, spacing, mask.shape, _AIR_CLEARANCE_MM)
        neighborhood = signal[bounds]
        if neighborhood.size > _MAX_NEIGHBORHOOD_CELLS:
            continue
        nearby_signal = ~np.isfinite(neighborhood) | (neighborhood > evidence["threshold"])
        if np.any(nearby_signal):
            distance = ndi.distance_transform_edt(~nearby_signal, sampling=spacing)
            if np.any(distance[tuple((cells - origin).T)] <= _AIR_CLEARANCE_MM):
                continue
        excluded.append(np.ravel_multi_index(cells.T, mask.shape))
    return excluded, report


def supported_tof_layer(collection, row, *, check=lambda: None):
    """Get an exterior-air display derivative bound to the original TOF layer."""
    provenance = row.get("provenance", {})
    if (row["kind"] != "vessels" or provenance.get("model") != "vesselfm"
            or row.get("edits") or not row.get("probability")):
        return row, None
    from .anatomy_model_registry import source_volume

    series = collection.series(row["source_series"])
    if series["modality"] != "TOF" or not source_volume(series):
        return row, None
    if (row["affine"] != series["affine"] or row["data"]["shape"] != series["data"]["shape"]
            or provenance.get("source") != series["original"]["sha256"]):
        return row, {"scope": "exterior-air", "reason": "Source identity or native lattice could not be verified"}
    try:
        collection.require_native_geometry(series)
        collection.require_native_geometry(row)
    except ValueError:
        return row, {"scope": "exterior-air", "reason": "The source lattice requires geometry review"}
    identity = {
        "version": EXTERIOR_DISPLAY,
        "sourceSha256": series["original"]["sha256"],
        "sourceSignalSha256": series["data"]["sha256"],
        "rawDataSha256": row["data"]["sha256"],
        "sourceProbabilitySha256": row["probability"]["sha256"],
        "affine": series["affine"],
        "label": row["label_value"],
    }
    key = digest(identity)
    if not hasattr(collection, "_tof_display_cache"):
        collection._tof_display_cache = {}
    cache = collection._tof_display_cache
    data, receipt = cache.get(key, (None, None))
    if data is not None:
        try:
            collection.array(data)
        except FileNotFoundError:
            receipt = None
    if receipt is None:
        mask = np.ascontiguousarray(collection.mask(row))
        groups, evidence = exterior_air_components(mask, collection.array(series["data"]), series["affine"], check=check)
        hidden = sum(len(group) for group in groups)
        if 0 < hidden < evidence["sourceVoxels"]:
            for group in groups:
                mask.ravel()[group] = False
            check()
            data = collection.put_array(mask.astype(np.uint8))
            reason = (
                f"Exterior-air view hides {len(groups):,} detached source-dark findings. "
                "Use Full source estimate to inspect them."
            )
        else:
            data = None
            hidden = 0
            reason = evidence.get("reason", "No detached finding passed the exterior-air screen")
        receipt = {
            **{k: v for k, v in identity.items() if k not in {"affine", "label"}},
            "scope": "exterior-air",
            "hiddenVoxels": hidden,
            "excludedFragments": len(groups) if hidden else 0,
            "sampledDataSha256": data["sha256"] if data else row["data"]["sha256"],
            "evidence": evidence,
            "reason": reason,
        }
        cache[key] = data, receipt
    return ({**row, "data": data, "label_value": 1} if data else row), receipt


ROOTED_DISPLAY = "rooted-vessel-display-v1"
# Name of the stored display beside a study. Raise it whenever rooted_vessel_layer or
# root_vessels would prepare a different display from the same inputs and policy.
ROOTED_CACHE = "rooted-vessel-display-cache-v1"
ROOTED_VESSEL_KINDS = {"vessel", "vessels", "artery", "vein", "sinus"}
ROOTED_REVIEW_PREFIX = "__vessel_review:"
ROOTED_REVIEW_COLOR = "#ffad54"
# One vessel colour. Arteries and veins are still told apart in the receipts and
# as separate display parts, but they are not drawn in different colours.
VESSEL_CLASS_COLORS = {"artery": "#aa352b", "vein": "#aa352b"}
# Vessel detail of the connected display. "major" is the calibre rule of
# RootingPolicy (1 mm and wider everywhere, 0.5 to 1 mm near a tumour or planned
# corridor) without the image-derived fine branches; "fine" shows 0.5 mm and
# wider everywhere, with those branches. CT defaults to fine, where thin
# enhanced branches are followed in the acquired image.
VESSEL_DETAILS = ("major", "fine")
# Layers that are only fine branches (resectionlab.vessel_fine_growth): hidden whole at the major detail.
FINE_BRANCH_MODELS = {"signal-grown-fine-branches"}
_PARENT_DEPTH = 3


def vessel_detail(modality, requested=None):
    """The requested detail, or the default for the acquisition."""
    from .vascular_rooting import image_modality

    if requested is not None:
        if requested not in VESSEL_DETAILS:
            raise ValueError(f"Vessel detail must be one of {VESSEL_DETAILS}")
        return requested
    try:
        return "fine" if image_modality(modality) == "ct" else "major"
    except ValueError:
        return "major"
_TUMOUR_KINDS = {"tumor", "tumor-whole", "tumor-core", "enhancing-tumor", "nonenhancing-tumor"}
_TOPBRAIN_RECORD = "https://zenodo.org/records/21972006"


def _same_grid(row, other):
    return row["affine"] == other["affine"] and row["data"]["shape"] == other["data"]["shape"]


def rooted_named_labels(collection, row, series, *, load=True):
    """Named vessel labels bound to this layer: ``(labels, scheme, sha256, reason)``.

    A layer may carry ``named_labels: {"data": <array>, "scheme": "topbrain-v1-ct"
    | "topbrain-v1-mr"}`` on its own grid (model ensembles, Phase 2). A supplied
    TopBrain v1 annotation is read from its imported original only when the layer
    is exactly the original's listed label values on the scan grid. Labels of one
    modality are never bound to a scan of the other. With ``load=False`` only the
    binding is checked and ``labels`` is None.
    """
    import nibabel as nib

    from .vascular_rooting import SCHEMES, image_modality

    try:
        modality = image_modality(series["modality"])
    except ValueError:
        return None, None, None, "Unknown acquisition modality"
    attached = row.get("named_labels")
    if isinstance(attached, dict):
        scheme = attached.get("scheme")
        data = attached.get("data")
        if scheme not in SCHEMES or not isinstance(data, dict) or data.get("shape") != row["data"]["shape"]:
            return None, None, None, "Attached named labels are not on this layer grid"
        if SCHEMES[scheme].modality != modality:
            return None, None, None, "Named labels of one modality are never applied to another"
        return (np.asarray(collection.array(data)) if load else None), scheme, data["sha256"], None
    provenance = row.get("provenance") or {}
    original = provenance.get("source")
    grouping = provenance.get("source_label_values")
    if (provenance.get("kind") not in {"supplied_annotation", "observed_source_annotation"}
            or not isinstance(original, dict) or not isinstance(original.get("path"), str)
            or not isinstance(grouping, list) or not grouping
            or any(type(value) is not int or value <= 0 for value in grouping)):
        return None, None, None, "No named labels are bound to this layer"
    dictionary = str(provenance.get("publisher_label_dictionary", ""))
    name = str(original.get("name", ""))
    by_dictionary = {c for c in ("ct", "mr") if dictionary.endswith(f"labels_topbrain_v1_{c}.json")}
    by_name = {c for c in ("ct", "mr") if series.get("publisher_dataset") == _TOPBRAIN_RECORD
               and name.startswith(f"topcow_{c}_")}
    if by_dictionary and by_name and by_dictionary != by_name:
        return None, None, None, "The label dictionary and the label file name disagree on modality"
    found = by_dictionary or by_name
    scheme = f"topbrain-v1-{found.pop()}" if len(found) == 1 else None
    if scheme is None:
        return None, None, None, "The supplied labels are not a recognised TopBrain v1 label map"
    if scheme != f"topbrain-v1-{modality}":
        return None, None, None, "Supplied labels of one modality are never applied to another"
    if not _same_grid(row, series):
        return None, None, None, "The supplied labels are not on the scan grid"
    if not load:
        return None, scheme, original.get("sha256"), None
    path = collection.path(original["path"])
    if not path.is_file() or file_hash(path) != original.get("sha256"):
        return None, None, None, "The supplied label original is missing or changed"
    values = np.asanyarray(nib.load(path).dataobj)
    if values.shape != tuple(row["data"]["shape"]) or not np.issubdtype(values.dtype, np.integer):
        return None, None, None, "The supplied label original is not on this grid"
    inside = np.isin(values, grouping)
    if not np.array_equal(inside, collection.mask(row)):
        return None, None, None, "The layer differs from its supplied label values"
    return np.where(inside, values, 0).astype(np.uint8), scheme, original["sha256"], None


ROOTED_CLASS_PREFIXES = {"vein": "__vessel_veins:", "artery": "__vessel_arteries:"}
_BONE_KINDS = {"bone", "skull"}


def _same_session(collection, row, other):
    return collection.series(other["source_series"])["timepoint"] == collection.series(row["source_series"])["timepoint"]


def _registered(collection, layer):
    try:
        return np.asarray(collection.patient_affine(layer)).round(9).tolist()
    except ValueError:
        return None


def _context_mask(collection, row, layers, frame, *, check=lambda: None):
    """Union of context layers on this vessel grid: same-grid layers directly, others
    resampled through their registration. Returns (mask or None, used ids, reasons)."""
    union, used, reasons = None, [], []
    shape = tuple(row["data"]["shape"])
    for layer in layers:
        check()
        if _same_grid(row, layer) and layer["source_series"] == row["source_series"]:
            current = collection.mask(layer)
        elif frame is None:
            reasons.append(f"{layer['id']}: this vessel layer has no registered patient frame")
            continue
        else:
            try:
                source = collection.patient_affine(layer)
            except ValueError as error:
                reasons.append(f"{layer['id']}: no registration to this grid ({error})")
                continue
            current = resample(collection.mask(layer).astype(np.uint8), source, shape, frame, order=0) > 0
        union = current if union is None else union | current
        used.append(layer["id"])
    return union, used, reasons


def rooted_vessel_layer(collection, row, *, review=False, detail=None, smooth=True, check=lambda: None, _depth=0):
    """Root-reachable display derivative of one vessel layer, with its receipt.

    Returns ``(display row, receipt, review row or None, class row or None)``.
    The display row keeps the rooted, bridged and level-of-detail-kept voxels of
    the layer's own class (arteries with unclassified vessels, or veins). When
    the layer also holds the other class, those voxels form the class row, drawn
    in that class's colour. The review row (only when ``review`` is set) holds
    the detached and level-of-detail-hidden source voxels. The source layer,
    its labels and the manifest are never changed. ``detail`` is "major" or
    "fine" (see ``VESSEL_DETAILS``); without it the acquisition decides. With
    ``smooth`` the display rows also carry the packed tube field of their thin
    branches under ``DISPLAY_TUBES`` (see ``vessel_display_tubes``): surfaces
    and volume tags draw those branches as round tubes; the mask is unchanged.
    """
    if row["kind"] not in ROOTED_VESSEL_KINDS:
        return row, None, None, None
    from .anatomy_approaches import read_approaches
    from .anatomy_geometry import vascular_cleanup_source
    from .vascular_rooting import ARTERY, UNCLASSIFIED, VEIN, RootingPolicy, root_vessels

    scope = "rooted-review" if review else "rooted"
    series = collection.series(row["source_series"])
    detail = vessel_detail(series["modality"], detail)
    signal = series["data"] if _same_grid(row, series) else None
    score = row if row.get("probability") else vascular_cleanup_source(collection, row)
    probability = score["probability"] if score and score.get("probability") and _same_grid(row, score) else None
    _, scheme, named_sha, named_reason = rooted_named_labels(collection, row, series, load=False)
    named_binding = {"scheme": scheme, "sha256": named_sha} if scheme else None
    try:
        frame = collection.patient_affine(row)
    except ValueError:
        frame = None
    # Saved approaches are capsules in the registered patient frame.
    corridors, corridor_reason = [], None
    try:
        approaches = read_approaches(collection)
        if approaches and frame is None:
            corridor_reason = "Corridors unavailable: this vessel layer has no registered patient frame"
        elif approaches:
            corridors = [{"entry": approach["entry"], "target": approach["target"],
                          "radiusMm": float(approach["radiusMm"])} for approach in approaches]
    except (ValueError, KeyError, OSError) as error:
        corridor_reason = f"Corridors unavailable: {error}"
    tumours = [layer for layer in collection.document["layers"]
               if layer["kind"] in _TUMOUR_KINDS and _same_session(collection, row, layer)]
    bones = [layer for layer in collection.document["layers"] if layer["kind"] in _BONE_KINDS
             and layer["source_series"] == row["source_series"] and _same_grid(row, layer)]
    # A model addition (for example the vesselFM candidates) never joins the
    # named tree it extends: it is shown only where it reaches the shown display
    # of that tree, and only under the calibre rule. An addition may extend
    # another addition (fine branches on the candidates); its roots are then the
    # shown display of the whole chain down to the named tree.
    # Whatever goes wrong with that chain, the addition shows nothing by default.
    parent_rows, parent_issue, parent = None, None, None
    parent_id = row.get("rooted_parent")
    if parent_id is not None:
        parent = next((layer for layer in collection.document["layers"] if layer["id"] == parent_id), None)
        if parent is None:
            parent_issue = f"the tree it extends ({parent_id}) is not in this study"
        elif (parent["kind"] not in ROOTED_VESSEL_KINDS or parent["id"] == row["id"]
                or parent["source_series"] != row["source_series"] or not _same_grid(row, parent)):
            parent_issue = f"the tree it extends ({parent_id}) is not a vessel tree on this grid"
        elif parent.get("rooted_parent") and _depth + 2 >= _PARENT_DEPTH:
            parent_issue = f"the chain of trees it extends is longer than {_PARENT_DEPTH} layers"
        else:
            try:
                parent_rows = _shown_chain(collection, parent, detail, check, _depth + 1)
            except InterruptedError:
                raise
            except Exception as error:
                check()  # A cancellation still cancels.
                parent_issue = f"the tree it extends ({parent_id}) could not be prepared ({type(error).__name__}: {error})"
            else:
                if not any(collection.mask(part).any() for part in parent_rows):
                    parent_issue, parent_rows = f"the tree it extends ({parent_id}) shows nothing", None
    if parent_issue is not None:
        return _detached_addition(collection, row, scope, review, parent_issue, detail=detail)
    if detail == "major" and (row.get("provenance") or {}).get("model") in FINE_BRANCH_MODELS:
        return _detached_addition(collection, row, scope, review,
                                  "these image-derived fine branches are shown at the fine vessel detail", detail=detail,
                                  key="detailHidden")
    policy = RootingPolicy() if detail == "major" else replace(RootingPolicy(), always_mm=RootingPolicy().local_mm)
    from .vessel_display_tubes import DISPLAY_TUBES, TubeDisplay, thin_tubes

    tube_rule = TubeDisplay() if smooth else None
    identity = {
        "version": ROOTED_DISPLAY, "policy": asdict(policy), "rawDataSha256": row["data"]["sha256"],
        "label": row["label_value"], "kind": row["kind"], "affine": row["affine"],
        "signalSha256": signal["sha256"] if signal else None,
        "probabilitySha256": probability["sha256"] if probability else None,
        "named": named_binding, "corridors": corridors,
        "frame": None if frame is None else np.asarray(frame).round(9).tolist(),
        "tumours": [[layer["data"]["sha256"], layer["affine"], layer["source_series"], _registered(collection, layer)]
                    for layer in tumours],
        "bones": sorted(layer["data"]["sha256"] for layer in bones),
        "parent": None if parent_rows is None else [part["data"]["sha256"] for part in parent_rows],
        "tubes": None if tube_rule is None else asdict(tube_rule),
    }
    key = digest(identity)
    if not hasattr(collection, "_rooted_display_cache"):
        collection._rooted_display_cache = {}
    cache = collection._rooted_display_cache
    entry = cache.get(key)
    # The display is also kept beside the study, so reopening does not prepare it again.
    stored = collection.root / "display-cache" / f"{ROOTED_CACHE}-{key}.json"
    if entry is None and stored.is_file():
        try:
            saved = json.loads(stored.read_text())
            if saved["identity"] == json.loads(json.dumps(identity)) and saved["checksum"] == digest(saved["result"]):
                entry = saved["result"]
        except (ValueError, KeyError, TypeError):
            entry = None  # An unreadable cache file is prepared again and replaced.
    try:
        for data in (entry or {}).get("arrays", ()):
            collection.array(data)
    except FileNotFoundError:
        entry = None
    if entry is None:
        mask = np.ascontiguousarray(collection.mask(row, check=check))
        named = None
        if named_binding:
            named, scheme, _, named_reason = rooted_named_labels(collection, row, series)
        tumour, tumour_ids, tumour_reasons = _context_mask(collection, row, tumours, frame, check=check)
        bone, bone_ids, _ = _context_mask(collection, row, bones, frame, check=check)
        tree = vein_tree = wall = None
        own_mask = mask.copy()
        if parent_rows is not None:
            tree = np.zeros(mask.shape, bool)
            vein_tree = np.zeros(mask.shape, bool)
            for part in parent_rows:
                part_mask = collection.mask(part)
                tree |= part_mask
                if part["kind"] in {"vein", "sinus"}:
                    vein_tree |= part_mask
            mask |= tree
        result = root_vessels(
            mask, affine=frame if frame is not None else affine(row["affine"]),
            named=named, scheme=scheme, modality=series["modality"], layer_kind=row["kind"],
            signal=None if signal is None else collection.array(signal),
            probability=None if probability is None else collection.array(probability),
            tumour=tumour, corridors=corridors, bone=bone, root_mask=tree, policy=policy, check=check)
        check()
        if tree is not None:
            # Keep only this layer's own voxels and bridges; the tree is drawn by its own layer.
            for output in (result.shown, result.review, result.lod_hidden):
                output &= ~tree
            result.classes[tree] = 0
            wall = _wall_thickening(result, tree)
            _addition_classes(result, tree, vein_tree)
        details = result.receipt
        if parent_id is not None:
            _addition_receipt(details, result, own_mask, parent, wall)
        details["lod"]["tumourLayers"] = tumour_ids
        details["lod"]["tumourReason"] = ("; ".join(tumour_reasons) if tumour_reasons else
                                          None if tumour_ids else "No tumour layer in this session")
        details["bridged"]["boneLayers"] = bone_ids
        display = result.display
        if wall and wall["overlapVoxels"]:
            # Display-only: the shown branches reach one voxel into the tree, in the class of the part they enter.
            box, overlap = wall["box"], wall["overlap"]
            display[box] |= overlap
            result.classes[box][overlap] = np.where(vein_tree[box][overlap], VEIN, ARTERY)
            del overlap
        own_vein = row["kind"] in {"vein", "sinus"}
        other = (result.classes == ARTERY) | (result.classes == UNCLASSIFIED) if own_vein else result.classes == VEIN
        other &= display
        own = display & ~other
        shown = collection.put_array(own.astype(np.uint8))
        split = collection.put_array(other.astype(np.uint8)) if other.any() else None
        # Thin branches are redrawn as smooth tubes, for display only; each class keeps its own.
        tubes = split_tubes = None
        if tube_rule is not None:
            spacing = np.linalg.norm(affine(row["affine"])[:3, :3], axis=0)
            drawn = {}
            for name, part, rest in (("own", own, other), ("split", other, own)):
                if not part.any():
                    continue
                beside = rest if tree is None else rest | tree
                try:
                    packed, tube_receipt = thin_tubes(part, spacing, context=beside if beside.any() else None,
                                                      rule=tube_rule, check=check)
                except InterruptedError:
                    raise
                except Exception as error:
                    # A display refinement must not cost the connected tree: the branches stay voxel-drawn.
                    check()  # A cancellation still cancels.
                    packed, tube_receipt = None, {"version": tube_rule.version, "failed": f"{type(error).__name__}: {error}"}
                drawn[name] = (None if packed is None else collection.put_array(packed), tube_receipt)
                del packed
            tubes, split_tubes = (drawn.get(name, (None, None))[0] for name in ("own", "split"))
            details["tubes"] = {name: item[1] for name, item in drawn.items()}
        hidden_mask = result.review | result.lod_hidden
        hidden = collection.put_array(hidden_mask.astype(np.uint8)) if hidden_mask.any() else None
        del display, own, other
        roots = details["roots"]
        named_roots = ", ".join(item["name"] for item in roots.get("labels", []))
        source = (f"named roots ({named_roots})" if roots["source"] == "named"
                  else f"{len(roots.get('acceptedRoots', [])):,} unnamed root components (largest, and entering from below)")
        review_count = details["review"]["components"]
        crust = details["review"].get("wallThickening", {}).get("components", 0)
        hidden_text = (f"{review_count - crust:,} detached components"
                       + (f", {crust:,} wall-thickening pieces that hug the tree wall without extending it" if crust else "")
                       + f" ({details['review']['volumeMm3'] / 1000:.2f} mL) and "
                       f"{details['lod']['voxels']:,} thin end-branch voxels are hidden for review. Source anatomy is preserved.")
        if parent_id is not None:
            reason = (f"Attached to the named tree of {parent['label']}: {details['display']['components']:,} of "
                      f"{details['input']['components']:,} pieces shown, {details['bridged']['components']:,} joined "
                      f"across signal-supported gaps; {hidden_text}")
        else:
            reason = (f"Connected tree from {source}. {details['bridged']['components']:,} fragments joined across "
                      f"signal-supported gaps; {hidden_text}")
        if details["classes"].get("note"):
            reason += " " + details["classes"]["note"]
        receipt = {
            "rawDataSha256": row["data"]["sha256"], "sampledDataSha256": shown["sha256"],
            "hiddenVoxels": int(hidden_mask.sum()), "excludedFragments": int(review_count),
            "excludedVolumeMm3": float(details["review"]["volumeMm3"] + details["lod"]["volumeMm3"]),
            "sourcePreserved": True, "reason": reason,
            **({"classSplitSha256": split["sha256"]} if split else {}),
            **({"namedLabelsUnavailable": named_reason} if named_reason else {}),
            **({"corridorsUnavailable": corridor_reason} if corridor_reason else {}),
            "rooting": details,
        }
        entry = {"arrays": [data for data in (shown, split, hidden, tubes, split_tubes) if data], "shown": shown,
                 "split": split, "hidden": hidden, "tubes": tubes, "splitTubes": split_tubes, "receipt": receipt}
        # A temporarily missing label original must not pin a label-less view for the session.
        if not (named_binding and named_reason):
            cache[key] = entry
            try:
                stored.parent.mkdir(exist_ok=True)
                atomic_json(stored, {"identity": identity, "result": entry, "checksum": digest(entry)})
            except OSError:
                pass  # A study that cannot be written to keeps the display for this session only.
    elif key not in cache:
        cache[key] = entry
    receipt = {**entry["receipt"], "scope": scope, "detail": detail}
    own_class = "vein" if row["kind"] in {"vein", "sinus"} else "artery"
    other_class = "artery" if own_class == "vein" else "vein"
    display = {**deepcopy(row), "data": entry["shown"], "label_value": 1, "color": VESSEL_CLASS_COLORS[own_class],
               "edits": [{"display_only": ROOTED_DISPLAY, "scope": scope}]}
    display.pop("probability", None)
    display.pop(DISPLAY_TUBES, None)
    if entry.get("tubes"):
        display[DISPLAY_TUBES] = entry["tubes"]
    class_row = None
    if entry["split"] is not None:
        class_row = {**deepcopy(row), "data": entry["split"], "label_value": 1, "kind": other_class,
                     "color": VESSEL_CLASS_COLORS[other_class],
                     "edits": [{"display_only": ROOTED_DISPLAY, "class": other_class}]}
        class_row.pop("probability", None)
        class_row.pop(DISPLAY_TUBES, None)
        if entry.get("splitTubes"):
            class_row[DISPLAY_TUBES] = entry["splitTubes"]
    hidden_row = None
    if review and entry["hidden"] is not None:
        hidden_row = {**deepcopy(row), "data": entry["hidden"], "label_value": 1, "color": ROOTED_REVIEW_COLOR,
                      "edits": [{"display_only": ROOTED_DISPLAY, "review": True}]}
        hidden_row.pop("probability", None)
        hidden_row.pop(DISPLAY_TUBES, None)
    return display, receipt, hidden_row, class_row


def _shown_chain(collection, layer, detail, check, depth):
    """Display rows of a layer and of every tree it extends, nearest first."""
    display, receipt, _, class_row = rooted_vessel_layer(collection, layer, detail=detail, check=check, _depth=depth)
    if (receipt or {}).get("parentUnavailable"):
        raise ValueError(receipt["parentUnavailable"])
    rows = [display] + ([class_row] if class_row else [])
    above = layer.get("rooted_parent")
    if above is not None:
        parent = next((row for row in collection.document["layers"] if row["id"] == above), None)
        if parent is None:
            raise ValueError(f"the tree it extends ({above}) is not in this study")
        rows += _shown_chain(collection, parent, detail, check, depth + 1)
    return rows


def _detached_addition(collection, row, scope, review, issue, *, detail, key="parentUnavailable"):
    """An addition that cannot be shown (its tree is unavailable, or the detail hides it): nothing by default,
    everything under review."""
    mask = collection.mask(row)
    shown = collection.put_array(np.zeros(mask.shape, np.uint8))
    hidden = collection.put_array(mask.astype(np.uint8)) if mask.any() else None
    receipt = {"rawDataSha256": row["data"]["sha256"], "sampledDataSha256": shown["sha256"], "scope": scope,
               "hiddenVoxels": int(mask.sum()), "excludedFragments": int(ndi.label(mask, CONNECTIVITY)[1]),
               "sourcePreserved": True, "detail": detail, key: issue,
               "reason": f"Not shown: {issue}. Its voxels are available under review. Source anatomy is preserved."}
    display = {**deepcopy(row), "data": shown, "label_value": 1, "color": VESSEL_CLASS_COLORS["artery"],
               "edits": [{"display_only": ROOTED_DISPLAY, "scope": scope}]}
    display.pop("probability", None)
    hidden_row = None
    if review and hidden is not None:
        hidden_row = {**deepcopy(row), "data": hidden, "label_value": 1, "color": ROOTED_REVIEW_COLOR,
                      "edits": [{"display_only": ROOTED_DISPLAY, "review": True}]}
        hidden_row.pop("probability", None)
    return display, receipt, hidden_row, None


def _addition_classes(result, tree, vein_tree):
    """Each shown piece of an addition takes the class of the tree part it joins."""
    from .vascular_rooting import UNCLASSIFIED, VEIN

    shown = result.shown | result.bridge
    if not shown.any():
        return
    pieces, count = ndi.label(shown, CONNECTIVITY)
    near_vein = ndi.binary_dilation(vein_tree, CONNECTIVITY) & shown
    near_artery = ndi.binary_dilation(tree & ~vein_tree, CONNECTIVITY) & shown
    vein = np.bincount(pieces[near_vein], minlength=count + 1)
    artery = np.bincount(pieces[near_artery], minlength=count + 1)
    venous = vein > artery
    venous[0] = False
    result.classes[shown] = UNCLASSIFIED
    result.classes[venous[pieces] & shown] = VEIN


def _wall_thickening(result, tree):
    """Move addition pieces that only thicken the tree wall to review; return them and the display overlap,
    as masks over ``box``, with their counts.

    A face-connected piece is a branch only when some voxel lies outside the
    tree dilated twice (26-connected). The overlap is the tree voxels
    face-adjacent to the shown branches, so their surface is drawn into the tree.
    """
    from scipy.sparse.csgraph import connected_components

    display = result.shown | result.bridge
    found = ndi.find_objects(display.view(np.uint8))
    wall = {"box": None, "crust": None, "overlap": None, "components": 0, "voxels": 0, "bridgeVoxels": 0, "overlapVoxels": 0}
    if not found:
        return wall
    box = tuple(slice(max(0, s.start - 3), min(n, s.stop + 3)) for s, n in zip(found[0], tree.shape))
    local = display[box]
    del display
    # Boolean propagation from the voxels beyond the wall; a label volume would not fit large TOF grids.
    seeds = local & ~ndi.binary_dilation(tree[box], CONNECTIVITY, iterations=2)
    branch = ndi.binary_propagation(seeds, ndi.generate_binary_structure(3, 1), mask=local)
    del seeds
    crust = local & ~branch
    points = np.argwhere(crust)
    if len(points):
        pairs = cKDTree(points).query_pairs(1.01, output_type="ndarray")
        graph = sparse.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2)
        wall.update(components=int(connected_components(graph, directed=False)[0]), crust=crust,
                    voxels=int((result.shown[box] & crust).sum()), bridgeVoxels=int((result.bridge[box] & crust).sum()))
        result.review[box] |= result.shown[box] & crust
        result.shown[box] &= ~crust
        result.bridge[box] &= ~crust
    del points
    wall["overlap"] = ndi.binary_dilation(branch) & tree[box]
    wall.update(box=box, overlapVoxels=int(wall["overlap"].sum()))
    return wall


def _addition_receipt(details, result, own, parent, wall=None):
    """Counts of the addition itself; the union with its tree is kept apart."""
    details["union"] = {key: details[key] for key in ("input", "attached", "display")}
    shown = result.shown | result.bridge
    voxel = details["input"]["volumeMm3"] / max(details["input"]["voxels"], 1)
    details["input"] = {"voxels": int(own.sum()), "volumeMm3": round(float(own.sum() * voxel), 4),
                        "components": int(ndi.label(own, CONNECTIVITY)[1])}
    details["display"] = {**details["display"], "voxels": int(shown.sum()), "sourceVoxels": int(result.shown.sum()),
                          "components": int(ndi.label(shown, CONNECTIVITY)[1])}
    details["attached"] = {"components": details["display"]["components"], "voxels": int(result.shown.sum()),
                           "volumeMm3": round(float(result.shown.sum() * voxel), 4)}
    wall = wall or {"crust": None, "components": 0, "voxels": 0, "bridgeVoxels": 0, "overlapVoxels": 0}
    rest = result.review
    if wall["crust"] is not None:
        rest = result.review.copy()
        rest[wall["box"]] &= ~wall["crust"]
    detached = int(ndi.label(rest, CONNECTIVITY)[1])
    del rest
    crust_voxels = wall["voxels"]
    details["review"] = {**details["review"], "components": detached + wall["components"],
                         "detachedComponents": detached,
                         "voxels": int(result.review.sum()), "volumeMm3": round(float(result.review.sum() * voxel), 4),
                         "wallThickening": {"components": wall["components"], "voxels": crust_voxels,
                                            "volumeMm3": round(float(crust_voxels * voxel), 4),
                                            "bridgeVoxelsDropped": wall["bridgeVoxels"],
                                            "reason": "Hugs the tree wall without extending it: no voxel lies more than "
                                                      "two voxels outside the tree."}}
    details["display"]["overlapVoxels"] = wall["overlapVoxels"]
    details["display"]["overlapRule"] = ("Display only: tree voxels face-adjacent to the shown branches, in the class of "
                                         "the tree part they belong to, so the branch surface joins the tree surface.")
    details["lod"] = {**details["lod"], "voxels": int(result.lod_hidden.sum())}
    details["classes"] = {**details["classes"], "vein": int((result.classes == 2).sum()),
                          "unclassified": int((result.classes == 3).sum())}
    details["rootedOn"] = {"layer": parent["id"], "label": parent["label"],
                           "rule": "shown only where it reaches that tree's shown display, directly or by a bridge of at most 3 mm"}


def _weak_components(points, shape, check):
    """Keep sparse weak-signal components off the full native-sized label grid."""
    flat = np.ravel_multi_index(points.T, shape)
    strides = np.cumprod((1, *shape[:0:-1]))[::-1]
    starts, stops = [], []
    for step in np.argwhere(CONNECTIVITY) - 1:
        if tuple(step) <= (0, 0, 0):
            continue
        check()
        valid = np.ones(len(points), dtype=bool)
        for axis in np.flatnonzero(step):
            limit = shape[axis] - 1 if step[axis] > 0 else 0
            valid &= points[:, axis] != limit
        source = np.flatnonzero(valid)
        target = flat[source] + step @ strides
        index = np.searchsorted(flat, target).clip(max=len(flat) - 1)
        adjacent = flat[index] == target
        starts.append(source[adjacent].astype(np.int32))
        stops.append(index[adjacent].astype(np.int32))
    row, col = np.concatenate(starts), np.concatenate(stops)
    graph = sparse.coo_matrix((np.ones(len(row), dtype=bool), (row, col)), shape=(len(flat), len(flat)))
    return sparse.csgraph.connected_components(graph, directed=False)[1]


def attached_vessel_uncertainty(mask, signal, scores, native_affine, *, check=lambda: None):
    """Locate thick, weak-signal regions attached to a brighter TOF estimate.

    Returned C-order source cells are an inspection footprint. A low-signal
    lumen is possible, so these findings carry no exclusion or correction.
    """
    mask = np.asarray(mask)
    signal = np.asarray(signal)
    scores = np.asarray(scores)
    if (mask.dtype != np.bool_ or mask.ndim != 3 or mask.shape != signal.shape
            or signal.dtype.kind not in "fui" or scores.shape != mask.shape
            or scores.dtype.kind != "f"):
        raise ValueError("Expected a native binary prediction, TOF signal and model scores")
    frame = affine(native_affine)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    axes = frame[:3, :3] / spacing
    limits = {
        "nativeSignalPercentile": _ATTACHED_DARK_PERCENTILE,
        "minimumCoreRadiusMm": _ATTACHED_CORE_RADIUS_MM,
        "minimumFootprintMm3": _ATTACHED_MINIMUM_VOLUME_MM3,
        "contextRadiusMm": _ATTACHED_CONTEXT_MM,
        "minimumNeighborContrastForegroundIqr": _ATTACHED_ANCHOR_CONTRAST_IQR,
        "maximumWeakForegroundVoxels": _MAX_ATTACHED_WEAK_CELLS,
        "maximumFindings": _MAX_ATTACHED_FINDINGS,
    }
    result = {
        "algorithm": ATTACHED_REVIEW,
        "status": "unavailable",
        "sourceForegroundVoxels": int(np.count_nonzero(mask)),
        "limits": limits,
        "findings": [],
    }
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        return {}, {**result, "reason": "Native TOF lattice is sheared"}
    if spacing.max() > _ATTACHED_CORE_RADIUS_MM / 1.5:
        return {}, {**result, "reason": "Native lattice undersamples the physical screen"}
    foreground = result["sourceForegroundVoxels"]
    if not 100 <= foreground <= _MAX_EXTERIOR_FOREGROUND:
        return {}, {**result, "reason": "Source foreground is too sparse or exceeds review workspace"}

    check()
    points = np.column_stack(np.nonzero(mask))
    image_values = signal[tuple(points.T)]
    if not np.isfinite(image_values).all():
        return {}, {**result, "reason": "Native foreground signal has missing measurements"}
    dark, lo, hi = np.percentile(image_values, [_ATTACHED_DARK_PERCENTILE, 25, 75])
    scale = float(hi - lo)
    if scale <= np.finfo(np.float32).eps * max(abs(float(lo)), abs(float(hi)), 1.0):
        return {}, {**result, "reason": "Native foreground signal has insufficient contrast"}
    weak_cells = image_values <= dark
    if np.count_nonzero(weak_cells) > _MAX_ATTACHED_WEAK_CELLS:
        return {}, {**result, "reason": "Weak native foreground exceeds review workspace"}
    weak = points[weak_cells]
    del points, image_values
    labels = _weak_components(weak, mask.shape, check)
    ordered = np.argsort(labels, kind="stable")
    sizes = np.bincount(labels)
    offsets = np.r_[0, sizes.cumsum()]
    voxel_volume = float(abs(np.linalg.det(frame[:3, :3])))
    component_ids = np.flatnonzero(sizes * voxel_volume >= _ATTACHED_MINIMUM_VOLUME_MM3)
    component_ids = component_ids[np.argsort(-sizes[component_ids], kind="stable")]
    spacing = np.asarray(spacing)
    lattice_shape = np.array(mask.shape)
    footprints = {}
    review_limited = 0
    for label in component_ids:
        check()
        cells = weak[ordered[offsets[label]:offsets[label + 1]]]
        lower, upper = cells.min(axis=0), cells.max(axis=0)
        local_shape = upper - lower + 1
        if np.any((local_shape + 1) * spacing < 2 * _ATTACHED_CORE_RADIUS_MM):
            continue
        if np.prod(local_shape + 2) > _MAX_NEIGHBORHOOD_CELLS:
            review_limited += 1
            continue
        group = np.zeros(local_shape, dtype=bool)
        group[tuple((cells - lower).T)] = True
        distance = ndi.distance_transform_edt(np.pad(group, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
        core = distance >= _ATTACHED_CORE_RADIUS_MM
        if not core.any():
            continue
        clearance = np.minimum(lower + .5, lattice_shape - upper - .5) * spacing
        if np.any(clearance <= _ATTACHED_CONTEXT_MM):
            continue
        start, context = _bounds(cells, spacing, mask.shape, _ATTACHED_CONTEXT_MM)
        source = signal[context]
        if source.size > _MAX_NEIGHBORHOOD_CELLS or not np.isfinite(source).all():
            review_limited += 1
            continue
        local_mask = mask[context]
        weak_region = np.zeros(source.shape, bool)
        weak_region[tuple((cells - start).T)] = True
        # A physical opening localizes the thick portion and leaves
        # peripheral low-signal filaments outside the inspection footprint.
        opened = group & (ndi.distance_transform_edt(~core, sampling=spacing) < _ATTACHED_CORE_RADIUS_MM)
        stations = np.argwhere(opened) + lower
        volume = len(stations) * voxel_volume
        if volume < _ATTACHED_MINIMUM_VOLUME_MM3:
            continue
        station_in_context = np.ones(source.shape, dtype=bool)
        station_in_context[tuple((stations - start).T)] = False
        ring = ndi.distance_transform_edt(station_in_context, sampling=spacing) <= _ATTACHED_CONTEXT_MM
        touching = ndi.binary_dilation(weak_region) & ring & ~weak_region & local_mask & (source > dark)
        if not touching.any():
            continue
        location = tuple(stations.T)
        measured = signal[location]
        candidate90 = float(np.percentile(measured, 90))
        attachment75 = float(np.percentile(source[touching], 75))
        contrast = (attachment75 - candidate90) / scale
        if contrast < _ATTACHED_ANCHOR_CONTRAST_IQR:
            continue
        values = scores[location]
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            review_limited += 1
            continue
        check()
        background = source[~local_mask & ring]
        center = np.array(np.unravel_index(distance.argmax(), distance.shape)) + lower
        footprint = np.ravel_multi_index(location, mask.shape)
        finding_id = f"attached-{int(footprint[0])}"
        footprints[finding_id] = footprint
        result["findings"].append({
            "id": finding_id,
            "centerIjk": center.tolist(),
            "centerRasMm": world(frame, center).tolist(),
            "boundsIjk": [stations.min(axis=0).tolist(), (stations.max(axis=0) + 1).tolist()],
            "voxels": len(stations),
            "volumeMm3": volume,
            "peakCoreRadiusMm": float(distance.max()),
            "foregroundContactVoxels": int(np.count_nonzero(touching)),
            "signalP90": candidate90,
            "backgroundSignalMedian": float(np.median(background)) if background.size else None,
            "attachmentSignalP75": attachment75,
            "attachmentContrastForegroundIqr": contrast,
            "uncalibratedModelScoreMedian": float(np.median(values)),
        })
        if len(footprints) == _MAX_ATTACHED_FINDINGS:
            break
    result.update(
        status="partial" if review_limited or len(footprints) == _MAX_ATTACHED_FINDINGS else "complete",
        foregroundDarkSignalCeiling=float(dark),
        foregroundSignalIqr=scale,
        groupsSkippedForMissingEvidenceOrWorkspace=review_limited,
    )
    result["findings"].sort(key=lambda finding: -finding["voxels"])
    return footprints, result


def verify_native_tof_review_source(collection, series, check):
    original = collection.path(series["original"]["path"])
    status = original.stat()
    identity = (str(original), status.st_dev, status.st_ino, status.st_size,
                status.st_mtime_ns, status.st_ctime_ns, series["original"]["sha256"])
    if identity not in collection._verified:
        if file_hash(original, check=check) != series["original"]["sha256"]:
            raise ValueError("Original native TOF checksum differs")
        collection._verified.add(identity)


def review_attached_tof_layer(collection, row, *, check=lambda: None):
    """Bind read-only attachment findings to the native acquisition and scores."""
    from .anatomy_model_registry import source_volume

    provenance = row.get("provenance", {})
    series = collection.series(row["source_series"])
    if (row["kind"] != "vessels" or provenance.get("model") != "vesselfm"
            or row.get("edits") or not row.get("probability")
            or series["modality"] != "TOF" or not source_volume(series)):
        return {}, {"status": "unavailable", "reason": "Select an unedited native TOF vesselFM estimate"}
    if (row["affine"] != series["affine"] or row["data"]["shape"] != series["data"]["shape"]
            or row["probability"]["shape"] != series["data"]["shape"]
            or provenance.get("source") != series["original"]["sha256"]):
        return {}, {"status": "unavailable", "reason": "Source identity or native lattice could not be verified"}
    try:
        collection.require_native_geometry(series)
        collection.require_native_geometry(row)
    except ValueError:
        return {}, {"status": "unavailable", "reason": "The source lattice requires geometry review"}
    check()
    verify_native_tof_review_source(collection, series, check)
    identity = {
        "algorithm": ATTACHED_REVIEW,
        "sourceSeries": series["id"],
        "sourceLayer": row["id"],
        "sourceSha256": series["original"]["sha256"],
        "sourceSignalSha256": series["data"]["sha256"],
        "sourceDataSha256": row["data"]["sha256"],
        "sourceScoreSha256": row["probability"]["sha256"],
        "sourceShape": series["data"]["shape"],
        "sourceAffineRasMm": series["affine"],
        "sourceLabel": row["label_value"],
    }
    # Resolve and verify all array members even for a previously seen identity.
    source_array = collection.array(series["data"])
    predicate_array = collection.array(row["data"])
    scores = collection.array(row["probability"])
    check()
    review_id = digest(identity)
    cached = getattr(collection, "_attached_tof_review_cache", None)
    if cached is not None and cached[0] == review_id:
        return cached[1], cached[2]
    footprints, analysis = attached_vessel_uncertainty(
        predicate_array == row["label_value"], source_array, scores, series["affine"], check=check,
    )
    receipt = {**analysis, **identity, "reviewId": review_id}
    collection._attached_tof_review_cache = (review_id, footprints, receipt)
    return footprints, receipt


def _attached_review_point(point, shape, cells):
    if (not isinstance(point, list) or len(point) != 3
            or any(type(value) is not int or not 0 <= value < extent for value, extent in zip(point, shape))
            or np.ravel_multi_index(point, shape) not in cells):
        raise ValueError("Select an amber voxel inside the reviewed region")
    return np.asarray(point)


def attached_tof_sections(collection, row, cells, finding, *, center_ijk=None, padding_mm=2., check=lambda: None):
    """Return small, paired source and review overlays through the native core."""
    series = collection.series(row["source_series"])
    signal = collection.array(series["data"], check=check)
    labels = collection.array(row["data"], check=check)
    spacing = np.linalg.norm(affine(series["affine"])[:3, :3], axis=0)
    center = _attached_review_point(finding["centerIjk"] if center_ijk is None else center_ijk, signal.shape, cells)
    lower, upper = np.asarray(finding["boundsIjk"])
    half_extent = np.clip(np.maximum(center - lower, upper - 1 - center)
                          + np.ceil(padding_mm / spacing), 1, 48).astype(int)
    lo = np.maximum(0, center - half_extent)
    hi = np.minimum(np.shape(signal), center + half_extent + 1)

    planes, measurements = [], []
    for axis, index in enumerate(center):
        check()
        section = [slice(a, b) for a, b in zip(lo, hi)]
        section[axis] = int(index)
        section = tuple(section)
        gray = signal[section].T
        estimate = labels[section].T == row["label_value"]
        outline = estimate & ~ndi.binary_erosion(estimate, border_value=1)
        u, v = [dimension for dimension in range(3) if dimension != axis]
        ijk = np.repeat(center[:, None], gray.size, axis=1)
        uu, vv = np.meshgrid(np.arange(lo[u], hi[u]), np.arange(lo[v], hi[v]))
        ijk[u], ijk[v] = uu.ravel(), vv.ravel()
        inside = np.isin(np.ravel_multi_index(ijk, signal.shape), cells).reshape(gray.shape)
        overlay = np.where(inside, 2, outline).astype(np.uint8)
        measurements.append(gray[np.isfinite(gray)])
        planes.append((axis, index, [spacing[v], spacing[u]], gray, overlay))
    available = np.concatenate(measurements)
    low, high = map(float, np.percentile(available, [1, 99]))
    scale = max(high - low, np.finfo(np.float32).eps * max(abs(high), 1.0))
    sections = []
    for axis, index, pixel_spacing, gray, overlay in planes:
        normalized = np.nan_to_num((gray.astype(float) - low) / scale, nan=0, posinf=0, neginf=0)
        pixels = np.clip(normalized * 255, 0, 255).astype(np.uint8)
        sections.append({
            "axis": axis, "index": int(index), "shape": list(gray.shape), "spacingMm": pixel_spacing,
            "startIjk": [int(index) if d == axis else int(lo[d]) for d in range(3)],
            "image": b64encode(pixels.tobytes()).decode("ascii"),
            "overlay": b64encode(overlay.tobytes()).decode("ascii"),
        })
    return {"window": [low, high], "sections": sections}


def erase_attached_tof(collection, row, mask, *, review_id, finding_id, points, radius_mm, check):
    """Apply one explicit native section pick within its source-bound footprint."""
    if (not np.isfinite(radius_mm) or not .25 <= radius_mm <= 4
            or not isinstance(points, list) or len(points) != 1):
        raise ValueError("Select one native review voxel and a radius from 0.25 to 4 mm")
    footprints, receipt = review_attached_tof_layer(collection, row, check=check)
    if type(finding_id) is not str or finding_id not in footprints or review_id != receipt["reviewId"]:
        raise ValueError("Attached vessel review changed; refresh the findings")
    cells = footprints[finding_id]
    center = _attached_review_point(points[0], mask.shape, cells)
    positions = np.column_stack(np.unravel_index(cells, mask.shape))
    delta = (positions - center) @ affine(row["affine"])[:3, :3].T
    erase = cells[np.einsum("ij,ij->i", delta, delta) <= radius_mm ** 2]
    erase = erase[mask.ravel()[erase]]
    if not len(erase):
        raise ValueError("The selected region is already erased in this preview")
    check()
    after = mask.copy()
    after.ravel()[erase] = False
    return after, {"operation": ATTACHED_ERASE, "reviewId": review_id, "findingId": finding_id,
                   "centerIjk": center.tolist(), "radiusMm": radius_mm, "voxels": len(erase),
                   "sourceDataSha256": receipt["sourceDataSha256"], "sourceScoreSha256": receipt["sourceScoreSha256"],
                   "sourceSignalSha256": receipt["sourceSignalSha256"]}


def classify_components(mask, signal, scores, native_affine, *, policy=ComponentPolicy()):
    """Classify whole 26-connected components without changing the prediction.

    This is a review proposal. In particular, an isolated bright fragment may
    be a distal vessel or extracranial anatomy and stays available. The rule
    abstains near the image extent and on low-information scans.
    """
    mask = np.asarray(mask)
    signal = np.asarray(signal)
    if mask.dtype != np.bool_ or mask.ndim != 3 or mask.shape != signal.shape:
        raise ValueError("Expected a binary prediction and matched native TOF signal")
    if scores is not None and np.shape(scores) != mask.shape:
        raise ValueError("Model scores must match the native TOF grid")
    frame = affine(native_affine)
    spacing = np.linalg.norm(frame[:3, :3], axis=0)
    axes = frame[:3, :3] / spacing
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5):
        raise ValueError("Component measurements require an orthogonal native TOF grid")
    voxel_volume = float(abs(np.linalg.det(frame[:3, :3])))
    points = np.column_stack(np.nonzero(mask))
    partition = np.zeros(mask.shape, dtype=np.uint8)
    if not len(points):
        raise ValueError("Empty vessel prediction")

    start, slices = _bounds(points, spacing, mask.shape, 0)
    labels, count = ndi.label(mask[slices], structure=CONNECTIVITY)
    memberships = labels[tuple((points - start).T)]
    del labels
    permutation = np.argsort(memberships, kind="stable")
    sizes = np.bincount(memberships, minlength=count + 1)
    groups = np.split(permutation, np.cumsum(sizes[1:-1]))
    network = sizes[memberships] * voxel_volume >= policy.small_volume_mm3
    vessels = signal[tuple(points[network].T)]
    vessels = vessels[np.isfinite(vessels)]
    have_signal_reference = len(vessels) >= 100 and np.ptp(vessels) > 0
    low, very_low = np.percentile(vessels, [
        policy.baseline_signal_percentile, policy.very_low_signal_percentile
    ]) if have_signal_reference else (0, 0)
    coordinates = world(frame, points)
    tree = cKDTree(coordinates[network]) if network.any() else None
    head, head_evidence = _head_support(signal, frame, lambda: None)
    outside_head = None
    if head is not None:
        support, _, stride = head
        lattice = np.minimum((points + stride // 2) // stride, np.array(support.shape) - 1)
        outside_head = ~support[tuple(lattice.T)]
    records = []
    for group in groups:
        cells = points[group]
        extent = (np.ptp(cells, axis=0) + 1) * spacing
        volume = len(cells) * voxel_volume
        source_id = int(np.ravel_multi_index(cells[0], mask.shape))
        component_id = f"component-{source_id}"
        state = RETAINED if volume >= policy.small_volume_mm3 else UNCERTAIN
        reason = "retained prediction" if state == RETAINED else "detached component needs review"
        edge = bool(np.any(
            np.minimum((cells + 0.5) * spacing, (np.array(mask.shape) - 0.5 - cells) * spacing)
            < policy.outer_ring_mm
        ))
        ratio = float(extent.max() / extent.min())
        evidence = None
        score = None
        separation = None
        wholly_outside = outside_head is not None and bool(outside_head[group].all())
        if wholly_outside:
            signal_values = signal[tuple(cells.T)]
            if (not edge and np.isfinite(signal_values).all()
                    and float(signal_values.max()) <= head_evidence["threshold"]):
                state = HIDDEN
                reason = "separate component wholly in dark space beyond the native head envelope"
            else:
                state = UNCERTAIN
                reason = "extracranial or boundary finding needs review"
        if (state == UNCERTAIN and not wholly_outside and volume <= policy.compact_volume_mm3
                and ratio <= policy.compact_axis_ratio):
            evidence = _source_contrast(cells, signal, mask, spacing, policy)
            separation = float(tree.query(coordinates[group])[0].min()) if tree is not None else None
            if scores is not None:
                values = scores[tuple(cells.T)]
                if np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)):
                    score = float(np.median(values))
            if (
                not edge
                and have_signal_reference
                and separation is not None
                and separation >= policy.isolation_mm
                and evidence is not None
                and evidence["foreground_peak"] <= low
                and evidence["peak_contrast_iqr"] <= policy.maximum_peak_contrast
                and (
                    (score is not None and score <= policy.score_median_ceiling)
                    or evidence["foreground_peak"] <= very_low
                )
            ):
                state = HIDDEN
                reason = "isolated compact finding with weak native TOF support"
        partition[tuple(cells.T)] = state
        records.append({
            "id": component_id,
            "source_anchor_cell": source_id,
            "state": state,
            "proposed_state": state,
            "reason": reason,
            "voxels": len(cells),
            "volume_mm3": volume,
            "bounds_ijk": [cells.min(axis=0).tolist(), (cells.max(axis=0) + 1).tolist()],
            "centroid_ras_mm": coordinates[group].mean(axis=0).tolist(),
            "axis_extent_mm": extent.tolist(),
            "near_source_extent": edge,
            "outside_native_head_envelope": wholly_outside,
            "separation_from_larger_candidate_mm": separation,
            "model_score_median": score,
            "source_contrast": evidence,
        })
    counts = {
        state: sum(row["voxels"] for row in records if row["state"] == state)
        for state in (RETAINED, UNCERTAIN, HIDDEN)
    }
    return partition, {
        "policy": asdict(policy),
        "reference_signal_available": bool(have_signal_reference),
        "reference_low_signal": float(low) if have_signal_reference else None,
        "reference_very_low_signal": float(very_low) if have_signal_reference else None,
        "native_head_envelope": head_evidence,
        "component_count": count,
        "foreground_voxels": len(points),
        "retained_voxels": int(counts[RETAINED]),
        "uncertain_voxels": int(counts[UNCERTAIN]),
        "hidden_voxels": int(counts[HIDDEN]),
        "hidden_components": sum(row["state"] == HIDDEN for row in records),
        "new_positive_voxels": 0,
        "geometry": "native TOF, whole-component visibility only",
        "source_prediction_modified": False,
        "scores_calibrated": False,
        "mechanics_admitted": False,
        "components": records,
    }


def review_component_visibility(partition, report, *, restore=(), exclude=()):
    """Move reviewed whole components between uncertain and hidden visibility."""
    if set(restore) & set(exclude):
        raise ValueError("A source component has conflicting review decisions")
    result = np.array(partition, copy=True)
    analysis = {**report, "components": [row.copy() for row in report["components"]]}
    components = {row["id"]: row for row in analysis["components"]}
    actions = [(name, HIDDEN, UNCERTAIN) for name in sorted(set(restore))]
    actions.extend((name, UNCERTAIN, HIDDEN) for name in sorted(set(exclude)))
    for name, previous, current in actions:
        row = components.get(name)
        if row is None or row["state"] != previous:
            raise ValueError(f"Unknown source component or incompatible visibility state: {name}")
        start, stop = np.asarray(row["bounds_ijk"], dtype=int)
        anchor = np.asarray(np.unravel_index(row["source_anchor_cell"], result.shape))
        if (np.any(start < 0) or np.any(stop > result.shape)
                or np.any(anchor < start) or np.any(anchor >= stop)):
            raise ValueError(f"Invalid source component bounds: {name}")
        region = result[tuple(slice(a, b) for a, b in zip(start, stop))]
        labels, _ = ndi.label(region == previous, structure=CONNECTIVITY)
        selected = labels[tuple(anchor - start)]
        if not selected or np.count_nonzero(labels == selected) != row["voxels"]:
            raise ValueError(f"Source component differs from the review receipt: {name}")
        region[labels == selected] = current
        row.update(state=current, reason="explicit component review")
        direction = 1 if current == HIDDEN else -1
        analysis["hidden_voxels"] += direction * row["voxels"]
        analysis["uncertain_voxels"] -= direction * row["voxels"]
        analysis["hidden_components"] += direction
    return result, analysis


def separate_reviewed_components(mask, components):
    """Return two disjoint, exhaustive groups of whole native components."""
    foreground = np.asarray(mask)
    if foreground.ndim != 3 or foreground.dtype != np.bool_ or not components:
        raise ValueError("A native binary foreground and reviewed components are required")
    cells = np.flatnonzero(foreground)
    labels, count = ndi.label(foreground, structure=CONNECTIVITY)
    groups = labels.ravel()[cells]
    sizes = np.bincount(groups, minlength=count + 1)
    anchors = np.full(count + 1, foreground.size, dtype=np.int64)
    np.minimum.at(anchors, groups, cells)
    selected = np.zeros(count + 1, dtype=bool)
    for anchor, size in components.items():
        if type(anchor) is not int or type(size) is not int or not 0 <= anchor < foreground.size:
            raise ValueError("Invalid reviewed component anchor or voxel count")
        label = labels.ravel()[anchor]
        if label == 0 or anchors[label] != anchor or sizes[label] != size:
            raise ValueError(f"Whole native component differs from the reviewed selection: {anchor}")
        selected[label] = True
    compartment = selected[labels]
    main = foreground & ~compartment
    if not np.any(main):
        raise ValueError("The separation leaves no main network")
    return main, compartment
