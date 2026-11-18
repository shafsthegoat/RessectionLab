"""Sample native labels in a target-centred RAS frame for visual rehearsal."""

import json
from itertools import product

import numpy as np

from .anatomy_approaches import FILENAME, _read_plan
from .anatomy_evidence import VASCULAR_KINDS, inference_bounds, is_dural_proxy, vascular_evidence_supported
from .anatomy_operative_intensity import operative_intensity
from .anatomy_planning import native_target_binding
from .anatomy_siam_display import supported_tissue_layer
from .native_consequence_evidence import _source_class
from .patient_anatomy import file_hash, world


MATERIAL_BITS = {"bone": 1, "dura": 2, "brain": 4, "tumor": 8, "vessels": 16}


def _saved_route(collection, args):
    snapshot = collection.identity()
    if args.get("snapshotId") != snapshot:
        raise ValueError("Anatomy changed; reload the patient")
    routes, _, active, selection, revision = _read_plan(collection)
    route = next((item for item in routes if item["id"] == args.get("approachId")), None)
    if route is None or route["assessmentStatus"] != "current" or route["sourceSnapshotId"] != snapshot:
        raise ValueError("The operative field requires a currently assessed saved approach")
    if route["parameterHash"] != args.get("parameterHash"):
        raise ValueError("The saved approach parameters changed")
    target = route.get("targetLayerId")
    if not target or target != args.get("targetLayerId") or active != target:
        raise ValueError("Select the saved approach for the active target layer")
    row = collection.layer(target)
    binding = native_target_binding(collection, row)
    # Old role=target routes predate explicit selection. Their assessment is
    # already bound to this exact snapshot and names the target independently.
    route_binding = route.get("targetCandidateBinding")
    if (route["measurements"].get("targetLayerId") != target
            or (selection is not None and selection["candidateBinding"] != binding)
            or (route_binding is not None and route_binding != binding)
            or (route_binding is None and row["role"] != "target")
            or ("candidateBinding" in args and args["candidateBinding"] != binding)):
        raise ValueError("The saved target evidence binding changed")
    return route, binding, revision


def _outward_exit(collection, source, target, direction):
    """Last positive ray depth inside this aligned acquisition, if intersected."""
    inverse = np.linalg.inv(collection.patient_affine(source))
    point, ray = world(inverse, target), inverse[:3, :3] @ direction
    lower, upper = -.5, np.asarray(source["data"]["shape"]) - .5
    moving = np.abs(ray) > 1.e-10
    if np.any(~moving & ((point < lower) | (point >= upper))):
        return 0.
    endpoints = np.stack([(lower - point[moving]) / ray[moving], (upper[moving] - point[moving]) / ray[moving]])
    near, far = np.max(endpoints.min(axis=0)), np.min(endpoints.max(axis=0))
    return float(far) if far >= max(0., near) else 0.


def _field_grid(collection, route):
    target = np.asarray(route["target"])
    outward = np.asarray(route["entry"]) - target
    length = float(np.linalg.norm(outward))
    z = outward / length
    anchor = np.eye(3)[np.argmin(np.abs(z))]
    x = anchor - (anchor @ z) * z
    x /= np.linalg.norm(x)
    basis = np.stack([x, np.cross(z, x), z])

    # A saved entry may lie on cortex. Include the outward ray through the
    # acquired fields to capture skull beyond it, with a bounded display depth.
    exit_depth = length
    for source in collection.document["series"]:
        try:
            exit_depth = max(exit_depth, _outward_exit(collection, source, target, z))
        except (ValueError, OSError):
            continue  # Unaligned acquisitions cannot extend the display field.
    outer = length + np.clip(exit_depth - length + 3, 32, 160)
    radius = max(24., route["radiusMm"] + 7., route.get("entryPatch", {}).get("radiusMm", 0.) + 5.)
    behind = max(15., min(radius, 30.))
    # Anchor the target at a display voxel center. Ordinary fields use
    # 0.3125 mm spacing; large trajectories stay below 161 x 161 x 399 cells.
    spacing = float(max(.3125, radius / 80, (outer + behind) / 396))
    half, below, above = (int(np.ceil(value / spacing)) for value in (radius, behind, outer))
    origin = -spacing * np.array([half, half, below])
    shape = (2 * half + 1, 2 * half + 1, below + above + 1)
    grid = np.eye(4)
    grid[:3, :3] = basis.T * spacing
    grid[:3, 3] = target + basis.T @ origin
    return grid, shape, {
        "origin": origin.tolist(), "shape": list(shape), "spacingMm": spacing,
        "radiusMm": half * spacing, "routeRadiusMm": route["radiusMm"], "entryDepthMm": length,
        "targetRasMm": target.tolist(), "basisRas": basis.tolist(), "units": "mm",
    }


def _material(row, target):
    kind = row["kind"]
    model = row.get("provenance", {}).get("model")
    if row["id"] == target:
        return "tumor"
    if kind == "brain":
        return "brain"
    if model == "siam-v2" and kind in {"brain-region", "cerebellar-cortex"}:
        # Only these SIAM tissue kinds denote parenchyma. CSF and
        # ventricular CSF share the categorical volume, but stay unlabelled.
        return "brain"
    if kind in {"bone", "skull"}:
        return "bone"
    # The EDEN legacy importer typed these model codes as brain-region.
    if (kind == "brain-region" and str(row["provenance"].get("model", "")).lower() == "supersynth"
            and row["label_value"] in {915, 916} and row["id"] == f"head-{row['label_value']}"):
        return "bone"
    if kind in VASCULAR_KINDS:
        return "vessels"
    if kind == "dura":
        return "dura"
    return None


def _evidence(row):
    if is_dural_proxy(row):
        return "Geometric dural envelope proxy"
    return {"generated_evidence": "Synthetic engineering label", "supplied_annotation": "Supplied annotation",
            "model_estimate": "Source-derived estimate", "derived_geometry": "Geometric estimate",
            "unclassified_source": "Source-associated label"}[_source_class(row)]


def _coordinate(matrix, axis, coords):
    return (matrix[axis, 0] * coords[0] + matrix[axis, 1] * coords[1]
            + matrix[axis, 2] * coords[2] + matrix[axis, 3])


def _sample_values(data, native, source_affine, source_shape, grid, shape, bounds, slices, check):
    """Pull a categorical grid once, preserving source-cell and acquisition support."""
    layer_from_field = np.linalg.solve(native, grid)
    source_from_layer = np.linalg.solve(source_affine, native)
    source_from_field = np.linalg.solve(source_affine, grid)
    low, high = map(np.asarray, bounds)
    # Omit redundant acquisition tests only when whole label cells are
    # contained. Cropped or tilted labels can protrude past acquired voxels.
    corners = world(source_from_layer, list(product(*zip(low - .5, high + .5))))
    contained = ((np.array_equal(native, source_affine) and np.all(high < source_shape))
                 or (np.all(corners > -.5) and np.all(corners < source_shape - .5)))
    axial = None
    if slices is not None:
        axial = np.zeros(data.shape[2], dtype=bool)
        axial[list(slices)] = True

    values = np.zeros(shape, dtype=data.dtype)
    covered = np.zeros(shape, dtype=bool)
    y = np.arange(shape[1])[None, :, None]
    z = np.arange(shape[2])[None, None, :]
    # Bound intermediate coordinates independently of trajectory and study size.
    slab_size = max(1, 262144 // (shape[1] * shape[2]))
    for start in range(0, shape[0], slab_size):
        check()
        stop = min(start + slab_size, shape[0])
        coords = (np.arange(start, stop)[:, None, None], y, z)
        cells = [np.floor(_coordinate(layer_from_field, axis, coords) + .5).astype(np.int32) for axis in range(3)]
        valid = np.ones((stop - start, *shape[1:]), dtype=bool)
        for axis, cell in enumerate(cells):
            valid &= (cell >= low[axis]) & (cell <= high[axis])
        if not valid.any():
            continue
        if not contained:
            for axis, size in enumerate(source_shape):
                point = _coordinate(source_from_field, axis, coords)
                center = _coordinate(source_from_layer, axis, cells)
                valid &= (point >= -.5) & (point < size - .5) & (center >= -.5) & (center < size - .5)
        for axis, cell in enumerate(cells):
            np.clip(cell, 0, data.shape[axis] - 1, out=cell)
        if axial is not None:
            valid &= axial[cells[2]]
        covered[start:stop] = valid
        values[start:stop] = data[tuple(cells)]
    return values, covered


def _sample(collection, row, grid, shape, check, cache):
    source = collection.series(row["source_series"])
    native = collection.patient_affine(row)
    source_affine = collection.patient_affine(source)
    data = collection.array(row["data"])
    bounds = inference_bounds(row) or ((0, 0, 0), tuple(size - 1 for size in data.shape))
    slices = row.get("coverage_fields", {}).get("supported_axial_indices")
    if slices is not None:
        if (not isinstance(slices, list)
                or any(type(index) is not int or not 0 <= index < data.shape[2] for index in slices)):
            raise ValueError("Invalid supported axial slice indices")
        slices = tuple(slices)
    source_shape = tuple(source["data"]["shape"])
    key = (row["data"]["sha256"], native.tobytes(), source_affine.tobytes(), source_shape, bounds, slices)
    if cache.get("key") != key:
        # A segmentation can represent many separately attributed tissues in
        # one categorical array. Keep just the current native grid in memory.
        cache.clear()
        values, covered = _sample_values(data, native, source_affine, np.asarray(source_shape),
                                         grid, shape, bounds, slices, check)
        cache.update(key=key, values=values, covered=covered)
    return (cache["values"] == row["label_value"]) & cache["covered"], cache["covered"]


def _runs(mask, depths):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    stops = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return [[float(depths[i]), float(depths[j])] for i, j in zip(starts, stops)]


def _station_summary(labels, depths, layers, entry_depth, target_endpoint_supported):
    middle = labels.shape[0] // 2
    line = labels[middle, middle]
    runs = {role: _runs((line & bit) != 0, depths) for role, bit in MATERIAL_BITS.items()}
    stations, sources, evidence, issues = {}, {}, {}, []
    last_route_sample = float(depths[depths <= entry_depth][-1])
    previous = None
    for role in ("bone", "dura", "brain", "tumor"):
        outward_runs = [run for run in runs[role] if run[1] >= -1.e-7]
        if role == "tumor":
            # A single supplied label may contain disconnected foci along the
            # same ray. Use only the interval through the assessed endpoint.
            candidates = [(min(hi, last_route_sample), layer) for layer in layers if layer["role"] == role
                          for lo, hi in layer["centerlineRunsMm"] if target_endpoint_supported and lo <= 0 <= hi]
        else:
            candidates = [(hi, layer) for layer in layers if layer["role"] == role
                          for _, hi in layer["centerlineRunsMm"] if hi >= -1.e-7]
        if role == "dura" and any(not layer["proxy"] for _, layer in candidates):
            candidates = [(hi, layer) for hi, layer in candidates if not layer["proxy"]]
        station = max((hi for hi, _ in candidates), default=None)
        stations[role] = station
        contributors = [layer for hi, layer in candidates if hi == station]
        sources[role] = [layer["id"] for layer in contributors]
        evidence[role] = ("proxy" if contributors[0]["proxy"] else "source") if contributors else None
        if station is None:
            issues.append("Aspiration unavailable because the saved endpoint lacks supported selected-target geometry"
                          if role == "tumor" else f"No {role} label sampled on the outward centerline")
            continue
        if len(outward_runs) > 1:
            issues.append(f"Multiple {role} intervals on the outward centerline")
        if previous and station > stations[previous]:
            issues.append(f"{role.capitalize()} label lies outward of the {previous} label")
        previous = role
    return stations, runs, sources, evidence, issues


def _encounters(measured):
    result = []
    for row in measured["structures"]:
        covered = (row.get("coverage", {}).get("wholeEnvelopeWithinField", False)
                   and row.get("acquisitionCoverage", {}).get("wholeEnvelopeWithinField", True))
        if row.get("confirmedIntersection"):
            status = "contact"
        elif row.get("intersectsOrUnresolved"):
            status = "unresolved"
        elif not row["available"] or not covered:
            status = "unknown"
        else:
            status = "no-labelled-contact"
        result.append({"id": row["layerId"], "label": row["label"], "kind": row["kind"],
                       "status": status, "reason": row.get("reason"), "covered": covered})
    return result


def operative_field(collection, args, *, check=lambda: None):
    route, binding, revision = _saved_route(collection, args)
    target = route["targetLayerId"]
    grid, shape, frame = _field_grid(collection, route)
    labels = np.zeros(shape, np.uint8)
    depths = frame["origin"][2] + np.arange(shape[2]) * frame["spacingMm"]
    middle = shape[0] // 2
    layers, excluded = [], []
    direct_dura = np.zeros(shape, bool)
    cache = {}
    source_rows = sorted(collection.document["layers"], key=is_dural_proxy)
    for row in sorted(source_rows, key=lambda item: (is_dural_proxy(item), item["data"]["sha256"])):
        role = _material(row, target)
        if not role:
            continue
        identity = {"id": row["id"], "label": row["label"], "role": role}
        model = row["provenance"].get("model")
        vascular_context = role == "vessels" and not vascular_evidence_supported(collection, row)
        try:
            if row.get("availability") == "empty_output" or row.get("evidence_status") == "empty-output":
                raise ValueError("Source reported an empty output")
            if vascular_context and str(model).lower() not in {"siam-v2", "supersynth"}:
                raise ValueError("Vascular estimate lacks supported positive geometry")
            display_row, support = row, None
            if role != "tumor" and model == "siam-v2":
                collection.patient_affine(row)
                display_row, support = supported_tissue_layer(collection, row, check=check)
            occupied, covered = _sample(collection, display_row, grid, shape, check, cache)
        except InterruptedError:
            raise
        except (ValueError, OSError, StopIteration) as error:
            if row["id"] == target:
                raise ValueError(f"Selected target source is unavailable: {error}") from error
            excluded.append({**identity, "reason": str(error) or "Source unavailable"})
            continue
        proxy = is_dural_proxy(row)
        if role == "dura":
            if proxy:
                occupied &= ~direct_dura
                covered = covered & ~direct_dura
                if not covered.any():
                    excluded.append({**identity, "reason": "Proxy support coincides with direct dural labels"})
                    continue
            else:
                # An annotation can be partial within its own acquisition.
                # Only positive source labels take precedence over proxy cells.
                direct_dura |= occupied
        labels[occupied] |= MATERIAL_BITS[role]
        source = collection.series(row["source_series"])
        evidence = f"{model} vascular estimate, display only" if vascular_context else _evidence(row)
        layers.append({**identity, "kind": row["kind"], "proxy": proxy, "evidence": evidence,
                       "vascularContextOnly": vascular_context,
                       "model": row["provenance"].get("model"), "sourceSeries": source["id"],
                       "sourceSha256": source["original"]["sha256"], "dataSha256": row["data"]["sha256"],
                       "sampledDataSha256": display_row["data"]["sha256"], "sourceSupport": support,
                       "labelValue": row["label_value"], "revision": row.get("revision", 0),
                       "coverageMeaning": row.get("coverage"), "reviewStatus": row.get("review_status"),
                       "sourceModality": source["modality"],
                       "registrationStatus": collection.document["transforms"][source["id"]]["status"],
                       "coveredCenterlineRunsMm": _runs(covered[middle, middle], depths),
                       "timepoint": source["timepoint"], "sampledVoxels": int(np.count_nonzero(occupied)),
                       "coveredVoxels": int(np.count_nonzero(covered)), "centerlineRunsMm": _runs(occupied[middle, middle], depths)})
    # Group native reads by array while preserving collection order in the UI.
    source_order = {row["id"]: index for index, row in enumerate(source_rows)}
    layers.sort(key=lambda item: source_order[item["id"]])
    excluded.sort(key=lambda item: source_order[item["id"]])
    cache.clear()
    measured = route["measurements"]
    target_endpoint_supported = any(item.get("targetPointInsideLabel") and item["layerId"] == target
                                    for item in measured["targets"])
    stations, runs, station_ids, station_evidence, issues = _station_summary(
        labels, depths, layers, frame["entryDepthMm"], target_endpoint_supported)
    for layer in layers:
        if layer["role"] == "vessels" and any(hi > frame["entryDepthMm"] for _, hi in layer["centerlineRunsMm"]):
            issues.append(f"{layer['label']} sampled on the centerline beyond the assessed entry")
    intensity = operative_intensity(collection, collection.layer(target)["source_series"], grid, frame, check=check)
    check()
    if (json.loads((collection.root / "patient-anatomy.json").read_text())["snapshot_id"] != args["snapshotId"]
            or file_hash(collection.root / FILENAME) != revision):
        raise ValueError("The anatomy or saved plan changed while sampling the operative field")
    return {**frame, **({"intensity": intensity} if intensity is not None else {}), "snapshotId": args["snapshotId"], "approachId": route["id"], "parameterHash": route["parameterHash"],
            "targetLayerId": target, "candidateBinding": binding, "labels": labels, "materialBits": dict(MATERIAL_BITS),
            "layers": layers, "excluded": excluded, "stations": stations, "stationRuns": runs, "stationIssues": issues,
            "stationLayerIds": station_ids, "stationEvidence": station_evidence,
            "encounters": _encounters(measured), "blockers": measured["constraintFailures"], "unknowns": measured["unknowns"],
            "sampling": "Nearest native label cell at display voxel centers. Unlabelled anatomy is unassessed.",
            "simulationExecutionAuthorized": False}
