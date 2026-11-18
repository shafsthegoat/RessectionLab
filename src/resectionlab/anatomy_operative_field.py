"""Sample native labels in a target-centred RAS frame for visual rehearsal."""

import json

import numpy as np

from .anatomy_adjudication import BoneEvidence
from .anatomy_sampling import sample_layer as _sample
from .anatomy_approaches import FILENAME, _read_plan
from .anatomy_evidence import VASCULAR_KINDS, is_dural_proxy, vascular_evidence_supported
from .anatomy_operative_intensity import operative_intensity
from .anatomy_planning import native_target_binding
from .anatomy_siam_display import supported_tissue_layer
from .native_consequence_evidence import _source_class
from .patient_anatomy import file_hash, world


MATERIAL_BITS = {"bone": 1, "dura": 2, "brain": 4, "tumor": 8, "vessels": 16}


# Overlapping display classes preserve native evidence. Only the selected target sets the tumor bit.
DETAIL_BITS = {"grayMatter": 32, "whiteMatter": 64, "csf": 128, "ventricles": 256,
               "artery": 512, "vein": 1024, "sinus": 2048,
               "enhancing": 4096, "nonenhancingCore": 8192, "perilesionalSignal": 16384, "tumorCore": 32768,
               "wholeAbnormality": 65536}


def _detail(row):
    kind = row["kind"]
    explicit = {"gray-matter": "grayMatter", "white-matter": "whiteMatter",
                "cerebellar-cortex": "grayMatter", "cerebellar-white-matter": "whiteMatter",
                "csf": "csf", "ventricles": "ventricles", "artery": "artery", "vein": "vein", "sinus": "sinus",
                "enhancing-tumor": "enhancing", "nonenhancing-tumor": "nonenhancingCore",
                "edema": "perilesionalSignal", "tumor-core": "tumorCore", "tumor-whole": "wholeAbnormality"}
    if kind in explicit:
        return explicit[kind]
    provenance = row.get("provenance", {})
    ontology = provenance.get("label_ontology", {})
    if not isinstance(ontology, dict):
        ontology = {}
    model = str(provenance.get("model", "")).lower()
    if model == "siam-v2":
        return {"GM": "grayMatter", "WM": "whiteMatter"}.get(ontology.get("name"))
    if model in {"supersynth", "synthseg", "fastsurfer"} and kind == "brain-region":
        code = ontology.get("value", row.get("original_label_value", row["label_value"]))
        if code in {2, 7, 41, 46}:
            return "whiteMatter"
        if code in {3, 8, 42, 47}:
            return "grayMatter"
        if code in {4, 5, 14, 15, 43, 44}:
            return "ventricles"
        if code == 24:
            return "csf"
    # Retained explicit source mapping in the prepared UCSF mirror studies.
    if provenance.get("kind") == "observed_source_annotation":
        if provenance.get("method") == "Pinned source BraTS compartment labels":
            return {1: "nonenhancingCore", 2: "perilesionalSignal", 4: "enhancing"}.get(provenance.get("annotation_value"))
        return {(1, "Non-enhancing tumor core / necrosis · source label 1"): "nonenhancingCore",
                (2, "Peritumoral edema · source label 2"): "perilesionalSignal",
                (4, "Enhancing tumor · source label 4"): "enhancing"}.get(
                    (provenance.get("source_label_value"), row["label"]))
    return None


def _inference_spacing(row):
    provenance = row["provenance"]
    # The retained SIAM DS715 3d_fullres plans specify a 0.75 mm isotropic network lattice.
    # Earlier study receipts contain the plans checksum but omit the spacing field.
    if provenance.get("model") == "siam-v2":
        from .siam_worker import ASSETS
        config = provenance.get("configuration", {})
        if (isinstance(config, dict) and config.get("files", {}).get("plans.json") == ASSETS["plans.json"]
                and config.get("parameters", {}).get("configuration") == "3d_fullres"):
            return [.75, .75, .75]
    pitch = provenance.get("inference_spacing_mm")
    if (isinstance(pitch, (list, tuple)) and len(pitch) == 3
            and all(type(value) in (float, int) and np.isfinite(value) and value > 0 for value in pitch)):
        return list(pitch)
    return None


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


def _material(row, target, detail):
    kind = row["kind"]
    model = row.get("provenance", {}).get("model")
    if row["id"] == target:
        return "tumor"
    if detail in {"csf", "ventricles"}:
        return "fluid"
    if detail in {"enhancing", "nonenhancingCore", "perilesionalSignal", "tumorCore", "wholeAbnormality"}:
        return "lesionContext"
    if kind == "brain" or detail in {"grayMatter", "whiteMatter"}:
        return "brain"
    if model == "siam-v2" and kind in {"brain-region", "cerebellar-cortex"}:
        # Deep nuclei without a specific display class retain their source name.
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
    has_whole = any(row["kind"] == "tumor-whole" for row in collection.document["layers"])
    labels = np.zeros(shape, np.uint32 if has_whole else np.uint16)
    depths = frame["origin"][2] + np.arange(shape[2]) * frame["spacingMm"]
    middle = shape[0] // 2
    layers, excluded = [], []
    direct_dura = np.zeros(shape, bool)
    cache = {}
    source_rows = sorted(collection.document["layers"], key=is_dural_proxy)
    bone = BoneEvidence(collection, source_rows, shape)
    for row in sorted(source_rows, key=lambda item: (is_dural_proxy(item), bone.order(item), item["data"]["sha256"])):
        detail = _detail(row)
        role = _material(row, target, detail)
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
        adjudication = None
        if role == "bone":
            occupied, covered, adjudication = bone.apply(row, occupied, covered)
        if vascular_context:
            detail = None
        labels[occupied] |= MATERIAL_BITS.get(role, 0) | DETAIL_BITS.get(detail, 0)
        source = collection.series(row["source_series"])
        evidence = f"{model} vascular estimate, display only" if vascular_context else _evidence(row)
        layers.append({**identity, "kind": row["kind"], "proxy": proxy, "evidence": evidence,
                       "vascularContextOnly": vascular_context, "detail": detail,
                       "sourceSpacingMm": np.linalg.norm(np.asarray(source["affine"])[:3, :3], axis=0).tolist(),
                       "labelSpacingMm": np.linalg.norm(np.asarray(row["affine"])[:3, :3], axis=0).tolist(),
                       "inferenceSpacingMm": _inference_spacing(row),
                       **({"adjudication": adjudication} if adjudication else {}),
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
            "targetLayerId": target, "candidateBinding": binding, "labels": labels, "materialBits": {**MATERIAL_BITS, **DETAIL_BITS},
            "layers": layers, "excluded": excluded, "stations": stations, "stationRuns": runs, "stationIssues": issues,
            "stationLayerIds": station_ids, "stationEvidence": station_evidence,
            "encounters": _encounters(measured), "blockers": measured["constraintFailures"], "unknowns": measured["unknowns"],
            "sampling": "Nearest native label cell at display voxel centers. Unlabelled anatomy is unassessed.",
            "simulationExecutionAuthorized": False}
