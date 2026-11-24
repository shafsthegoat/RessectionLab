"""Desktop patient anatomy operations, using existing bounded binary transfers."""

from copy import deepcopy
from pathlib import Path
import uuid
import hashlib
import json
import re
import numpy as np
from .patient_anatomy import (
    PatientImagingCollection,
    PatientAnatomySnapshot,
    affine,
    resample,
    world,
)
from .anatomy_editor import AnatomyDraft
from .anatomy_workspace import read_view, save_view, component_review
from .anatomy_geometry import mesh, normals, appearance_kind, vascular_cleanup_source
from .anatomy_qa import layer_quality
from .anatomy_display import (
    brain_vessel_display, display_affine, patient_display_context, source_display_grid,
    source_lattice_annotation_preview,
)


VASCULAR_REVIEW_MANIFEST = "vascular-review.anatomy.json"
VASCULAR_REVIEW_REVISION = re.compile(r"vascular-review\.[0-9a-f]{12}\.anatomy\.json")
VASCULAR_REVIEW_READS = {
    "anatomyLocalAlignment",
    "anatomyView", "anatomySample", "anatomyGraph", "anatomySection", "anatomyComponents",
    "anatomyAttachedVesselReview",
}


def _linked_vascular_review(path):
    document = json.loads(path.read_text())
    data = document.get("vascular_review", {})
    if data.get("schema") != "eden2020-restricted-vascular-companion-v1" or data.get("linked_source") is not True:
        raise ValueError("Invalid source-linked vascular review")
    source = path.parent / "patient-anatomy.json"
    if hashlib.sha256(source.read_bytes()).hexdigest() != data.get("source_manifest", {}).get("sha256"):
        raise ValueError("The vascular review's pinned source manifest has changed")
    baseline = PatientImagingCollection(path.parent)
    if (document.get("patient_id") != baseline.document["patient_id"]
            or data["source_manifest"].get("snapshot_id") != baseline.identity()):
        raise ValueError("The vascular review belongs to different source anatomy")
    return PatientImagingCollection(path.parent, document)


def open_anatomy_manifest(path):
    """Open the selected native manifest and verify linked review provenance."""
    path = Path(path).resolve()
    if path.name == "patient-anatomy.json":
        return PatientImagingCollection(path.parent)
    if path.name == VASCULAR_REVIEW_MANIFEST or VASCULAR_REVIEW_REVISION.fullmatch(path.name):
        return _linked_vascular_review(path)
    raise ValueError("Select a native study or vascular review manifest")


def _retain(session, result):
    if isinstance(result, dict):
        if "path" in result and "byteLength" in result:
            session.anatomy_transfer_paths.add(result["path"])
        for x in result.values():
            _retain(session, x)
    elif isinstance(result, list):
        for x in result:
            _retain(session, x)


def dispatch(session, operation, args, request, progress):
    # A source read does not change the cached view or its transfer descriptors.
    if operation not in {"anatomyView", "anatomySample", "anatomyAttachedVesselReview", "anatomyLocalAlignment"}:
        session.anatomy_view_cache = None
    if not hasattr(session, "anatomy_collections"):
        session.anatomy_packages = {}
        session.anatomy_collections = {}
        session.anatomy_drafts = {}
        session.anatomy_transfer_paths = set()
        session.anatomy_removal = {}
    if operation == "anatomyCreate":
        session.anatomy_plan_cache = None
        session.anatomy_plan_target_cache = None
        c = PatientImagingCollection.create(args["path"], args["patientId"], timepoint=args.get("timepoint"))
        c.document["preparation_workflow"] = "native-adaptive-v1"
        request.begin_commit()
        c.save("create patient imaging collection")
        key = str(uuid.uuid4())
        session.anatomy_packages = {}
        session.anatomy_collections = {key: c}
        session.anatomy_drafts = {}
        session.anatomy_removal = {}
        return {"collectionId": key, "document": c.document, "viewState": read_view(c)}
    if operation == "anatomyOpen":
        session.anatomy_plan_cache = None
        session.anatomy_plan_target_cache = None
        path = Path(args["path"]).resolve()
        package_path = path if path.suffix == ".rsworkspace" else None
        if path.suffix == ".rsworkspace":
            from .anatomy_package import open_package
            path = open_package(path, args["workingRoot"], check=request.check)
        c = open_anatomy_manifest(path)
        linked = path.name != "patient-anatomy.json"
        key = str(uuid.uuid4())
        session.anatomy_collections = {key: c}
        session.anatomy_linked_review = (key, path) if linked else None
        session.anatomy_drafts = {}
        session.anatomy_removal = {}
        session.anatomy_transfer_paths = set()
        session.anatomy_packages = {key: package_path} if package_path and not args.get("readOnlyPackage") else {}
        return {"collectionId": key, "document": c.document,
                "viewState": None if linked else read_view(c), "readOnlyStudy": linked}
    key = args.get("collectionId")
    if key not in session.anatomy_collections:
        raise ValueError("Patient collection is not open")
    c = session.anatomy_collections[key]
    manifest = c.root / "patient-anatomy.json"
    linked_review = getattr(session, "anatomy_linked_review", None)
    if linked_review and linked_review[0] == key:
        manifest = linked_review[1]
        if operation not in VASCULAR_REVIEW_READS:
            raise ValueError("This source-linked vascular review is read-only")
    current_id = json.loads(manifest.read_text()).get(
        "snapshot_id"
    )
    if operation == "anatomyPrepare" and args.get("action") == "plan" and args.get("snapshotId") is None:
        # A cancelled worker may have completed some stages before stopping.
        if current_id != c.identity():
            if key in session.anatomy_drafts:
                raise ValueError("Resolve the active correction before refreshing")
            c = PatientImagingCollection(c.root)
            session.anatomy_collections[key] = c
        args = {**args, "snapshotId": c.identity()}
    if args.get("snapshotId") != c.identity() or current_id != c.identity():
        raise ValueError("Anatomy changed; reload the patient")
    if key in session.anatomy_removal and session.anatomy_removal[key][0] != current_id:
        session.anatomy_removal.pop(key)
    request.check()
    if operation == "anatomyLocalAlignment":
        from .anatomy_local_alignment import local_alignment_evidence

        progress(.1, "Checking coverage and image correspondence in the local field")
        result = local_alignment_evidence(c, args.get("seriesId"), args.get("center"), check=request.check)
        if json.loads(manifest.read_text()).get("snapshot_id") != current_id:
            raise ValueError("Anatomy changed during the local alignment screen")
        return result
    if operation == "anatomyInspectIntake":
        from .anatomy_intake import inspect_intake
        return inspect_intake(args["kind"], args["paths"], check=request.check)
    if operation == "anatomyDiffusionBaseline":
        if key in session.anatomy_drafts:
            raise ValueError("Accept or reject the active correction first")
        from .anatomy_diffusion import add_native_diffusion_baseline

        candidate = PatientImagingCollection(c.root, c.document)
        progress(.15, "Reading the native diffusion baseline")
        series = add_native_diffusion_baseline(candidate, args.get("assetId"), check=request.check)
        if candidate.identity() != current_id:
            request.begin_commit()
            candidate.save("retain native diffusion baseline for inspection", expected=current_id)
            session.anatomy_collections[key] = candidate
        return {"document": candidate.document, "seriesId": series["id"]}
    if operation in {"anatomyImportBatch", "anatomyPrepare"}:
        if key in session.anatomy_drafts:
            raise ValueError("Accept or reject the active correction first")
        if operation == "anatomyImportBatch":
            from .anatomy_intake import reviewed_intake
            reference_id = c.document.get("reference_series")
            c = reviewed_intake(c, args, check=request.check, commit=request.begin_commit, progress=progress)
            if c.document.get("reference_series") != reference_id:
                session.anatomy_removal.pop(key, None)
            session.anatomy_collections[key] = c
            return {"document": c.document}
        from .anatomy_preparation import preparation_plan, prepare_anatomy
        options = dict(indication=args.get("tumorIndication"),
                       full_head_confirmed=args.get("fullHeadConfirmed") is True,
                       include_dural_proxy=args.get("includeDuralProxy") is True)
        if args.get("action") == "plan":
            return {"document": c.document, "plan": preparation_plan(c, **options)}
        if args.get("action") != "run" or not isinstance(args.get("planId"), str):
            raise ValueError("Review the proposed preparation plan before running")
        return prepare_anatomy(c, plan_id=args["planId"], cancelled=request.cancelled.is_set,
                               progress=progress, retry_failed=args.get("retryFailed") is True, **options)
    if operation == "anatomySavePackage":
        if key in session.anatomy_drafts:
            raise ValueError("Accept or reject the correction before saving")
        from .anatomy_package import save_package
        result = save_package(c, args["path"], check=request.check, progress=progress, commit=request.begin_commit)
        session.anatomy_packages[key] = Path(args["path"])
        return result
    if operation == "anatomySaveView":
        if key in session.anatomy_drafts:
            raise ValueError("Accept or reject the correction before saving")
        request.begin_commit()
        save_view(c, args["viewState"])
        archive = session.anatomy_packages.get(key)
        if archive and args.get("saveArchive") is not False:
            from .anatomy_package import save_package
            save_package(c, archive, check=request.check, progress=progress)
        return {"saved": True, "portable": bool(archive)}
    if operation == "anatomyComponents":
        draft = session.anatomy_drafts.get(key)
        if draft and draft.layer_id != args["layerId"]:
            raise ValueError("Finish the current layer correction first")
        report, _ = component_review(c, args["layerId"], draft.mask if draft else None)
        return report
    if operation == "anatomyAttachedVesselReview":
        from .vascular_cleanup import attached_tof_sections, review_attached_tof_layer

        row = c.layer(args["layerId"])
        if row["source_series"] != args["seriesId"]:
            raise ValueError("Open the vessel estimate's native TOF acquisition")
        footprints, receipt = review_attached_tof_layer(c, row, check=request.check)
        report = {**receipt, "snapshotId": current_id}
        finding_id = args.get("findingId")
        findings = report.get("findings", [])
        if finding_id is None:
            finding = findings[0] if findings else None
        else:
            if type(finding_id) is not str or finding_id not in footprints or args.get("reviewId") != report.get("reviewId"):
                raise ValueError("Attached vessel review changed; refresh the findings")
            finding = next(item for item in findings if item["id"] == finding_id)
        if finding is not None:
            report["preview"] = {"findingId": finding["id"], **attached_tof_sections(
                c, row, footprints[finding["id"]], finding, check=request.check,
            )}
        request.check()
        if json.loads(manifest.read_text()).get("snapshot_id") != current_id:
            raise ValueError("Anatomy changed; reload the patient")
        return report
    if operation == "anatomyView":
        reference = c.series(args.get("seriesId", c.document["reference_series"]))
        requested = args.get("layers", [])
        if (
            not isinstance(requested, list)
            or len(requested) > 128
            or len(set(requested)) != len(requested)
        ):
            raise ValueError(
                "Select up to 128 anatomical layers"
            )
        tract_ids = {row["id"] for row in c.document.get("tractography", [])}
        known_ids = {row["id"] for row in c.document["layers"]} | tract_ids
        if not set(requested) <= known_ids:
            raise ValueError("Unknown anatomical layer or tract bundle")
        requested = [name for name in requested if name not in tract_ids]
        native = affine(reference["affine"])
        preview_alignment = args.get("previewAlignment") is True
        vessel_scope = args.get("vesselScope", "all")
        if vessel_scope not in {"all", "brain", "intracranial", "excluded", "exterior-air"}:
            raise ValueError("Unknown vessel display scope")
        native_vessel_review = c.document.get("vascular_review", {}).get("schema") == "eden2020-restricted-vascular-companion-v1"
        vessel_surface_mode = args.get("vesselSurfaceMode", "source" if native_vessel_review else "conditioned")
        if vessel_surface_mode not in {"source", "conditioned"}:
            raise ValueError("Unknown vessel surface representation")
        source_frame_issue = None
        try:
            displayed, estimated = display_affine(c, reference, preview=preview_alignment)
            registered = True
        except ValueError as error:
            source_frame_issue = str(error)
            displayed, estimated, registered = native, False, False
        patient_context = None
        if args.get("patientModel") is True and registered:
            context_key = (key, c.identity(), preview_alignment)
            cached = getattr(session, "anatomy_context_cache", None)
            if cached is None or cached[0] != context_key:
                patient_context = patient_display_context(c, preview=preview_alignment, check=request.check)
                session.anatomy_context_cache = (context_key, patient_context)
            else:
                patient_context = cached[1]
        source_displayed = displayed
        alignment_previews = [reference["id"]] if estimated else []
        if patient_context:
            alignment_previews.extend(s["seriesId"] for s in patient_context["sources"]
                                      if s["alignment"] == "estimated" and s["inPatientBounds"])
        # MRI acceleration is reversible and never feeds source measurements.
        # Native layer geometry is transformed into this displayed voxel frame.
        image_data, native, displayed, preview = source_display_grid(reference, displayed)
        shape = tuple(image_data["shape"])
        source_cache_key = (key, c.identity(), reference["id"], image_data["sha256"],
                            displayed.tobytes(), bool(args.get("original")), preview_alignment, bool(patient_context))
        cache_key = (source_cache_key, vessel_scope, vessel_surface_mode)
        cacheable = not (session.anatomy_drafts.get(key) or key in session.anatomy_removal
                         or args.get("highlightComponents"))
        cache = getattr(session, "anatomy_view_cache", None)
        if not cacheable or not cache or cache["key"] != cache_key:
            previous_mri = cache.get("mri") if cache and cache["key"][0] == source_cache_key else None
            if previous_mri and not Path(previous_mri["path"]).is_file():
                previous_mri = None
            support = {"headSupport": cache["headSupport"]} if previous_mri and "headSupport" in cache else {}
            cache = {"key": cache_key, "layers": {}, "label_sets": {}, "mri": previous_mri, "tracts": None, **support}
        # Publish only a complete response. A cancelled/failed preparation may
        # have temporary transfer files pruned and must not cache those paths.
        cache = {**cache, "layers": dict(cache["layers"]), "label_sets": dict(cache["label_sets"])}
        layers = []
        label_sets = {}
        unavailable = []
        transfer = session.transfers.array
        for i, name in enumerate(requested):
            request.check()
            if name in cache["layers"]:
                record = deepcopy(cache["layers"][name])
                layers.append(record)
                alignment_previews.extend(record.get("alignmentPreviews", []))
                if record.get("labelSet"):
                    set_id = record["labelSet"]
                    label_sets[set_id] = cache["label_sets"][set_id]
                continue
            row = deepcopy(c.layer(name))
            draft = session.anatomy_drafts.get(key)
            if draft and draft.layer_id == name and not args.get("original"):
                row["data"] = c.put_array(draft.mask.astype(np.uint8))
                row["label_value"] = 1
                row["edits"] = [{"draft": True}]
                row["revision"] += 1
            row_previews = [reference["id"]] if estimated and row["source_series"] == reference["id"] else []
            source_grid_preview = False
            try:
                if row["source_series"] == reference["id"]:
                    if registered:
                        source = display_affine(c, row, preview=preview_alignment)[0]
                    else:
                        c.require_native_geometry(reference)
                        c.require_native_geometry(row)
                        source = affine(row["affine"])
                else:
                    if not registered:
                        raise ValueError("selected image is unregistered")
                    if c.series(row["source_series"])["timepoint"] != reference["timepoint"]:
                        raise ValueError("Separate study timepoint")
                    source, layer_estimated = display_affine(c, row, preview=preview_alignment)
                    if layer_estimated:
                        row_previews.append(row["source_series"])
                        alignment_previews.extend(row_previews)
            except ValueError:
                source_grid_preview = preview_alignment and source_lattice_annotation_preview(c, row, reference)
                source = source_displayed
                if not source_grid_preview:
                    unavailable.append(name)
                    continue
            progress(
                (i + 0.1) / max(1, len(requested)), f"Preparing native {row['label']}"
            )
            if key in session.anatomy_removal and row["role"] == "target":
                _, removal_preview = session.anatomy_removal[key]
                ref = c.series(c.document["reference_series"])
                cut = (
                    resample(
                        removal_preview.astype(np.uint8),
                        c.patient_affine(ref),
                        tuple(row["data"]["shape"]),
                        source,
                        order=0,
                    )
                    > 0
                )
                row["data"] = c.put_array((c.mask(row) & ~cut).astype(np.uint8))
                row["label_value"] = 1
                row["edits"] = [{"removal_preview": True}]
            source_surface_only = vessel_surface_mode == "source" and row["kind"] in {"vessel", "vessels", "artery", "vein", "sinus"}
            # The native extraction verifies contour signs before use. A raw
            # cached contour does not require a second full score-volume read.
            quality = layer_quality(c, row, verify_probability=not source_surface_only)
            source_volume = None if source_grid_preview else float(
                c.mask(row).sum() * abs(np.linalg.det(affine(row["affine"])[:3, :3]))
            )
            if source_grid_preview:
                message = "Annotation header omits spatial units. Preview on its supplied source grid; physical measurements unavailable."
                quality = {**quality, "issues": [message, *quality["issues"]]}
            from .anatomy_siam_display import supported_tissue_layer
            source_support = None
            if not (draft and draft.layer_id == name) and args.get("highlightLayer") != name:
                row, source_support = supported_tissue_layer(c, row, check=request.check, raw_qc=bool(args.get("original")))
            bone_review = None
            if (registered and not row_previews and not source_grid_preview and not draft
                    and not args.get("original") and not args.get("highlightLayer") and args.get("patientModel")):
                from .anatomy_adjudication import composite_bone_layer

                row, bone_review = composite_bone_layer(c, row, request.check)
                if bone_review:
                    count = bone_review["alternativePositiveVoxels"]
                    disagreement = bone_review["alternativeOnlyVoxels"]
                    ids = ", ".join(bone_review["preferredLayerIds"])
                    quality = {**quality, "issues": [*quality["issues"],
                               f"CT bone display from {ids} takes precedence at {count} MRI bone cells; "
                               f"{disagreement} are unlabelled on CT. Raw source QC retains the MRI alternative."]}
            raw_siam = bool(source_support and source_support["status"] == "unfiltered")
            display_filter = None
            unfiltered = row
            # Correction review always shows the complete editable source.
            if (vessel_scope == "exterior-air" and not native_vessel_review and not source_grid_preview
                    and not draft and not args.get("original") and not args.get("highlightLayer")):
                from .vascular_cleanup import supported_tof_layer

                row, display_filter = supported_tof_layer(c, row, check=request.check)
            if (not raw_siam and not source_grid_preview and vessel_scope == "brain" and row["kind"] in {"vessels", "artery", "vein", "sinus"}
                    and not draft and not args.get("highlightLayer")):
                row, display_filter = brain_vessel_display(c, row)
            if (not raw_siam and not source_grid_preview and vessel_scope in {"intracranial", "excluded"} and row["kind"] in {"vessels", "artery", "vein", "sinus"}
                    and not draft and not args.get("highlightLayer")):
                from .anatomy_intracranial import intracranial_vessels
                row, display_filter = intracranial_vessels(c, row, excluded=vessel_scope == "excluded", cancelled=request.cancelled.is_set)
            score_source = vascular_cleanup_source(c, unfiltered)
            if score_source is None and row is not unfiltered:
                score_source = unfiltered
            geometry = mesh(c, row, cancelled=request.cancelled.is_set,
                            display_source=score_source, conditioned=not source_surface_only)
            request.check()
            shared = row.get("kind") == "brain-region" and not row.get("edits")
            # Identical categorical values may originate on different physical grids.
            set_id = hashlib.sha256(row["data"]["sha256"].encode() + source.tobytes()).hexdigest() if shared else None
            if shared and set_id not in label_sets:
                if set_id not in cache["label_sets"]:
                    values = resample(c.array(row["data"]), source, shape, displayed, order=0).astype(np.uint16)
                    cache["label_sets"][set_id] = {"id": set_id, "data": transfer(values, "uint16")}
                label_sets[set_id] = cache["label_sets"][set_id]
            mask = None if shared else resample(c.mask(row).astype(np.uint8), source, shape, displayed, order=0)
            record = {
                "name": name,
                "kind": row["kind"],
                "displayKind": appearance_kind(row),
                "sourceSeries": row["source_series"],
                "alignment": ("source-grid preview" if source_grid_preview else "independent" if not registered
                              else "estimated" if row_previews else c.document["transforms"][row["source_series"]]["status"]),
                "sourceIdentity": quality["sourceIdentity"],
                "qa": quality,
                "surfaceDiagnostics": {"policy": str(geometry.get("policy_version", "legacy")),
                    "kind": "tumor" if row["kind"] in {"tumor", "lesion-context"} else "vessel" if row["kind"] in {"vessels", "artery", "vein", "sinus"} else "context",
                    "maxDisplacementMm": float(geometry["max_displacement_mm"]),
                    "restoredComponents": int(geometry.get("restored_components", 0)),
                    "backedOffVertices": int(geometry.get("backed_off_vertices", 0)),
                    "fallbackReason": str(geometry.get("fallback_reason", "")),
                    "cacheHit": bool(geometry["cache_hit"]),
                    "preparationMs": float(geometry["preparation_ms"]),
                    "triangles": int(len(geometry["indices"])),
                    **({"sourceSurfaceOnly": True} if source_surface_only else {}),
                    **({"surfaceField": str(geometry["surface_field"])} if "surface_field" in geometry else {}),
                    **({"publisherSourceSha256": geometry["publisher_source_sha256"]}
                       if "publisher_source_sha256" in geometry else {}),
                    **({
                        "maxComponentVolumeChangeFraction": float(geometry["max_component_volume_change_fraction"]),
                        "volumeRestoredComponents": int(geometry["volume_restored_components"]),
                    } if "max_component_volume_change_fraction" in geometry else {}),
                    "intersectionCheck": "source-restored" if bool(geometry.get("smoothing_rejected", False)) or str(geometry.get("fallback_reason", "")) else "passed"},
                "label": row["label"] + (" · Source-grid preview" if source_grid_preview else ""),
                "color": row["color"],
                "alignmentPreviews": row_previews,
                "displayFilter": display_filter,
                **({"sourceSupport": source_support} if source_support else {}),
                **({"boneAdjudication": bone_review} if bone_review else {}),
                **({"labelSet": set_id, "labelValue": row["label_value"]} if shared else {"mask": transfer(mask, "uint8")}),
                **({"volumeMm3": source_volume} if source_volume is not None else {}),
            }
            if True:  # Empty geometry is explicit; never rebuild it from a categorical placeholder.
                transform = np.linalg.inv(displayed) @ source
                raw = world(transform, geometry["source"]).astype(np.float32)
                p = raw if source_surface_only else world(transform, geometry["positions"]).astype(np.float32)
                faces = geometry["indices"]
                if np.linalg.det(transform[:3, :3]) < 0:
                    faces = faces[:, ::-1]
                record["surface"] = {"empty": True}
                if p.size:
                    raw_transfer = transfer(raw, "float32")
                    raw_normals = transfer(normals(raw, faces), "float32")
                    record["surface"] = {
                        "positions": raw_transfer if source_surface_only else transfer(p, "float32"),
                        "source": raw_transfer,
                        "normals": raw_normals if source_surface_only else transfer(normals(p, faces), "float32"),
                        "sourceNormals": raw_normals,
                        "indices": transfer(faces, "uint32"),
                    }
            layers.append(record)
            if cacheable:
                cache["layers"][name] = deepcopy(record)
            # A review highlight is a display-only overlay; it is never included in
            # quantitative layers, revisions or planning. Exact component ids are
            # bound to the draft's immutable starting mask.
            review_ids = args.get("highlightComponents", [])
            if name == args.get("highlightLayer") and review_ids:
                draft = session.anatomy_drafts.get(key)
                base = draft.original if draft and draft.layer_id == name else c.mask(c.layer(name))
                report, components = component_review(c, name, base)
                if report["reviewId"] != args.get("reviewId"):
                    raise ValueError("Fragment review changed; refresh the component list")
                known = {r["id"] for r in report["components"]}
                if not isinstance(review_ids, list) or any(type(i) is not int or i not in known for i in review_ids):
                    raise ValueError("Unknown highlighted component")
                highlight = np.isin(components, review_ids)
                hrow = deepcopy(c.layer(name))
                hrow["data"] = c.put_array(highlight.astype(np.uint8))
                hrow["label_value"] = 1
                hrow["edits"] = [{"review_highlight": True}]
                hgeo = mesh(c, hrow, cancelled=request.cancelled.is_set)
                transform = np.linalg.inv(displayed) @ source
                hp = world(transform, hgeo["positions"]).astype(np.float32)
                hr = world(transform, hgeo["source"]).astype(np.float32)
                hi = hgeo["indices"] if np.linalg.det(transform[:3, :3]) >= 0 else hgeo["indices"][:, ::-1]
                layers.append({"name": "__fragment_review", "color": "#ffb454", "mask": transfer(resample(highlight.astype(np.uint8), source, shape, displayed, order=0), "uint8"), "surface": {"positions": transfer(hp,"float32"), "source": transfer(hr,"float32"), "normals": transfer(normals(hp,hi),"float32"), "sourceNormals": transfer(normals(hr,hi),"float32"), "indices": transfer(hi,"uint32")}})
        tracts = []
        unavailable_tracts = []
        for tract in ([] if cache["tracts"] is not None else c.document.get("tractography", [])):
            try:
                sid = tract["source_series"]
                t = c.document["transforms"].get(sid)
                if sid == reference["id"]:
                    c.require_native_geometry(reference)
                    transform = np.linalg.inv(native)
                elif registered and t:
                    _, estimated_tract = display_affine(c, c.series(sid), preview=preview_alignment)
                    transform = np.linalg.inv(displayed) @ c.registration_matrix(sid, require_geometry=True)
                    if estimated_tract:
                        alignment_previews.append(sid)
                else:
                    unavailable_tracts.append(tract["id"])
                    continue
                points = world(transform, c.array(tract["points_ras_mm"])).astype(
                    np.float32
                )
                if len(points):
                    tracts.append(
                        {
                            "id": tract["id"],
                            "label": tract["label"], "sourceSeries": sid,
                            "alignment": ("independent" if not registered else "estimated"
                                          if (estimated if sid == reference["id"] else estimated_tract)
                                          else t["status"]),
                            "points": transfer(points, "float32"),
                            "offsets": transfer(c.array(tract["offsets"]), "uint32"),
                        }
                    )
            except ValueError:
                unavailable_tracts.append(tract["id"])
        if cache["tracts"] is None:
            cache["tracts"] = tracts
            cache["unavailable_tracts"] = unavailable_tracts
        unavailable.extend(cache["unavailable_tracts"])
        alignment_previews.extend(t["sourceSeries"] for t in cache["tracts"] if t["alignment"] == "estimated")
        if cache["mri"] is None:
            cache["mri"] = transfer(c.array(image_data), "float32")
            from .anatomy_siam_display import source_head_support
            support = None if args.get("original") else source_head_support(c, reference, check=request.check)
            if support is not None:
                if preview:
                    support = resample(support, affine(reference["affine"]), shape, native, order=0)
                cache["headSupport"] = transfer(support, "uint8")
        # Resolve source-textured cortical and operative surroundings. Mesh
        # vertices are in the selected display lattice, while these affines map
        # the acquisition's actual display samples into the same physical frame.
        appearance_sources = []
        source_ids = dict.fromkeys(layer["sourceSeries"] for layer in layers
                if (layer.get("displayKind") in {"gray-matter", "cerebral-cortex", "cerebellar-cortex"}
                    and layer["alignment"] != "source-grid preview")
                or (layer.get("displayKind") in {"bone", "skull", "brain"}
                    and layer["alignment"] in {"reference", "reviewed", "source-provided"}))
        for sid in source_ids:
            scan = c.series(sid)
            if sid == reference["id"]:
                scalar, texture_frame, frame = cache["mri"], displayed, source_displayed
            else:
                # The layer admission above already requires its frame and this
                # timepoint. Acquire the scalar's own lattice, not the label's.
                frame, _ = display_affine(c, scan, preview=preview_alignment)
                data, _, texture_frame, _ = source_display_grid(scan, frame)
                scalar = transfer(c.array(data), "float32")
            units = str(scan.get("intensity_units") or scan.get("metadata", {}).get("Units", "")).upper()
            appearance_sources.append({"seriesId": sid, "modality": scan["modality"],
                    "intensityUnits": "HU" if units in {"HU", "HOUNSFIELD", "HOUNSFIELD UNITS"} else "source",
                    "sourceSha256": scan["original"]["sha256"], "affine": texture_frame.tolist(),
                    "signal": scalar, "nativeShape": scan["data"]["shape"], "nativeAffine": frame.tolist()})
        result = {
            "tractography": cache["tracts"],
            **({"appearanceSources": appearance_sources} if appearance_sources else {}),
            **({"patientContext": patient_context} if patient_context else {}),
            "caseId": c.document["patient_id"],
            "caseHash": "sha256:" + hashlib.sha256((c.identity()+":"+reference["id"]).encode()).hexdigest(),
            "frame": "RAS+",
            "shape": list(shape),
            "affine": displayed.tolist(),
            "mri": cache["mri"],
            **({"headSupport": cache["headSupport"]} if "headSupport" in cache else {}),
            "compartments": layers,
            "labelSets": list(label_sets.values()),
            "modality": reference["modality"],
            "displayPreview": None if preview is None else {"factor": 2, "meaning": "reversible source-image display preview; anatomical measurements retain native grids"},
            "unavailableLayers": unavailable,
            "sourceFrameIssue": source_frame_issue,
            "patientFrameReady": registered and not estimated,
            "registrationStatus": ("alignment preview" if alignment_previews else "patient frame")
            if registered else "independent source frame",
            "alignmentPreviews": sorted(set(alignment_previews)),
            "alignmentQuality": {sid: deepcopy(c.document["transforms"][sid]["quality"]) for sid in set(alignment_previews) if c.document["transforms"].get(sid, {}).get("quality")},
            "snapshotId": c.identity(),
            "draft": bool(session.anatomy_drafts.get(key)),
        }
        session.anatomy_transfer_paths = set()
        _retain(session, result)
        if cacheable:
            session.anatomy_view_cache = cache
            _retain(session, cache)
        else:
            session.anatomy_view_cache = None
        return result
    if operation == "anatomyEdit":
        action = args.get("action")
        name = args.get("layerId")
        session.anatomy_removal.pop(key, None)
        if action == "reject":
            session.anatomy_drafts.pop(key, None)
        elif action == "commit":
            draft = session.anatomy_drafts[key]
            c = PatientImagingCollection(c.root, c.document)
            request.begin_commit()
            c.commit_edit(
                draft.layer_id,
                draft.mask,
                author=args.get("author", ""),
                expected=draft.snapshot_id,
                operations=draft.operations,
            )
            session.anatomy_collections[key] = c
            session.anatomy_drafts.pop(key)
        else:
            draft = session.anatomy_drafts.get(key) or AnatomyDraft(c, name)
            if draft.layer_id != name:
                raise ValueError("Commit or reject the current layer correction first")
            if action == "remove-components":
                report, labels = component_review(c, name, draft.original)
                if args.get("reviewId") != report["reviewId"]:
                    raise ValueError("Fragment review changed; refresh before previewing")
                ids = args.get("componentIds", [])
                known = {r["id"] for r in report["components"]}
                if not isinstance(ids, list) or any(type(i) is not int or i not in known for i in ids):
                    raise ValueError("Unknown component selection")
                after = draft.mask & ~np.isin(labels, ids)
                draft.replace(after, {"operation": action, "reviewId": report["reviewId"], "componentIds": ids})
            elif action in {"undo", "redo"}:
                draft.step(action)
            else:
                draft.apply(
                    c,
                    action,
                    cancelled=request.cancelled.is_set,
                    points=args.get("points", []),
                    radius_mm=float(args.get("radiusMm", 2)),
                    threshold=float(args.get("threshold", 0.5)),
                )
            session.anatomy_drafts[key] = draft
        draft = session.anatomy_drafts.get(key)
        return {
            "document": c.document,
            "draft": None
            if draft is None
            else {
                "layerId": draft.layer_id,
                "undo": len(draft.undo),
                "redo": len(draft.redo),
                "changedVoxels": int(np.count_nonzero(draft.mask != draft.original)),
            },
        }
    if operation == "anatomyGraph":
        from .anatomy_geometry import inspect_vessel_geometry

        return inspect_vessel_geometry(
            c, args.get("layerId", "vessels"), args.get("point"), args.get("displaySeriesId")
        )
    if operation == "anatomyPlan":
        if key in session.anatomy_drafts:
            raise ValueError("Commit or reject the correction before planning")
        if args.get("action") == "operativeField":
            from .anatomy_operative_field import operative_field

            result = operative_field(c, args, check=request.check)
            result["labels"] = session.transfers.array(result["labels"], result["labels"].dtype.name)
            if "intensity" in result:
                image = result["intensity"]
                image["signal"] = session.transfers.array(image["signal"], "uint8")
                overview = image["overview"]
                overview["signal"] = session.transfers.array(overview["signal"], "uint8")
            _retain(session, result)
            return result
        if args.get("action"):
            from .anatomy_approaches import dispatch_plan, NativeRouteGeometry

            if args["action"] in {"list", "selectTarget"}:
                from .anatomy_planning import native_target_options
                from .surgical_patient import surgical_environment_options

                result = dispatch_plan(c, args, commit=request.begin_commit, check=request.check)
                target_cache = getattr(session, "anatomy_plan_target_cache", None)
                if target_cache is None or target_cache[:2] != (c.root, c.identity()):
                    target_cache = (c.root, c.identity(), native_target_options(c, check=request.check),
                                    surgical_environment_options(c, check=request.check))
                    session.anatomy_plan_target_cache = target_cache
                from .anatomy_adjudication import acquisition_contributions
                if "acquisitions" not in target_cache[3]:
                    target_cache[3]["acquisitions"] = acquisition_contributions(c, target_cache[2], target_cache[3])
                result["targetOptions"] = target_cache[2]
                result["rehearsalOptions"] = target_cache[3]
                return result
            cache = getattr(session, "anatomy_plan_cache", None)
            if cache is None or cache.snapshot.snapshot_id != c.identity() or cache.c.root != c.root:
                cache = NativeRouteGeometry(PatientAnatomySnapshot.capture(c), check=request.check)
                session.anatomy_plan_cache = cache
            else:
                cache.check = request.check
            return dispatch_plan(c, args, geometry=cache, commit=request.begin_commit)
        from .anatomy_planning import route_clearance

        return route_clearance(
            PatientAnatomySnapshot.capture(c),
            args["start"],
            args["end"],
            args.get("radiusMm", 1.0),
        )
    if operation == "anatomyRemoval":
        if key in session.anatomy_drafts:
            raise ValueError(
                "Commit or reject the correction before geometric removal preview"
            )
        from .anatomy_planning import removal_preview

        if args.get("reset"):
            session.anatomy_removal.pop(key, None)
        else:
            session.anatomy_removal[key] = (current_id, removal_preview(
                PatientAnatomySnapshot.capture(c), args["point"], args["radiusMm"]
            ))
        return {
            "snapshotId": c.identity(),
            "preview": key in session.anatomy_removal,
            "meaning": "reversible geometric target removal only",
        }
    if operation == "anatomySection":
        from .anatomy_sections import section

        result = section(
            c,
            series_id=args["seriesId"],
            center=args["center"],
            normal=args["normal"],
            spacing_mm=args.get("spacingMm", 0.5),
            moving_id=args.get("movingId"),
            checkerboard=args.get("checkerboard", True),
            registration_review=bool(args.get("registrationReview")),
            original=bool(args.get("original")),
            check=request.check,
            layers=args.get("layers", []),
        )
        result["pixels"] = session.transfers.array(result["pixels"], "uint8")
        _retain(session, result)
        return result
    if operation == "anatomySample":
        cached = getattr(session, "anatomy_sample_snapshot", None)
        if cached is None or cached.collection.root != c.root or cached.snapshot_id != current_id:
            cached = session.anatomy_sample_snapshot = PatientAnatomySnapshot.capture(c)
        return cached.sample(args["point"], layer_id=args.get("layerId"), independent_series=args.get("independentSeries"),
                             inspect_sources=True)
    if operation == "anatomyReconstruct":
        if key in session.anatomy_drafts:
            raise ValueError("Commit or reject the correction before reconstruction")
        from .anatomy_models import reconstruct, import_proposal

        result = reconstruct(
            c,
            args["seriesId"],
            args["model"],
            cancelled=request.cancelled.is_set,
            synthseg_version=args.get("synthsegVersion", 2),
            brain_layer_id=args.get("brainLayerId"),
        )
        c = PatientImagingCollection(c.root, c.document)
        import_proposal(c, args["seriesId"], result)
        request.begin_commit()
        c.save("reconstruction " + args["model"], expected=args["snapshotId"])
        session.anatomy_collections[key] = c
        return {"document": c.document, "result": result}
    if operation == "anatomyRegister":
        if args.get("action") == "ensure" and (
            c.document.get("preparation_workflow") or c.document.get("migration_source")
            or c.path("anatomy-preparation.json").is_file()
        ):
            # Prepare owns registration and its retry checkpoints. Source viewing
            # revalidates stored transforms and can open an independent scan.
            return {"document": c.document, "result": {"changed": False}}
        reference_id = c.document.get("reference_series")
        c = PatientImagingCollection(c.root, c.document)
        previous_view = read_view(c)
        from .anatomy_import import review_registration

        if args.get("action") == "reviewNativeWall":
            from .anatomy_local_wall_review import PROPERTY, review_native_wall_alignment

            previous = c.document["transforms"].get(args["seriesId"], {}).get(PROPERTY)
            result = review_native_wall_alignment(c, args["seriesId"], args["movingPoints"],
                args["referencePoints"], args["reviewer"], args.get("nativeWallScope"))
            if result == previous:
                return {"document": c.document, "result": result}
        elif args.get("action") == "review":
            result = review_registration(
                c,
                args["seriesId"],
                args["movingPoints"],
                args["referencePoints"],
                args["reviewer"],
            )
        else:
            from .anatomy_alignment import ensure_alignment
            estimate = args.get("action") == "estimate"
            changed = ensure_alignment(c, force=estimate, series_id=args.get("seriesId") if estimate else None,
                                       cancelled=request.cancelled.is_set, progress=progress)
            result = {"changed": changed}
            if not changed:
                return {"document": c.document, "result": result}
        request.begin_commit()
        c.save("registration " + args.get("seriesId", "study"), expected=args["snapshotId"])
        session.anatomy_collections[key] = c
        session.anatomy_drafts.pop(key, None)
        retired_clipping = []
        if c.document.get("reference_series") != reference_id:
            session.anatomy_removal.pop(key, None)
        elif previous_view:
            previous_view["snapshotId"] = c.identity()
            retired_clipping = save_view(c, previous_view, after_registration=True)
        return {"document": c.document, "result": result,
                **({"retiredClipping": retired_clipping} if retired_clipping else {})}
    if operation == "anatomyImport":
        reference_id = c.document.get("reference_series")
        c = PatientImagingCollection(c.root, c.document)
        if args.get("kind") in {
            "labels",
            "tractography",
            "motor",
            "language",
            "functional",
        }:
            from .anatomy_import import import_anatomy_derivative

            import_anatomy_derivative(
                c,
                args["path"],
                source_series=args["seriesId"],
                kind=args["kind"],
                label=args.get("label", "Imported anatomy"),
                role_map=args.get("roleMap"),
            )
        elif args.get("kind") == "dicom":
            from .anatomy_import import import_dicom

            import_dicom(
                c,
                args["path"],
                modality=args.get("modality"),
                sequence_by_series=args.get("sequenceBySeries"),
                timepoint=args.get("timepoint", "preoperative"),
            )
        else:
            if args.get("kind") != "nifti":
                raise ValueError("Unsupported anatomy import kind")
            from .anatomy_import import import_nifti_series

            imported = import_nifti_series(
                c,
                args["path"],
                series_id=str(uuid.uuid4()),
                modality=args["modality"],
                timepoint=args.get("timepoint", "preoperative"),
                patient_id=args["patientId"],
            )
            imported["label"] = Path(args["path"]).name.removesuffix(".gz").removesuffix(".nii")
        if args.get("kind") in {"nifti", "dicom"}:
            from .anatomy_import import select_intake_reference

            select_intake_reference(c)
        request.begin_commit()
        c.save("import patient imaging", expected=args["snapshotId"])
        if c.document.get("reference_series") != reference_id:
            session.anatomy_removal.pop(key, None)
        session.anatomy_collections[key] = c
        return {"document": c.document}
    if operation == "anatomyExport":
        snapshot = PatientAnatomySnapshot.capture(c)
        grid = snapshot.simulation_grid()
        destination = Path(args["path"]) / ("anatomy-" + snapshot.snapshot_id)
        destination.mkdir(parents=True, exist_ok=True)
        arrays = {f"target_{k}": v for k, v in grid["targets"].items()}
        arrays.update({f"critical_{k}": v for k, v in grid["critical"].items()})
        np.savez_compressed(
            destination / "anatomy-simulation.npz", affine=grid["affine"], **arrays
        )
        receipt = {
            k: v for k, v in grid.items() if k not in {"targets", "critical", "affine"}
        }
        receipt.update(
            targets=list(grid["targets"]),
            critical=list(grid["critical"]),
            affine=grid["affine"].tolist(),
        )
        (destination / "snapshot.json").write_text(json.dumps(receipt, indent=2))
        return receipt
    raise ValueError("Unsupported anatomy operation")
