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
from .vascular_cleanup import (
    ATTACHED_ERASE, ATTACHED_SCORE_SURFACE, ROOTED_CLASS_PREFIXES, ROOTED_REVIEW_COLOR, ROOTED_REVIEW_PREFIX,
    ROOTED_VESSEL_KINDS, VESSEL_DETAILS, rooted_vessel_layer,
)
from .anatomy_display import (
    add_cta_arterial_coverage, brain_vessel_display, cta_coverage_is_current, display_affine, is_cta_coverage,
    patient_display_context, source_display_grid,
    source_lattice_annotation_preview,
)


VASCULAR_REVIEW_MANIFEST = "vascular-review.anatomy.json"
VASCULAR_REVIEW_REVISION = re.compile(r"vascular-review\.[0-9a-f]{12}\.anatomy\.json")
VASCULAR_REVIEW_READS = {
    "anatomyLocalAlignment",
    "anatomyView", "anatomySample", "anatomyGraph", "anatomySection", "anatomyComponents",
    "anatomyAttachedVesselReview", "anatomyVesselGapReview",
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


# "rooted" is the default display; "rooted-review" adds the hidden parts as an overlay.
VESSEL_SCOPES = {"rooted", "rooted-review", "all", "brain", "intracranial", "excluded", "exterior-air"}
# The viewer packs at most this many compartments; derived overlays never push past it.
MAX_DISPLAY_COMPARTMENTS = 128
_DERIVED = "__derived:"


def _overlay_record(c, row, name, color, source, shape, displayed, transfer, cancelled, *, mask=None):
    """A display-only overlay compartment; it never enters layers, revisions or planning."""
    geometry = mesh(c, row, cancelled=cancelled)
    transform = np.linalg.inv(displayed) @ source
    positions = world(transform, geometry["positions"]).astype(np.float32)
    raw = world(transform, geometry["source"]).astype(np.float32)
    indices = geometry["indices"] if np.linalg.det(transform[:3, :3]) >= 0 else geometry["indices"][:, ::-1]
    mask = c.mask(row) if mask is None else mask
    return {"name": name, "color": color,
            "mask": transfer(resample(mask.astype(np.uint8), source, shape, displayed, order=0), "uint8"),
            "surface": {"positions": transfer(positions, "float32"), "source": transfer(raw, "float32"),
                        "normals": transfer(normals(positions, indices), "float32"),
                        "sourceNormals": transfer(normals(raw, indices), "float32"),
                        "indices": transfer(indices, "uint32")}}


def _surface_assets(geometry, transform, source_only, transfer):
    raw = world(transform, geometry["source"]).astype(np.float32)
    positions = raw if source_only else world(transform, geometry["positions"]).astype(np.float32)
    if not positions.size:
        return {"empty": True}
    compatible_winding = np.linalg.det(transform[:3, :3]) >= 0
    faces = geometry["indices"] if compatible_winding else geometry["indices"][:, ::-1]
    shared_normals = geometry.get("shared_publisher_normals")
    same_positions = positions is raw or np.array_equal(positions.view(np.uint32), raw.view(np.uint32))
    normal_cache = geometry["normals"] if compatible_winding and not shared_normals else None
    if normal_cache is not None and np.array_equal(raw, geometry["positions"]):
        raw_normals = normal_cache
    else:
        raw_normals = normals(raw, faces)
    raw_asset = transfer(raw, "float32")
    raw_normal_asset = transfer(raw_normals, "float32")
    if shared_normals and not source_only:
        # Carry whole-surface normals into display IJK for continuous shading
        # across named seams.
        display_normals = geometry["normals"] @ np.linalg.inv(transform[:3, :3])
        display_normals /= np.maximum(np.linalg.norm(display_normals, axis=1, keepdims=True), 1e-20)
        normal_asset = transfer(display_normals, "float32")
    elif same_positions:
        normal_asset = raw_normal_asset
    else:
        display_normals = (normal_cache if normal_cache is not None
                           and np.array_equal(positions, geometry["positions"]) else normals(positions, faces))
        normal_asset = transfer(display_normals, "float32")
    return {
        "positions": raw_asset if same_positions else transfer(positions, "float32"),
        "source": raw_asset, "normals": normal_asset, "sourceNormals": raw_normal_asset,
        "indices": transfer(faces, "uint32"),
    }


def dispatch(session, operation, args, request, progress):
    # A source read does not change the cached view or its transfer descriptors.
    if operation not in {"anatomyView", "anatomySample", "anatomyComponents", "anatomyAttachedVesselReview", "anatomyVesselGapReview",
                         "anatomyLocalAlignment"}:
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
        session.anatomy_vessel_field_paths = set()
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
        session.anatomy_vessel_field_paths = set()
        session.anatomy_packages = {key: package_path} if package_path and not args.get("readOnlyPackage") else {}
        return {"collectionId": key, "document": c.document,
                "viewState": None if linked else read_view(c), "readOnlyStudy": linked}
    key = args.get("collectionId")
    inspection = getattr(session, 'native_wall_source', None)
    inspection = inspection if inspection and key == inspection.key else None
    if inspection:
        if operation != 'anatomyView':
            raise ValueError('Native-wall source inspection is read-only')
        if 'localVesselField' in args:
            raise ValueError('Select an ordinary Anatomy study for this inspection')
        scan = inspection.scans.get(args.get('seriesId'))
        if scan is None:
            raise ValueError('This scan is outside the saved rehearsal source binding')
        from .surgical_native_wall_source import check_source_current, local_source_view

        extent = args.get('sourceExtent', 'full')
        if extent == 'local':
            return local_source_view(inspection, scan, args.get('snapshotId'), session.transfers.array, request.check, progress)
        if extent != 'full':
            raise ValueError('Choose a local field or full acquisition')

        c = inspection.collection
        check_source_current(inspection, scan['seriesId'])
        args = {'collectionId': key, 'snapshotId': args.get('snapshotId'), 'seriesId': scan['seriesId'],
                'layers': [layer['id'] for layer in scan['layers']], 'original': True}
    else:
        if key not in session.anatomy_collections:
            raise ValueError("Patient collection is not open")
        if 'sourceExtent' in args:
            raise ValueError('Source inspection requires a saved rehearsal binding')
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
    if operation == 'anatomyView' and 'localVesselField' in args:
        if key in session.anatomy_drafts:
            raise ValueError('Resolve the current correction before opening the vessel field')
        from .native_source_field import anatomy_vessel_field

        manifest_hash = hashlib.sha256(manifest.read_bytes()).digest()
        view, paths = anatomy_vessel_field(c, args.get('seriesId'), args['localVesselField'],
                                        session.transfers.array, request.check, progress)
        if hashlib.sha256(manifest.read_bytes()).digest() != manifest_hash:
            raise ValueError('The source study changed while opening the vessel field')
        request.check()
        session.anatomy_vessel_field_paths = paths
        return view
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
    if operation == "anatomyCtaArterialCoverage":
        if key in session.anatomy_drafts:
            raise ValueError("Accept or reject the active correction first")
        candidate = PatientImagingCollection(c.root, c.document)
        progress(.15, "Verifying supplied arteries on the acquired CTA grid")
        layer, counts = add_cta_arterial_coverage(candidate, args.get("seriesId"), args.get("focusLayerIds"),
                                                  args.get("widerLayerIds"), check=request.check)
        request.check()
        if candidate.identity() != current_id:
            request.begin_commit()
            candidate.save("combine supplied arterial coverage for display", expected=current_id)
            session.anatomy_collections[key] = candidate
        return {"document": candidate.document, "layerId": layer["id"], "counts": counts}
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
        if is_cta_coverage(c.layer(args["layerId"])):
            raise ValueError("Select an input annotation to inspect vessel components")
        draft = session.anatomy_drafts.get(key)
        if draft and draft.layer_id != args["layerId"]:
            raise ValueError("Finish the current layer correction first")
        focus = args.get("focus")
        report, _ = component_review(c, args["layerId"], draft.mask if draft else None,
                                     focus=focus, check=request.check)
        if focus is not None:
            if json.loads(manifest.read_text()).get("snapshot_id") != current_id:
                raise ValueError("Anatomy changed; reload the patient")
            report["snapshotId"] = current_id
        request.check()
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
                c, row, footprints[finding["id"]], finding, center_ijk=args.get("centerIjk"), check=request.check,
            )}
        request.check()
        if json.loads(manifest.read_text()).get("snapshot_id") != current_id:
            raise ValueError("Anatomy changed; reload the patient")
        return report
    if operation == "anatomyVesselGapReview":
        from .anatomy_tof_gap_review import tof_gap_review

        if key in session.anatomy_drafts:
            raise ValueError("Resolve the current correction before reviewing TOF gaps")
        report = tof_gap_review(c, args["layerId"], args["seriesId"], finding_id=args.get("findingId"),
                                review_id=args.get("reviewId"), check=request.check, progress=progress)
        request.check()
        if json.loads(manifest.read_text()).get("snapshot_id") != current_id:
            raise ValueError("Anatomy changed; reload the patient")
        return {**report, "snapshotId": current_id}
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
        if vessel_scope not in VESSEL_SCOPES:
            raise ValueError("Unknown vessel display scope")
        # None lets the acquisition decide (fine on CT, major vessels otherwise).
        vessel_detail = args.get("vesselDetail")
        if vessel_detail is not None and vessel_detail not in VESSEL_DETAILS:
            raise ValueError("Unknown vessel detail")
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
        if inspection:
            image_data, preview = reference['data'], None
        else:
            image_data, native, displayed, preview = source_display_grid(reference, displayed)
        shape = tuple(image_data["shape"])
        source_cache_key = (key, c.identity(), reference["id"], image_data["sha256"],
                            displayed.tobytes(), bool(args.get("original")), preview_alignment, bool(patient_context))
        corridor_key = None
        if vessel_scope in {"rooted", "rooted-review"}:
            # Saved corridors widen the fine-vessel level of detail near them.
            from .anatomy_approaches import _read_plan
            try:
                corridor_key = _read_plan(c)[4]
            except (ValueError, OSError):
                corridor_key = "unreadable"
        cache_key = (source_cache_key, vessel_scope, vessel_detail, vessel_surface_mode, corridor_key)
        cacheable = not (session.anatomy_drafts.get(key) or key in session.anatomy_removal
                         or args.get("highlightComponents"))
        cache = getattr(session, "anatomy_view_cache", None)
        if not cacheable or not cache or cache["key"] != cache_key:
            previous_mri = cache.get("mri") if cache and cache["key"][0] == source_cache_key else None
            if previous_mri and not Path(previous_mri["path"]).is_file():
                previous_mri = None
            support = {"headSupport": cache["headSupport"]} if previous_mri and "headSupport" in cache else {}
            if previous_mri and "ctExterior" in cache:
                previous_exterior = cache["ctExterior"]
                if previous_exterior is None or Path(previous_exterior["mask"]["path"]).is_file():
                    support["ctExterior"] = previous_exterior
            cache = {"key": cache_key, "layers": {}, "label_sets": {}, "mri": previous_mri, "tracts": None, **support}
        # Publish only a complete response. A cancelled/failed preparation may
        # have temporary transfer files pruned and must not cache those paths.
        cache = {**cache, "layers": dict(cache["layers"]), "label_sets": dict(cache["label_sets"])}
        layers = []
        derived = []
        label_sets = {}
        unavailable = []
        transfer = session.transfers.array
        for i, name in enumerate(requested):
            request.check()
            if name in cache["layers"]:
                record = deepcopy(cache["layers"][name])
                layers.append(record)
                derived.extend(deepcopy(cache["layers"].get(_DERIVED + name, [])))
                alignment_previews.extend(record.get("alignmentPreviews", []))
                if record.get("labelSet"):
                    set_id = record["labelSet"]
                    label_sets[set_id] = cache["label_sets"][set_id]
                continue
            row = deepcopy(c.layer(name))
            if is_cta_coverage(row) and not cta_coverage_is_current(c, row):
                unavailable.append(name)
                continue
            draft = session.anatomy_drafts.get(key)
            if draft and draft.layer_id == name and not args.get("original"):
                if draft.mode == ATTACHED_ERASE:
                    row.setdefault("original_data", row["data"])
                    row.setdefault("original_label_value", row["label_value"])
                row["data"] = c.put_array(draft.mask.astype(np.uint8))
                row["label_value"] = 1
                row["edits"] = [{"draft": True, **({"surface_policy": ATTACHED_SCORE_SURFACE}
                                                    if draft.mode == ATTACHED_ERASE else {})}]
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
            mask_binding = (row["data"], row["label_value"])
            native_mask = c.mask(row, check=request.check)
            quality = layer_quality(c, row, verify_probability=not source_surface_only,
                                    mask=native_mask, check=request.check)
            source_volume = None if source_grid_preview else float(
                quality["voxelCount"] * abs(np.linalg.det(affine(row["affine"])[:3, :3]))
            )
            if source_grid_preview:
                message = "Annotation header omits spatial units. Preview on its supplied source grid; physical measurements unavailable."
                quality = {**quality, "issues": [message, *quality["issues"]]}
            from .anatomy_siam_display import supported_tissue_layer
            source_support = None
            if not inspection and not (draft and draft.layer_id == name) and args.get("highlightLayer") != name:
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
            rooted_review = rooted_class = None
            # The default view: root-reachable vessels, bridged where acquired
            # signal supports it; detached and thin terminal parts go to review.
            if (not raw_siam and not source_grid_preview and vessel_scope in {"rooted", "rooted-review"}
                    and row["kind"] in ROOTED_VESSEL_KINDS and not draft and not args.get("original")
                    and not args.get("highlightLayer")):
                try:
                    row, display_filter, rooted_review, rooted_class = rooted_vessel_layer(
                        c, row, review=vessel_scope == "rooted-review", detail=vessel_detail, check=request.check)
                except InterruptedError:
                    raise
                except Exception as error:  # One layer's derivative must not fail the whole view.
                    request.check()  # A cancellation still cancels.
                    display_filter = {"scope": vessel_scope, "failed": True, "sourcePreserved": True,
                                      "reason": f"Connected tree unavailable for this layer ({type(error).__name__}: "
                                                f"{error}). Showing the full source estimate."}
            score_source = vascular_cleanup_source(c, unfiltered)
            if score_source is None and row is not unfiltered:
                score_source = unfiltered
            if (row["data"], row["label_value"]) != mask_binding:
                native_mask = c.mask(row, check=request.check)
            geometry = mesh(c, row, cancelled=request.cancelled.is_set,
                            display_source=score_source, conditioned=not source_surface_only,
                            cache_directory=inspection.workspace / 'meshes' if inspection else None,
                            mask=native_mask, check=request.check)
            request.check()
            shared = row.get("kind") == "brain-region" and not row.get("edits")
            # Identical categorical values may originate on different physical grids.
            set_id = hashlib.sha256(row["data"]["sha256"].encode() + source.tobytes()).hexdigest() if shared else None
            if shared and set_id not in label_sets:
                if set_id not in cache["label_sets"]:
                    values = resample(c.array(row["data"]), source, shape, displayed, order=0).astype(np.uint16)
                    cache["label_sets"][set_id] = {"id": set_id, "data": transfer(values, "uint16")}
                label_sets[set_id] = cache["label_sets"][set_id]
            mask = None if shared else resample(native_mask, source, shape, displayed, order=0)
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
            record["surface"] = _surface_assets(geometry, np.linalg.inv(displayed) @ source, source_surface_only, transfer)
            layers.append(record)
            if cacheable:
                cache["layers"][name] = deepcopy(record)
            extra = []
            if rooted_class is not None:
                # The other vessel class of this layer, drawn in its own colour.
                kind = rooted_class["kind"]
                extra.append({"label": f"{row['label']} · {kind} part", "kind": kind, "displayKind": kind,
                              **_overlay_record(c, rooted_class, ROOTED_CLASS_PREFIXES[kind] + name, rooted_class["color"],
                                                source, shape, displayed, transfer, request.cancelled.is_set)})
            if rooted_review is not None:
                extra.append({"label": "Detached and thin vessel parts (review)",
                              **_overlay_record(c, rooted_review, ROOTED_REVIEW_PREFIX + name, ROOTED_REVIEW_COLOR,
                                                source, shape, displayed, transfer, request.cancelled.is_set)})
            derived.extend(extra)
            if cacheable and extra:
                cache["layers"][_DERIVED + name] = deepcopy(extra)
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
                layers.append(_overlay_record(c, hrow, "__fragment_review", "#ffb454", source, shape, displayed,
                                              transfer, request.cancelled.is_set, mask=highlight))
        tracts = []
        unavailable_tracts = []
        for tract in ([] if inspection or cache["tracts"] is not None else c.document.get("tractography", [])):
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
        if ("ctExterior" not in cache and reference["modality"] in {"CT", "CTA"} and
                not preview and not args.get("original")):
            from .anatomy_ct_exterior import ct_exterior_sheet

            exterior = ct_exterior_sheet(c, reference, check=request.check)
            cache["ctExterior"] = None if exterior is None else {
                **exterior, "mask": transfer(exterior["mask"], "uint8"), "signalSha256": cache["mri"]["sha256"],
            }
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
            **({"ctExterior": cache["ctExterior"]} if cache.get("ctExterior") is not None else {}),
            "compartments": layers + derived[:max(0, MAX_DISPLAY_COMPARTMENTS - len(layers))],
            **({"omittedDerivedCompartments": len(derived) - max(0, MAX_DISPLAY_COMPARTMENTS - len(layers))}
               if len(layers) + len(derived) > MAX_DISPLAY_COMPARTMENTS else {}),
            "labelSets": list(label_sets.values()),
            "modality": reference["modality"],
            "sourceSeriesId": reference["id"],
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
        if inspection:
            check_source_current(inspection, reference['id'])
            inspection.transfer_paths.clear()
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
        if action != 'reject' and name and is_cta_coverage(c.layer(name)):
            raise ValueError("Select an input annotation to make a correction")
        session.anatomy_removal.pop(key, None)
        if action == "reject":
            session.anatomy_drafts.pop(key, None)
        elif action == "commit":
            draft = session.anatomy_drafts[key]
            if draft.mode == ATTACHED_ERASE:
                from .vascular_cleanup import review_attached_tof_layer

                if np.array_equal(draft.mask, draft.original):
                    raise ValueError("The native TOF preview has no change to accept")
                _, receipt = review_attached_tof_layer(c, c.layer(draft.layer_id), check=request.check)
                if receipt.get("reviewId") != draft.operations[0]["reviewId"]:
                    raise ValueError("Attached vessel review changed; discard the draft and refresh")
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
            mode = ATTACHED_ERASE if action == ATTACHED_ERASE else None
            draft = session.anatomy_drafts.get(key) or AnatomyDraft(c, name, mode=mode)
            if draft.layer_id != name:
                raise ValueError("Commit or reject the current layer correction first")
            if action not in {"undo", "redo"} and draft.mode != mode:
                raise ValueError("Finish the current correction before changing editing tools")
            if action == ATTACHED_ERASE:
                from .vascular_cleanup import erase_attached_tof

                row = c.layer(name)
                if row["source_series"] != args.get("seriesId"):
                    raise ValueError("Open the vessel estimate's native TOF acquisition")
                after, receipt = erase_attached_tof(c, row, draft.mask, review_id=args.get("reviewId"),
                    finding_id=args.get("findingId"), points=args.get("points"),
                    radius_mm=float(args.get("radiusMm", 1)), check=request.check)
                draft.replace(after, receipt)
            elif action == "remove-components":
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
                **({"mode": draft.mode} if draft.mode else {}),
            },
        }
    if operation == "anatomyGraph":
        from .anatomy_geometry import inspect_vessel_geometry

        name = args.get("layerId", "vessels")
        if is_cta_coverage(c.layer(name)):
            raise ValueError("Select an input annotation to inspect vessel branches")
        return inspect_vessel_geometry(c, name, args.get("point"), args.get("displaySeriesId"))
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
