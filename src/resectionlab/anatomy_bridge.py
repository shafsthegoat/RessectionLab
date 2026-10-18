"""Desktop patient anatomy operations, using existing bounded binary transfers."""

from copy import deepcopy
from pathlib import Path
import uuid
import hashlib
import json
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
from .anatomy_geometry import mesh, normals
from .anatomy_qa import layer_quality
from .anatomy_display import display_affine, brain_vessel_display


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
    # Only one immutable displayed source is cached. Any operation outside a
    # view conservatively invalidates descriptors (including edits/registration).
    if operation != "anatomyView":
        session.anatomy_view_cache = None
    if not hasattr(session, "anatomy_collections"):
        session.anatomy_packages = {}
        session.anatomy_collections = {}
        session.anatomy_drafts = {}
        session.anatomy_transfer_paths = set()
        session.anatomy_removal = {}
    if operation == "anatomyCreate":
        c = PatientImagingCollection.create(args["path"], args["patientId"], timepoint=args.get("timepoint"))
        request.begin_commit()
        c.save("create patient imaging collection")
        key = str(uuid.uuid4())
        session.anatomy_packages = {}
        session.anatomy_collections = {key: c}
        session.anatomy_drafts = {}
        session.anatomy_removal = {}
        return {"collectionId": key, "document": c.document, "viewState": read_view(c)}
    if operation == "anatomyOpen":
        path = Path(args["path"]).resolve()
        package_path = path if path.suffix == ".rsworkspace" else None
        if path.suffix == ".rsworkspace":
            from .anatomy_package import open_package
            path = open_package(path, args["workingRoot"], check=request.check)
        if path.name != "patient-anatomy.json":
            raise ValueError("Select patient-anatomy.json")
        c = PatientImagingCollection(path.parent)
        key = str(uuid.uuid4())
        session.anatomy_collections = {key: c}
        session.anatomy_drafts = {}
        session.anatomy_removal = {}
        session.anatomy_transfer_paths = set()
        session.anatomy_packages = {key: package_path} if package_path else {}
        return {"collectionId": key, "document": c.document, "viewState": read_view(c)}
    key = args.get("collectionId")
    if key not in session.anatomy_collections:
        raise ValueError("Patient collection is not open")
    c = session.anatomy_collections[key]
    current_id = json.loads((c.root / "patient-anatomy.json").read_text()).get(
        "snapshot_id"
    )
    if args.get("snapshotId") != c.identity() or current_id != c.identity():
        raise ValueError("Anatomy changed; reload the patient")
    request.check()
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
        native = affine(reference["affine"])
        t = c.document["transforms"].get(reference["id"])
        preview_alignment = args.get("previewAlignment") is True
        vessel_scope = args.get("vesselScope", "all")
        if vessel_scope not in {"all", "brain", "intracranial", "excluded"}:
            raise ValueError("Unknown vessel display scope")
        try:
            displayed, estimated = display_affine(c, reference, preview=preview_alignment)
            registered = True
        except ValueError:
            displayed, estimated, registered = native, False, False
        alignment_previews = [reference["id"]] if estimated else []
        shape = tuple(reference["data"]["shape"])
        cache_key = (key, c.identity(), reference["id"], reference["data"]["sha256"],
                     displayed.tobytes(), bool(args.get("original")), preview_alignment, vessel_scope)
        cacheable = not (session.anatomy_drafts.get(key) or key in session.anatomy_removal
                         or args.get("highlightComponents"))
        cache = getattr(session, "anatomy_view_cache", None)
        if not cacheable or not cache or cache["key"] != cache_key:
            cache = {"key": cache_key, "layers": {}, "label_sets": {}, "mri": None, "tracts": None}
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
            row_previews = []
            if row["source_series"] == reference["id"]:
                source = (
                    affine(row["affine"]) if not registered else display_affine(c, row, preview=preview_alignment)[0]
                )
            else:
                try:
                    if not registered:
                        raise ValueError("selected image is unregistered")
                    if c.series(row["source_series"])["timepoint"] != reference["timepoint"]:
                        raise ValueError("Separate study timepoint")
                    source, layer_estimated = display_affine(c, row, preview=preview_alignment)
                    if layer_estimated:
                        row_previews.append(row["source_series"])
                        alignment_previews.extend(row_previews)
                except ValueError:
                    unavailable.append(name)
                    continue
            progress(
                (i + 0.1) / max(1, len(requested)), f"Preparing native {row['label']}"
            )
            if key in session.anatomy_removal and row["role"] == "target":
                preview = session.anatomy_removal[key]
                ref = c.series(c.document["reference_series"])
                cut = (
                    resample(
                        preview.astype(np.uint8),
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
            quality = layer_quality(c, row)
            source_volume = float(c.mask(row).sum() * abs(np.linalg.det(affine(row["affine"])[:3, :3])))
            display_filter = None
            # Correction review always shows the complete editable source.
            if (vessel_scope == "brain" and row["kind"] in {"vessels", "artery", "vein", "sinus"}
                    and not draft and not args.get("highlightLayer")):
                row, display_filter = brain_vessel_display(c, row)
            if (vessel_scope in {"intracranial", "excluded"} and row["kind"] in {"vessels", "artery", "vein", "sinus"}
                    and not draft and not args.get("highlightLayer")):
                from .anatomy_intracranial import intracranial_vessels
                row, display_filter = intracranial_vessels(c, row, excluded=vessel_scope == "excluded", cancelled=request.cancelled.is_set)
            geometry = mesh(c, row, cancelled=request.cancelled.is_set)
            request.check()
            shared = row.get("kind") == "brain-region" and not row.get("edits")
            set_id = row["data"]["sha256"] if shared else None
            if shared and set_id not in label_sets:
                if set_id not in cache["label_sets"]:
                    values = resample(c.array(row["data"]), source, shape, displayed, order=0).astype(np.uint16)
                    cache["label_sets"][set_id] = {"id": set_id, "data": transfer(values, "uint16")}
                label_sets[set_id] = cache["label_sets"][set_id]
            mask = None if shared else resample(c.mask(row).astype(np.uint8), source, shape, displayed, order=0)
            record = {
                "name": name,
                "kind": row["kind"],
                "sourceIdentity": quality["sourceIdentity"],
                "qa": quality,
                "surfaceDiagnostics": {"policy": str(geometry.get("policy_version", "legacy")),
                    "kind": "tumor" if row["kind"] == "tumor" else "vessel" if row["kind"] in {"vessels", "artery", "vein", "sinus"} else "context",
                    "maxDisplacementMm": float(geometry["max_displacement_mm"]),
                    "restoredComponents": int(geometry.get("restored_components", 0)),
                    "backedOffVertices": int(geometry.get("backed_off_vertices", 0)),
                    "fallbackReason": str(geometry.get("fallback_reason", "")),
                    "intersectionCheck": "source-restored" if bool(geometry.get("smoothing_rejected", False)) or str(geometry.get("fallback_reason", "")) else "passed"},
                "label": row["label"],
                "color": row["color"],
                "alignmentPreviews": row_previews,
                "displayFilter": display_filter,
                **({"labelSet": set_id, "labelValue": row["label_value"]} if shared else {"mask": transfer(mask, "uint8")}),
                "volumeMm3": source_volume,
            }
            if True:  # Empty geometry is explicit; never rebuild it from a categorical placeholder.
                transform = np.linalg.inv(displayed) @ source
                p = world(transform, geometry["positions"]).astype(np.float32)
                raw = world(transform, geometry["source"]).astype(np.float32)
                faces = geometry["indices"]
                record["surface"] = {"empty": True} if not p.size else {
                    "positions": transfer(p, "float32"),
                    "source": transfer(raw, "float32"),
                    "normals": transfer(normals(p, faces), "float32"),
                    "sourceNormals": transfer(normals(raw, faces), "float32"),
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
                hi = hgeo["indices"]
                layers.append({"name": "__fragment_review", "color": "#ffb454", "mask": transfer(resample(highlight.astype(np.uint8), source, shape, displayed, order=0), "uint8"), "surface": {"positions": transfer(hp,"float32"), "source": transfer(hr,"float32"), "normals": transfer(normals(hp,hi),"float32"), "sourceNormals": transfer(normals(hr,hi),"float32"), "indices": transfer(hi,"uint32")}})
        tracts = []
        for tract in ([] if cache["tracts"] is not None else c.document.get("tractography", [])):
            try:
                sid = tract["source_series"]
                t = c.document["transforms"].get(sid)
                if sid == reference["id"]:
                    transform = np.linalg.inv(native)
                elif (
                    registered
                    and t
                    and t["status"] in {"reference", "reviewed", "source-provided"}
                ):
                    transform = np.linalg.inv(displayed) @ affine(t["matrix"])
                else:
                    continue
                points = world(transform, c.array(tract["points_ras_mm"])).astype(
                    np.float32
                )
                if len(points):
                    tracts.append(
                        {
                            "id": tract["id"],
                            "label": tract["label"],
                            "points": transfer(points, "float32"),
                            "offsets": transfer(c.array(tract["offsets"]), "uint32"),
                        }
                    )
            except ValueError:
                continue
        if cache["tracts"] is None:
            cache["tracts"] = tracts
        if cache["mri"] is None:
            cache["mri"] = transfer(c.array(reference["data"]), "float32")
        result = {
            "tractography": cache["tracts"],
            "caseId": c.document["patient_id"],
            "caseHash": "sha256:" + hashlib.sha256((c.identity()+":"+reference["id"]).encode()).hexdigest(),
            "frame": "RAS+",
            "shape": list(shape),
            "affine": displayed.tolist(),
            "mri": cache["mri"],
            "compartments": layers,
            "labelSets": list(label_sets.values()),
            "modality": reference["modality"],
            "unavailableLayers": unavailable,
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
            c, args.get("layerId", "vessels"), args.get("point")
        )
    if operation == "anatomyPlan":
        if key in session.anatomy_drafts:
            raise ValueError("Commit or reject the correction before planning")
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
            session.anatomy_removal[key] = removal_preview(
                PatientAnatomySnapshot.capture(c), args["point"], args["radiusMm"]
            )
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
            layers=args.get("layers", []),
        )
        result["pixels"] = session.transfers.array(result["pixels"], "uint8")
        _retain(session, result)
        return result
    if operation == "anatomySample":
        return PatientAnatomySnapshot.capture(c).sample(args["point"])
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
        )
        c = PatientImagingCollection(c.root, c.document)
        import_proposal(c, args["seriesId"], result)
        request.begin_commit()
        c.save("reconstruction " + args["model"], expected=args["snapshotId"])
        session.anatomy_collections[key] = c
        return {"document": c.document, "result": result}
    if operation == "anatomyRegister":
        c = PatientImagingCollection(c.root, c.document)
        previous_view = read_view(c)
        from .anatomy_import import review_registration

        if args.get("action") == "review":
            result = review_registration(
                c,
                args["seriesId"],
                args["movingPoints"],
                args["referencePoints"],
                args["reviewer"],
            )
        else:
            from .anatomy_alignment import ensure_alignment
            changed = ensure_alignment(c, force=args.get("action") == "estimate", cancelled=request.cancelled.is_set, progress=progress)
            result = {"changed": changed}
            if not changed:
                return {"document": c.document, "result": result}
        request.begin_commit()
        c.save("registration " + args.get("seriesId", "study"), expected=args["snapshotId"])
        if previous_view:
            previous_view["snapshotId"] = c.identity()
            save_view(c, previous_view)
        session.anatomy_collections[key] = c
        session.anatomy_drafts.pop(key, None)
        return {"document": c.document, "result": result}
    if operation == "anatomyImport":
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
            )
        elif args.get("kind") == "dicom":
            from .anatomy_import import import_dicom

            import_dicom(
                c,
                args["path"],
                modality=args.get("modality"),
                timepoint=args.get("timepoint", "preoperative"),
            )
        else:
            imported = c.add_series(
                args["path"],
                series_id=str(uuid.uuid4()),
                modality=args["modality"],
                timepoint=args.get("timepoint", "preoperative"),
                patient_id=args["patientId"],
            )
            imported["label"] = Path(args["path"]).name.removesuffix(".gz").removesuffix(".nii")
        request.begin_commit()
        c.save("import patient imaging", expected=args["snapshotId"])
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
