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
    if not hasattr(session, "anatomy_collections"):
        session.anatomy_collections = {}
        session.anatomy_drafts = {}
        session.anatomy_transfer_paths = set()
        session.anatomy_removal = {}
    if operation == "anatomyCreate":
        c = PatientImagingCollection.create(args["path"], args["patientId"])
        request.begin_commit()
        c.save("create patient imaging collection")
        key = str(uuid.uuid4())
        session.anatomy_collections = {key: c}
        session.anatomy_drafts = {}
        session.anatomy_removal = {}
        return {"collectionId": key, "document": c.document, "viewState": read_view(c)}
    if operation == "anatomyOpen":
        path = Path(args["path"]).resolve()
        if path.name != "patient-anatomy.json":
            raise ValueError("Select patient-anatomy.json")
        c = PatientImagingCollection(path.parent)
        key = str(uuid.uuid4())
        session.anatomy_collections = {key: c}
        session.anatomy_drafts = {}
        session.anatomy_removal = {}
        session.anatomy_transfer_paths = set()
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
    if operation == "anatomySaveView":
        if key in session.anatomy_drafts:
            raise ValueError("Accept or reject the correction before saving")
        request.begin_commit()
        save_view(c, args["viewState"])
        return {"saved": True}
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
        registered = t and t["status"] in {"reference", "reviewed", "source-provided"}
        displayed = c.patient_affine(reference) if registered else native
        shape = tuple(reference["data"]["shape"])
        layers = []
        label_sets = {}
        unavailable = []
        transfer = session.transfers.array
        for i, name in enumerate(requested):
            request.check()
            row = deepcopy(c.layer(name))
            draft = session.anatomy_drafts.get(key)
            if draft and draft.layer_id == name and not args.get("original"):
                row["data"] = c.put_array(draft.mask.astype(np.uint8))
                row["label_value"] = 1
                row["edits"] = [{"draft": True}]
                row["revision"] += 1
            if row["source_series"] == reference["id"]:
                source = (
                    affine(row["affine"]) if not registered else c.patient_affine(row)
                )
            else:
                try:
                    if not registered:
                        raise ValueError("selected image is unregistered")
                    source = c.patient_affine(row)
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
            geometry = mesh(c, row, cancelled=request.cancelled.is_set)
            request.check()
            shared = row.get("kind") == "brain-region" and not row.get("edits")
            set_id = row["data"]["sha256"] if shared else None
            if shared and set_id not in label_sets:
                values = resample(c.array(row["data"]), source, shape, displayed, order=0).astype(np.uint16)
                label_sets[set_id] = {"id": set_id, "data": transfer(values, "uint16")}
            mask = None if shared else resample(c.mask(row).astype(np.uint8), source, shape, displayed, order=0)
            record = {
                "name": name,
                "label": row["label"],
                "color": row["color"],
                **({"labelSet": set_id, "labelValue": row["label_value"]} if shared else {"mask": transfer(mask, "uint8")}),
                "volumeMm3": float(
                    c.mask(row).sum()
                    * abs(np.linalg.det(affine(row["affine"])[:3, :3]))
                ),
            }
            if True:  # Empty geometry is explicit; never rebuild it from a categorical placeholder.
                transform = np.linalg.inv(displayed) @ source
                p = world(transform, geometry["positions"]).astype(np.float32)
                raw = world(transform, geometry["source"]).astype(np.float32)
                faces = geometry["indices"]
                record["surface"] = {
                    "positions": transfer(p, "float32"),
                    "source": transfer(raw, "float32"),
                    "normals": transfer(normals(p, faces), "float32"),
                    "sourceNormals": transfer(normals(raw, faces), "float32"),
                    "indices": transfer(faces, "uint32"),
                }
            layers.append(record)
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
        for tract in c.document.get("tractography", []):
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
        result = {
            "tractography": tracts,
            "caseId": c.document["patient_id"],
            "caseHash": "sha256:" + hashlib.sha256((c.identity()+":"+reference["id"]).encode()).hexdigest(),
            "frame": "RAS+",
            "shape": list(shape),
            "affine": displayed.tolist(),
            "mri": transfer(c.array(reference["data"]), "float32"),
            "compartments": layers,
            "labelSets": list(label_sets.values()),
            "modality": reference["modality"],
            "unavailableLayers": unavailable,
            "registrationStatus": "patient frame"
            if registered
            else "independent source frame",
            "snapshotId": c.identity(),
            "draft": bool(session.anatomy_drafts.get(key)),
        }
        session.anatomy_transfer_paths = set()
        _retain(session, result)
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
        from .anatomy_import import rigid_registration, review_registration

        if args.get("action") == "review":
            result = review_registration(
                c,
                args["seriesId"],
                args["movingPoints"],
                args["referencePoints"],
                args["reviewer"],
            )
        else:
            result = rigid_registration(
                c, args["seriesId"], cancelled=request.cancelled.is_set
            )
        request.begin_commit()
        c.save("registration " + args["seriesId"], expected=args["snapshotId"])
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
            c.add_series(
                args["path"],
                series_id=str(uuid.uuid4()),
                modality=args["modality"],
                timepoint=args.get("timepoint", "preoperative"),
                patient_id=args["patientId"],
            )
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
