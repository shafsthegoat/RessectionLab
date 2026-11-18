"""Inspect and publish one reviewed group of native imaging and annotations."""

from collections import defaultdict
import json
from math import prod
from pathlib import Path
import uuid

from nibabel.filebasedimages import ImageFileError
import numpy as np

from .anatomy_import import (
    _dicom_sources,
    _inspect_nifti,
    _timepoint,
    _unique_identity,
    _validated_role_map,
    import_anatomy_derivative,
    import_dicom,
    import_nifti_series,
    select_intake_reference,
)
from .patient_anatomy import MAX_ARRAY, PatientImagingCollection, digest, file_hash


KINDS = {"nifti", "dicom", "labels"}
MAX_PATHS = 128
MAX_ITEMS = 256


def _sources(kind, path, *, selected_series=None, check=lambda: None):
    path = Path(path).resolve(strict=True)
    if kind == "dicom":
        from pydicom.errors import InvalidDicomError

        try:
            for details, _header, files in _dicom_sources(path, selected_series):
                check()
                checksums = []
                for source in files:
                    check()
                    checksums.append(file_hash(source))
                members = {"dicom": sorted(checksums)}
                yield details, {}, {}, members
        except InvalidDicomError as error:
            raise ValueError("A selected DICOM series has unreadable metadata") from error
        return
    details, metadata, companions = _inspect_nifti(path)
    if kind == "labels":
        bounded = len(details["shape"]) == 3 and 0 < prod(details["shape"]) <= MAX_ARRAY // 4
        valid_units = details["units_status"] in {"mm", "unknown"}
        details.update(modality=None, candidates=[], storage="series", supported=bool(bounded and valid_units),
                       issues=[] if bounded and valid_units else ["Select a bounded 3D NIfTI annotation with millimeter geometry"])
        for key, source_key in (("patient_id", "SourcePatientID"), ("study_uid", "SourceStudyInstanceUID"),
                                ("visit_id", "SourceSessionID")):
            details[key] = _unique_identity([details.get(key), metadata.get(source_key)], key.replace("_", " "))
        details["timepoint"] = _timepoint(details["timepoint"], metadata.get("SourceTimepoint"))
    elif details["storage"] == "series" and (
        details["units_status"] not in {"mm", "unknown"} or not 0 < prod(details["shape"]) <= MAX_ARRAY // 4
    ):
        details.update(supported=False, issues=["Select a bounded 3D NIfTI scan with millimeter geometry"])
    check()
    members = {"original": file_hash(path)}
    members.update({key: checksum for key, (_path, checksum) in companions.items()})
    yield details, metadata, companions, members


def _receipt(kind, details, members):
    return digest({"kind": kind, "details": details, "members": members})


def inspect_intake(kind, paths, *, check=lambda: None):
    """Return source suggestions and a content receipt for each selectable input."""
    if not isinstance(kind, str) or kind not in KINDS or not isinstance(paths, list) or not 1 <= len(paths) <= MAX_PATHS:
        raise ValueError("Choose a bounded set of imaging files or DICOM folders")
    if any(not isinstance(path, str) or not path.strip() for path in paths):
        raise ValueError("Choose valid imaging paths")
    items = []
    for value in dict.fromkeys(paths):
        check()
        path = str(Path(value).resolve())
        try:
            for details, _metadata, _companions, members in _sources(kind, path, check=check):
                item = {
                    "path": path, "kind": kind, "label": details["label"],
                    "suggestedRole": details["modality"], "candidates": details["candidates"],
                    "storage": details["storage"], "patientId": details["patient_id"],
                    "studyId": details.get("visit_id") or details["study_uid"],
                    "timepoint": details["timepoint"], "warning": " · ".join(details["issues"]),
                    "supported": details["supported"], "sourceReceipt": _receipt(kind, details, members),
                }
                if kind == "dicom":
                    item["seriesUid"] = details["series_uid"]
                items.append(item)
        except (ValueError, OSError, RuntimeError, ImageFileError, ImportError) as error:
            check()
            items.append({"path": path, "kind": kind, "label": Path(path).name,
                          "supported": False, "warning": str(error)})
        if len(items) > MAX_ITEMS:
            raise ValueError("Choose a smaller set of scan series")
    return {"items": items}


def _reviewed_sources(items, check):
    selected, dicom_by_path = set(), defaultdict(list)
    for item in items:
        if (not isinstance(item, dict) or not isinstance(item.get("kind"), str)
                or item["kind"] not in KINDS or not isinstance(item.get("path"), str)):
            raise ValueError("Choose a reviewed imaging input")
        kind, path = item["kind"], str(Path(item["path"]).resolve(strict=True))
        uid = item.get("seriesUid")
        if not item.get("sourceReceipt") or (kind == "dicom") != isinstance(uid, str):
            raise ValueError("Inspect each selected input before import")
        selection = kind, path, uid
        if selection in selected:
            raise ValueError("Choose each reviewed input once")
        selected.add(selection)
        if kind == "dicom":
            dicom_by_path[path].append(uid)
    dicom = {
        (path, record[0]["series_uid"]): record
        for path, ids in dicom_by_path.items()
        for record in _sources("dicom", path, selected_series=ids, check=check)
    }
    reviewed = []
    for item in items:
        check()
        kind, path = item["kind"], str(Path(item["path"]).resolve(strict=True))
        record = dicom[path, item["seriesUid"]] if kind == "dicom" else next(_sources(kind, path, check=check))
        details, _metadata, _companions, members = record
        if not details["supported"] or _receipt(kind, details, members) != item["sourceReceipt"]:
            raise ValueError("Selected source changed or is unsupported; review it again before importing")
        reviewed.append(({**item, "path": path}, *record))
    return reviewed, dicom_by_path


def _check_association(collection, reviewed, patient_id, timepoint, alias):
    collection.verify_source_association(timepoint, patient_id)
    prior = [collection.document.get("intake_association", {}),
             *(row.get("acquisition_identity", {}) for row in collection.source_records())]
    details = [detail for _item, detail, *_rest in reviewed]
    source_patient = _unique_identity([detail["patient_id"] for detail in details], "source patient")
    if alias is not None and (not isinstance(alias, str) or not alias or alias != source_patient):
        raise ValueError("Enter the exact source patient ID shown during review")
    if source_patient is not None and source_patient != (alias or patient_id):
        raise ValueError("Source patient ID differs; confirm the specific source patient ID")
    result = {"patient_key": _unique_identity([*(item.get("patient_key") for item in prior),
                                                digest(source_patient) if source_patient else None], "source patient")}
    for key in ("visit_id", "study_uid"):
        result[key] = _unique_identity([*(item.get(key) for item in prior), *(detail.get(key) for detail in details)],
                                       "study visit")
    if any(detail["timepoint"] and detail["timepoint"] != timepoint for detail in details):
        raise ValueError("Selected source belongs to a different operative timepoint")
    return {key: value for key, value in result.items() if value is not None}


def _verify_imported(row, members):
    if "dicom" in members:
        actual = {"dicom": sorted(item["sha256"] for item in row.get("dicom_originals", []))}
    else:
        actual = {"original": row["original"]["sha256"],
                  **{key: value["sha256"] for key, value in row.get("companions", {}).items()}}
    if actual != members:
        raise ValueError("Selected source changed during import; review it again")


def _source_series_id(collection, item, admitted):
    series_id, input_index = item.get("seriesId"), item.get("sourceInputIndex")
    if input_index is not None:
        if series_id is not None or type(input_index) is not int:
            raise ValueError("Bind each annotation to one scalar source scan")
        series_id = admitted.get(input_index)
    source = next((row for row in collection.document["series"] if row["id"] == series_id), None)
    if source is None:
        raise ValueError("Bind each annotation to an imported scalar source scan")
    return source


def _verify_annotation_source(collection, source, metadata):
    annotated_issuer = metadata.get("IssuerOfPatientID")
    source_issuer = source.get("acquisition_identity", {}).get("issuer_key")
    if annotated_issuer and source_issuer and digest(str(annotated_issuer).strip()) != source_issuer:
        raise ValueError("Annotation patient issuer differs from the source scan")
    declared = _unique_identity([metadata.get("SourceSeriesInstanceUID"),
                                 metadata.get("ReferencedSeriesInstanceUID")], "annotation source series")
    if declared is None:
        return
    if source.get("dicom_originals"):
        uid = source["id"]
    else:
        sidecar = source.get("companions", {}).get("metadata")
        if sidecar is not None:
            original = collection.path(sidecar["path"])
            if file_hash(original) != sidecar["sha256"]:
                raise ValueError("Stored source sidecar changed")
            uid = json.loads(original.read_text()).get("SeriesInstanceUID")
        else:
            uid = None
    if declared != uid:
        raise ValueError("Annotation metadata identify a different source series")


def _attach_annotation(collection, item, metadata, companions, members, admitted):
    source = _source_series_id(collection, item, admitted)
    _verify_annotation_source(collection, source, metadata)
    mapping = _validated_role_map(item.get("roleMap"), None)
    if not mapping:
        raise ValueError("Declare the ontology of the supplied nonzero label values")
    label = item.get("label", Path(item["path"]).name)
    if not isinstance(label, str) or not label.strip() or len(label) > 256:
        raise ValueError("Annotation label must be 1–256 characters")
    identity = digest({"members": members, "source_series": source["id"], "ontology": mapping})
    if any(row.get("provenance", {}).get("intake_annotation_key") == identity for row in collection.document["layers"]):
        return
    if len(members) == 1:
        legacy = {
            row["provenance"].get("source_label_value"): row for row in collection.document["layers"]
            if row["source_series"] == source["id"]
            and row.get("provenance", {}).get("source", {}).get("sha256") == members["original"]
            and row.get("provenance", {}).get("kind") == "observed_source_annotation"
            and "intake_annotation_key" not in row["provenance"]
        }
        if legacy.keys() == mapping.keys() and all(
            row["kind"] == mapping[value]["kind"] and (mapping[value]["label"] in {None, row["label"]})
            for value, row in legacy.items()
        ):
            return
    original = collection.original(item["path"])
    originals = {key: collection.original(name) for key, (name, _checksum) in companions.items()}
    _verify_imported({"original": original, "companions": originals}, members)
    rows = import_anatomy_derivative(
        collection, collection.path(original["path"]), kind="labels",
        source_series=source["id"], label=label, role_map=item["roleMap"], retain_empty=True,
    )
    for row in rows:
        specification = mapping[row["label_value"]]
        if row["kind"] == "tumor":
            row["role"] = "context"
        row["provenance"].update(
            intake_annotation_key=identity, source_companions=originals,
            label_ontology={"value": row["label_value"], **specification, "role": row["role"]},
            planning_target_candidate=row["kind"] == "tumor",
        )
        row["source_grid_spacing_mm"] = source["grid_spacing_mm"]
        row["output_grid_spacing_mm"] = np.linalg.norm(np.asarray(row["affine"])[:3, :3], axis=0).tolist()


def reviewed_intake(collection, args, *, check=lambda: None, commit=lambda: None, progress=lambda *x: None):
    """Publish all reviewed scan and label associations in one anatomy revision."""
    items = args.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_ITEMS:
        raise ValueError("Review the selected imaging sources")
    if args.get("confirmedAssociation") is not True or "timepoint" not in args:
        raise ValueError("Confirm the selected patient and study visit before import")
    snapshot, patient_id, timepoint = args.get("snapshotId"), args.get("patientId"), args["timepoint"]
    if (timepoint == "source-session-unknown" and collection.document.get("study_timepoint") is None
            and collection.document["series"] and collection.document["series"][0]["timepoint"] is None):
        timepoint = None  # Keep the unspecified phase of older source-bound studies.
    if patient_id != collection.document["patient_id"]:
        raise ValueError("Selected patient differs from the open patient")
    if snapshot is None or snapshot != collection.identity() or snapshot != PatientImagingCollection(collection.root).identity():
        raise ValueError("Patient changed; reload before committing")
    reviewed, dicom_by_path = _reviewed_sources(items, check)
    alias = args.get("sourcePatientId")
    association = _check_association(collection, reviewed, patient_id, timepoint, alias)

    c = PatientImagingCollection(collection.root, collection.document)
    admitted, dicom_rows = {}, {}
    for index, (item, details, _metadata, _companions, members) in enumerate(reviewed):
        check()
        progress(index / len(reviewed), f"Importing reviewed source {index + 1} of {len(reviewed)}")
        kind, path = item["kind"], item["path"]
        if kind == "labels":
            continue
        if details["storage"] == "series" and not item.get("modality"):
            raise ValueError("Confirm the sequence role of each scalar scan")
        if kind == "nifti":
            row = import_nifti_series(
                c, path, series_id=str(uuid.uuid4()), modality=item.get("modality"), patient_id=patient_id,
                timepoint=timepoint, source_patient_id=alias if details["patient_id"] else None,
            )
        else:
            if path not in dicom_rows:
                sequence_map = {candidate["seriesUid"]: candidate.get("modality")
                                for candidate, *_rest in reviewed
                                if candidate["kind"] == "dicom" and candidate["path"] == path}
                dicom_rows[path] = {
                    row["id"]: row for row in import_dicom(
                        c, path, selected_series=dicom_by_path[path], sequence_by_series=sequence_map,
                        timepoint=timepoint, source_patient_id=alias,
                    )
                }
            row = dicom_rows[path][item["seriesUid"]]
        _verify_imported(row, members)
        if row.get("storage") != "source-asset":
            admitted[index] = row["id"]
    select_intake_reference(c)
    for item, _details, metadata, companions, members in reviewed:
        check()
        if item["kind"] == "labels":
            _attach_annotation(c, item, metadata, companions, members, admitted)
    check()
    if c.identity() != snapshot:
        c.document["preparation_workflow"] = "native-adaptive-v1"
        if association:
            c.document["intake_association"] = association
        commit()
        c.save("Reviewed imaging and annotation intake", expected=snapshot)
    elif PatientImagingCollection(c.root).identity() != snapshot:
        raise ValueError("Patient changed; reload before committing")
    return c
