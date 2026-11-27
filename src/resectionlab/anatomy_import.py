"""Local imaging intake and reviewable rigid registration."""

from pathlib import Path
import hashlib
import json
import re
from tempfile import TemporaryDirectory
import numpy as np
from .patient_anatomy import affine, world, reconstruction_lock, file_hash, digest, nifti_geometry, nifti_geometry_issue, projection_or_localizer


MRI_SEQUENCES = {
    "T1", "T1CE", "T2", "FLAIR", "MRA", "TOF", "MRV",
    "SWI", "SWI_MAGNITUDE", "SWI_PHASE",
}
CT_SEQUENCES = {"CT", "CTA"}
IMAGING_MODALITIES = MRI_SEQUENCES | CT_SEQUENCES
OPERATIVE_LABELS = {
    "tumor": ("tumor", "target", "#c56df1"),
    "brain": ("brain", "context", "#91c4b6"),
    "bone": ("skull", "context", "#d8e7f8"),
    "skull": ("skull", "context", "#d8e7f8"),
    "vessels": ("vessels", "critical", "#ef5679"),
    "vessel": ("vessels", "critical", "#ef5679"),
    "artery": ("artery", "critical", "#e86c62"),
    "vein": ("vein", "critical", "#678fde"),
    "sinus": ("sinus", "critical", "#8078ce"),
    "dura": ("dura", "context", "#eccca7"),
    "context": ("context", "context", "#97bdcf"),
}


def _validated_role_map(role_map, present):
    if not isinstance(role_map, dict):
        raise ValueError("roleMap must map numeric label values to operative roles")
    result = {}
    for raw_value, raw_spec in role_map.items():
        if isinstance(raw_value, bool) or not str(raw_value).isdigit():
            raise ValueError("roleMap keys must be positive integer label values")
        value = int(raw_value)
        if value < 1 or value > 65535 or value in result:
            raise ValueError("roleMap contains a duplicate or out-of-range label value")
        if isinstance(raw_spec, str):
            spec = {"kind": raw_spec}
        elif isinstance(raw_spec, dict) and set(raw_spec) <= {"kind", "label", "color"}:
            spec = raw_spec
        else:
            raise ValueError(f"Invalid roleMap entry for label {value}")
        key = spec.get("kind")
        if not isinstance(key, str) or key not in OPERATIVE_LABELS:
            raise ValueError(f"Unsupported operative role {key!r} for label {value}")
        kind, role, color = OPERATIVE_LABELS[key]
        display_label = spec.get("label")
        if display_label is not None and (not isinstance(display_label, str) or not display_label.strip() or len(display_label) > 128):
            raise ValueError(f"Invalid roleMap label for value {value}")
        if "color" in spec:
            color = spec["color"]
            if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                raise ValueError(f"Invalid roleMap color for value {value}")
        result[value] = {"kind": kind, "role": role, "color": color, "label": display_label}
    if present is not None and set(result) != present:
        missing = sorted(present - set(result))
        extra = sorted(set(result) - present)
        raise ValueError(f"roleMap must cover every nonzero label exactly; missing={missing}, absent={extra}")
    return result


def imaging_modality(modality, *, dicom_modality=None):
    """Require a semantic sequence; DICOM's MR is only an acquisition family."""
    source = str(dicom_modality).strip().upper() if dicom_modality is not None else None
    if source is not None and source not in {"MR", "CT"}:
        raise ValueError(f"Unsupported DICOM Modality {source!r}; import an MR or CT image series")
    if modality is None and source == "CT":
        return "CT"
    value = str(modality).strip().upper() if modality is not None else ""
    if value not in IMAGING_MODALITIES:
        raise ValueError(
            f"Unsupported imaging sequence {modality!r}; choose {', '.join(sorted(IMAGING_MODALITIES))} explicitly"
        )
    if source == "MR" and value not in MRI_SEQUENCES:
        raise ValueError(f"DICOM MR cannot be imported as {value}; choose an MRI sequence")
    if source == "CT" and value not in CT_SEQUENCES:
        raise ValueError(f"DICOM CT cannot be imported as {value}; choose CT or CTA")
    return value


def _normalized(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _sequence_hints(value):
    value = _normalized(value)
    patterns = {
        "MIP": r"(?:tof)?mip(?:s| 3d)?|(?:max(?:imum)?|min(?:imum)?) intensity projection",
        "DWI": r"dwi|dti|diffusion(?: weighted)?|adc",
        "MRV": r"mrv|mr venogra(?:m|phy)|venogram",
        "TOF": r"tof(?:2d|3d)?|time of flight",
        "MRA": r"mra|mr angiogra(?:m|phy)|angiomr",
        "CTA": r"cta|ct angio(?:gra(?:m|phy))?|angioct",
        "SWI_MAGNITUDE": r"swi(?:p)? (?:mag|magnitude)|(?:mag|magnitude) swi(?:p)?",
        "SWI_PHASE": r"swi(?:p)? phase|phase swi(?:p)?",
        "SWI": r"swi(?:p)?|susceptibility weighted(?: imaging)?",
        "FLAIR": r"(?:t2)?flair|fluid attenuated inversion recovery",
        "T1CE": r"t1(?:w)?(?:ce|gd|c|post|tce)|t1(?:w)? (?:ce|gd|c|post|postcontrast|post contrast|gad(?:olinium)?|contrast enhanced)|t1wce|(?:ce|post contrast|postcontrast) t1w?",
        "T1": r"t1w?|t1 weighted|mprage|bravo|tfe",
        "T2": r"t2w?|t2 weighted",
        "CT": r"ct|computed tomography",
    }
    hints = {role for role, pattern in patterns.items() if re.search(rf"\b(?:{pattern})\b", value)}
    for specific, general in (
        ("CTA", "CT"), ("T1CE", "T1"), ("FLAIR", "T2"), ("TOF", "MRA"),
        ("SWI_MAGNITUDE", "SWI"), ("SWI_PHASE", "SWI"),
    ):
        if specific in hints:
            hints.discard(general)
    return hints


def _describe_sequence(name, metadata, *, source_family=None, shape=None, gradients=False):
    fields = ("SequenceRole", "Modality", "SeriesDescription", "ProtocolName",
              "SequenceName", "PulseSequenceType", "ImageType", "ScanOptions")
    metadata_hints = set().union(*(_sequence_hints(metadata.get(key, "")) for key in fields))
    name_hints = _sequence_hints(name)
    if source_family == "CT" and metadata_hints - {"CT"}:
        metadata_hints.discard("CT")
    processed_swi = re.search(r"\b(?:swip|processed swi)\b", _normalized(
        " ".join([name, *(str(metadata.get(key, "")) for key in fields)])
    ))
    if "SWI" in metadata_hints | name_hints and not processed_swi:
        component = _normalized(metadata.get("ComplexImageComponent", metadata.get("Part", "")))
        image_type = _normalized(metadata.get("ImageType", "")).split()
        for tokens, role in (({"mag", "magnitude"}, "SWI_MAGNITUDE"),
                             ({"p", "phase"}, "SWI_PHASE")):
            if component in tokens or tokens & set(image_type):
                metadata_hints.add(role)
                metadata_hints.discard("SWI")
                name_hints.discard("SWI")
    all_hints = metadata_hints | name_hints
    coverage = {"status": "unassessed", "limitations": []}
    for names, pattern, flag in (
        (("SkullStripped", "BrainExtracted"), r"\b(?:skull ?stripped|brain ?extracted|brain only)\b", "skull-stripped"),
        (("Defaced", "PrivacyRedacted"), r"\b(?:defaced|privacy redacted)\b", "privacy-redacted"),
        (("Cropped",), r"\b(?:cropped|limited fov)\b", "cropped"),
    ):
        if any(metadata.get(key) is True for key in names):
            coverage["limitations"].append({"kind": flag, "basis": "source metadata"})
        elif re.search(pattern, _normalized(name)):
            coverage["limitations"].append({"kind": flag, "basis": "source name; verify on slices"})
    if coverage["limitations"]:
        coverage["status"] = "limited"
    elif metadata.get("FullHeadAcquired") is True:
        coverage["full_head_acquired"] = True
    if metadata.get("WholeBrainAcquired") is True:
        coverage["whole_brain"] = True
    issues = []
    if "MIP" in all_hints or "DWI" in all_hints or gradients:
        role = "MIP" if "MIP" in all_hints else "DWI"
        issues.append("Projection retained as a source asset" if role == "MIP" else
                      "Diffusion retained on its original grid with available gradient companions")
        return dict(modality=role, candidates=[role], storage="source-asset", coverage=coverage,
                    issues=issues, supported=True)
    if shape and len(shape) != 3:
        return dict(modality=None, candidates=[], storage="source-asset", coverage=coverage,
                    issues=["Select a bounded 3D anatomical volume or identified diffusion source"], supported=False)
    if source_family in {"MR", "CT"}:
        allowed = MRI_SEQUENCES if source_family == "MR" else CT_SEQUENCES
        if not all_hints and source_family == "CT":
            metadata_hints.add("CT")
        metadata_hints &= allowed
        name_hints &= allowed
    candidates = sorted(metadata_hints | name_hints)
    role = candidates[0] if len(candidates) == 1 else None
    if not role:
        issues.append("Conflicting sequence hints; choose a role" if candidates else
                      "Sequence role is ambiguous; choose a role")
    return dict(modality=role, candidates=candidates, storage="series", coverage=coverage,
                issues=issues, supported=source_family in {None, "CT", "MR"})


def _unique_identity(values, field):
    known = {str(value).strip() for value in values if value is not None and str(value).strip()}
    if len(known) > 1:
        raise ValueError(f"Source metadata disagree about {field}")
    return next(iter(known), None)


def _is_hounsfield(value):
    return _normalized(value) in {"hu", "hounsfield unit", "hounsfield units"}


def _timepoint(*values):
    texts = [_normalized(value) for value in values]
    return _unique_identity([
        phase for pattern, phase in ((r"\bpre ?op(?:erative)?\b", "preoperative"),
                                    (r"\bpost ?op(?:erative)?\b", "postoperative"))
        if any(re.search(pattern, value) for value in texts)
    ], "operative timepoint")


def _inspect_nifti(path, sidecar=None):
    import nibabel as nib

    path = Path(path)
    if not path.name.endswith((".nii", ".nii.gz")):
        raise ValueError("Select a NIfTI image")
    suffixless = path.name.removesuffix(".gz").removesuffix(".nii")
    if sidecar is not None and not Path(sidecar).is_file():
        raise ValueError("Specified NIfTI sidecar is missing")
    sidecar = Path(sidecar) if sidecar is not None else path.with_name(suffixless + ".json")
    metadata, companions = {}, {}
    if sidecar.exists():
        with sidecar.open("rb") as source:
            raw = source.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("NIfTI sidecar exceeds 1 MiB")
        metadata = json.loads(raw)
        if not isinstance(metadata, dict):
            raise ValueError("NIfTI sidecar must contain a JSON object")
        companions["metadata"] = (sidecar, hashlib.sha256(raw).hexdigest())
    for name in ("bval", "bvec"):
        gradient = path.with_name(suffixless + "." + name)
        if gradient.is_file():
            companions[name] = (gradient, file_hash(gradient))
    im = nib.load(path)
    geometry = nifti_geometry(im)
    source_family = str(metadata.get("Modality", "")).upper()
    coordinates = re.findall(r"(?:^|_)(sub|ses)-([a-zA-Z0-9]+)(?=_|$)", suffixless)
    subjects = [value for key, value in coordinates if key == "sub"]
    sessions = [value for key, value in coordinates if key == "ses"]
    # BIDS entities are externally declared identifiers. Header description
    # and acquisition dates alone do not identify a person or visit.
    subject_id = metadata.get("SubjectID")
    session_id = metadata.get("SessionID")
    patient = _unique_identity([metadata.get("PatientID"), str(subject_id).removeprefix("sub-") if subject_id else None,
                                *subjects], "patient")
    visit = _unique_identity([str(session_id).removeprefix("ses-") if session_id else None,
                              metadata.get("VisitID"), *sessions], "visit")
    study = metadata.get("StudyInstanceUID")
    if study is not None and not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", str(study)):
        raise ValueError("Invalid StudyInstanceUID in NIfTI sidecar")
    details = _describe_sequence(
        suffixless, metadata, source_family=source_family if source_family in {"MR", "CT"} else None,
        shape=im.shape, gradients=bool({"bval", "bvec"} & companions.keys()),
    )
    details.update(label=suffixless, patient_id=patient, visit_id=visit, study_uid=study,
                   timepoint=_timepoint(metadata.get("Timepoint"), metadata.get("OperativeTimepoint"), visit, suffixless),
                   shape=list(im.shape), units_status=geometry["spatial_units"], native_geometry=geometry)
    issue = nifti_geometry_issue(geometry)
    if issue and details["storage"] == "series":
        if geometry["spatial_units"] in {"mm", "unknown"}:
            details["issues"].append(issue + "; independent source viewing only")
        else:
            details.update(supported=False, issues=[issue, *details["issues"]])
    return details, metadata, companions


def inspect_nifti_series(path, *, sidecar=None):
    """Describe one NIfTI input and matching sidecar without importing voxels."""
    return _inspect_nifti(path, sidecar)[0]


def _source_identity(collection, details, *, timepoint, patient_id, source_patient_id, visit_id, namespace):
    collection.verify_source_association(timepoint, patient_id)
    patient = details["patient_id"]
    if patient is not None and patient != (source_patient_id or collection.document["patient_id"]):
        raise ValueError("Source patient ID differs; explicitly confirm the deidentified association")
    if source_patient_id and not patient:
        raise ValueError("Source patient ID could not be verified from metadata")
    visit = _unique_identity([details.get("visit_id"), visit_id], "visit")
    source_timepoint = _timepoint(details["timepoint"], visit)
    if source_timepoint and timepoint != source_timepoint:
        raise ValueError("Source metadata identify a different operative timepoint")
    study = details["study_uid"]
    identity = {
        "namespace": namespace, "study_uid": study, "visit_id": visit,
        "patient_key": digest(patient) if patient else None,
        "association": "source identifier and local confirmation" if patient else "explicit local study association",
    }
    for row in collection.source_records():
        prior = row.get("acquisition_identity", {})
        conflicts = [
            prior.get(key) and identity[key] and prior[key] != identity[key]
            for key in ("study_uid", "visit_id")
        ]
        if any(conflicts):
            raise ValueError("Source belongs to a different visit; open a separate study")
        if (prior.get("namespace") == namespace and prior.get("patient_key")
                and identity["patient_key"] and prior["patient_key"] != identity["patient_key"]):
            raise ValueError("Source belongs to a different patient")
    return identity


def import_nifti_series(collection, path, *, series_id, modality=None, timepoint="preoperative",
                        patient_id=None, sidecar=None, source_patient_id=None, visit_id=None):
    """Add a typed scalar source or preserve identified raw diffusion/projection."""
    import nibabel as nib

    details, metadata, companions = _inspect_nifti(path, sidecar)
    identity = _source_identity(collection, details, timepoint=timepoint, patient_id=patient_id,
                                source_patient_id=source_patient_id, visit_id=visit_id, namespace="nifti")
    explicit_role = str(modality).strip().upper() if modality is not None else None
    if details["modality"] is None and explicit_role in {"DWI", "MIP"}:
        details.update(storage="source-asset", modality=explicit_role, supported=True)
    if not details["supported"]:
        raise ValueError(details["issues"][0])
    is_asset = details["storage"] == "source-asset"
    if is_asset:
        sequence = details["modality"]
        if explicit_role is not None and explicit_role != sequence:
            raise ValueError("Select a bounded 3D anatomical volume; diffusion and projection cannot supply this sequence")
    else:
        family = str(metadata.get("Modality", "")).upper()
        sequence = imaging_modality(modality if modality is not None else details["modality"],
                                    dicom_modality=family if family in {"CT", "MR"} else None)
    source_hash = file_hash(path)
    companion_hashes = {key: checksum for key, (_, checksum) in companions.items()}
    for row in collection.source_records():
        if (row.get("timepoint") == timepoint and row["original"]["sha256"] == source_hash
                and row["modality"] == sequence
                and {key: value["sha256"] for key, value in row.get("companions", {}).items()} == companion_hashes
                and (not identity["visit_id"] or row.get("acquisition_identity", {}).get("visit_id") == identity["visit_id"])
                and (not identity["study_uid"] or row.get("acquisition_identity", {}).get("study_uid") == identity["study_uid"])):
            return row
    collection.verify_new_source_id(series_id)
    original = collection.original(path)
    retained = {key: collection.original(item) for key, (item, _) in companions.items()}
    if original["sha256"] != source_hash or {key: row["sha256"] for key, row in retained.items()} != companion_hashes:
        raise ValueError("Source or companion changed during import")
    imaging = {
        "label": details["label"], "coverage": details["coverage"],
        "acquisition_identity": identity,
        "sequence_assignment": {"basis": "explicit" if modality is not None else "metadata and name",
                                "candidates": details["candidates"], "issues": details["issues"]},
    }
    if projection_or_localizer({"metadata": metadata, "label": Path(path).name}):
        imaging["projection_or_localizer"] = True
    if retained:
        imaging["companions"] = retained
    acquisition = metadata.get("AcquisitionDateTime")
    if isinstance(acquisition, str) and acquisition:
        imaging["acquisition_datetime"] = acquisition
    unit_values = [metadata.get(key) for key in ("IntensityUnits", "RescaleType", "Units")]
    if sequence in CT_SEQUENCES and any(_is_hounsfield(value) for value in unit_values):
        imaging["intensity_units"] = "HU"
        imaging["intensity_units_basis"] = "NIfTI sidecar declaration"
    if is_asset:
        im = nib.load(collection.path(original["path"]))
        matrix = affine(im.affine)
        row = dict(id=series_id, patient_id=collection.document["patient_id"], original=original,
                   modality=sequence, timepoint=timepoint, storage="source-asset", affine=matrix.tolist(),
                   shape=list(im.shape), units_status=im.header.get_xyzt_units()[0], native_geometry=nifti_geometry(im),
                   grid_spacing_mm=nib.affines.voxel_sizes(matrix).tolist(), **imaging)
        collection.document.setdefault("source_assets", []).append(row)
    else:
        row = collection.add_series(path, series_id=series_id, modality=sequence, timepoint=timepoint,
                                    patient_id=patient_id, original=original)
        row.update(imaging)
        _swi_component(row)
    return row


def select_intake_reference(collection):
    """Keep an established patient frame, or replace an independent scout anchor."""
    document = collection.document
    current = document.get("reference_series")
    if current is None:
        return None
    reference_source = collection.series(current)
    baseline_reference = reference_source.get("diffusion_baseline") is not None
    provisional = baseline_reference or projection_or_localizer(reference_source)
    if not provisional and (collection.published_reference or len(document["transforms"]) != 1):
        return current
    order = ("T1", "T1CE", "T2", "FLAIR", "CT", "CTA", "TOF", "MRA", "MRV",
             "SWI", "SWI_MAGNITUDE", "SWI_PHASE")
    candidates = [row for row in document["series"] if row["modality"] in order and not projection_or_localizer(row)]
    candidates.sort(key=lambda row: (
        (nifti_geometry_issue(row["native_geometry"]) is not None if "native_geometry" in row
         else row.get("units_status") != "mm"),
        row.get("coverage", {}).get("status") == "limited",
        order.index(row["modality"]), row["id"],
    ))
    if provisional:
        # A placeholder can be displayed independently, but only a verified
        # metric acquisition can establish the replacement patient frame.
        candidates = (row for row in candidates if _verified_reference_geometry(collection, row))
    reference = next((row["id"] for row in candidates), current)
    if reference != current:
        transforms = document["transforms"]
        if provisional and (collection.published_reference or set(transforms) != {current}
                            or transforms[current]["status"] != "reference"):
            document.setdefault("registration_history", []).extend(
                {"series_id": series_id, "reference_series": current, "transform": transform,
                 "invalidation_reason": ("Previous reference was a diffusion baseline for inspection"
                                         if baseline_reference else "Previous reference was a projection or localizer")}
                for series_id, transform in transforms.items()
            )
        document["transforms"] = {
            reference: {"matrix": np.eye(4).tolist(), "status": "reference", "uncertainty_mm": None}
        }
        document["reference_series"] = reference
        if provisional and (display := document.get("display")):
            display.pop("cursor_ras_mm", None)
            display.pop("entry_ras_mm", None)
    return reference


def _verified_reference_geometry(collection, row):
    try:
        collection.require_native_geometry(row)
        original = row["original"]
        return file_hash(collection.path(original["path"])) == original["sha256"]
    except (OSError, ValueError):
        return False


def _swi_component(row):
    components = {"SWI": "processed susceptibility-weighted image",
                  "SWI_MAGNITUDE": "magnitude", "SWI_PHASE": "phase"}
    if row["modality"] in components:
        row["susceptibility_component"] = components[row["modality"]]
        row["susceptibility_interpretation"] = (
            "supplied source contrast; no venous segmentation or phase units inferred"
        )


def tractography_geometry(path):
    """Decode supported streamline formats into their declared physical RAS frame.

    Nibabel applies the TrackVis voxel-centre convention and header transform;
    callers must separately establish registration to the associated source MRI.
    """
    import nibabel as nib
    from .patient_anatomy import MAX_ARRAY

    tractogram = nib.streamlines.load(path, lazy_load=False).tractogram.to_world()
    streamlines = tractogram.streamlines
    if streamlines._data.nbytes > MAX_ARRAY:
        raise ValueError("Tractography exceeds local geometry limit")
    points = np.asarray(streamlines._data, dtype=np.float32)
    lengths = np.asarray([len(s) for s in streamlines], dtype=np.uint64)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or np.any(lengths < 2) or int(lengths.sum()) != len(points)):
        raise ValueError("Tractography requires finite RAS-mm polylines with at least two points")
    offsets = np.concatenate([np.zeros(1, np.uint64), np.cumsum(lengths)])
    if offsets.nbytes > MAX_ARRAY or offsets[-1] > np.iinfo(np.uint32).max:
        raise ValueError("Tractography exceeds local geometry limit")
    return points, offsets.astype(np.uint32)


def _dicom_sources(directory, selected_series=None):
    import pydicom
    import SimpleITK as sitk

    directory = str(Path(directory))
    ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory) or [])
    if not ids:
        raise ValueError("No readable DICOM image series")
    if selected_series is not None:
        if not isinstance(selected_series, (list, tuple, set)) or not selected_series \
                or len(set(selected_series)) != len(selected_series) or not set(selected_series) <= set(ids):
            raise ValueError("Select distinct known DICOM SeriesInstanceUIDs")
        ids = [uid for uid in ids if uid in selected_series]
    for uid in ids:
        if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", uid):
            raise ValueError("Invalid DICOM SeriesInstanceUID")
        files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, uid)
        if not files:
            raise ValueError(f"DICOM series {uid} contains no readable images")
        headers = [pydicom.dcmread(path, stop_before_pixels=True) for path in files]
        first = headers[0]
        fields = ("PatientID", "IssuerOfPatientID", "StudyInstanceUID", "SeriesInstanceUID",
                  "Modality", "FrameOfReferenceUID")
        if any(len({str(getattr(header, name, "")) for header in headers}) != 1 for name in fields):
            raise ValueError(f"DICOM series {uid} mixes patient, study, frame or acquisition identity")
        patient, study, family = (str(getattr(first, key, "")).strip() for key in
                                  ("PatientID", "StudyInstanceUID", "Modality"))
        if str(getattr(first, "SeriesInstanceUID", "")) != uid:
            raise ValueError(f"DICOM series {uid} contains a mismatched SeriesInstanceUID")
        meta_fields = ("SeriesDescription", "ProtocolName", "SequenceName", "ImageType",
                       "ScanOptions", "ComplexImageComponent", "ContrastBolusAgent")
        metadata = {key: getattr(first, key, None) for key in meta_fields}
        source_timepoint = _timepoint(*(getattr(first, key, None) for key in ("StudyDescription", "ProtocolName")))
        # A late slice may carry an explicit projection or diffusion image type.
        descriptions = [*metadata.values(), *[getattr(header, "ImageType", None) for header in headers[1:]]]
        header_hints = set().union(*(_sequence_hints(value) for value in descriptions))
        metadata["SequenceRole"] = " ".join(sorted(header_hints & {"MIP", "DWI"}))
        details = _describe_sequence("", metadata, source_family=family.upper())
        details["projection_or_localizer"] = (projection_or_localizer({"metadata": metadata}) or
            any(projection_or_localizer({"ImageType": getattr(h, "ImageType", "")}) for h in headers[1:]))
        if not patient or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", study):
            details["issues"].append("DICOM patient and StudyInstanceUID are required for study association")
            details["supported"] = False
        label = getattr(first, "SeriesDescription", None) or getattr(first, "ProtocolName", None) or "DICOM series"
        details.update(label=str(label), series_uid=uid, study_uid=study or None,
                       patient_id=patient or None, timepoint=source_timepoint)
        if family.upper() == "CT" and all(_is_hounsfield(getattr(header, "RescaleType", "")) for header in headers):
            details["intensity_units"] = "HU"
        yield details, first, files


def inspect_dicom_series(directory):
    """List each image series for explicit selection and ambiguous sequence review."""
    return [details for details, _, _ in _dicom_sources(directory)]


def import_dicom(collection, directory, *, modality=None, sequence_by_series=None, timepoint="preoperative",
                 selected_series=None, source_patient_id=None, visit_id=None):
    import SimpleITK as sitk

    sources = list(_dicom_sources(directory, selected_series))
    ids = {details["series_uid"] for details, _, _ in sources}
    if sequence_by_series is not None and not isinstance(sequence_by_series, dict):
        raise ValueError("sequenceBySeries must map DICOM SeriesInstanceUID to an explicit or suggested sequence")
    if modality is not None and sequence_by_series:
        raise ValueError("Use either modality for one DICOM series or sequenceBySeries for multiple series")
    if modality is not None and len(ids) != 1:
        raise ValueError("Multiple DICOM series require sequenceBySeries keyed by SeriesInstanceUID")
    if sequence_by_series is not None and set(sequence_by_series) != ids:
        raise ValueError("sequenceBySeries must identify every DICOM SeriesInstanceUID exactly once")
    _unique_identity([details["patient_id"] for details, _, _ in sources], "DICOM patient")
    _unique_identity([details["study_uid"] for details, _, _ in sources], "DICOM study")
    _unique_identity([str(getattr(header, "IssuerOfPatientID", "")) for _, header, _ in sources], "DICOM patient issuer")
    c = collection
    prepared = []
    for details, header, files in sources:
        uid = details["series_uid"]
        family = str(header.Modality).upper()
        if family not in {"MR", "CT"}:
            imaging_modality(None, dicom_modality=family)
        if not details["supported"]:
            raise ValueError("DICOM patient and StudyInstanceUID are required for study association")
        identity = _source_identity(c, details, timepoint=timepoint, patient_id=None,
                                    source_patient_id=source_patient_id, visit_id=visit_id, namespace="dicom")
        issuer = str(getattr(header, "IssuerOfPatientID", "")).strip()
        if issuer:
            identity["issuer_key"] = digest(issuer)
        for row in c.source_records():
            previous = row.get("acquisition_identity", {}).get("issuer_key")
            if previous and issuer and previous != identity["issuer_key"]:
                raise ValueError("DICOM patient issuer differs from the open study")
        explicit = sequence_by_series[uid] if sequence_by_series is not None else modality
        if details["storage"] == "source-asset":
            sequence = details["modality"]
            if explicit is not None and str(explicit).strip().upper() != sequence:
                raise ValueError("Diffusion and projection DICOM cannot supply a structural or vascular volume")
        elif explicit is None and details["modality"] is None:
            raise ValueError(f"Unsupported imaging sequence for DICOM series {uid}; {details['issues'][0]}")
        else:
            sequence = imaging_modality(explicit or details["modality"], dicom_modality=family)
        checksums = sorted(file_hash(path) for path in files)
        existing = next((row for row in c.source_records() if row["id"] == uid), None)
        if existing:
            stored = sorted(item["sha256"] for item in existing.get("dicom_originals", []))
            if stored != checksums or existing["modality"] != sequence or existing["timepoint"] != timepoint:
                raise ValueError(f"DICOM series {uid} conflicts with the existing source")
            prepared.append((existing, None))
            continue
        c.verify_new_source_id(uid)
        prepared.append((None, (uid, files, header, family, sequence, identity, details, checksums, explicit)))

    # Publish the candidate only after all selected series were decoded. A
    # failed read may leave unreferenced content-addressed files for reuse.
    staged = type(c)(c.root, {k: v for k, v in c.document.items() if k != "snapshot_id"})
    imported = []
    for existing, pending in prepared:
        if existing is not None:
            imported.append(next(row for row in staged.source_records() if row["id"] == existing["id"]))
            continue
        uid, files, header, family, sequence, identity, details, checksums, explicit = pending
        originals = [c.original(path) for path in files]
        if sorted(item["sha256"] for item in originals) != checksums:
            raise ValueError("DICOM source changed during import")
        retained_paths = [str(c.path(item["path"])) for item in originals]
        acquisition = str(getattr(header, "AcquisitionDateTime", "")) or None
        if not acquisition and getattr(header, "AcquisitionDate", None):
            acquisition = str(header.AcquisitionDate) + str(getattr(header, "AcquisitionTime", ""))
        metadata = dict(
            acquisition_identity=identity, label=details["label"], coverage=details["coverage"],
            dicom_modality=family, dicom_originals=originals, acquisition_datetime=acquisition,
            dicom_frame_uid=str(getattr(header, "FrameOfReferenceUID", "")) or None,
            sequence_assignment={"basis": "explicit" if explicit is not None else "acquisition metadata",
                                 "candidates": details["candidates"], "issues": details["issues"]},
        )
        if details.get("projection_or_localizer"):
            metadata["projection_or_localizer"] = True
        if details["storage"] == "source-asset":
            row = dict(id=uid, patient_id=c.document["patient_id"], modality=sequence, timepoint=timepoint,
                       storage="source-asset", original=originals[0], **metadata)
            # Original DICOM instances keep geometry including non-scalar
            # frames. No scalar decode or substitute affine is produced.
            staged.document.setdefault("source_assets", []).append(row)
        else:
            reader = sitk.ImageSeriesReader()
            reader.SetFileNames(retained_paths)
            image = reader.Execute()
            if image.GetDimension() != 3:
                raise ValueError("Select a single 3D DICOM series")
            with TemporaryDirectory(prefix=".dicom-", dir=c.root) as temporary:
                out = Path(temporary) / (uid + ".nii.gz")
                sitk.WriteImage(image, str(out))
                row = staged.add_series(out, series_id=uid, modality=sequence, timepoint=timepoint)
            row.update(metadata)
            if details.get("intensity_units") == "HU":
                row["intensity_units"] = "HU"
                row["intensity_units_basis"] = "DICOM rescale type declaration"
            _swi_component(row)
        imported.append(row)
    for key in ("series", "source_assets", "reference_series", "transforms"):
        if key in staged.document:
            c.document[key] = staged.document[key]
    return imported


def rigid_registration(collection, moving_id, *, cancelled=lambda: False):
    import SimpleITK as sitk

    moving = collection.series(moving_id)
    fixed = collection.series(collection.document["reference_series"])
    if moving["timepoint"] != fixed["timepoint"]:
        raise ValueError(
            "Different timepoints require separate registration review; automatic rigid fusion withheld"
        )
    for row in [moving, fixed]:
        if (
            file_hash(collection.path(row["original"]["path"]))
            != row["original"]["sha256"]
        ):
            raise ValueError("Original source changed before registration")
        collection.require_native_geometry(row)
    with reconstruction_lock():
        f = sitk.ReadImage(
            str(collection.path(fixed["original"]["path"])), sitk.sitkFloat32
        )
        m = sitk.ReadImage(
            str(collection.path(moving["original"]["path"])), sitk.sitkFloat32
        )
        initial = sitk.CenteredTransformInitializer(
            f,
            m,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMattesMutualInformation(50)
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(0.02, 20261014)
        registration.SetInterpolator(sitk.sitkLinear)
        registration.SetOptimizerAsGradientDescent(1.0, 100, 1e-6, 10)
        registration.SetOptimizerScalesFromPhysicalShift()
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        registration.SetInitialTransform(initial, inPlace=False)
        registration.AddCommand(
            sitk.sitkIterationEvent,
            lambda: registration.StopRegistration() if cancelled() else None,
        )
        result = registration.Execute(f, m)
        if cancelled():
            raise InterruptedError("Registration cancelled")
        # ITK transform samples fixed -> moving in LPS. Store moving -> fixed RAS.
        origin = np.array(result.TransformPoint((0.0, 0.0, 0.0)))
        matrix = np.eye(4)
        matrix[:3, 3] = origin
        for axis in range(3):
            matrix[:3, axis] = (
                np.array(result.TransformPoint(tuple(np.eye(3)[axis]))) - origin
            )
        lps = np.diag([-1.0, -1.0, 1.0, 1.0])
        matrix = lps @ np.linalg.inv(matrix) @ lps
        collection.set_registration(
            moving_id,
            matrix,
            status="unreviewed",
            method="Mattes MI rigid; independent landmarks pending",
        )
        return {
            "matrix": matrix.tolist(),
            "metric": float(registration.GetMetricValue()),
            "status": "unreviewed",
        }


def registration_landmarks(
    collection, series_id, moving_points, reference_points, reviewer
):
    if (
        not reviewer.strip()
        or len(moving_points) < 3
        or len(moving_points) != len(reference_points)
    ):
        raise ValueError(
            "Reviewer and at least three paired anatomical landmarks required"
        )
    m = np.asarray(moving_points, float)
    r = np.asarray(reference_points, float)
    if (
        m.shape != r.shape
        or m.ndim != 2
        or m.shape[1] != 3
        or not np.isfinite(m).all()
        or not np.isfinite(r).all()
    ):
        raise ValueError("Invalid landmarks")
    if (
        np.linalg.matrix_rank(m - m.mean(axis=0)) < 2
        or np.linalg.matrix_rank(r - r.mean(axis=0)) < 2
    ):
        raise ValueError("Landmarks must not be coincident or collinear")
    matrix = collection.registration_matrix(series_id, require_geometry=True)
    t = collection.document["transforms"][series_id]
    if t["status"] == "reference":
        raise ValueError("The reference series defines the patient frame")
    errors = np.linalg.norm(world(matrix, m) - r, axis=1)
    if errors.max() > 2.0:
        raise ValueError("Landmark residual exceeds 2 mm; improve alignment before accepting review")
    for sid, points in ((series_id, m), (collection.document["reference_series"], r)):
        source = collection.series(sid)
        indices = world(np.linalg.inv(affine(source["affine"])), points)
        if np.any(indices < 0) or np.any(indices > np.asarray(source["data"]["shape"]) - 1):
            raise ValueError(f"Landmarks outside acquired source coverage in {sid}")
    return {"moving": m.tolist(), "reference": r.tolist(), "errors_mm": errors.tolist(),
            "reviewer": reviewer, "meaning": "sampled landmark residuals, not a whole-image error bound"}


def review_registration(collection, series_id, moving_points, reference_points, reviewer):
    landmarks = registration_landmarks(collection, series_id, moving_points, reference_points, reviewer)
    t = collection.document["transforms"][series_id]
    collection.set_registration(
        series_id,
        t["matrix"],
        status="reviewed",
        method=t["method"],
        uncertainty_mm=max(landmarks["errors_mm"]),
        landmarks=landmarks,
    )
    for key in ("direction", "reference_series", "parent_series", "source_to_parent", "algorithm_version", "input_key", "quality", "transform_digest"):
        if key in t:
            collection.document["transforms"][series_id][key] = t[key]
    return landmarks["errors_mm"]


def import_anatomy_derivative(
    collection, path, *, source_series, kind, label, role="context", role_map=None,
    retain_empty=False,
):
    """Explicit source-associated labels, functional scores or tractography.

    Coordinates are preserved; source registration must independently be reviewed.
    No tract or functional data is inferred from structural MRI.
    """
    import nibabel as nib
    import uuid
    from .patient_anatomy import MAX_ARRAY

    c = collection
    c.series(source_series)
    p = Path(path)
    identity = str(uuid.uuid4())
    if role_map is not None and kind != "labels":
        raise ValueError("roleMap applies only to categorical labels")
    if kind == "tractography":
        points, offsets = tractography_geometry(p)
        original = c.original(p)
        item = {
            "id": identity,
            "patient_id": c.document["patient_id"],
            "kind": kind,
            "label": label,
            "source_series": source_series,
            "original": original,
            "points_ras_mm": c.put_array(points),
            "offsets": c.put_array(offsets),
            "review_status": "unreviewed source registration and tractography",
            "role": "context",
            "coordinate_basis": "source physical RAS+ millimeters; file header interpreted by nibabel",
            "identity_basis": "explicit supplied label; functional identity not independently verified",
        }
        c.document.setdefault("tractography", []).append(item)
        return item
    im = nib.load(p)
    if len(im.shape) != 3 or np.prod(im.shape) * 4 > MAX_ARRAY:
        raise ValueError("Select a bounded single-volume derivative")
    geometry = nifti_geometry(im)
    units = geometry["spatial_units"]
    if units not in {"mm", "unknown"}:
        raise ValueError("Imported anatomy requires millimeter geometry")
    values = im.get_fdata(dtype=np.float32)
    if kind in {"motor", "language", "functional"}:
        # Continuous maps are retained as scores, without inventing a calibrated
        # probability or automatically turning positive scores into constraints.
        item = c.add_layer(
            np.zeros(im.shape, np.uint8),
            im.affine,
            layer_id=identity,
            source_series=source_series,
            kind=kind,
            role="context",
            color="#f7bc65",
            label=label,
            provenance={
                "source": c.original(p),
                "threshold": "unset; explicitly review score meaning before labeling",
            },
        )
        item["score"] = c.put_array(values)
        item["score_meaning"] = (
            "supplied functional values; units and threshold require review"
        )
        item["units_status"] = units
        item["native_geometry"] = geometry
        return item
    if (
        not np.array_equal(values, np.rint(values))
        or np.any(values < 0)
        or values.max() > 65535
    ):
        raise ValueError("Categorical derivative must contain integer labels")
    present = {int(value) for value in np.unique(values) if value != 0}
    mapping = _validated_role_map(role_map, None if retain_empty and not present else present) if role_map is not None else None
    original = c.original(p)
    ref = c.put_array(values.astype(np.uint16))
    items = []
    for value in sorted(present or (mapping if retain_empty else ()) or ()):
        specification = mapping[value] if mapping is not None else None
        row = {
            "id": identity + "-" + str(value),
            "patient_id": c.document["patient_id"],
            "source_series": source_series,
            "label": specification["label"] if specification and specification["label"] else f"{label} {value}",
            "kind": specification["kind"] if specification else kind,
            "role": specification["role"] if specification else role,
            "color": specification["color"] if specification else "#97bdcf",
            "affine": affine(im.affine).tolist(),
            "units_status": units,
            "native_geometry": geometry,
            "data": ref,
            "label_value": value,
            "revision": 0,
            "review_status": "unreviewed imported annotation",
            "coverage": "supplied annotation field; negative voxels unassessed",
            "provenance": {"source": original, "source_label_value": value,
                           "kind": "observed_source_annotation"},
        }
        if retain_empty:
            row["evidence_status"] = "available" if value in present else "empty-output"
        c.document["layers"].append(row)
        items.append(row)
    return items
