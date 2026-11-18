"""Source precedence for research bone display on an explicitly sampled field."""

import numpy as np

from .anatomy_model_registry import ct_intensities
from .anatomy_evidence import inference_bounds


def _bone_method(collection, row):
    p = row.get("provenance", {})
    model = str(p.get("model", "")).lower()
    legacy = row["kind"] == "brain-region" and model == "supersynth" and row.get("label_value") in (915, 916)
    if row["kind"] not in {"bone", "skull"} and not legacy:
        return None
    if row.get("availability") == "empty_output" or row.get("evidence_status") == "empty-output":
        return None
    source = collection.series(row["source_series"])
    if source.get("modality") in {"CT", "CTA"}:
        dense = (model == "totalsegmentator-skull" and p.get("parameters", {}).get("positive_ct_bone_gate_hu") == 250
                 or p.get("method") in {
                     "native CT 300 HU boundary proposal",
                     "Native acquired CT >=300 HU bone-density proposal; teeth and extracranial bone included",
                 })
        if dense:
            try:
                collection.patient_affine(row)
                inference_bounds(row)
                data = collection.array(row["data"])
                slices = row.get("coverage_fields", {}).get("supported_axial_indices")
                if slices is not None and (not isinstance(slices, list) or any(
                        type(z) is not int or not 0 <= z < data.shape[2] for z in slices)):
                    return None
                key = (row["data"]["sha256"], row["label_value"], source["data"]["sha256"], source.get("intensity_units"))
                if not hasattr(collection, "_bone_candidates"):
                    collection._bone_candidates = {}
                if key not in collection._bone_candidates:
                    step = max(1, 262144 // (data.shape[1] * data.shape[2]))
                    collection._bone_candidates[key] = ct_intensities(collection, source) and any(
                        np.any(data[start:start+step] == row["label_value"]) for start in range(0, len(data), step))
                if collection._bone_candidates[key]:
                    return "ct"
            except (ValueError, OSError, StopIteration):
                return None
    elif model in {"supersynth", "siam-v2"}:
        return "mri"
    return None


class BoneEvidence:
    """Call apply only on registered, coverage-limited samples in order().

    A zero CT estimate withholds the competing MRI display claim. It cannot
    establish clearance. Partial annotations do not define dense background.
    """

    def __init__(self, collection, rows, shape):
        self.methods = {r["id"]: _bone_method(collection, r) for r in rows}
        self.shape = shape
        self.ct_coverage = None
        self.ct_positive = None
        self.sources = []

    def order(self, row):
        return int(self.methods.get(row["id"]) == "mri")

    def apply(self, row, occupied, covered):
        method = self.methods.get(row["id"])
        if method == "ct" and np.any(covered):
            if self.ct_coverage is None:
                self.ct_coverage = np.zeros(self.shape, bool)
                self.ct_positive = np.zeros(self.shape, bool)
            self.ct_coverage |= covered
            self.ct_positive |= occupied
            self.sources.append(row["id"])
        if method != "mri" or self.ct_coverage is None:
            return occupied, covered, None
        overlap = covered & self.ct_coverage
        if not np.any(overlap):
            return occupied, covered, None
        return occupied & ~overlap, covered & ~overlap, {
            "policy": "covered-ct-before-mri-bone-v1", "preferredLayerIds": list(self.sources),
            "overlapVoxels": int(overlap.sum()),
            "disagreementVoxels": int(np.count_nonzero(overlap & (occupied != self.ct_positive))),
            "alternativePositiveVoxels": int(np.count_nonzero(overlap & occupied)),
            "meaning": "CT bone estimate used in overlap. Unlabelled CT is unknown anatomy. MRI retained in original layer.",
        }


def sampled_bone_evidence(collection, readings, point):
    rows = collection.document["layers"]
    evidence = BoneEvidence(collection, rows, (1,))
    preferred, alternatives = [], []
    for row in sorted(rows, key=evidence.order):
        sample = readings[row["id"]]
        if not sample.get("known") or sample.get("occupied") is None or not evidence.methods[row["id"]]:
            continue
        positive = sample["occupied"]
        if row.get("provenance", {}).get("model") == "siam-v2":
            from .anatomy_siam_display import supported_tissue_layer
            from .patient_anatomy import world
            try:
                display, _ = supported_tissue_layer(collection, row)
                voxel = world(np.linalg.inv(collection.patient_affine(row)), point)
                index = tuple(np.floor(voxel + .5).astype(int))
                positive = collection.array(display["data"])[index] == display["label_value"]
            except (ValueError, OSError):
                continue
        occupied = np.asarray([positive])
        _, _, decision = evidence.apply(row, occupied, np.ones(1, bool))
        item = {"layerId": row["id"], "seriesId": row["source_series"], "model": str(row["provenance"].get("model") or row["provenance"].get("method")), "positiveLabel": bool(occupied[0])}
        if decision:
            alternatives.append(item)
        else:
            preferred.append(item)
    if not alternatives:
        return None
    # Complementary cortical/cancellous classes from one prediction describe
    # a single bone estimate. Class-level background is not another vote.
    predictions = {}
    for item in preferred + alternatives:
        identity = (item["seriesId"], item["model"])
        predictions[identity] = predictions.get(identity, False) or item["positiveLabel"]
    return {"structure": "bone", "policy": "covered-ct-before-mri-bone-v1",
            "preferred": preferred, "alternatives": alternatives,
            "disagreement": len(set(predictions.values())) > 1,
            "meaning": "CT estimate used within sampled coverage. Zero is unlabelled. Clearance is unassessed."}


def structure_disagreements(collection, readings):
    """Compare positive label support only among like structures at the sampled point."""
    comparable = {"brain", "dura", "vessel", "vessels", "artery", "vein", "sinus"}
    groups = {}
    for row in collection.document["layers"]:
        state = readings[row["id"]]
        kind = row["kind"]
        if kind not in comparable or not state.get("known") or state.get("occupied") is None:
            continue
        kind = "vessels" if kind == "vessel" else kind
        p = row.get("provenance", {})
        groups.setdefault(kind, []).append({
            "layerId": row["id"], "seriesId": row["source_series"], "positiveLabel": state["occupied"],
            "method": str(p.get("model") or p.get("method") or p.get("kind") or "source-associated label"),
            "dataSha256": row["data"]["sha256"], "labelValue": row["label_value"],
        })
    return [{"structure": kind, "sources": sources,
             "meaning": "Positive and unlabelled readings differ. Partial labels and correlated estimates establish no anatomical absence or independent confirmation."}
            for kind, sources in groups.items() if len({s["positiveLabel"] for s in sources}) > 1]


def acquisition_contributions(collection, target_options, environment_options):
    """Describe observed outcomes and the precise geometric uses already preflighted."""
    from .anatomy_evidence import is_siam_context, vascular_evidence_supported

    targets = {r["layerId"] for r in target_options if r.get("target") is not None}
    ready = {r["layerId"] for r in environment_options["layers"] if r["status"] == "ready"}
    doc = collection.document
    rows = []
    for source in collection.source_records():
        sid = source["id"]
        modality = source.get("modality", "unknown")
        transform = doc["transforms"].get(sid, {})
        status = transform.get("status", "unregistered")
        asset = source.get("storage") == "source-asset"
        reason = ""
        try:
            if asset:
                raise ValueError("Archived source; no registered scalar derivative")
            collection.patient_affine(source)
        except (ValueError, StopIteration) as error:
            reason = str(error) or "Source registration unavailable"
        layers = [r for r in doc["layers"] if r["source_series"] == sid]
        model_inputs = []
        for layer in doc["layers"]:
            channels = (layer.get("provenance", {}).get("inputs") or {})
            if not isinstance(channels, dict):
                continue
            if any(ch.get("series_id") == sid and ch.get("original_sha256") == source["original"]["sha256"]
                   for ch in channels.get("channels", []) if isinstance(ch, dict)):
                model_inputs.append(layer["label"])
        tracts = [r["label"] for r in doc.get("tractography", []) if r["source_series"] == sid]
        uses, unavailable = [], []
        for layer in layers:
            if layer.get("availability") == "empty_output" or layer.get("evidence_status") == "empty-output":
                unavailable.append(layer["label"])
            elif not reason and layer["id"] in targets:
                uses.append(layer["label"] + " (geometric target)")
            elif (not reason and layer["id"] in ready and not is_siam_context(layer)
                  and vascular_evidence_supported(collection, layer)
                  and layer.get("provenance", {}).get("evidence_use") != "context_only"):
                uses.append(layer["label"] + " (positive geometry for limited rehearsal)")
        quality = transform.get("quality", {})
        rows.append({
            "seriesId": sid, "label": source.get("label") or sid, "modality": modality,
            "anatomicalRoles": sorted({r["kind"] for r in layers}),
            "registration": status, "qc": quality.get("status"), "warnings": quality.get("warnings", []),
            "disposition": "unsupported" if asset else "quantitative geometry" if uses else "inspection context",
            "reason": reason, "layerCount": len(layers), "emptyLayers": unavailable,
            "uses": uses, "suppliedTracts": tracts, "modelInputs": model_inputs, "asset": asset,
        })
    # Migrated studies may retain external diffusion volumes as inventory notes.
    for item in doc.get("source_assets", []):
        if item.get("storage") == "source-asset" or item.get("kind") not in {"diffusion_4d", "reverse_phase_reference"}:
            continue
        name = item.get("name") or item.get("sha256")
        rows.append({"seriesId": name, "label": name, "modality": item["kind"].replace("_", " "),
                     "anatomicalRoles": [], "registration": "unregistered", "qc": None, "warnings": [],
                     "disposition": "unsupported", "reason": "External acquisition inventory; no reconstructed derivative",
                     "layerCount": 0, "emptyLayers": [], "uses": [], "suppliedTracts": [], "modelInputs": [], "asset": True})
    return rows


def composite_bone_layer(collection, layer, check):
    """Make a display-only MRI exterior on the layer's native lattice.

    Eligible study sources determine applicability, including hidden CT layers.
    Correction drafts and raw source inspection bypass this derivative.
    """
    if _bone_method(collection, layer) != "mri":
        return layer, None
    try:
        grid = collection.patient_affine(layer)
    except ValueError:
        return layer, None
    ct_rows = [r for r in collection.document["layers"] if _bone_method(collection, r) == "ct"]
    if not ct_rows:
        return layer, None
    from .anatomy_sampling import sample_layer

    data = collection.array(layer["data"])
    output = np.empty(data.shape, np.uint8)
    kept = total = different = 0
    sources = set()
    step = max(1, 262144 // (data.shape[1] * data.shape[2]))
    for start in range(0, len(data), step):
        check()
        foreground = data[start:start+step] == layer["label_value"]
        out = output[start:start+step]
        out[:] = foreground
        total += int(np.count_nonzero(foreground))
        if not foreground.any():
            continue
        field = grid.copy()
        field[:3, 3] += start * grid[:3, 0]
        covered = np.zeros(out.shape, bool)
        positive = np.zeros(out.shape, bool)
        for row in ct_rows:
            ct, support = sample_layer(collection, row, field, out.shape, check, {})
            if (support & foreground).any():
                sources.add(row["id"])
            covered |= support
            positive |= ct
        different += int(np.count_nonzero(foreground & covered & ~positive))
        out[covered] = 0
        kept += int(np.count_nonzero(out))
    if not sources:
        return layer, None
    receipt = {"preferredLayerIds": sorted(sources), "alternativePositiveVoxels": total - kept,
               "alternativeOnlyVoxels": different, "retainedVoxels": kept}
    return {**layer, "data": collection.put_array(output), "label_value": 1}, receipt
