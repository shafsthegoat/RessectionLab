import numpy as np

from resectionlab.anatomy_ct_exterior import ct_exterior_sheet


class SourceStudy:
    def __init__(self, signal, labels, affine):
        self.signal = signal
        self.labels = labels
        self.scan = {"id": "ct", "modality": "CTA", "intensity_units": "HU", "affine": affine.tolist(),
                     "data": {"path": "signal", "shape": list(signal.shape)}}
        self.document = {"layers": [{"id": "brain", "kind": "brain", "source_series": "ct",
                                    "affine": affine.tolist(), "label_value": 1,
                                    "data": {"path": "labels", "shape": list(labels.shape)}}]}

    def array(self, ref):
        return self.signal if ref["path"] == "signal" else self.labels

    def require_native_geometry(self, row):
        assert row is self.scan


def source_fixture():
    signal = np.full((110, 120, 64), -1000, dtype=np.float32)
    signal[20:91, 26:104, 4:60] = 45
    signal[20:23, 33:101, 8:55] = 1100
    brain = np.zeros(signal.shape, dtype=np.uint8)
    brain[32:82, 37:92, 10:54] = 1
    signal[:, :5, 30:] = 280
    signal[40:48, 10:17, 45:53] = 350
    return signal, brain


def expand(result, shape):
    mask = np.zeros(shape, dtype=bool)
    if result is not None:
        slices = tuple(slice(x, x + n) for x, n in zip(result["offset"], result["mask"].shape, strict=True))
        mask[slices] = result["mask"]
    return mask


def test_sheet_keeps_anatomical_and_other_detached_foreground_on_reordered_native_grid():
    signal, brain = source_fixture()
    study = SourceStudy(signal, brain, np.eye(4))
    raw = signal.copy()
    exclusion = expand(ct_exterior_sheet(study, study.scan, check=lambda: None), signal.shape)
    assert exclusion[:, :5, 30:].all()
    assert not exclusion[40:48, 10:17, 45:53].any()
    assert not exclusion[brain == 1].any()
    np.testing.assert_array_equal(study.signal, raw)

    reordered = lambda data: data[::-1, :, :].transpose(2, 0, 1)
    affine = np.array([[0., -1., 0., 109.], [0., 0., 1., 0.], [1., 0., 0., 0.], [0., 0., 0., 1.]])
    other = SourceStudy(reordered(signal), reordered(brain), affine)
    moved = expand(ct_exterior_sheet(other, other.scan, check=lambda: None), other.signal.shape)
    np.testing.assert_array_equal(moved, reordered(exclusion))


def test_sheet_refuses_contact_mineral_density_missing_evidence_and_nearby_labels():
    signal, labels = source_fixture()
    for change in ("contact", "dense", "no anchor", "near label"):
        intensity, brain = signal.copy(), labels.copy()
        if change == "contact":
            intensity[45:51, 5:27, 35:41] = 180
        elif change == "dense":
            intensity[40, 2, 45] = 1200
        elif change == "no anchor":
            brain.fill(0)
        else:
            brain[40, 6, 47] = 1
        study = SourceStudy(intensity, brain, np.eye(4))
        assert ct_exterior_sheet(study, study.scan, check=lambda: None) is None, change
