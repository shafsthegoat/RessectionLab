import itertools
import numpy as np
import pytest
from resectionlab.head_segmentation import tile_starts,blend_window


def test_irregular_full_fields_have_positive_coverage_without_edge_cropping():
    shape=(176,208,224);patch=160
    coverage=np.zeros(shape,np.float32);weights=blend_window(patch)
    for start in itertools.product(*(tile_starts(n,patch,96)for n in shape)):
        roi=tuple(slice(a,a+patch)for a in start);coverage[roi]+=weights
    assert np.isfinite(coverage).all()and coverage.min()>0
    assert np.allclose(weights,weights[::-1,::-1,::-1])
    assert weights[patch//2,patch//2,patch//2]>weights[0,0,0]


def test_field_padding_and_stride_are_explicit_and_last_tile_is_anchored():
    assert tile_starts(128)==[0]
    assert tile_starts(192)==[0,64]
    assert tile_starts(256)==[0,80,128]
    with pytest.raises(ValueError):tile_starts(100)
    with pytest.raises(ValueError):tile_starts(200,stride=200)
