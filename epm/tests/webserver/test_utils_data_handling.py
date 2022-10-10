import pytest
import numpy as np
from epm.experiment_utils.data_handling import warp, unwarp

def test_warp():
    x = 2 * np.ones((4, 2))

    assert np.array_equal(warp(x, True), x)
    assert np.array_equal(warp(x, False), np.log10(x))
    with pytest.raises(AssertionError):
        warp(x, '1')
    with pytest.raises(AssertionError):
        warp(x, 1)
    with pytest.raises(AssertionError):
        warp(x, 'True')


def test_unwarp():
    x = 2 * np.ones((4, 2))

    assert np.array_equal(unwarp(x, True), x)
    assert np.array_equal(unwarp(x, False), np.power(10, x))
    with pytest.raises(AssertionError):
        warp(x, '1')
    with pytest.raises(AssertionError):
        warp(x, 1)
    with pytest.raises(AssertionError):
        warp(x, 'True')
