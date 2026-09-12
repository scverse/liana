import numpy as np
import pytest
from scipy.sparse import csr_matrix

from liana.preprocessing import neg_to_zero, zi_minmax


def test_zi_minmax_after_neg_to_zero() -> None:
    """`neg_to_zero` stores explicit zeros; `zi_minmax` must not mis-align `.data` against them."""
    x = np.array([[-1.0, 0.5], [2.0, 4.0], [5.5, 7.1]])
    zeroed = neg_to_zero(x)
    # the negative value is stored as an explicit zero, so `.data` is longer than `.nonzero()`
    assert zeroed.data.size > zeroed.nonzero()[0].size

    scaled = zi_minmax(zeroed, cutoff=0.0)
    np.testing.assert_allclose(scaled.toarray()[:, 0], [0.0, 2.0 / 5.5, 1.0])


def test_zi_minmax_raises_on_constant_column() -> None:
    x = np.array([[1.0, 0.5], [1.0, 4.0], [1.0, 7.1]])
    with pytest.raises(ValueError, match="zero range"):
        zi_minmax(x)


def test_zi_minmax_allows_all_zero_column() -> None:
    """An unexpressed gene has a zero range too, but carries no stored values -- it is legitimate."""
    x = csr_matrix(np.array([[0.0, 0.5], [0.0, 4.0], [0.0, 7.1]]))
    scaled = zi_minmax(x, cutoff=0.0)
    np.testing.assert_allclose(scaled.toarray()[:, 0], 0.0)
