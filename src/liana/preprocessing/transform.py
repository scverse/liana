from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr

if TYPE_CHECKING:
    from liana._core._types import MatrixLike


def zi_minmax(X: MatrixLike, cutoff: float = 0.5) -> csr_matrix:
    """
    Zero-inflated min-max scaling, adopted from CiteFuse (Kim et al., 2020; https://academic.oup.com/bioinformatics/article/36/14/4137/5827474).

    This function scales the data to the range [0, 1] for each column of a
    two-dimensional array and sets values below a specified cutoff to 0 (after
    scaling).

    Parameters
    ----------
    X
        Data to be scaled.
    cutoff
        Cutoff value for zero-inflation - values less than this are set to 0.
        Default is 0.5.

    Returns
    -------
    X
        The scaled data matrix

    Examples
    --------
    >>> import numpy as np
    >>> import liana as li
    >>> x = np.array([[0.1, 0.3], [2.0, 4.0], [5.5, 7.1]])
    >>> li.pp.zi_minmax(x).toarray().round(3)
    array([[0.   , 0.   ],
           [0.   , 0.544],
           [1.   , 1.   ]])

    `cutoff` is applied after scaling, so lowering it keeps more of the middle:

    >>> li.pp.zi_minmax(x, cutoff=0.1).toarray().round(3)
    array([[0.   , 0.   ],
           [0.352, 0.544],
           [1.   , 1.   ]])
    """
    copied = X.copy()
    mat = copied if isspmatrix_csr(copied) else csr_matrix(copied)

    min_vals = np.asarray(mat.min(axis=0).todense())[0]
    max_vals = np.asarray(mat.max(axis=0).todense())[0]
    nonzero_rows, nonzero_cols = mat.nonzero()
    scaled_values = (mat.data - min_vals[nonzero_cols]) / (max_vals[nonzero_cols] - min_vals[nonzero_cols])

    scaled_values[scaled_values < cutoff] = 0
    nonzero_rows, nonzero_cols = mat.nonzero()

    return csr_matrix((scaled_values, (nonzero_rows, nonzero_cols)), shape=mat.shape)


def neg_to_zero(X: MatrixLike, cutoff: float = 0) -> csr_matrix:
    """
    Set negative values to 0.

    Parameters
    ----------
    X
        Data to be transformed.
    cutoff
        Cutoff value for zero-inflation - values less than
        this are set to 0. Default is 0.

    Returns
    -------
    The modified data matrix

    Examples
    --------
    >>> import numpy as np
    >>> import liana as li
    >>> x = np.array([-1, -0.5, 0.1, 0.4, 2])
    >>> li.pp.neg_to_zero(x).toarray()
    array([[0. , 0. , 0.1, 0.4, 2. ]])

    `cutoff` raises the threshold above 0:

    >>> li.pp.neg_to_zero(x, cutoff=0.5).toarray()
    array([[0., 0., 0., 0., 2.]])
    """
    copied = X.copy()
    mat = copied if isspmatrix_csr(copied) else csr_matrix(copied)
    mat.data[mat.data < cutoff] = 0

    return mat
