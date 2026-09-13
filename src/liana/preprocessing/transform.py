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

    Raises
    ------
    ValueError
        If a column's stored values are all identical *and* non-zero, since min-max scaling it would
        be a division by a zero range. A column whose stored values are identical but sit alongside
        implicit zeros still has a range, and is scaled.

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
    # `mat.data` is per *stored* value while `mat.nonzero()` skips explicit zeros, so the two only line
    # up once the explicit zeros are pruned -- `neg_to_zero` stores them without pruning.
    mat.eliminate_zeros()

    min_vals = np.asarray(mat.min(axis=0).todense())[0]
    max_vals = np.asarray(mat.max(axis=0).todense())[0]
    ranges = max_vals - min_vals
    # A constant column would scale to `0/0`. `min`/`max` above are over the whole column, implicit zeros
    # included, so a zero range on a column with no stored values is just an all-zero (unexpressed)
    # column -- legitimate, and it never reaches the division below.
    has_stored = np.bincount(mat.indices, minlength=mat.shape[1]) > 0
    constant = np.flatnonzero((ranges == 0) & has_stored)
    if constant.size:
        raise ValueError(
            f"Cannot min-max scale {constant.size} column(s) whose values are all identical "
            f"(zero range), at index/indices {constant.tolist()}: the scaling is undefined (0/0). "
            "Drop these columns before scaling."
        )

    nonzero_rows, nonzero_cols = mat.nonzero()
    scaled_values = (mat.data - min_vals[nonzero_cols]) / ranges[nonzero_cols]

    scaled_values[scaled_values < cutoff] = 0

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
