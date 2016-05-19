"""
Classes that implement linear projection operators and their transposes.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg.interface import MatrixLinearOperator


def get_subset_lin_op(lin_op, sub_idx):
    """ Subset a linear operator to the indices in `sub_idx`. Equivalent to A' = A[sub_idx, :]
    :param LinearOperator lin_op: input linear operator
    :param np.ndarray[int] sub_idx: subset index
    :return: the subset linear operator
    :rtype: LinearOperator
    """
    if lin_op is None:
        return None
    if type(lin_op) is IndexOperator:
        # subsetting IndexOperator yields a new IndexOperator
        return IndexOperator(lin_op.index_map[sub_idx], dim_x=lin_op.dim_x)
    elif isinstance(lin_op, MatrixLinearOperator):
        # subsetting a matrix multiplication operation yields a new matrix
        return MatrixLinearOperator(lin_op.A[sub_idx, :])
    # in the general case, append a sub-indexing operator
    return IndexOperator(sub_idx, dim_x=lin_op.shape[0]) * lin_op


def rmatvec_nd(lin_op, x):
    """
    Project a 1D or 2D numpy or sparse array using rmatvec. This is different from rmatvec
    because it applies rmatvec to each row and column. If x is n x n and lin_op is n x k,
    the result will be k x k.

    :param LinearOperator lin_op: The linear operator to apply to x
    :param np.ndarray|sp.spmatrix x: array/matrix to be projected
    :return: the projected array
    :rtype: np.ndarray|sp.spmatrix
    """
    if x is None or lin_op is None:
        return x
    if isinstance(x, sp.spmatrix):
        y = x.toarray()
    elif np.isscalar(x):
        y = np.array(x, ndmin=1)
    else:
        y = np.copy(x)
    proj_func = lambda z: lin_op.rmatvec(z)
    for j in range(y.ndim):
        if y.shape[j] == lin_op.shape[0]:
            y = np.apply_along_axis(proj_func, j, y)
    return y


class IndexOperator(LinearOperator):
    """
    A linear one-to-many operator equivalent to ``y_j = A_jx = x_{index_j}``,
    i.e. ``y = x[index]``.
    The inverse operation is ``x_i = A_i^T y = \sum_j y_j \delta(index_j - i)``.
    When computing the inverse ``x = A^T y``, it is not assumed that all x's were used to generate
    y, so the operator can be initialized with ``max_idx = len(x) - 1``, in which case the inverse
    operator will produce a vector of size len(x).
    """

    def _index(self, x):
        return x[self.index_map]

    def _reverse_index(self, x):
        """ Helper method for summing over elements with shared indices."""
        count_func = lambda z: np.bincount(self.index_map, weights=z, minlength=self.dim_x)
        return np.apply_along_axis(count_func, 0, x)

    def __init__(self, index_map, dim_x=None):
        """ Set up the linear transform and its transpose with the index map
        :param np.ndarray[int] index_map: indicates, for each element of the output, which input
            element is applied
        :param int|None dim_x: dimension of the projected array x
        """
        if dim_x is not None and dim_x < np.max(index_map) - 1:
            raise ValueError("dim_x ({}) must be None or at least max(index_map)+1 ({})".format(
                dim_x, np.max(index_map) + 1))
        self.dim_x = dim_x or np.max(index_map) + 1
        self.index_map = index_map.astype(int)
        if index_map.dtype != int:
            raise ValueError("Index map must be a numpy array of integers")
        if np.any(index_map < 0):
            raise ValueError("Index map must be positive")
        super(IndexOperator, self).__init__(shape=(len(index_map), self.dim_x),
                                            matvec=self._index,
                                            rmatvec=self._reverse_index,
                                            dtype=bool)
