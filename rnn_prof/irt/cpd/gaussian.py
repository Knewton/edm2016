"""
A module containing Gaussian conditional probability distributions.
"""
import numpy as np
from scipy import sparse as sp

from .cpd import CPD, CPDTerms, FunctionInfo
from ..linear_operators import IndexOperator, get_subset_lin_op, rmatvec_nd
from ..updaters import UpdateTerms

MAX_SIZE_FOR_DENSIFICATION = 1000


class GaussianCPD(CPD):
    MEAN_KEY = 'mean'
    PRECISION_KEY = 'precision'

    PARAM_KEYS = (MEAN_KEY, PRECISION_KEY)
    dim = None
    mean = None
    precision = None

    def __init__(self, dim=None, mean=None, precision=None, mean_lin_op=None, support=None):
        """
        :param int dim: dimensionality of the distribution. If None, inferred from mean or precision
        :param float|np.ndarray|None mean: the mean, float or 1D array. If not supplied on init or
            call-time, assumed to be 0.  If scalar, assumes the mean is the same for all dimensions.
        :param float|np.ndarray|sp.spmatrix|None precision: the precision matrix.  If not supplied
            on init or at call-time, assumed to be the identity.
            If 1D, assumes that the diagonal of the precision matrix is passed in; if scalar
            and CPD dimensionality > 1, assumes that the precision matrix is precision * identity
            matrix.
        :param None|IndexOperator mean_lin_op: linear transform operator for the mean parameter,
            whose is shape is (dim, len(mean_vector))
        :param tuple(float) support: Defines the support for the probability distribution. Passed
            to solver pars updater to prevent the parameter from being set outside the range of the
            supports.
        """
        mean, precision, const = self._validate_args(dim=dim, mean=mean, precision=precision)
        self.mean = mean.ravel() if self.dim > 1 else mean
        self.precision = precision
        self.hessian_cache = None
        self.hessian_mean_cache = None
        self.const = const
        self.mean_lin_op = mean_lin_op
        self.support = support

    def __call__(self, x, mean=None, precision=None, terms_to_compute=None):
        """
        :param float|np.ndarray x: the point at which to evaluate the Gaussian log-probability
        :param float|np.ndarray|None mean: the distribution mean. If None, obtained at
            initialization, or set to 0 (scalar/vector) if not available at initialization.
            If scalar, assumes the mean is the same for all dimensions.
        :param float|np.ndarray|sp.spmatrix precision: precision matrix. If None, obtained at
            initialization, or set to the identity if not available at initialization.
            If 1D, assumes that the diagonal of the precision matrix is passed in; if scalar
            and CPD dimensionality > 1, assumes that the precision matrix is precision * identity
            matrix.
        :param dict[str, UpdateTerms] terms_to_compute: which data/parameter gradients and Hessians
            to compute
        :return: the value, the gradients, and the Hessians of the CPD
        :rtype: CPDTerms
        """
        if mean is not None and self.mean_lin_op is not None:
            mean = self.mean_lin_op * mean
        mean, precision, const = self._validate_args(mean=mean, precision=precision)

        # compute the data terms
        if np.isscalar(x):
            if mean.size > 1:
                raise ValueError('x must be a vector of same shape as the mean vector')
            z = mean - x
            if isinstance(precision, sp.spmatrix):
                data_gradient = precision.toarray() * z
            else:
                data_gradient = precision * z
            value = const - 0.5 * z * data_gradient
        else:
            if not isinstance(x, np.ndarray):
                raise ValueError('x must be a numpy array')
            if x.size != mean.size:
                raise ValueError('x ({}) must have same size as the mean ({})'.format(x.size,
                                                                                      mean.size))
            z = mean - x.ravel()
            if np.isscalar(precision):
                data_gradient = precision * z
            elif isinstance(precision, np.ndarray) and precision.size == self.dim:
                data_gradient = precision.ravel() * z
            else:
                data_gradient = precision.dot(z)
            value = const - 0.5 * z.T.dot(data_gradient)
            # if x is 2D, return 2D gradient
            data_gradient = data_gradient.reshape(x.shape)

        wrt = {}
        if terms_to_compute:
            for par_key, to_compute in terms_to_compute.iteritems():
                gradient = None
                hessian = None
                if to_compute >= UpdateTerms.grad:
                    if par_key == self.DATA_KEY:
                        gradient = data_gradient
                        if to_compute == UpdateTerms.grad_and_hess:
                            hessian = self.hessian
                    elif par_key == self.MEAN_KEY:
                        gradient = -data_gradient
                        if self.mean_lin_op is not None:
                            gradient = rmatvec_nd(self.mean_lin_op, gradient)
                        if to_compute == UpdateTerms.grad_and_hess:
                            hessian = self.hessian_wrt_mean
                    elif par_key == self.PRECISION_KEY:
                        raise NotImplementedError("Precision gradient/Hessian not implemented")
                    elif par_key not in self.PARAM_KEYS:
                        raise ValueError("Terms requested for non-parameter {}".format(par_key))
                wrt[par_key] = FunctionInfo(value, gradient, hessian)
        return CPDTerms(log_prob=value, wrt=wrt)

    @property
    def hessian_wrt_mean(self):
        """ The Hessian of the multivariate Gaussian w.r.t. its mean, potentially including the
        linear projection.

        :return: The Hessian w.r.t. the mean
        :rtype: float|np.ndarray|sp.spmatrix
        """
        if self.hessian_mean_cache is None:
            hessian = self.hessian
            if self.mean_lin_op is not None:
                if np.isscalar(hessian) and isinstance(self.mean_lin_op, IndexOperator):
                    # index operator preserves diagonality
                    hessian = sp.diags(self.mean_lin_op.rmatvec(hessian * np.ones(self.dim)), 0)
                elif np.isscalar(hessian):
                    hessian = hessian * np.eye(self.dim)
                    hessian = rmatvec_nd(self.mean_lin_op, hessian)
                else:
                    hessian = rmatvec_nd(self.mean_lin_op, hessian)
            self.hessian_mean_cache = hessian
        return self.hessian_mean_cache

    @property
    def hessian(self):
        """ The Hessian of the multivariate Gaussian
        :return: The Hessian of the Gaussian distribution.
        :rtype: float|np.ndarray|sp.spmatrix
        """
        if self.hessian_cache is None:
            self.hessian_cache = -self.precision
        return self.hessian_cache

    def _validate_args(self, dim=None, mean=None, precision=None):
        """ Check the types and shapes of mean, precision, and dimensionality.  If passed values
        are None, use the ones provided at initialization or infer from other values."""

        # dimensionality is set once at initialization
        self.dim = self.dim or dim
        # check/infer the mean and the dimensionality
        if mean is not None:
            if np.isscalar(mean):
                mean = np.asarray(mean)
                if mean != float(mean):
                    raise ValueError("Scalar mean must be a float")
                if self.dim is None:
                    self.dim = 1
                elif self.dim > 1:
                    # mean is scalar but dimensionality > 1; assume we want constant mean
                    tmp_mean = np.empty(self.dim, dtype=float)
                    tmp_mean.fill(mean)
                    mean = tmp_mean
            else:
                if not isinstance(mean, np.ndarray) or mean.ndim > 2:
                    raise ValueError("Vector mean must be a 1D or 2D numpy array")
                if self.dim is None:
                    self.dim = len(mean)
                elif mean.size != self.dim:
                    raise ValueError("Mean has length {}, but cpd's dim is {}".format(
                        mean.size, self.dim))
                # convert to a 1D numpy array
                mean = mean.ravel()
        elif self.mean is not None:
            # if here, must have already initialized and set self.dim
            mean = self.mean
        else:
            # mean neither passed in nor set at initialization; assume 0 mean
            if self.dim is None and precision is None:
                raise ValueError("Cannot initialize Gaussian with unknown dimensionality")
            else:
                if self.dim is None:
                    self.dim = 1 if np.isscalar(precision) else precision.shape[0]
                mean = np.zeros(self.dim)

        # check/infer precision and compute the normalization constant
        if precision is not None:
            if np.isscalar(precision) or precision.size == 1:
                # scalar precision
                if isinstance(precision, sp.spmatrix):
                    precision = precision.toarray()[0, 0]
                if precision != float(precision) or precision <= 0.:
                    raise ValueError("Scalar precision must be a positive float")
                # precision is scalar; assume iid
                log_precision_det = self.dim * np.log(precision)
            elif (isinstance(precision, np.ndarray) and precision.size == self.dim and
                  precision.shape[0] == self.dim):
                # diagonal of the precision is passed in
                log_precision_det = np.sum(np.log(precision))
            elif (isinstance(precision, (np.ndarray, sp.spmatrix)) and precision.ndim == 2 and
                  precision.shape == (self.dim, self.dim)):
                # 2D precision is passed in
                if isinstance(precision, np.ndarray):
                    log_precision_det = np.log(np.linalg.det(precision))
                elif isinstance(precision, sp.spmatrix):
                    # TODO: Get rid of this densification (sp.linalg.splu may work for large matrix)
                    if precision.shape[0] < MAX_SIZE_FOR_DENSIFICATION:
                        log_precision_det = np.log(np.linalg.det(precision.todense()))
                    else:
                        # If it's too big, just ignore the determinant.
                        log_precision_det = 1.0
                if not np.isfinite(log_precision_det):
                    raise ValueError("Precision must be positive definite.")
            else:
                raise ValueError("Precision matrix must be a ({0}, 1) or ({0}, {0}) array, was {1}"
                                 .format(self.dim, precision.shape))
            const = -0.5 * (self.dim * np.log(2 * np.pi) - log_precision_det)
        elif self.precision is not None:
            precision = self.precision
            const = self.const
        else:
            # precision neither passed in nor set at initialization; assume it's the identity
            precision = 1.
            const = -0.5 * self.dim * np.log(2 * np.pi)

        return mean, precision, const

    def get_dependent_vars(self, var_idx):
        """ Given the indices of the query variables, var_idx, returns the set of dependent
        variables (including var_idx); i.e., the smallest set S containing var_idx for which
        (complement of S) indep of (S).  This is done by finding all non-zero columns of the
        `var_idx` rows of the precision matrix.

        :param np.ndarray[int]|np.ndarray[bool] var_idx: indices of the query variables
        :return: indices of the dependent variables
        :rtype: np.ndarray[int]
        """
        if isinstance(self.precision, sp.spmatrix):
            prec = self.precision.tocsr()
        elif np.isscalar(self.precision):
            return var_idx
        else:
            prec = self.precision
        return np.unique(np.nonzero(prec[var_idx, :])[1])

    def get_subset_cpd(self, sub_idx):
        """ Get the cpd over a subset of the variables.
        :param np.ndarray[int]|np.ndarray[bool] sub_idx: indices of variables to keep
        :return: a new Gaussian CPD
        :rtype: GaussianCPD
        """
        if len(sub_idx) == 0 or (sub_idx.dtype == bool and not np.sum(sub_idx)):
            raise ValueError("sub_idx must not be empty")
        sub_mean = self.mean[sub_idx]
        sub_dim = len(sub_mean)
        if isinstance(self.precision, sp.dia_matrix):
            sub_precision = sp.dia_matrix((self.precision.diagonal()[sub_idx], np.zeros(1)),
                                          shape=(sub_dim, sub_dim))
        elif np.isscalar(self.precision):
            sub_precision = self.precision
        elif isinstance(self.precision, np.ndarray):
            if np.prod(self.precision.shape) == self.dim:
                sub_precision = self.precision[sub_idx]
            else:
                # We do the indexing this way for performance reasons.
                sub_precision = self.precision[sub_idx, :][:, sub_idx]
        else:
            # We do the indexing this way for performance reasons.
            sub_precision = self.precision.tocsr()[sub_idx, :][:, sub_idx]
        return GaussianCPD(dim=sub_dim, mean=sub_mean, precision=sub_precision,
                           mean_lin_op=get_subset_lin_op(self.mean_lin_op, sub_idx))
