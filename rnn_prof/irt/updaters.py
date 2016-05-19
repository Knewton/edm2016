"""
Classes for computing an update step direction based on gradients (and optionally the Hessian).
"""
from enum import IntEnum
import logging

import numpy as np
from scipy.sparse.linalg import spsolve

import utils

LOGGER = logging.getLogger(__name__)


FLOAT_DTYPE = np.dtype('float64')
THETAS_KEY = 'thetas'
OFFSET_COEFFS_KEY = 'offset_coeffs'
NONOFFSET_COEFFS_KEY = 'nonoffset_coeffs'
INSTR_COEFFS_KEY = 'instructional_coeffs'
THETA_OFFSETS_KEY = 'theta_offsets'
OPTIMIZED_VARS = (THETAS_KEY, NONOFFSET_COEFFS_KEY, OFFSET_COEFFS_KEY, INSTR_COEFFS_KEY,
                  THETA_OFFSETS_KEY)
SHOULD_TERMINATE_KEY = 'should_terminate'

_logger = logging.getLogger(__name__)


class SolverPars(object):
    """
    Class holding generic parameters for iterative updates on a parameter.  It attempts to prevent
    the user from setting attributes that are not ``learn``, ``num_steps``, ``grad_tol``,
    ``diff_tol``, or ``updater``.
    """
    __slots__ = ['learn', 'num_steps', 'grad_tol', 'diff_tol', 'updater']

    def __init__(self, learn=True, num_steps=None, grad_tol=1e-2, diff_tol=1e-2,
                 updater=None):
        """
        :param bool learn: True if the parameters associated with this instance are to be learned
                            If True, step_size and num_steps must be greater than zero.
        :param int num_steps: number of steps to take in this parameter before updating the next.
        :param float grad_tol: stopping tolerance for gradient
        :param float diff_tol: stopping tolerance for value change
        :param ParameterUpdater updater: a function that takes the gradient and the Hessian of coeff
            log-priors, and returns a step direction
        """
        if not learn:
            if num_steps is not None and num_steps != 0:
                _logger.warn("The argument learn=False was set along with non-zero values of "
                             "num_steps={0}! This may represent a "
                             "misunderstanding of the user. We will not learn the parameters "
                             "associated with these SolverPars and are setting num_steps"
                             "to zero as a precaution.".format(num_steps))
            # To be safe, we set these to zero, even though they should never be used:
            num_steps = 0
        else:
            if num_steps is None:
                num_steps = 1
        if updater is None:
            updater = NewtonRaphson()
        utils.check_positive_float(grad_tol, 'grad_tol')
        utils.check_positive_float(diff_tol, 'diff_tol')
        utils.check_nonnegative_int(num_steps, 'num_steps')
        if not isinstance(updater, NewtonRaphson):
            raise TypeError('updater must be a NewtonRaphson')
        if learn and num_steps == 0:
            raise ValueError("num_steps must be greater than zero if learn=True")
        self.learn = learn
        self.num_steps = num_steps
        self.grad_tol = grad_tol
        self.diff_tol = diff_tol
        self.updater = updater

    def copy(self):
        """
        Make a copy of this object. This is trivially a deep copy.

        :return: A (deep) copy.
        :rtype: SolverPars
        """
        return SolverPars(learn=self.learn, num_steps=self.num_steps,
                          grad_tol=self.grad_tol, diff_tol=self.diff_tol, updater=self.updater)


class UpdateTerms(IntEnum):
    """Indicates which log-probability terms (gradient, Hessian) are required by an updater."""
    none = 0
    grad = 1
    grad_and_hess = 2


class NewtonRaphson(object):
    """
    Newton Raphson update that solves the quadratic problem x = Hessian^-1 grad.
    """
    def __init__(self, step_size=1e-1, ravel_order='F'):
        """
        :param float step_size: step size
        :param str ravel_order: the ravel order for gradient and Hessian reshaping. Used for
            backward-compatibility with IRTLearner which uses order='F', whereas BayesNet uses
            order='C'.
        """
        utils.check_nonnegative_float(step_size, 'step_size')
        self.step_size = step_size
        self.ravel_order = ravel_order

    def __call__(self, x, gradient, hessian, support=None):
        """
        :param np.ndarray x: current estimate of the parameter
        :param np.ndarray gradient: parameter gradient.
        :param np.ndarray hessian: parameter Hessian.
        :param tuple(float) support: the bounds of the variable being updated, used to truncate
            the updated value
        :return: the new estimate after moving in the direction of the Newton step.
        :rtype: np.ndarray
        """
        if hessian is None:
            raise ValueError('Hessian required for second order methods')
        else:
            if np.isscalar(hessian):
                step_vec = -gradient / hessian
            elif isinstance(hessian, np.ndarray):
                # dense matrix
                if hessian.size == gradient.size:
                    # assume Hessian diagonal is stored
                    step_vec = -gradient / np.asarray(hessian)
                else:
                    step_vec = -np.linalg.solve(hessian, gradient.ravel(order=self.ravel_order))
            else:
                # sparse matrix
                if hessian.shape[0] == 1:
                    # sp.linalg.spsolve cannot handle 1D matrices
                    step_vec = -gradient / hessian.toarray()
                else:
                    step_vec = -spsolve(hessian, gradient.ravel(order=self.ravel_order))
            self.step = step_vec.reshape(x.shape, order=self.ravel_order)
            value = x + self.step_size * self.step
            if np.any(~np.isfinite(value)):
                raise RuntimeError("Newly computed values are not all finite!")
            if support is not None:
                np.clip(value, support[0], support[1], out=value)
            return value
