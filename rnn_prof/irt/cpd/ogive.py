"""
A module containing Ogive item response functions.
"""
from __future__ import division
from abc import ABCMeta
import logging

import numpy as np
from scipy import stats as st

from .cpd import CPDTerms, CPD, FunctionInfo
from ..constants import THETAS_KEY, OFFSET_COEFFS_KEY, NONOFFSET_COEFFS_KEY
from ..linear_operators import IndexOperator
from ..updaters import UpdateTerms

LOGGER = logging.getLogger(__name__)


class OgiveCPD(CPD):
    """
    Abstract base class for the correctness probability distributions that use the ogive item
    response function (Normal CDF).  Contains common methods for 1PO and 2PO IRT response funcs.
    """
    __metaclass__ = ABCMeta
    _irf_arg_grad_cache = None
    _irf_arg_hessian_cache = None

    @staticmethod
    def bernoulli_logli(trues, probs, average=False):
        """ Compute the log-likelihood of all the data, given the Bernoulli probabilities.
        :param np.ndarray trues: array of boolean data values
        :param np.ndarray probs: array of probability of true
        :param bool average: whether to return the average log-likelihood, rather than the sum
        :return: the log-likelihood
        :rtype: float
        """
        if trues.shape != probs.shape:
            raise ValueError("trues and probs have shapes {} and {}, must be numpy arrays of same "
                             "shape".format(trues.shape, probs.shape))
        falses = np.logical_not(trues)
        log_li = np.sum(np.log(probs[trues])) + np.sum(np.log(1.0 - probs[falses]))
        if average:
            return log_li / trues.size
        else:
            return log_li

    # a dictionary of transformations from parameters to data, keyed on parameter string
    # for example, lin_operators['thetas'].dot(thetas) gives a vector of thetas in response-space
    lin_operators = {}

    def _to_data_space(self, par_key, x):
        """ A thin wrapper around forward projection operation that is a little more legible. """
        return self.lin_operators[par_key] * x

    def _all_to_data_space(self, **params):
        """ Convenience function to map all parameters provided as arguments into response space """
        return {par_key: self._to_data_space(par_key, par) for par_key, par in params.iteritems()}

    def _to_par_space(self, par_key, x):
        """ A thin wrapper around backward projection operation that is a little more legible. """
        return self.lin_operators[par_key].rmatvec(x)

    def _validate_args(self, correct, terms_to_compute, **input_params):
        # check key alignment
        if terms_to_compute:
            self._validate_param_keys(input_params.keys(), terms_to_compute.keys())
        # check input dimensions
        if not isinstance(correct, np.ndarray) or correct.ndim != 1:
            raise ValueError("correct must be 1D numpy array")
        for par_key, param in input_params.iteritems():
            if not isinstance(param, np.ndarray) or param.ndim != 2:
                raise ValueError("{} must be a 2D numpy array".format(par_key))

    def compute_prob_correct(self, **params):
        """ Compute the probability of correct for each response, given student and item parameters.
        :return: the rps (probabilities that each response is correct)
        :rtype: np.ndarray
        """
        params_dataspace = self._all_to_data_space(**params)
        irf_arg = self._irf_arg(**params_dataspace)
        return self._prob_correct_from_irf_arg(irf_arg, **params_dataspace)

    def compute_prob_true(self, **params):
        """ Wrapper for compute_prob_correct that implements the BinaryCPD interface. Used by
        the Metrics class."""
        return self.compute_prob_correct(**params)

    def index_map(self, par_key):
        """ Utility function for extracting the remapping index from data to params.  For example,
        ``index_map('thetas')`` gets the theta index associated with each interaction.
        :param str par_key: which index map to get
        :return: the reindexing map
        :rtype: np.ndarray[int]
        """
        linear_operator = self.lin_operators[par_key]
        return linear_operator * np.arange(linear_operator.shape[1])

    def _irf_arg_grad(self, key, **params):
        """ Compute the gradient of the item response function argument w.r.t. parameter indicated
        by key. """
        raise NotImplementedError

    @staticmethod
    def _irf_arg(**params):
        """ Compute the argument of the item response function, e.g., for the OnePO model:
        (thetas + offset_coeffs). """
        raise NotImplementedError

    @staticmethod
    def _prob_correct_from_irf_arg(irf_arg, **params_per_response):
        """ Compute the probability of correct for each interaction from the item response function
        argument.
        :param np.ndarray irf_arg: argument of the item response function
        :param dict params_per_response: additional arguments, to allow flexibility
            for subclasses to apply transformations to prob correct outside of the IRF
        :return: probability of correct for each response
        :rtype: np.ndarray
        """
        return st.norm.cdf(irf_arg)

    @staticmethod
    def _d_prob_correct_from_irf_arg(irf_arg, **params_per_response):
        """ Compute the gradient of the probability of correct for each interaction w.r.t. the item
            response function argument.
        :param np.ndarray irf_arg: argument of the item response function
        :param dict params_per_response: additional arguments, to allow flexibility
            for subclasses to apply transformations to prob correct outside of the IRF
        :return: the gradient of probability of each correct response
        :rtype: np.ndarray
        """
        return st.norm.pdf(irf_arg)

    @staticmethod
    def _grad_from_irf_arg_grad(correct, prob_correct, d_prob_correct, irf_arg_grad=None):
        """
        Compute gradient of log posterior of each response given gradient of argument of the
        item response function.

        TODO: see if storing corrects/incorrects separately speeds up this function

        :param np.ndarray correct: 1D array of boolean correctness values, one for each response
        :param np.ndarray prob_correct: 1D array of probability of correct for each response
        :param np.ndarray d_prob_correct: 1D array of derivative of probability of correct
        :param np.ndarray|None irf_arg_grad: gradient of argument of IRF. If we're optimizing
            thetas, this is nonoffset_coeffs and vice versa.  If None, treated as 1.0.
        :return: the gradient of the log likelihood for each response. Size same as irf_arg_grad
        :rtype: np.ndarray
        """
        incorrect = np.logical_not(correct)
        if irf_arg_grad is None:
            grad_per_response = np.empty((len(correct), 1))
            grad_per_response[correct, :] = (d_prob_correct[correct] /
                                             prob_correct[correct])[:, np.newaxis]
            grad_per_response[incorrect, :] = (d_prob_correct[incorrect] /
                                               (prob_correct[incorrect] - 1.0))[:, np.newaxis]
        else:
            grad_per_response = np.copy(irf_arg_grad)
            grad_per_response[correct, :] *= (d_prob_correct[correct] /
                                              prob_correct[correct])[:, np.newaxis]
            grad_per_response[incorrect, :] *= (d_prob_correct[incorrect] /
                                                (prob_correct[incorrect] - 1.0))[:, np.newaxis]
        return grad_per_response

    def _cached_grad_from_irf_arg_grad(self, *args):
        """ Interface for an optional cached version of `_grad_from_irf_arg_grad` implemented in
        the subclass. Defaults to re-computing every time if not overridden in the subclass."""
        return self._grad_from_irf_arg_grad(*args)

    @staticmethod
    def _hessian_from_irf_arg_grad(correct, prob_correct, d_prob_correct, irf_arg,
                                   irf_arg_grad=None):
        """
        Compute the diagonal entries of the hessian of the log posterior for each response given
        gradient of the argument of the item response function. Note that the contribution of each
        response to the hessian of the log-likelihood w.r.t. proficiencies is only to the diagonal
        element(s) corresponding to the assessed concept(s) and nowhere else.

        :param np.ndarray correct: 1D array of boolean correctness values, one for each response
        :param np.ndarray prob_correct: 1D array of probability of correct for eaceh response
        :param np.ndarray d_prob_correct: 1D array of derivative of probability of correct
        :param np.ndarray irf_arg: argument of IRF -- typically,
            thetas * nonoffset_coeffs + offset_coeffs
        :param np.ndarray|None irf_arg_grad: gradient of argument of IRF. If we're optimizing
            thetas, this is nonoffset_coeffs and vice versa. If None, treated as np.ones, but
            unncessary multiplication by ones is avoided.
        :return: Diagonal entries of 2nd deriv. of log likelihood for each response. Shape is same
                 as irf_arg_grad.
        :rtype: np.ndarray
        """
        incorrect = np.logical_not(correct)
        if irf_arg_grad is None:
            neg_irf_arg_grad_sq_dpc = -d_prob_correct[:, np.newaxis]
            hess_per_response = np.empty((len(correct), 1))
        else:
            neg_irf_arg_grad_sq_dpc = -d_prob_correct[:, np.newaxis] * irf_arg_grad * irf_arg_grad
            hess_per_response = np.empty_like(irf_arg_grad)
        irf_arg_pc_plus_dpc = irf_arg * prob_correct + d_prob_correct

        hess_per_response[correct, :] = (neg_irf_arg_grad_sq_dpc[correct, :] *
                                         (irf_arg_pc_plus_dpc[correct] /
                                          np.square(prob_correct[correct]))[:, np.newaxis])
        hess_per_response[incorrect, :] = (neg_irf_arg_grad_sq_dpc[incorrect, :] *
                                           ((irf_arg_pc_plus_dpc[incorrect] -
                                             irf_arg[incorrect]) /
                                            np.square(1. - prob_correct[incorrect]))
                                           [:, np.newaxis])
        return hess_per_response

    def _cached_hessian_from_irf_arg_grad(self, *args):
        """ Interface for an optional cached version of `_hessian_from_irf_arg_grad` implemented
        in the subclass. Defaults to re-computing every time if not overridden in the subclass."""
        return self._hessian_from_irf_arg_grad(*args)

    def __call__(self, correct, terms_to_compute=None, **params):
        """
        :param np.ndarray correct: the data point at which to evaluate the CPD and its gradients
        :param np.ndarray thetas: student proficiencies
        :param np.ndarray offset_coeffs: item negative difficulties
        :param np.ndarray nonoffset_coeffs: item discriminabilities
        :param dict[str, UpdateTerms] terms_to_compute: which data/parameter gradients and Hessians
            to compute
        :return: the value, the gradients, and the Hessians of the CPD
        :rtype: CPDTerms
        """
        self._validate_args(correct, terms_to_compute, **params)
        params_per_response = self._all_to_data_space(**params)

        # compute re-used quantities
        irf_arg = self._irf_arg(**params_per_response)
        prob_correct = self._prob_correct_from_irf_arg(irf_arg, **params_per_response)
        # clip RP for stability
        np.clip(prob_correct, 1e-16, 1 - 1e-16, out=prob_correct)
        log_likelihood = self.bernoulli_logli(correct, prob_correct)

        wrt = {}
        if terms_to_compute:
            if terms_to_compute.get(self.DATA_KEY, UpdateTerms.none) > UpdateTerms.none:
                raise NotImplementedError("Data gradient/Hessian not implemented in OgiveCPDs.")
            # cache response-space grad/hess for models (e.g., OnePO) with identical irf_arf_grads
            self._irf_arg_grad_cache = None
            self._irf_arg_hessian_cache = None
            for key, what_to_compute in terms_to_compute.iteritems():
                gradient = None
                hessian = None
                if what_to_compute > UpdateTerms.none:
                    d_prob_correct = self._d_prob_correct_from_irf_arg(irf_arg,
                                                                       **params_per_response)
                    irf_arg_grad = self._irf_arg_grad(key, **params_per_response)
                    grad_per_response = self._cached_grad_from_irf_arg_grad(correct,
                                                                            prob_correct,
                                                                            d_prob_correct,
                                                                            irf_arg_grad)

                    gradient = self._to_par_space(key, grad_per_response)
                if what_to_compute == UpdateTerms.grad_and_hess:
                    hessian = self._cached_hessian_from_irf_arg_grad(correct, prob_correct,
                                                                     d_prob_correct,
                                                                     irf_arg,
                                                                     irf_arg_grad)
                    hessian = self._to_par_space(key, hessian)
                wrt[key] = FunctionInfo(log_likelihood, gradient, hessian)

        return CPDTerms(log_prob=log_likelihood, wrt=wrt)


class OnePOCPD(OgiveCPD):
    """
    One parameter ogive response cpd, irf_arg = f(theta + offset_coeff)
    """

    PARAM_KEYS = (THETAS_KEY, OFFSET_COEFFS_KEY)

    def __init__(self, theta_idx, item_idx, num_thetas=None, num_items=None):
        """
        :param np.ndarray theta_idx: a 1D array of indices mapping `correct` to `thetas`
        :param np.ndarray item_idx: a 1D array of indices mapping `correct` to items
        :param int|None num_thetas: optional number of students. Default is one plus
                                      the maximum index.
        :param int|None num_items: optional number of items. Default is one plus
                                   the maximum index.
        """
        self.lin_operators = {THETAS_KEY: IndexOperator(theta_idx, num_thetas),
                              OFFSET_COEFFS_KEY: IndexOperator(item_idx, num_items)}

    @staticmethod
    def _irf_arg(thetas, offset_coeffs):
        """ Compute the item response function argument.
        :param np.ndarray thetas: An array of thetas, of shape (num_responses, 1)
        :param np.ndarray offset_coeffs: The difficulties (betas), of shape (num_responses, 1)
        :return: thetas + offset_coeffs, raveled, with shape (num_responses, )
        :rtype: np.ndarray
        """
        return (thetas + offset_coeffs).ravel()

    def _irf_arg_grad(self, key, **params):
        """ Compute the gradient of the irf argument w.r.t. parameter indicated by `key`,
        which is always None (corresponding to all ones for OnePO CPD).

        :param str key: parameter for which the gradient is requested
        :param params: parameters of the item response function.
        :return: the gradient
        :rtype: None
        """
        return None


class TwoPOCPD(OgiveCPD):
    """
    One parameter ogive response cpd, irf_arg = f(nonoffset_coeff * theta + offset_coeff)
    """
    PARAM_KEYS = (THETAS_KEY, OFFSET_COEFFS_KEY, NONOFFSET_COEFFS_KEY)

    def __init__(self, theta_idx, item_idx, num_thetas=None, num_items=None):
        """
        :param np.ndarray theta_idx: a 1D array of indices mapping `correct` to `thetas`
        :param np.ndarray item_idx: a 1D array of indices mapping `correct` to items
        :param int|None num_thetas: optional number of students. Default is one plus
                                    the maximum index.
        :param int|None num_items: optional number of items. Default is one plus
                                   the maximum index.
        """
        self.lin_operators = {THETAS_KEY: IndexOperator(theta_idx, num_thetas),
                              OFFSET_COEFFS_KEY: IndexOperator(item_idx, num_items),
                              NONOFFSET_COEFFS_KEY: IndexOperator(item_idx, num_items)}

    @staticmethod
    def _irf_arg(thetas, offset_coeffs, nonoffset_coeffs):
        """ Compute the item response function argument.
        :param np.ndarray thetas: An array of thetas, of shape (num_responses, num_latent)
        :param np.ndarray offset_coeffs: The difficulties (betas), of shape (num_responses, 1)
        :param np.ndarray nonoffset_coeffs: The discriminabilities (alphas), of shape (
            num_responses, num_latent)
        :return: An array of dimension num_responses where the i'th entry is
         \sum_j thetas_ij*nonoffset_coeffs_ij + offset_coeffs_i, with shape (num_responses, )
        :rtype: np.ndarray
        """
        return np.sum(thetas * nonoffset_coeffs, axis=1) + offset_coeffs.ravel()

    def _irf_arg_grad(self, key, **params):
        """ Compute the gradient of the irf argument w.r.t. parameter indicated by `key`.
        :param str key: parameter for which the gradient is requested
        :param params: parameters of the item response function.
        :return: the gradient
        :rtype: None|np.ndarray
        """
        if key == THETAS_KEY:
            return params[NONOFFSET_COEFFS_KEY]
        elif key == NONOFFSET_COEFFS_KEY:
            return params[THETAS_KEY]
        else:
            return None
