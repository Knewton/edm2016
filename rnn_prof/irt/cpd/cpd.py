"""
A module containing the abstract base class for a conditional probability distribution (CPD) in a
Bayes Net.
"""
from __future__ import division
from abc import ABCMeta, abstractmethod
from collections import namedtuple

import numpy as np


FLOAT_DTYPE = np.dtype('float64')

FunctionInfo = namedtuple('FunctionInfo', ['value', 'gradient', 'hessian'])


class CPDTerms(object):
    """A class containing the possible outputs of a conditional probability distribution:
     the value (log_probability), the gradients and Hessians w.r.t. the data and the parameters."""
    def __init__(self, log_prob, wrt=None):
        """
        :param float log_prob: the value of the CPD at the input data and parameters
        :param dict[str, FunctionInfo] wrt: gradients and Hessian w.r.t. data/params indicated by
            the keys
        """
        if wrt is not None and not isinstance(wrt, dict):
            raise TypeError("wrt must be a dict")
        self.log_prob = log_prob
        self.wrt = wrt or {}


class CPD(object):
    """
    The abstract base class for a conditional probability distribution Pr(data|{params}).
    Parameters may be passed in during initialization (if storing or pre-computing intermediate
    quantities is desirable) or at call-time.
    """
    __metaclass__ = ABCMeta

    # Keys in ``terms_to_compute`` input argument and ``CPDTerms.wrt`` output structure. The data
    # key is reserved and cannot be one of parameter keys.
    DATA_KEY = 'data'
    PARAM_KEYS = ()
    support = None

    @abstractmethod
    def __call__(self, data, params, terms_to_compute=None):
        """
        :param data: the data point at which to evaluate the CPD and its gradients
        :param params: keyword arguments for distribution parameters
        :param dict[str, UpdateTerms] terms_to_compute: which data/parameter gradients and Hessians
            to compute
        :return: the value and gradients of the CPD
        :rtype: CPDTerms
        """
        raise NotImplementedError

    def _validate_param_keys(self, input_keys, param_term_keys):
        """
        Check that all required parameters are available (from init or args) and check that
        requested gradients are for parameters that exist.

        :param list|None input_keys: keys of parameters passed into the CPD
        :param list|None param_term_keys: keys of parameters for which gradients/Hessians are
            requested
        """
        input_keys = input_keys or {}
        param_term_keys = param_term_keys or {}

        for par_key in self.PARAM_KEYS:
            if par_key == self.DATA_KEY:
                raise ValueError("{} is a reserved key, cannot be a parameter".format(
                    self.DATA_KEY))
            # check that all the parameters have been initialized or passed in
            if getattr(self, par_key, None) is None and par_key not in input_keys:
                raise ValueError("must initialize with %s or pass it in as a parameter" % par_key)

        # check that only valid param gradients are requested
        for par_key in param_term_keys:
            if par_key != self.DATA_KEY and par_key not in self.PARAM_KEYS:
                raise ValueError("Terms requested for non-parameter {}".format(par_key))


class DistributionInfo(object):
    """
    Base data structure for distributions. The main usage is a base class for priors.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """ The dimension variable represents the number of variables this distribution is over and
        must be set in the subclass. It is just initialized here. """
        self.dim = 0

    @abstractmethod
    def log_prob(self, x):
        """ Return a FunctionInfo object containing the value, gradient,
        and hessian of the log probability of x according to this distribution.
        Subclasses should implement this method. The returned hessian can be None,
        in which case only gradient will be used.

        :param numpy.ndarray x: A 1D numpy array at which to evaluate the distribution
        :rtype: FunctionInfo
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, num_samples):
        """ Draw samples from the distribution.  Subclasses may implement this method.
        :param int num_samples: number of samples to draw
        :return: samples from the distribution.
        :rtype: np.ndarray
        """
        raise NotImplementedError
