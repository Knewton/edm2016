"""
A module containing the basic Bayes Net object: the node.
"""
import copy
import logging
import sys

import numpy as np
from scipy import sparse as sp

from .constants import DEFAULT_STEP_SIZE
from .cpd import CPD, GaussianCPD
from .metrics import Metrics
from .updaters import UpdateTerms, SolverPars, NewtonRaphson

LOGGER = logging.getLogger(__name__)


class Node(object):
    """
    A node in a Bayes net, containing point estimates of its "data" -- random variable X, links to
    parent nodes representing "parameters" -- random variables Y, and a likelihood function Pr(X|Y).
    It is responsible for integrating evidence for X from child nodes (bottom up info) with prior
    information Pr(X|Y) to update the point estimates \hat{X}.
    A node contains the following (and no other attributes):
        - its name
        - the conditional probability distribution Pr(X|Y) (the prior Pr(X) if there are no parents)
        - its data
        - ids (meta information associated with the data (e.g., UUIDs)
        - parent nodes in the Bayes net, representing parameters (Y)
        - a boolean indicating whether it is "held-out" for cross-validation
        - a SolverPars object that holds optimization settings and the updater
        - convergence state indicators (max gradient, max change in values, boolean indicator)
        - prediction accuracy metrics
        - last value of its log-CPD and the delta, to check for convergence
    """
    __slots__ = ['name', 'cpd', 'data', 'ids', 'param_nodes', 'held_out', 'solver_pars',
                 'converged', 'max_diff', 'max_grad', 'metrics', 'log_prob_delta', 'log_prob']

    def __init__(self, name, data, cpd, solver_pars=None, param_nodes=None, held_out=False,
                 ids=None):
        """
        :param str name: the node's name (only used for debugging)
        :param object data: the node's data
        :param CPD cpd: the conditional probability distribution of the data given params
        :param SolverPars solver_pars: optimization parameters, including whether to optimize the
            data at all (solver_pars.learn), the type of updater (first/second order), and
            termination conditions.
        :param dict[str, Node] param_nodes: parent nodes holding the parameters
        :param bool held_out: Whether the data is a held_out test set
        :param np.ndarray|str|None ids: an identifier associated with each element in the data.
            If string, treat as a single ID; if supplied, should match the length of the data.
        """
        # check that likelihood param keys are included in parent keys
        if not isinstance(cpd, CPD):
            raise TypeError("cpd must be of type CPD, but it is {}".format(type(cpd)))
        self.name = name
        self.cpd = cpd
        self.param_nodes = param_nodes or {}
        self.held_out = held_out
        self._check_and_set_data_ids(data, ids)

        updater = NewtonRaphson(ravel_order='C', step_size=DEFAULT_STEP_SIZE)
        self.solver_pars = solver_pars or SolverPars(learn=not self.held_out,
                                                     updater=updater)
        if self.held_out and self.solver_pars.learn:
            raise ValueError('This is a held out test set, but solver_pars.learn is True.')
        self.converged = not self.solver_pars.learn
        self.max_diff = None
        self.max_grad = None

        if not isinstance(self.solver_pars, SolverPars):
            raise TypeError("solver_pars must be a SolverPars")
        if self.solver_pars.learn and self.solver_pars.num_steps != 1:
            raise ValueError("If doing learning, num_steps must be 1.")

        # check that the CPD uses parent nodes' values as params and can compute gradient wrt to
        # them (if the CPD does not use them as params, they should not be connected to this node)
        for param_key in self.param_nodes:
            if param_key not in self.cpd.PARAM_KEYS:
                raise ValueError("CPD does not use {} as a parameter".format(param_key))
        self.metrics = Metrics(self)
        self.log_prob_delta = None
        self.log_prob = None

    def _check_and_set_data_ids(self, data, ids):
        """ Do some basic checks on dimensionality and Sized-ness of data and ids, to better handle
        None and unsized objects.  If ids is a string, treat as singleton.
        Sets the class member variables ``data`` and ``ids``.

        :param object data: the data
        :param np.ndarray|str|None ids: the ids
        """
        if ids is not None:
            # if IDs provided, assume data is Sized
            if isinstance(ids, str):
                ids_len = 1
            elif hasattr(ids, '__len__'):
                ids_len = len(ids)
            else:
                ids_len = 1
            data_len = len(data) if hasattr(data, '__len__') else 1
            if ids_len != data_len:
                raise ValueError("Number of unique ids (%d) should match length of data (%d)" %
                                 (ids_len, data_len))
            self.ids = np.array(ids, ndmin=1)  # in case another iterable passed in
        else:
            # if IDs are None, no checks required
            self.ids = ids
        self.data = data

    @property
    def param_data(self):
        """ Returns a dictionary of current values of parameter nodes' data, keyed on the name of
        the associated CPD parameter.
        :return: values of the parameters stored in self.param_nodes.
        :rtype: dict[str, np.ndarray]
        """
        return {k: n.data for k, n in self.param_nodes.iteritems()}

    @property
    def required_update_terms(self):
        """ Indicates which terms are required by the node's updater.  A child node calls this
        method to determine which terms of the evidence (none, gradient, Hessian) to compute
        :return: required update terms
        :rtype: RequiredUpdateTerms
        """
        if not self.solver_pars.learn:
            return UpdateTerms.none
        else:
            return UpdateTerms.grad_and_hess

    def compute_log_prob(self):
        """
        Compute the log-probability of the data given any parameters. Updates the stored value and
        returns it.

        :return: the log-probability
        :rtype: float
        """
        self.log_prob = self.cpd(self.data, **self.param_data).log_prob
        return self.log_prob

    def update(self, evidence_terms=None):
        """
        Given bottom-up information (evidence gradients) and gradient of own log-probability
        (likelihood of data given params), updates its data and the stored value of the log-prob,
        and checks whether its optimization termination conditions have been met. Returns the
        gradients of the log-prob with respect to all params (post-update to its data).

        :param dict[Node, FunctionInfo] evidence_terms: the update information (gradients/Hessians)
            from all the children's log Pr(child's data|this node's data), computed w.r.t. this
            node's data (which is effectively a parameter in child node's CPD).
        :return: parameter log-prob terms (incl gradients and Hessian) for all the parent nodes
        :rtype: dict[Node, FunctionInfo]
        """

        def is_square_mat(x):
            """
            :param float|np.ndarray|sp.spmatrix x:
            :return: True if x is a matrix (has dimension NxN with N > 1), False otherwise
            :rtype: bool
            """
            return not np.isscalar(x) and x.size > 1 and x.size == x.shape[0] ** 2

        def matrixify(non_mat, dim, to_sparse):
            """
            Convert a scalar/vector into a diagonal matrix (with the scalar/vector on the diagonal)
            :param float|np.ndarray non_mat: A scalar or vector
            :param int dim: The dimension of the resulting diagonal matrix
            :param boolean to_sparse: If True, make the diagonal matrix sparse
            :return: Diagonal matrix
            :rtype: np.ndarray|sp.spmatrix
            """
            is_vec = not np.isscalar(non_mat) and np.prod(non_mat.shape) > 1
            if to_sparse:
                return sp.diags(non_mat.ravel(), 0) if is_vec else non_mat * sp.eye(dim)
            else:
                return np.diag(non_mat.ravel(), 0) if is_vec else non_mat * np.eye(dim)

        def add_hessians(hess1, hess2):
            """
            Add two hessians. Each hessian can be either scalar, vector, or square matrix. In the
            case of vectors/matrices, dimensions are assumed to match. If a parameter
            scalar/vector, it is assumed that it represents a diagonal matrix with the
            scalar/vector as its diagonal values.

            :param float|np.ndarray|sp.spmatrix hess1: The first hessian
            :param float|np.ndarray|sp.spmatrix hess2: The second hessian
            :return: hess1 + hess2
            :rtype: float|np.ndarray|sp.spmatrix
            """
            if is_square_mat(hess1) and not is_square_mat(hess2):
                hess2 = matrixify(hess2, hess1.shape[0], sp.issparse(hess1))
            elif is_square_mat(hess2) and not is_square_mat(hess1):
                hess1 = matrixify(hess1, hess2.shape[0], sp.issparse(hess2))
            return hess1 + hess2

        if self.solver_pars.learn:
            # get the CPD terms required by the data updater
            data_term = {self.cpd.DATA_KEY: self.required_update_terms}
            log_prob_terms = self.cpd(self.data, terms_to_compute=data_term, **self.param_data)

            # update own values based on likelihood
            gradient = log_prob_terms.wrt[self.cpd.DATA_KEY].gradient
            hessian = log_prob_terms.wrt[self.cpd.DATA_KEY].hessian
            # and the evidence
            if evidence_terms is not None:
                for source_node, evidence_term in evidence_terms.iteritems():
                    if evidence_term.gradient is not None:
                        gradient += evidence_term.gradient
                    if evidence_term.hessian is not None:
                        hessian = add_hessians(hessian, evidence_term.hessian)
            new_data = self.solver_pars.updater(self.data, gradient, hessian, self.cpd.support)
            self.max_grad = np.max(np.abs(gradient))
            self.max_diff = np.max(np.abs(new_data - self.data))
            self.data = new_data
            self.converged = (self.max_grad < self.solver_pars.grad_tol and
                              self.max_diff < self.solver_pars.diff_tol)

        # get the CPD terms required by the parameter nodes
        old_log_prob = self.log_prob
        if self.held_out:
            param_terms = None
        else:
            param_terms = {k: n.required_update_terms for k, n in self.param_nodes.iteritems()}
        log_prob_terms = self.cpd(self.data, terms_to_compute=param_terms, **self.param_data)
        self.log_prob = log_prob_terms.log_prob
        if old_log_prob is None:
            self.log_prob_delta = None
        else:
            self.log_prob_delta = self.log_prob - old_log_prob
        return {v: log_prob_terms.wrt.get(k) for k, v in self.param_nodes.iteritems()}

    def subset(self, idx, inplace=False):
        """ Subset (optionally in place) the node's data and cpd to include only some of the
        variables. Returns a new node (if not inplace) and an array for remapping children's
        param_index, for example::

            test_set_thetas = np.unique(theta_idx[:1000])  # thetas for the first 1000 interactions
            param_idx = theta_node.subset(test_set_thetas, inplace=True)  # node modified in place
            trimmed_theta_idx = param_idx[theta_idx]  # used to create new 'responses' node

        :param np.ndarray idx: index of variables that should remain in the node
        :param bool inplace: whether to change the node's cpd and data in place
        :return: the new node (if not done inplace) and the re-index param_index
        :rtype: (Node, np.ndarray)|np.ndarray[int]
        """
        orig_dim = self.cpd.dim

        # construct new data
        data = self.data[idx]

        # trim the CPD if possible
        if hasattr(self.cpd, 'get_subset_cpd'):
            cpd = self.cpd.get_subset_cpd(idx)
        else:
            cpd = self.cpd
        new_dim = cpd.dim

        # reindex for the child's param indices
        param_idx = np.empty(orig_dim, dtype=int)
        param_idx.fill(sys.maxint)
        param_idx[idx] = np.arange(new_dim)
        if inplace:
            self.data = data
            if self.ids is not None:
                self.ids = self.ids[idx]
            self.cpd = cpd
            return param_idx
        else:
            ids = self.ids[idx] if self.ids is not None else None
            subset_node = Node(name=self.name, data=data, cpd=cpd, solver_pars=self.solver_pars,
                               param_nodes=self.param_nodes.copy(), ids=ids)
            return subset_node, param_idx

    def copy(self):
        """
        Make a copy of this object.  The node's data are deep-copied, but the cpd and the links
        to the param-nodes are not (it is up to the user to rewire the graph as desired).
        :return: A copy of the node
        :rtype: Node
        """
        return Node(name=self.name,
                    data=copy.deepcopy(self.data),
                    cpd=self.cpd,
                    solver_pars=self.solver_pars.copy(),
                    param_nodes=self.param_nodes.copy(),  # shallow copy of the dictionary
                    held_out=self.held_out,
                    ids=self.ids)

    def get_all_data_and_ids(self):
        """ Get a dictionary of all data values, keyed by id.  If data has only one non-singleton
        dimension, the values of the returned dictionary will be scalars.
        :return: all the data
        :rtype: dict[object, np.ndarray]
        """
        if self.ids is None:
            raise ValueError("IDs for not stored in node {}".format(self.name))
        if self.data is None:
            raise ValueError("No data in node {}".format(self.name))
        if not hasattr(self.data, '__len__') or len(self.data) < 2:
            # if data is a singleton, do not zip outputs, and unwrap
            if len(self.ids) == 1:
                # unwrap from iterable
                ids = self.ids.ravel()[0]
            else:
                # make hashable
                ids = tuple(self.ids)
            return {ids: self.data}
        return dict(zip(self.ids, np.squeeze(self.data)))

    def get_data_by_id(self, ids):
        """  Helper for getting current data values from stored identifiers
        :param float|list ids: ids for which data are requested
        :return: the stored ids
        :rtype: np.ndarray
        """
        if self.ids is None:
            raise ValueError("IDs not stored in node {}".format(self.name))
        if self.data is None:
            raise ValueError("No data in node {}".format(self.name))
        ids = np.array(ids, ndmin=1, copy=False)
        found_items = np.in1d(ids, self.ids)
        if not np.all(found_items):
            raise ValueError("Cannot find {} among {}".format(ids[np.logical_not(found_items)],
                                                              self.name))
        idx = np.empty(len(ids), dtype='int')
        for k, this_id in enumerate(ids):
            if self.ids.ndim > 1:
                idx[k] = np.flatnonzero(np.all(self.ids == this_id, axis=1))[0]
            else:
                idx[k] = np.flatnonzero(self.ids == this_id)[0]
        return np.array(self.data, ndmin=1)[idx]


class DefaultGaussianNode(Node):
    def __init__(self, name, dim, mean=0.0, precision=1.0, **node_kwargs):
        """ Make a node with a Gaussian CPD and all-zero data.
        :param str name: name of the node
        :param int dim: dimensionality of the data vector
        :param mean: The mean of the Gaussian. See ..py.module:`kirt.bayesnet.cpd.gaussian`
            for more details
        :param precision: The precision of the Gaussian. See
            ..py.module:`.cpd.gaussian` for more details
        :type mean: float|np.ndarray|sp.spmatrix|None
        :type precision: float|np.ndarray|sp.spmatrix|None
        :param node_kwargs: optional parameters to pass to the Node constructor
        :return: the node
        :rtype: Node
        """
        super(DefaultGaussianNode, self).__init__(
            name=name,
            data=np.zeros((dim, 1)),
            cpd=GaussianCPD(dim=dim, mean=mean, precision=precision),
            **node_kwargs)
