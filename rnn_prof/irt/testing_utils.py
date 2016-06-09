"""
Utilities for testing
"""
from collections import namedtuple

import numpy as np

from .callbacks import ConvergenceCallback
from .cpd import GaussianCPD
from .irt import BayesNetLearner
from .node import Node

EPSILON = 1e-3
NUM_TRIALS = 3

NUM_ITEMS = 50
NUM_INSTR_ITEMS = 0
NUM_LATENT = 1
NUM_CHOICES = 2
NUM_RESPONSES = 500
NUM_STUDENTS = 20
PROB_CORRECT = 0.5

THETA_MU = 0.0
THETA_SIGMA = 1.0

NONOFFSET_COEFF_SHAPE_GEN = 100.0
NONOFFSET_COEFF_SCALE_GEN = 0.01

NONOFFSET_COEFF_SHAPE = 2.0
NONOFFSET_COEFF_SCALE = 0.5

OFFSET_COEFF_MU = 0.0
OFFSET_COEFF_SIGMA = 1.0

INSTR_COEFF_MU = 0.05
INSTR_COEFF_SIGMA = 0.025

THETA_OFFSETS_SIGMA = 0.05

ResponseData = namedtuple('ResponseData', ['correct', 'student_idx', 'item_idx'])


def log_norm_pdf(x, mu=0.0, var=1.0):
    """ Evaluate the log normal pdf.
    :param np.ndarray|float x: point at which to evaluate the log norm pdf
    :param np.ndarray|float mu: mean of the normal distribution
    :param np.ndarray|float var: variance of the normal distribution.
    """
    return -0.5 * np.log(2. * np.pi * var) - 0.5 / var * (x - mu) ** 2


FINITE_DIFF_EPSILON = 1e-6
ALMOST_EQUAL_EPSILON = 1e-4


def finite_diff_grad(x, func, epsilon=FINITE_DIFF_EPSILON):
    """ Approximate the derivative of a function using finite difference.
     :param np.ndarray x: point at which to evaluate derivative
     :param function func: function with which to take finite differences.
     """
    fwd_x = np.copy(x)
    bwd_x = np.copy(x)
    fwd_xx = fwd_x.ravel()
    bwd_xx = bwd_x.ravel()
    y = np.zeros(x.shape)
    yy = y.ravel()
    for i in xrange(x.size):
        fwd_xx[i] += epsilon
        bwd_xx[i] -= epsilon
        yy[i] = (func(fwd_x) - func(bwd_x)) / 2.0 / epsilon
        fwd_xx[i] -= epsilon
        bwd_xx[i] += epsilon
    return y


def finite_diff_hessian(x, grad, epsilon=FINITE_DIFF_EPSILON):
    """ Approximate the Hessian of a function using finite difference in the partial gradient.
    :param np.ndarray x: point at which to evaluate derivative
    :param function grad: function that returns the gradient
    """
    fwd_x = np.copy(x)
    bwd_x = np.copy(x)
    fwd_xx = fwd_x.ravel()
    bwd_xx = bwd_x.ravel()
    y = np.zeros((x.size, x.size))
    for i in xrange(x.size):
        for j in xrange(x.size):
            fwd_xx[i] += epsilon
            bwd_xx[i] -= epsilon
            y[i, j] = (grad(fwd_x).ravel()[j] - grad(bwd_x).ravel()[j]) / 2.0 / epsilon
            fwd_xx[i] -= epsilon
            bwd_xx[i] += epsilon
    return y


def finite_diff_hessian_diag(x, grad, epsilon=FINITE_DIFF_EPSILON):
    """ Approximate the diagonal of the Hessian of a function using finite difference in the
    partial gradient.
    :param np.ndarray x: point at which to evaluate derivative
    :param function grad: function that returns the gradient
    """
    fwd_x = np.copy(x)
    bwd_x = np.copy(x)
    fwd_xx = fwd_x.ravel()
    bwd_xx = bwd_x.ravel()
    y = np.zeros(x.shape)
    yy = y.ravel()
    for i in xrange(x.size):
        fwd_xx[i] += epsilon
        bwd_xx[i] -= epsilon
        yy[i] = (grad(fwd_x).ravel()[i] - grad(bwd_x).ravel()[i]) / 2.0 / epsilon
        fwd_xx[i] -= epsilon
        bwd_xx[i] += epsilon
    return y


def generate_data(num_students=NUM_STUDENTS,
                  num_items=NUM_ITEMS,
                  num_responses=NUM_RESPONSES,
                  prob_correct=PROB_CORRECT):
    """ Simulate student response data (independently of any parameters).

    :param int num_students: Number of unique student ids.
    :param int num_items: number of assessment items
    :param int num_responses: number of responses to generate
    :param float prob_correct: probability of correct (probability of choosing first choice when
                               num_choices > 1; probability of other choices are all equal)
    :return: the response data
    :rtype: ResponseData
    """
    correct = np.random.rand(num_responses) < prob_correct
    num_responses_per_student, remainder = divmod(num_responses, num_students)
    unique_student_ids = range(num_students)
    student_idx = [reg_id for reg_id in unique_student_ids for _ in
                   range(num_responses_per_student)]
    # If num_responses can't be perfectly divided into students, add the remaining responses
    # to the last student id:
    student_idx.extend([unique_student_ids[-1]] * remainder)
    student_idx = np.array(student_idx)

    item_idx = np.random.random_integers(low=0, high=num_items-1, size=num_responses)
    np.random.shuffle(student_idx)

    return ResponseData(correct, student_idx, item_idx)


class MockNode(Node):
    """
    A test node class that stores the evidence terms passed into it and does nothing with them,
    and whose update method returns a dictionary with param node names
    """

    def __init__(self, *args, **kwargs):
        super(MockNode, self).__init__(*args, **kwargs)
        self.obtained_evidence_terms = {}

    def update(self, evidence_terms=None):
        """ An update function that stores all the evidence infos passed to it, and sets its
        log_prob to a random Gaussian value

        :param list evidence_terms: evidence information passed into the node
        :return: the names of all param nodes
        :rtype: dict[Node, str]
        """
        if evidence_terms is not None:
            self.obtained_evidence_terms.update(evidence_terms)
        self.log_prob = np.random.randn()
        return {v: v.name for k, v in self.param_nodes.iteritems()}


class MockLearner(BayesNetLearner):
    """
    A learner with the following graph of TestNodes (directed edges pointing down):

       A
       |
       B
      / \
     C  D
     \ / \
     E   F
    """

    def __init__(self):
        cpd = GaussianCPD(dim=1)
        node_a = MockNode(name='A', data=None, cpd=cpd)
        node_b = MockNode(name='B', data=None, cpd=cpd, param_nodes={'mean': node_a})
        node_c = MockNode(name='C', data=None, cpd=cpd, param_nodes={'mean': node_b})
        node_d = MockNode(name='D', data=None, cpd=cpd, param_nodes={'mean': node_b})
        node_e = MockNode(name='E', data=None, cpd=cpd, param_nodes={'mean': node_c,
                                                                     'precision': node_d})
        node_f = MockNode(name='F', data=None, cpd=cpd, param_nodes={'mean': node_d})
        super(MockLearner, self).__init__(nodes=[node_a, node_b, node_c, node_d, node_e, node_f],
                                          max_iterations=1, callback=ConvergenceCallback())
