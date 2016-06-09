"""
Unit tests for computing parameter update steps based on gradients and Hessians.
"""
from __future__ import division

import logging
import numpy as np
import unittest
from scipy.sparse.construct import csr_matrix

from rnn_prof.irt import updaters as undertest

LOGGER = logging.getLogger('test_irt')

NUM_TRIALS = 10


class TestUpdaters(unittest.TestCase):
    def setUp(self):
        """
        Make random gradients and Hessians for variables that are 2D numpy arrays.
        """
        def rand_posdef_matrix(dim):
            """random positive definite, well-conditioned matrix"""
            A = np.random.randn(dim, dim)
            u, s, v = np.linalg.svd(A)
            return u.dot(np.diag(np.clip(s, 0.1, np.inf))).dot(u.T)
        shapes = [(np.random.randint(1, 10), np.random.randint(1, 10)) for _ in range(NUM_TRIALS)]
        # make sure at least one trial uses (1,1) matrix
        shapes[0] = (1, 1)
        self.xs = [np.random.randn(*shape) for shape in shapes]
        self.grads = [np.random.randn(*shape) for shape in shapes]
        self.hessians = [rand_posdef_matrix(shape[0]*shape[1]) for shape in shapes]

    def test_support(self):
        """
        Test that the updated value is not outside the support.
        """
        lbound = 0.1
        ubound = 0.15
        updater = undertest.NewtonRaphson()
        for x, grad, hessian in zip(self.xs, self.grads, self.hessians):
            new_val = updater(x, grad, hessian, support=(lbound, ubound))
            self.assertTrue(np.all(new_val >= lbound))
            self.assertTrue(np.all(new_val <= ubound))

    def test_newton_raphson(self):
        """ Test that Newton-Raphson solves the quadratic problem. """
        updater = undertest.NewtonRaphson()
        for x, grad, hessian in zip(self.xs, self.grads, self.hessians):
            expected_step_vec = -np.linalg.inv(hessian).dot(grad.ravel(order='F'))
            expected_step_vec = expected_step_vec.reshape(grad.shape[1], grad.shape[0]).T
            expected_estimate = x + updater.step_size * expected_step_vec
            # test solution on toy matrix
            nr_estimate = updater(x, grad, hessian)
            np.testing.assert_almost_equal(expected_estimate, nr_estimate, decimal=4)
            # test when toy Hessian is sparse
            nr_sparse_estimate = updater(x, grad, csr_matrix(hessian))
            np.testing.assert_almost_equal(nr_sparse_estimate, expected_estimate, decimal=4)
