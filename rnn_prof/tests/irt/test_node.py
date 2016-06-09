"""
Tests for the Node class
"""
from __future__ import division
import sys

import numpy as np
from scipy import sparse as sp
import unittest
import uuid

from rnn_prof.irt import node as undertest
from rnn_prof.irt.cpd import CPD, GaussianCPD
from rnn_prof.irt.updaters import UpdateTerms, NewtonRaphson, SolverPars
from rnn_prof.irt.cpd.cpd import FunctionInfo
from rnn_prof.irt.testing_utils import EPSILON

NUM_TESTS = 10


class TestNode(unittest.TestCase):
    def gen_data(self, dim=None):
        dim = dim or np.random.randint(1, 10)
        data = np.random.randn(dim, 1)
        cpd = GaussianCPD(dim=dim, precision=np.random.rand(dim, 1))
        ids = np.array([str(uuid.uuid4()) for _ in range(dim)])
        return dim, data, cpd, ids

    def test_node(self):
        """ Test that a Node object correctly outputs its log-probability, updates its data, and
        computes gradients w.r.t. its parameters"""
        densify = lambda z: z.toarray() if sp.issparse(z) else z

        data_key = CPD.DATA_KEY
        mean_key = GaussianCPD.MEAN_KEY

        for trial in range(NUM_TESTS):
            # set up
            dim = np.random.random_integers(1, 10)
            x = np.random.randn(dim, 1)
            # precision is constant
            precision = np.random.randn(dim, dim)
            precision = precision.dot(precision.T)
            cpd = GaussianCPD(dim=dim, precision=precision)
            # mean is a latent parameter (parent node with a random updater)
            mean = np.random.randn(dim, 1)
            mean_node = undertest.Node(name='mean', data=mean, cpd=GaussianCPD(dim=dim),
                                       solver_pars=SolverPars(updater=NewtonRaphson()))
            node = undertest.Node(name='x', data=np.copy(x), cpd=cpd,
                                  param_nodes={mean_key: mean_node},
                                  solver_pars=SolverPars(updater=NewtonRaphson(step_size=1e-3)))
            data_terms_to_compute = {data_key: node.required_update_terms}
            param_terms_to_compute = {mean_key: mean_node.required_update_terms}

            # test the node's log probability
            self.assertAlmostEqual(node.compute_log_prob(),
                                   cpd(x, mean=mean, precision=precision).log_prob,
                                   places=6)
            # test that node correctly sets its required update terms
            self.assertEqual(node.required_update_terms, UpdateTerms.grad_and_hess)

            # test that node updated its data according to the updater
            expected_data_terms = cpd(node.data, mean=mean_node.data,
                                      terms_to_compute=data_terms_to_compute).wrt[data_key]
            evidence_terms = node.update()
            expected_data = NewtonRaphson(step_size=1e-3)(x, expected_data_terms.gradient,
                                                          expected_data_terms.hessian)
            np.testing.assert_allclose(node.data, expected_data)

            # test that node outputs correct evidence terms
            self.assertEqual(evidence_terms.keys(), [mean_node])
            expected_mean_terms = cpd(node.data, mean=mean_node.data,
                                      terms_to_compute=param_terms_to_compute).wrt[mean_key]
            mean_terms = evidence_terms[mean_node]
            np.testing.assert_allclose(mean_terms.gradient, expected_mean_terms.gradient)
            np.testing.assert_allclose(densify(mean_terms.hessian),
                                       densify(expected_mean_terms.hessian))

            # make some mock evidence and test that the node combines it upon update
            ev_grads = np.random.randn(2, 1)
            ev_hess = np.random.randn(2, 1)
            data_terms = node.cpd(node.data, mean=mean_node.data,
                                  terms_to_compute=data_terms_to_compute).wrt[data_key]
            expected_data = NewtonRaphson(step_size=1e-3)(node.data,
                                                          data_terms.gradient + np.sum(ev_grads),
                                                          densify(data_terms.hessian) + np.eye(
                                                              data_terms.hessian.shape[0]) *
                                                          np.sum(ev_hess))
            mock_node1 = undertest.Node('mock', 0, GaussianCPD(dim=1))
            mock_node2 = undertest.Node('mock', 0, GaussianCPD(dim=1))
            mock_evidence_terms = {mock_node1: FunctionInfo(0, ev_grads[0], ev_hess[0]),
                                   mock_node2: FunctionInfo(0, ev_grads[1], ev_hess[1])}
            _ = node.update(evidence_terms=mock_evidence_terms)
            np.testing.assert_allclose(node.data, expected_data)

    def test_converged(self):
        """ Test that node correctly sets its converged flag. """
        for _ in range(NUM_TESTS):
            dim, x, cpd, _ = self.gen_data()
            data_terms = {cpd.DATA_KEY: UpdateTerms.grad_and_hess}
            solver_pars = SolverPars(updater=NewtonRaphson())
            node = undertest.Node(name='test node', data=x, cpd=cpd, solver_pars=solver_pars)

            # add some random "evidence" gradient
            mock_node = undertest.Node('mock', 0, GaussianCPD(dim=1))
            evid_grad = np.random.randn(*x.shape)
            # add random diagonal hessians
            evid_hess = np.random.randn(*x.shape)
            data_grad = cpd(x, terms_to_compute=data_terms).wrt[cpd.DATA_KEY].gradient
            data_hess = cpd(x, terms_to_compute=data_terms).wrt[cpd.DATA_KEY].hessian
            expected_max_grad = np.max(np.abs((evid_grad + data_grad)))
            expected_max_diff = np.max(np.abs(solver_pars.updater(x, evid_grad + data_grad,
                                                                  evid_hess + data_hess) - x))

            for delta_diff, delta_grad in ((-EPSILON, -EPSILON), (-EPSILON, EPSILON),
                                           (EPSILON, -EPSILON), (EPSILON, EPSILON)):
                node.solver_pars.grad_tol = expected_max_grad + delta_grad
                node.solver_pars.diff_tol = expected_max_diff + delta_diff
                _ = node.update(evidence_terms={mock_node: FunctionInfo(0, evid_grad, evid_hess)})

                # node should flag "converged" if both gradients and parameter change tolerances
                # are above the actual values
                self.assertEqual(node.converged, delta_diff > 0 and delta_grad > 0)

                # reset node state
                node.data = np.copy(x)

    def test_subset(self):
        for _ in range(NUM_TESTS):
            for inplace in (False, True):
                dim, data, cpd, ids = self.gen_data()
                node = undertest.Node(name='test node', data=data, cpd=cpd, ids=ids)
                sub_idx = np.random.rand(dim) < 0.5
                # make sure subset_idx is not empty
                while not np.sum(sub_idx):
                    sub_idx = np.random.rand(dim) < 0.5
                if inplace:
                    param_idx = node.subset(sub_idx, inplace=True)
                else:
                    node, param_idx = node.subset(sub_idx, inplace=False)
                expected_param_idx = sys.maxint * np.ones(dim, dtype=int)
                expected_param_idx[sub_idx] = np.arange(np.sum(sub_idx))

                np.testing.assert_array_equal(param_idx, expected_param_idx)
                np.testing.assert_array_equal(node.data, data[sub_idx])
                np.testing.assert_array_equal(node.cpd.mean, cpd.mean[sub_idx])
                np.testing.assert_array_equal(node.cpd.precision, cpd.precision[sub_idx])
                np.testing.assert_array_equal(node.ids, ids[sub_idx])

    def test_copy(self):
        """ Test that node method makes deep copies of the data.  """
        dim, data, cpd, ids = self.gen_data()
        parent_test_node = undertest.Node(name='test parent', data=None, cpd=cpd)
        node = undertest.Node(name='test node', data=data, cpd=cpd,
                              param_nodes={cpd.MEAN_KEY: parent_test_node},
                              held_out=np.random.rand() > 0.5, ids=ids)
        node_copy = node.copy()

        # check that name is copied
        self.assertEquals(node_copy.name, node.name)
        node_copy.name += '_copy'
        self.assertFalse(node_copy.name == node.name)

        # check that held_out is copied
        self.assertEqual(node_copy.held_out, node.held_out)
        node_copy.held_out = not node.held_out
        self.assertFalse(node_copy.held_out == node.held_out)

        # check that cpd is copied by reference
        self.assertTrue(node_copy.cpd is node.cpd)

        # check that param_nodes dictionary is a copy with references to same objects
        self.assertFalse(node_copy.param_nodes is node.param_nodes)
        for par_key in node.param_nodes:
            self.assertTrue(node_copy.param_nodes[par_key] is node.param_nodes[par_key])

        # check that solver_pars is a copy with same member objects
        for key in node.solver_pars.__slots__:
            self.assertEqual(getattr(node_copy.solver_pars, key),
                             getattr(node.solver_pars, key))

        # test that data is copied, and mutating the copy's data leaves the original intact
        np.testing.assert_array_equal(node_copy.data, node.data)
        node_copy.data += np.random.randn(*node_copy.data.shape)
        np.testing.assert_array_equal(node.data, data)
        np.testing.assert_array_equal(node_copy.ids, node.ids)

    def test_get_data_by_id(self):
        dim, data, cpd, ids = self.gen_data()
        node = undertest.Node(name='test node', data=data, cpd=cpd, ids=ids)
        # test setting of ids
        np.testing.assert_array_equal(node.ids, ids)
        # test for one id
        idx = np.random.randint(0, dim)
        np.testing.assert_array_equal(node.get_data_by_id(ids[idx]).ravel(), node.data[idx])
        # test for a random set of ids
        ids_subset = np.random.choice(ids, dim, replace=True)
        np.testing.assert_array_equal(node.get_data_by_id(ids_subset),
                                      [node.data[np.flatnonzero(ids == x)[0]] for x in ids_subset])
        # test for all ids
        self.assertEqual(node.get_all_data_and_ids(), {x: node.get_data_by_id(x) for x in ids})
        # test when data are singleton
        dim, _, cpd, ids = self.gen_data(dim=1)
        node = undertest.Node(name='test node', data=1, cpd=cpd, ids=ids)
        self.assertEqual(node.get_all_data_and_ids(), {x: node.get_data_by_id(x) for x in ids})


class TestDefaultGaussianNode(unittest.TestCase):
    def test_init(self):
        for _ in range(NUM_TESTS):
            dim = np.random.randint(2, 10)
            node = undertest.DefaultGaussianNode('test', dim)
            np.testing.assert_array_equal(node.data, np.zeros((dim, 1)))
            self.assertEqual(node.name, 'test')
            self.assertEqual(node.cpd.__class__, GaussianCPD)
            self.assertEqual(node.cpd.dim, dim)
            np.testing.assert_array_equal(node.cpd.mean, np.zeros(dim))
            self.assertEqual(node.cpd.precision, 1.)

    def test_non_default_init(self):
        for _ in range(NUM_TESTS):
            dim = np.random.randint(2, 10)
            mean = np.random.randn()
            diagonal_precision = np.random.randn() ** 2
            precision = sp.eye(dim) * diagonal_precision
            node = undertest.DefaultGaussianNode('test', dim, mean=mean, precision=precision)
            np.testing.assert_array_equal(node.data, np.zeros((dim, 1)))
            self.assertEqual(node.name, 'test')
            self.assertEqual(node.cpd.__class__, GaussianCPD)
            self.assertEqual(node.cpd.dim, dim)
            np.testing.assert_array_equal(node.cpd.mean,
                                          np.ones(dim) * mean)
            np.testing.assert_array_equal(node.cpd.precision.toarray(),
                                          np.eye(dim) * diagonal_precision)
