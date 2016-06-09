"""
Tests for the BayesNetLearner abstract class and its implementations.
"""
from __future__ import division

import logging
import numpy as np
from scipy import sparse as sp
import unittest

from rnn_prof.irt import irt as undertest
from rnn_prof.irt.cpd import GaussianCPD
from rnn_prof.irt.node import Node
from rnn_prof.irt.testing_utils import MockLearner

LOGGER = logging.getLogger(__name__)
EQUIV_DECIMAL_PLACES = 8
MAX_ITER = 500
NUM_TESTS = 5


class TestBayesNetLearner(unittest.TestCase):
    def setUp(self):
        # use the test learner
        self.learner = MockLearner()

    def test_learn(self):
        """ Test that nodes get evidence from all their children (i.e. nodes are processed in
        topological order) and the summed log-posterior equals to the sum of the contributions from
        all nodes.
        """
        self.learner.learn()

        # test that nodes got evidence updates in the correct order
        self.assertEqual(self.learner.nodes['A'].obtained_evidence_terms,
                         {self.learner.nodes['B']: 'A'})
        self.assertEqual(self.learner.nodes['B'].obtained_evidence_terms,
                         {self.learner.nodes['C']: 'B', self.learner.nodes['D']: 'B'})
        self.assertEqual(self.learner.nodes['C'].obtained_evidence_terms,
                         {self.learner.nodes['E']: 'C'})
        self.assertEqual(self.learner.nodes['D'].obtained_evidence_terms,
                         {self.learner.nodes['E']: 'D', self.learner.nodes['F']: 'D'})
        self.assertEqual(self.learner.nodes['E'].obtained_evidence_terms, {})
        self.assertEqual(self.learner.nodes['F'].obtained_evidence_terms, {})

        # test that log-probs from all nodes have been added
        self.assertAlmostEqual(self.learner.log_posterior,
                               sum(n.log_prob for n in self.learner.nodes.values()),
                               places=EQUIV_DECIMAL_PLACES)

    def test_get_posterior_hessian(self):
        """ Tests the computation of the log-posterior Hessian with a simple graph,
              X
             /|
            / |
           v  v
           Y  Z
        where all nodes contain 2D Gaussians and node X encodes the mean for nodes Y and Z
        """
        for k in range(NUM_TESTS):
            prec_x = np.diag(np.random.rand(2), 0)
            prec_y = sp.diags(np.random.rand(2), 0)  # throw in a sparse precision
            prec_z = np.diag(np.random.rand(2), 0)
            node_x = Node(name='x', data=np.random.randn(2), cpd=GaussianCPD(precision=prec_x))
            node_y = Node(name='y', data=np.random.randn(2), cpd=GaussianCPD(precision=prec_y),
                          param_nodes={GaussianCPD.MEAN_KEY: node_x})
            node_z = Node(name='z', data=np.random.randn(2), cpd=GaussianCPD(precision=prec_z),
                          param_nodes={GaussianCPD.MEAN_KEY: node_x}, held_out=True)
            learner = undertest.BayesNetLearner(nodes=[node_x, node_y, node_z])
            np.testing.assert_almost_equal(learner.get_posterior_hessian('x', use_held_out=True),
                                           -prec_x - prec_y - prec_z)
            np.testing.assert_almost_equal(learner.get_posterior_hessian('x', use_held_out=False),
                                           -prec_x - prec_y)
            np.testing.assert_almost_equal(learner.get_posterior_hessian('x'), -prec_x - prec_y)
            np.testing.assert_almost_equal(learner.get_posterior_hessian('x', np.random.randn(2)),
                                           -prec_x - prec_y)
