"""
Tests for the callback functionality
"""

import itertools as its
import numpy as np
import unittest

from rnn_prof.irt.callbacks import ConvergenceCallback
from rnn_prof.irt.cpd import GaussianCPD
from rnn_prof.irt.irt import BayesNetLearner
from rnn_prof.irt.node import Node
from rnn_prof.irt.testing_utils import EPSILON

NUM_NODES = 4
NUM_HELD_OUT_NODES = 2


class TestConvergenceCallback(unittest.TestCase):
    def setUp(self):
        nodes = []
        for i in range(NUM_NODES):
            nodes.append(Node(name=str(i), data=np.random.randn(i + 1), cpd=GaussianCPD(dim=i + 1),
                              held_out=i < NUM_HELD_OUT_NODES))
        self.learner = BayesNetLearner(nodes=nodes)

    def test_call(self):
        """
        Test callback returns correct should_continue and that error is thrown if early_stopping but
        no held_out nodes.
        """
        # Test that should_continue from callback is correct
        for early_stopping in (False, True):
            callback = ConvergenceCallback(early_stopping=early_stopping)
            for converged_states in its.product(*((True, False),) * NUM_NODES):
                for held_out_log_prob_deltas in its.product(*((-EPSILON, EPSILON),) *
                                                            NUM_HELD_OUT_NODES):
                    for node, state in zip(self.learner.nodes.values(), converged_states):
                        node.converged = state
                    for (node, held_out_log_prob_delta) in zip(
                            self.learner.nodes.values()[:NUM_HELD_OUT_NODES],
                            held_out_log_prob_deltas):
                        node.log_prob_delta = held_out_log_prob_delta

                    expected_should_continue = not all(converged_states[NUM_HELD_OUT_NODES:])
                    if early_stopping and sum(held_out_log_prob_deltas) <= 0:
                        expected_should_continue = False

                    actual_should_continue = callback(self.learner)

                    self.assertEqual(expected_should_continue, actual_should_continue)

        # Test condition check
        for node in self.learner.nodes.values():
            node.held_out = False
        callback = ConvergenceCallback(early_stopping=True)
        with self.assertRaises(ValueError):
            callback(self.learner)

    def test_callback_interface(self):
        """ Test that all callback function (interfaces still work). """
        callback = ConvergenceCallback()
        _ = callback(self.learner)
