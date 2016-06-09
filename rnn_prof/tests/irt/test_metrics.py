"""
Unit tests for BayesNet prediction accuracy metrics
"""
from collections import defaultdict
import itertools as its
import logging
import numpy as np
import unittest
import uuid

from rnn_prof.irt import metrics as undertest
from rnn_prof.irt.constants import TRAIN_RESPONSES_KEY, THETAS_KEY, OFFSET_COEFFS_KEY
from rnn_prof.irt.learners import OnePOLearner

LOGGER = logging.getLogger(__name__)
EPS_DECIMAL = 6


class TestMetrics(unittest.TestCase):
    def setUp(self):
        num_responses = 100
        num_students = 10
        num_items = 10
        self.correct = np.random.rand(num_responses) > 0.5
        item_idx = np.random.choice(num_items, size=num_responses)
        theta_idx = np.random.choice(num_students, size=num_responses)
        self.learner = OnePOLearner(self.correct, theta_idx, item_idx)

        # set non-trivial thetas and offsets
        self.learner.nodes[THETAS_KEY].data = np.random.randn(num_students, 1)
        self.learner.nodes[OFFSET_COEFFS_KEY].data = np.random.randn(num_items, 1)

        params = self.learner.nodes[TRAIN_RESPONSES_KEY].param_data
        self.prob_correct = self.learner.nodes[TRAIN_RESPONSES_KEY].cpd.compute_prob_true(**params)

    def test_compute_logli(self):
        """ Test that the log-likelihood metric normalizes by the size of the node's data. """
        self.assertAlmostEqual(self.learner.nodes[TRAIN_RESPONSES_KEY].metrics.compute_logli(),
                               self.learner.nodes[TRAIN_RESPONSES_KEY].compute_log_prob(),
                               places=EPS_DECIMAL)
        self.assertAlmostEqual(self.learner.nodes[TRAIN_RESPONSES_KEY].metrics.compute_logli(False),
                               self.learner.nodes[TRAIN_RESPONSES_KEY].compute_log_prob(),
                               places=EPS_DECIMAL)
        self.assertAlmostEqual(self.learner.nodes[TRAIN_RESPONSES_KEY].metrics.compute_logli(True),
                               (self.learner.nodes[TRAIN_RESPONSES_KEY].compute_log_prob() /
                                len(self.correct)),
                               places=EPS_DECIMAL)

    def test_compute_naive(self):
        """ Test the Naive (predict most frequent response values) metric."""
        fraction_correct = np.mean(self.correct)
        expected = max(fraction_correct, 1 - fraction_correct)
        actual = self.learner.nodes[TRAIN_RESPONSES_KEY].metrics.compute_naive()
        self.assertAlmostEqual(actual, expected, places=EPS_DECIMAL)

    def test_map_accuracy(self):
        """ Test the MAP accuracy metric."""
        expected = np.mean((self.prob_correct > 0.5) ==
                           self.learner.nodes[TRAIN_RESPONSES_KEY].data)
        actual = self.learner.nodes[TRAIN_RESPONSES_KEY].metrics.compute_map_accuracy()
        self.assertAlmostEqual(actual, expected, places=EPS_DECIMAL)

    def test_d_prime(self):
        """ Test the d-prime statistic"""
        pc_correct = self.prob_correct[self.correct]
        pc_incorrect = self.prob_correct[np.logical_not(self.correct)]
        expected = (np.mean(pc_correct) - np.mean(pc_incorrect)) / \
            np.sqrt(0.5 * np.var(pc_correct) + 0.5 * np.var(pc_incorrect))
        actual = self.learner.nodes[TRAIN_RESPONSES_KEY].metrics.compute_d_prime()
        self.assertAlmostEqual(actual, expected, places=EPS_DECIMAL)

    def test_auc_helper(self):
        """ Test the math meat of the AUC metric computation. """
        num_responses = 50
        num_trials = 100
        for trial in range(num_trials):
            # Create random response correctnesses
            correct_prob = 0.1 + 0.8 * np.random.rand()
            corrects = np.zeros(num_responses, dtype=bool)
            # Make sure there's at least 1 correct and 1 incorrect
            while np.sum(corrects) in (0, num_responses):
                corrects = np.random.rand(num_responses) < correct_prob
                incorrects = np.logical_not(corrects)

            # Create some random response probabilities
            rps = np.random.rand(num_responses)
            num_correct = float(np.sum(corrects))
            num_incorrect = float(np.sum(incorrects))

            # Compute AUC the slow way by iterating through thresholds
            tprs = np.zeros(len(rps) + 2)
            fprs = np.zeros(len(tprs))
            for i, threshold in enumerate(np.r_[-1., np.sort(rps), 2.]):
                tprs[i] = np.sum(np.logical_and(corrects, rps > threshold)) / num_correct
                fprs[i] = np.sum(np.logical_and(incorrects, rps > threshold)) / num_incorrect
            expected_auc = np.trapz(tprs[::-1], fprs[::-1])
            actual_auc = undertest.Metrics.auc_helper(corrects, rps)
            self.assertAlmostEqual(expected_auc, actual_auc)

            # Now compute some edge cases (all rps are 0 or all rps are 1)
            self.assertAlmostEqual(0.5, undertest.Metrics.auc_helper(corrects,
                                                                     np.zeros(num_responses)))
            self.assertAlmostEqual(0.5, undertest.Metrics.auc_helper(corrects,
                                                                     np.ones(num_responses)))

            # Now construct a case where a perfect threshold is possible
            sorted_rps = np.sort(rps)
            corrects = rps > sorted_rps[np.random.randint(1, num_responses-1)]
            self.assertAlmostEqual(1.0, undertest.Metrics.auc_helper(corrects, rps))

    def test_compute_per_student_naive(self):
        num_students = 10
        num_responses = 1000
        unique_reg_ids = sorted([uuid.uuid4() for _ in range(num_students)])
        reg_ids = np.random.choice(unique_reg_ids, size=num_responses)
        corrects = np.random.rand(num_responses) > 0.5
        is_held_out = np.random.rand(num_responses) < 0.2
        # make sure first reg_id appears only in train set
        is_held_out[reg_ids == unique_reg_ids[0]] = False
        # make sure last reg_id appears only in test set
        is_held_out[reg_ids == unique_reg_ids[-1]] = True
        # test per-student naive using a naive implementation
        per_student_num_correct = defaultdict(float)
        per_student_num_resp = defaultdict(int)
        for (reg_id, correct, held_out) in its.izip(reg_ids, corrects, is_held_out):
            if held_out:
                continue
            per_student_num_correct[reg_id] += float(correct)
            per_student_num_resp[reg_id] += 1

        preds = {reg_id: (num_correct / per_student_num_resp[reg_id]) >= 0.5
                 for reg_id, num_correct in per_student_num_correct.iteritems()}
        train_psn = np.mean([c == preds[r]
                             for r, c in its.izip(reg_ids[~is_held_out], corrects[~is_held_out])])
        test_psn = np.mean([c == preds[r] if r in preds else c
                            for r, c in its.izip(reg_ids[is_held_out], corrects[is_held_out])])
        actual_psn = undertest.Metrics.compute_per_student_naive(reg_ids, corrects, is_held_out)
        self.assertEqual((train_psn, test_psn), actual_psn)
