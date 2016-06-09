"""
Tests for online cross-validation.
"""
import unittest

import numpy as np

from rnn_prof.irt.online_cross_validation import get_online_rps, _idx_to_occurrence_ordinal
from rnn_prof.irt.constants import THETAS_KEY, TEST_RESPONSES_KEY, OFFSET_COEFFS_KEY
from rnn_prof.irt.cpd import OnePOCPD
from rnn_prof.irt.learners import OnePOLearner
from rnn_prof.irt.metrics import Metrics


class TestOnlinePrediction(unittest.TestCase):
    @staticmethod
    def set_up_learner(zero_difficulties, learn_diffs, theta_variance):
        """ Make student interactions and train a 1PO learner. Interactions are not sorted by
            student.
        :param bool zero_difficulties: whether item difficulties should be 0 (otherwise, drawn
            from standard Normal.
        :param bool learn_diffs: whether to learn item difficulties
        :param float theta_variance: theta prior variance
        :return: the trained learner and the student indices for the test interactions
        :rtype: BayesNetLearner, np.ndarray
        """
        held_out_frac = 0.2
        num_students = 50
        num_items = 10
        num_interactions = 100

        # make all interactions
        student_idx = np.random.randint(0, num_students, num_interactions)
        item_idx = np.random.randint(0, num_items, num_interactions)
        difficulties = np.zeros(num_items) if zero_difficulties else np.random.randn(num_items)
        thetas = np.linspace(-2, 2, num_students)
        prob_correct = OnePOCPD._prob_correct_from_irf_arg(thetas[student_idx] -
                                                           difficulties[item_idx])
        correct = np.random.rand(num_interactions) < prob_correct
        held_out_students = np.arange(num_students)[np.random.rand(num_students) < held_out_frac]
        held_out_idx = np.in1d(student_idx, held_out_students)
        held_out_student_idx = student_idx[held_out_idx]

        learner = OnePOLearner(correct, student_idx, item_idx, is_held_out=held_out_idx,
                               max_iterations=10)
        # make weak theta prior and no item difficulty learning
        learner.nodes[THETAS_KEY].cpd.precision /= theta_variance
        learner.nodes[OFFSET_COEFFS_KEY].solver_pars.learn = learn_diffs
        learner.nodes[THETAS_KEY].solver_pars.grad_tol = 1e-12
        # train learner on training data
        learner.learn()
        return learner, held_out_student_idx

    def test_prediction(self):
        """ Test that online prediction yields the same probabilities as running percent correct
        when item parameters are default and priors are weak. """
        learner, held_out_student_idx = self.set_up_learner(True, False, 1e9)

        # store all node's data and reference IDs for CPDs and param_node dicts
        orig_datas = {key: node.data for key, node in learner.nodes.iteritems()}
        orig_cpd_ids = {key: id(node.cpd) for key, node in learner.nodes.iteritems()}
        orig_param_node_ids = {key: id(node.param_nodes)
                               for key, node in learner.nodes.iteritems()}
        orig_fields = {}
        for field in ('callback', 'max_iterations', 'log_posterior', 'iter'):
            orig_fields[field] = getattr(learner, field)

        prob_correct = get_online_rps(learner, held_out_student_idx, max_iterations=1000)

        # get the test node with all the appended test responses
        test_correct = learner.nodes[TEST_RESPONSES_KEY].data

        valid_idx = np.isfinite(prob_correct)
        num_nan_rp = len(prob_correct) - np.sum(valid_idx)
        # check that number of NaN RPs equals total number of students
        self.assertEqual(num_nan_rp, len(np.unique(held_out_student_idx)))
        online_pc = Metrics.online_perc_correct(test_correct, held_out_student_idx)
        np.testing.assert_array_almost_equal(prob_correct[valid_idx], online_pc[valid_idx],
                                             decimal=6)

        # test that the original quantities are not modified
        for key in orig_datas:
            self.assertTrue(learner.nodes[key].data is orig_datas[key])
            np.testing.assert_array_equal(learner.nodes[key].data, orig_datas[key])
            self.assertTrue(id(learner.nodes[key].cpd) == orig_cpd_ids[key])
            self.assertTrue(id(learner.nodes[key].param_nodes) == orig_param_node_ids[key])
        for field, value in orig_fields.iteritems():
            self.assertTrue(getattr(learner, field) is value)

        # test that running online prediction again yields the same result; this time modify learner
        prob_correct_mod = get_online_rps(learner, held_out_student_idx, max_iterations=1000,
                                          copy_learner=False)
        np.testing.assert_equal(prob_correct_mod, prob_correct)
        # original responses should not have been modified, but thetas should have been
        for key in orig_datas:
            if key == THETAS_KEY:
                self.assertFalse(learner.nodes[key].data is orig_datas[key])
            else:
                self.assertTrue(learner.nodes[key].data is orig_datas[key])
                self.assertTrue(id(learner.nodes[key].cpd) == orig_cpd_ids[key])
                self.assertTrue(id(learner.nodes[key].param_nodes) == orig_param_node_ids[key])

    def test_first_interaction_rps(self):
        """ Test that the predicted RP for students' first interactions is equal to the item
        offset parameter passed through the IRF. """
        learner, held_out_student_idx = self.set_up_learner(False, True, 1.0)
        prob_correct = get_online_rps(learner, held_out_student_idx, max_iterations=100,
                                      compute_first_interaction_rps=True)
        # check that there are no NaNs in RPs
        self.assertTrue(np.all(np.isfinite(prob_correct)))

        # figure out which interactions are a student's first
        seen_student_idx = set()
        first_int_idx = np.zeros(len(held_out_student_idx), dtype=bool)
        for k, idx in enumerate(held_out_student_idx):
            if idx not in seen_student_idx:
                first_int_idx[k] = True
                seen_student_idx.add(idx)

        # test that RPs are offset coeffs passed through the IRF
        test_offsets = learner.params_per_response()[TEST_RESPONSES_KEY][OFFSET_COEFFS_KEY]
        np.testing.assert_array_almost_equal(prob_correct[first_int_idx],
            OnePOCPD._prob_correct_from_irf_arg(test_offsets[first_int_idx]).ravel())

    def test_online_pred_contemporaneous_events(self):
        """ Test that online prediction for events with the same item parameters occurring at
        the same labeled "time" yield the same probability of correct (i.e. thetas do not change
        from one event to another).
        """
        learner, held_out_student_idx = self.set_up_learner(False, False, 1.0)
        num_test_interactions = len(held_out_student_idx)
        unique_student_idx = set(held_out_student_idx)

        # test without providing times
        prob_correct = get_online_rps(learner, held_out_student_idx, max_iterations=1000,
                                      compute_first_interaction_rps=True)
        for student_idx in unique_student_idx:
            # expect all predicted RPs to be different
            student_prob_correct = prob_correct[held_out_student_idx == student_idx]
            self.assertTrue(np.all(np.diff(student_prob_correct)))

        # test with all unique times
        unique_times = np.arange(num_test_interactions)
        prob_correct = get_online_rps(learner, held_out_student_idx, max_iterations=1000,
                                      compute_first_interaction_rps=True,
                                      test_student_time_idx=unique_times)
        for student_idx in unique_student_idx:
            # expect all predicted RPs to be different
            student_prob_correct = prob_correct[held_out_student_idx == student_idx]
            self.assertTrue(np.all(np.diff(student_prob_correct)))

        # test with contemporaneous interactions. last two of each student's interactions share time
        contemp_times = np.arange(num_test_interactions)
        for student_idx in unique_student_idx:
            contemp_times[np.flatnonzero(held_out_student_idx == student_idx)[-2:]] = -1
        prob_correct = get_online_rps(learner, held_out_student_idx, max_iterations=1000,
                                      compute_first_interaction_rps=True,
                                      test_student_time_idx=contemp_times)
        for student_idx in unique_student_idx:
            # expect last two predicted RPs to be same
            student_prob_correct = prob_correct[held_out_student_idx == student_idx]
            if len(student_prob_correct) > 1:
                self.assertFalse(np.any(np.diff(student_prob_correct[-2:])))

    def test_idx_to_occurrence_ordinal(self):
        """
        Test helper method for setting the ordinal based on student and time ids.
        """
        students = np.array([1, 0, 1, 2, 3, 3, 0, 1, 1])
        expected_ordinals = [0, 0, 1, 0, 0, 1, 1, 2, 3]
        np.testing.assert_array_equal(expected_ordinals, _idx_to_occurrence_ordinal(students))

        # same with string identifiers
        students = np.array(['B', 'A', 'B', 'C', 'D', 'D', 'A', 'B', 'B'])
        np.testing.assert_array_equal(expected_ordinals, _idx_to_occurrence_ordinal(students))

        # all unique times
        times = np.array([4, 1, 3, 2, 6, 5, 0, 8, 7])
        np.testing.assert_array_equal(expected_ordinals,
                                      _idx_to_occurrence_ordinal(students, times))
        # non-unique times, but unique for each student
        times = np.array([4, 1, 1, 1, 1, 5, 0, 6, 3])
        np.testing.assert_array_equal(expected_ordinals,
                                      _idx_to_occurrence_ordinal(students, times))
        # bundle the middle 2 interactions for student 'B'
        times = np.array([4, 1, 1, 1, 1, 5, 0, 1, 3])
        expected_ordinals = [0, 0, 1, 0, 0, 1, 1, 1, 2]
        np.testing.assert_array_equal(expected_ordinals,
                                      _idx_to_occurrence_ordinal(students, times))
