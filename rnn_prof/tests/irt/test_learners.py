"""
Tests for simple ogive and logistic learners.

@author Chaitu Ekanadham - chaitu@knewton.com

06/03/2015
"""
from __future__ import division
import itertools as its
import uuid

import numpy as np
import unittest

from rnn_prof.irt import learners as undertest
from rnn_prof.irt.callbacks import RecordingCallback, ITER_KEY, TRAIN_LOG_POST_KEY
from rnn_prof.irt.constants import (TRAIN_RESPONSES_KEY, TEST_RESPONSES_KEY, THETAS_KEY,
                                    OFFSET_COEFFS_KEY, NONOFFSET_COEFFS_KEY)
from rnn_prof.irt.cpd import OnePOCPD, TwoPOCPD
from rnn_prof.irt.testing_utils import EPSILON, generate_data

NUM_TRIALS = 3
# try combinations of a few (possibly) problematic scenarios
NUM_THETAS, NUM_ITEMS, NUM_RESPONSES, PROB_CORRECT = zip(*its.product((1, 10),
                                                                      (1, 50),
                                                                      (100,),
                                                                      (0.5,)))


class TestOgiveLearners(unittest.TestCase):
    @staticmethod
    def gen_data(trial):
        return generate_data(num_students=NUM_THETAS[trial],
                             num_items=NUM_ITEMS[trial],
                             num_responses=NUM_RESPONSES[trial],
                             prob_correct=PROB_CORRECT[trial])

    def learn_and_assert(self, learner):
        """ Run the learner and make sure log posteriors are finite and increasing.

        :param BayesNetLearner learner: learner to optimize parameters
        """
        learner.learn()
        iter_indices = np.isfinite(learner.callback.metrics[ITER_KEY])
        log_posteriors = learner.callback.metrics[TRAIN_LOG_POST_KEY][iter_indices]
        self.assertTrue(np.all(np.isfinite(log_posteriors)))
        np.testing.assert_array_less(-EPSILON, np.diff(log_posteriors))

    def test_irf(self):
        """
        Test that the correct IRF function is used in the learners.
        """
        data = self.gen_data(0)
        for (learner_class, cpd_class) in ((undertest.OnePOLearner, OnePOCPD),
                                           (undertest.TwoPOLearner, TwoPOCPD)):
            learner = learner_class(data.correct, student_idx=data.student_idx,
                                    item_idx=data.item_idx, num_students=NUM_THETAS[0],
                                    num_items=NUM_ITEMS[0], callback=RecordingCallback(),
                                    max_iterations=10)
            self.assertEqual(learner.nodes[TRAIN_RESPONSES_KEY].cpd.__class__, cpd_class)

    def test_irt_learning(self):
        """
        Test that log posteriors are finite and increasing for various trials for the 1PO/2PO model.
        """
        for trial in range(len(NUM_THETAS)) * NUM_TRIALS:
            data = self.gen_data(trial)
            for learner_class in (undertest.OnePOLearner, undertest.TwoPOLearner):
                learner = learner_class(data.correct, student_idx=data.student_idx,
                                        item_idx=data.item_idx, num_students=NUM_THETAS[trial],
                                        num_items=NUM_ITEMS[trial], callback=RecordingCallback(),
                                        max_iterations=10)
                self.learn_and_assert(learner)

                # test that params_per_response() returns the correct values
                pars = learner.params_per_response()
                resp_node = learner.nodes[TRAIN_RESPONSES_KEY]
                for key in (OFFSET_COEFFS_KEY, THETAS_KEY, NONOFFSET_COEFFS_KEY):
                    if key in learner.nodes:
                        expected_par = resp_node.cpd.lin_operators[key] * learner.nodes[key].data
                        np.testing.assert_array_equal(pars[TRAIN_RESPONSES_KEY][key],
                                                      expected_par)
                # test that posterior hessian computation does not break, Hessians are of right size
                self.assertEqual(learner.get_posterior_hessian(OFFSET_COEFFS_KEY).shape,
                                 (NUM_ITEMS[trial], 1))
                self.assertEqual(learner.get_posterior_hessian(THETAS_KEY).shape,
                                 (NUM_THETAS[trial], 1))

    def test_train_test_split(self):
        """ Test that the classes split train and test data properly.  """
        for trial in range(len(NUM_THETAS)) * NUM_TRIALS:
            if NUM_RESPONSES[trial] < 100:
                continue
            data = self.gen_data(trial)

            # test OnePO and TwoPO
            for learner_class in (undertest.OnePOLearner, undertest.TwoPOLearner):
                is_held_out = np.random.rand(NUM_RESPONSES[trial]) > 0.5
                train_idx = np.logical_not(is_held_out)
                learner = learner_class(data.correct, student_idx=data.student_idx,
                                        item_idx=data.item_idx, is_held_out=is_held_out,
                                        num_students=NUM_THETAS[trial], num_items=NUM_ITEMS[trial])
                # test that correctnesses are split up correctly
                np.testing.assert_array_equal(learner.nodes[TRAIN_RESPONSES_KEY].data,
                                              data.correct[train_idx])
                np.testing.assert_array_equal(learner.nodes[TEST_RESPONSES_KEY].data,
                                              data.correct[is_held_out])
                # test that each response node's cpd references the right thetas and items
                for (cpd, idx) in ((learner.nodes[TRAIN_RESPONSES_KEY].cpd, train_idx),
                                   (learner.nodes[TEST_RESPONSES_KEY].cpd, is_held_out)):
                    np.testing.assert_array_equal(cpd.index_map(THETAS_KEY),
                                                  data.student_idx[idx])
                    np.testing.assert_array_equal(cpd.index_map(OFFSET_COEFFS_KEY),
                                                  data.item_idx[idx])
                    if learner_class is undertest.TwoPOLearner:
                        np.testing.assert_array_equal(cpd.index_map(NONOFFSET_COEFFS_KEY),
                                                      data.item_idx[idx])
                # test that train and test node's held_out flag is set correctly
                self.assertFalse(learner.nodes[TRAIN_RESPONSES_KEY].held_out)
                self.assertTrue(learner.nodes[TEST_RESPONSES_KEY].held_out)

    def test_get_ids(self):
        for learner_class in (undertest.OnePOLearner, undertest.TwoPOLearner):
            # test the case with no IDs
            num_items = 50
            num_students = 20
            data = generate_data(num_items=num_items, num_students=num_students)
            learner = learner_class(data.correct, student_idx=data.student_idx,
                                    item_idx=data.item_idx)
            for bogus_id in ('bogus_id', data.item_idx[0], 0):
                with self.assertRaises(ValueError):
                    _ = learner.get_difficulty(bogus_id)

            uniq_item_ids = np.array([str(uuid.uuid4()) for _ in range(num_items)])
            item_ids = uniq_item_ids[data.item_idx]
            uniq_student_ids = np.array([str(uuid.uuid4()) for _ in range(num_students)])
            student_ids = uniq_student_ids[data.student_idx]

            # test that non-unique IDs for a single item raises an error
            bad_item_ids = [x for x in item_ids]
            # find an item that occurs more than once
            recurrent_item_idx, item_count = np.unique(data.item_idx, return_counts=True)
            recurrent_item_idx = recurrent_item_idx[(item_count > 1)][0]
            # set the first two of its occurrences to have different IDs
            occ_idx = np.flatnonzero(np.array(data.item_idx) == recurrent_item_idx)
            bad_item_ids[occ_idx[0]] = uniq_item_ids[0]
            bad_item_ids[occ_idx[1]] = uniq_item_ids[1]
            with self.assertRaises(ValueError):
                _ = learner_class(data.correct, student_idx=data.student_idx,
                                  item_idx=data.item_idx, item_ids=bad_item_ids)

            # test that mismatched ID/index lengths raise an error
            bad_item_ids = item_ids[:-1]
            with self.assertRaises(ValueError):
                _ = learner_class(data.correct, student_idx=data.student_idx,
                                  item_idx=data.item_idx, item_ids=bad_item_ids)

            learner = learner_class(data.correct, student_idx=data.student_idx,
                                    student_ids=student_ids, item_idx=data.item_idx,
                                    item_ids=item_ids,  max_iterations=10)
            learner.learn()
            for _ in range(NUM_TRIALS):
                query_len = np.random.randint(1, 3)

                # test item parameters
                query_item_id = np.random.choice(uniq_item_ids, size=query_len, replace=False)
                # unique item IDs in sort order of item_idx
                sorted_ids = item_ids[np.unique(data.item_idx, return_index=True)[1]]
                # find interactions that match query_item_ids
                idx = [k for x in query_item_id for k, sid in enumerate(sorted_ids) if x == sid]

                actual_diffs = learner.get_difficulty(query_item_id)
                if learner_class is undertest.OnePOLearner:
                    expected_diffs = -learner.nodes[OFFSET_COEFFS_KEY].data[idx]
                else:
                    expected_diffs = -(learner.nodes[OFFSET_COEFFS_KEY].data[idx] /
                                       learner.nodes[NONOFFSET_COEFFS_KEY].data[idx])
                np.testing.assert_array_equal(actual_diffs, expected_diffs)

                actual_offsets = learner.get_offset_coeff(query_item_id)
                expected_offsets = learner.nodes[OFFSET_COEFFS_KEY].data[idx]
                np.testing.assert_array_equal(actual_offsets, expected_offsets)

                if learner_class is undertest.TwoPOLearner:
                    actual_nonoffsets = learner.get_nonoffset_coeff(query_item_id)
                    expected_nonoffsets = learner.nodes[NONOFFSET_COEFFS_KEY].data[idx]
                    np.testing.assert_array_equal(actual_nonoffsets, expected_nonoffsets)

                # test student parameters
                query_student_id = np.random.choice(uniq_student_ids, size=query_len, replace=False)
                # unique student IDs in sort order of student_idx
                sorted_ids = student_ids[np.unique(data.student_idx, return_index=True)[1]]
                # find interactions that match query_student_ids
                idx = [k for x in query_student_id for k, sid in enumerate(sorted_ids) if x == sid]

                actual_thetas = learner.get_theta(query_student_id)
                expected_thetas = learner.nodes[THETAS_KEY].data[idx]
                np.testing.assert_array_equal(actual_thetas, expected_thetas)
