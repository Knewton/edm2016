"""
Tests for data splitting utilities
"""
import unittest
import uuid

import numpy as np
import pandas as pd

from rnn_prof.data import splitting_utils as undertest

USER_KEY = 'user'
QUESTION_KEY = 'question'


class TestSplittingUtils(unittest.TestCase):
    def test_split_data(self):
        num_folds = 5
        question_ids = [str(uuid.uuid4()) for _ in range(100)]
        user_ids = [str(uuid.uuid4()) for _ in range(20)]
        num_responses = 1000
        data = pd.DataFrame(data={undertest.USER_IDX_KEY: np.random.choice(user_ids, num_responses),
                                  USER_KEY: np.random.choice(user_ids, num_responses),
                                  QUESTION_KEY: np.random.choice(question_ids, num_responses)},
                            index=np.arange(num_responses))

        for train_data, test_data in undertest.split_data(data, num_folds=num_folds):
            # test that all students and all rows appear in train and test
            self.assertEqual(set.union(set(train_data[undertest.USER_IDX_KEY].values),
                                       set(test_data[undertest.USER_IDX_KEY].values)),
                             set(data[undertest.USER_IDX_KEY].values))
            self.assertEqual(set.union(set(train_data.index), set(test_data.index)),
                             set(data.index))
            # test that no students and no rows appear in both train and test
            self.assertEqual(set.intersection(set(train_data[undertest.USER_IDX_KEY].values),
                                              set(test_data[undertest.USER_IDX_KEY].values)),
                             set([]))
            self.assertEqual(set.intersection(set(train_data.index), set(test_data.index)),
                             set([]))

        # test number of folds
        self.assertEqual(len(list(undertest.split_data(data, num_folds=num_folds))), num_folds)

        # test that setting seed gives the same student partitions
        train1, test1 = undertest.split_data(data, num_folds=num_folds, seed=0).next()
        train2, test2 = undertest.split_data(data, num_folds=num_folds, seed=0).next()
        np.testing.assert_array_equal(train1[undertest.USER_IDX_KEY].values,
                                      train2[undertest.USER_IDX_KEY].values)
        np.testing.assert_array_equal(test1[undertest.USER_IDX_KEY].values,
                                      test2[undertest.USER_IDX_KEY].values)
