import unittest

import numpy as np
import pandas as pd

from rnn_prof.data import rnn
from rnn_prof.data.constants import ITEM_IDX_KEY, USER_IDX_KEY, TIME_IDX_KEY, CORRECT_KEY


class TestRnnDataTransforms(unittest.TestCase):
    common_data = [rnn.UserData(length=3, history=[0, 1, 2], next_answer=[1, 2, 3],
                                truth=[1, 1, 0]),
                   rnn.UserData(length=2, history=[3, 1], next_answer=[1, 2], truth=[0, 0]),
                   rnn.UserData(length=1, history=[4], next_answer=[5], truth=[1])]

    def test_build_nn_data(self):
        # Build fake data
        data = pd.DataFrame({
            USER_IDX_KEY: [0, 0, 0, 0, 1, 1, 1, 2, 2],
            ITEM_IDX_KEY: [0, 1, 2, 3, 2, 1, 3, 4, 5],
            TIME_IDX_KEY: [1, 2, 3, 4, 6, 5, 4, 4, 5],
            CORRECT_KEY:  [0, 1, 1, 0, 0, 0, 0, 1, 1]
        })
        num_questions = data[ITEM_IDX_KEY].max() + 1

        # Run it through the builder
        transformed = rnn.build_nn_data(data, num_questions)

        # What data we expect
        expected = [
            rnn.UserData(
                length=3,
                history=[0, 1 + num_questions, 2 + num_questions],
                next_answer=[1, 2, 3],
                truth=[1, 1, 0]
            ),
            rnn.UserData(
                length=2,
                history=[3, 1],
                next_answer=[1, 2],
                truth=[0, 0],
            ),
            rnn.UserData(
                length=1,
                history=[4 + num_questions],
                next_answer=[5],
                truth=[1]
            )
        ]

        # Assert!
        transformed.sort(key=lambda x: -x.length)
        for expected_entry, actual_entry in zip(expected, transformed):
            assert expected_entry == actual_entry

        # What if use_correct is False? Everything should be the same *except* history
        expected = [rnn.UserData(length=datum.length,
                                 history=[q % num_questions for q in datum.history],
                                 next_answer=datum.next_answer,
                                 truth=datum.truth) for datum in expected]
        transformed = rnn.build_nn_data(data, num_questions, use_correct=False)
        transformed.sort(key=lambda x: -x.length)
        for expected_entry, actual_entry in zip(expected, transformed):
            assert expected_entry == actual_entry

    def test__batch_dimension_list(self):

        # If the threshold is high, creates a different batch per group
        high_threshold_output = rnn._batch_dimension_list(self.common_data, threshold=1.1)
        assert high_threshold_output == [(3, 1), (2, 1), (1, 1)]

        # If the threshold is low, creates one batch
        low_threshold_output = rnn._batch_dimension_list(self.common_data, threshold=-0.1)
        assert low_threshold_output == [(3, 3)]

    def test_build_batches(self):
        num_questions = 6

        # If we use the stacked identity basis then we get back the questions
        basis = np.vstack([np.eye(num_questions), np.eye(num_questions)])
        output = rnn.build_batches(self.common_data, num_questions, basis, threshold=-0.1)

        assert len(output) == 1, "You should only have one batch with a negative threshold"

        output = output[0]
        # Test that the x's are correct. Note that our choice of basis means that the one-hot
        # position should be the number of the question
        for user_idx, datum in enumerate(self.common_data):
            for history_idx, actual_question_idx in enumerate(datum.history):
                for question_idx in range(num_questions):
                    if question_idx == actual_question_idx:
                        assert (output.history[history_idx,
                                               user_idx,
                                               question_idx % num_questions] == 1)
                    else:
                        assert (output.history[history_idx,
                                               user_idx,
                                               question_idx % num_questions] == 0)

        # Test that the y's are correct
        for user_idx, datum in enumerate(self.common_data):
            for history_idx, actual_question_idx in enumerate(datum.next_answer):
                for question_idx in range(num_questions):
                    if question_idx == actual_question_idx:
                        assert output.next_answer[history_idx, user_idx, question_idx] == 1
                    else:
                        assert output.next_answer[history_idx, user_idx, question_idx] == 0

        # Now for the truths t
        for user_idx, datum in enumerate(self.common_data):
            for history_idx, correct in enumerate(datum.truth):
                assert output.truth[history_idx, user_idx] == correct

        # Finally, make sure that we are masking the appropriate part of the history
        for user_idx, datum in enumerate(self.common_data):
            data_len = datum.length
            for history_idx in range(output.mask.shape[0]):
                if history_idx < data_len:
                    assert output.mask[history_idx, user_idx] == 1
                else:
                    assert output.mask[history_idx, user_idx] == 0
