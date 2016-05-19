from collections import defaultdict
import numpy as np
import os
import pandas as pd
import pickle
import tempfile
import unittest

from rnn_prof import run_irt as undertest
from rnn_prof.data.constants import CONCEPT_IDX_KEY, USER_IDX_KEY, ASSISTMENTS
from rnn_prof.data.splitting_utils import split_data
from rnn_prof.data.wrapper import load_data

ASSISTMENTS_TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__),
                                             'data', 'test_assist_data.csv.gz')
TEST_NUM_FOLDS = 2
# Columns expected in the pickled output
EXPECTED_COLS = ['auc', 'fold_num', 'global', 'is_two_po', 'map']


class TestRunIrt(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pckl') as f:
            cls.filename = f.name

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.filename)

    def test_irt(self):
        """ Make sure IRT can run on Assistments data and outputs results."""
        for data_file, data_source in [(ASSISTMENTS_TESTDATA_FILENAME, ASSISTMENTS)]:
            data, _, _, _, _ = load_data(data_file, data_source)
            for is_two_po in [True, False]:
                data_folds = split_data(data, num_folds=TEST_NUM_FOLDS)
                undertest.irt(data_folds, TEST_NUM_FOLDS, output=self.filename, is_two_po=is_two_po)

                with open(self.filename, 'rb') as output:
                    output = pickle.load(output)
                    for col in EXPECTED_COLS:
                        self.assertTrue(col in output)
                    self.assertTrue(np.all(output['is_two_po'].values == is_two_po))

    def test_compute_theta_idx(self):
        train_data = pd.DataFrame({USER_IDX_KEY: [0, 0, 0, 1, 1, 2],
                                  CONCEPT_IDX_KEY: [0, 1, 1, 1, 1, 0]})
        test_data = pd.DataFrame({USER_IDX_KEY: [3, 3, 3, 4, 5],
                                  CONCEPT_IDX_KEY: [0, 1, 2, 2, 2]})

        # For single concept, there should be a single idx per user
        expected_single_concept = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 5])
        actual_single_concept = undertest.compute_theta_idx(train_data, test_df=test_data,
                                                            single_concept=True)
        assert np.all(expected_single_concept == actual_single_concept)

        actual_multi_concept = undertest.compute_theta_idx(train_data, test_df=test_data,
                                                           single_concept=False)

        # Rearrange the input and output for assertions
        output_dict = defaultdict(list)
        for (_, row), computed in zip(pd.concat([train_data, test_data]).iterrows(),
                                      actual_multi_concept):
            output_dict[(row[USER_IDX_KEY], row[CONCEPT_IDX_KEY])].append(computed)
        values = []
        for value in output_dict.itervalues():
            # There is only one value per student/concept pair
            assert len(set(value)) == 1
            values.append(value)

        # Every value is unique per student/concept pair
        assert len(values) == len(set(v for vv in values for v in vv))
