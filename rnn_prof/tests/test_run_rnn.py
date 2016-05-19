import os
import pickle
import shutil
import tempfile
import unittest

from rnn_prof import run_rnn as undertest
from rnn_prof.data.constants import ASSISTMENTS
from rnn_prof.data.splitting_utils import split_data
from rnn_prof.data.wrapper import load_data, DEFAULT_DATA_OPTS

ASSISTMENTS_TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'data',
                                             'test_assist_data.csv.gz')
TEST_NUM_FOLDS = 2
TEST_NUM_ITERS = 2
OUTPUT_PREFIX = 'output'

# Columns expected in the pickled output
EXPECTED_COLS = ['auc', 'fold_num', 'global', 'is_two_po', 'map']


class TestRunRNN(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.output_dir = tempfile.mkdtemp('output')
        cls.output_prefix = os.path.join(cls.output_dir, OUTPUT_PREFIX)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.output_dir)

    def test_run(self):
        """ Make sure RNN can run on assistments data and outputs results."""
        for data_file, data_source in [(ASSISTMENTS_TESTDATA_FILENAME, ASSISTMENTS)]:
            data, _, item_ids, _, _ = load_data(data_file, data_source)
            data_folds = split_data(data, num_folds=TEST_NUM_FOLDS)
            undertest.run(data_folds, TEST_NUM_FOLDS, len(item_ids), TEST_NUM_ITERS,
                          DEFAULT_DATA_OPTS, output=self.output_prefix)

            # Check that output was dumped for each fold
            for i in range(1, TEST_NUM_FOLDS + 1):
                with open(self.output_prefix + str(i), 'rb') as outfile:
                    output = pickle.load(outfile)
                    self.assertTrue(len(output))

    def test_run_with_output_compression(self):
        """ Make sure RNN can run on assistments data and outputs results."""
        for data_file, data_source in [(ASSISTMENTS_TESTDATA_FILENAME, ASSISTMENTS)]:
            data, _, item_ids, _, _ = load_data(data_file, data_source)
            data_folds = split_data(data, num_folds=TEST_NUM_FOLDS)
            undertest.run(data_folds, TEST_NUM_FOLDS, len(item_ids), TEST_NUM_ITERS,
                          DEFAULT_DATA_OPTS, output=self.output_prefix,
                          output_compress_dim=20)

            # Check that output was dumped for each fold
            for i in range(1, TEST_NUM_FOLDS + 1):
                with open(self.output_prefix + str(i), 'rb') as outfile:
                    output = pickle.load(outfile)
                    self.assertTrue(len(output))
