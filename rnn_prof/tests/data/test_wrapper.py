from __future__ import division

import os
import unittest

from rnn_prof.data import assistments
from rnn_prof.data import wrapper as undertest
from rnn_prof.data.constants import USER_IDX_KEY


TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'test_assist_data.csv.gz')


class TestWrapper(unittest.TestCase):

    def test_proportion_students_retained(self):
        data_opts = undertest.DEFAULT_DATA_OPTS

        raw_output = assistments.load_data(
            TESTDATA_FILENAME,
            template_id_col=data_opts.template_id_col,
            concept_id_col=data_opts.concept_id_col,
            remove_nan_skill_ids=data_opts.remove_skill_nans,
            max_interactions_per_user=data_opts.max_interactions_per_user,
            min_interactions_per_user=data_opts.min_interactions_per_user,
            drop_duplicates=data_opts.drop_duplicates)

        output = undertest.load_data(TESTDATA_FILENAME,
                                     'assistments',
                                     data_opts=data_opts)

        self.assertEqual(len(raw_output[0]), len(output[0]))

        test_proportion_students_retained = 2 / 3
        data_opts = undertest.DataOpts(
            num_folds=2, item_id_col=None, template_id_col=None,
            concept_id_col=None,
            remove_skill_nans=False, seed=0, use_correct=True, use_hints=False,
            drop_duplicates=False,
            max_interactions_per_user=None, min_interactions_per_user=2,
            proportion_students_retained=test_proportion_students_retained)

        output = undertest.load_data(TESTDATA_FILENAME,
                                     'assistments',
                                     data_opts=data_opts)

        total_users = raw_output[0][USER_IDX_KEY].nunique()
        retained_users = output[0][USER_IDX_KEY].nunique()
        self.assertAlmostEquals(retained_users / total_users,
                                test_proportion_students_retained,
                                1e-5)
