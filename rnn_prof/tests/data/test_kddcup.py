import itertools as its
import os
import unittest

from rnn_prof.data import kddcup as undertest


TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'test_kddcup_data.csv.gz')


class TestLoadKddData(unittest.TestCase):

    def test_load_data(self):
        max_inter_values = (None, int(1e6), 10, 2)
        drop_duplicates_values = (False, True)
        min_inter_values = (2, 3)

        for (max_inter, drop_duplicates, min_inter) in \
                its.product(max_inter_values, drop_duplicates_values, min_inter_values):

            if max_inter is not None and max_inter < min_inter:
                # The maximum number of interactions must be greater than the minimum
                continue

            output = undertest.load_data(TESTDATA_FILENAME,
                                         concept_id_col=undertest.KC_NAME_STARTS_WITH,
                                         template_id_col=undertest.PROBLEM_NAME,
                                         item_id_col=undertest.STEP_NAME,
                                         max_interactions_per_user=max_inter,
                                         min_interactions_per_user=min_inter,
                                         drop_duplicates=drop_duplicates)
            output_data = output[0]
            self.assertGreater(len(output_data), 0)
            self.assertEqual(set(output_data.columns),
                             {undertest.USER_IDX_KEY, undertest.ITEM_IDX_KEY,
                              undertest.CORRECT_KEY, undertest.TIME_IDX_KEY,
                              undertest.CONCEPT_IDX_KEY, undertest.TEMPLATE_IDX_KEY})

            max_interactions = max_inter or int(1e6)
            self.assertLessEqual(output_data.groupby(undertest.USER_IDX_KEY).size().max(),
                                 max_interactions)
            self.assertGreaterEqual(output_data.groupby(undertest.USER_IDX_KEY).size().min(),
                                    min_inter)

            if drop_duplicates:
                num_dupes = output_data.groupby([undertest.USER_IDX_KEY, undertest.TIME_IDX_KEY,
                                                 undertest.ITEM_IDX_KEY]).size().values
                self.assertEqual(set(num_dupes), {1})
