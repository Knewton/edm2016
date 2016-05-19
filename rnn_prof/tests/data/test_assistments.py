import itertools as its
import os
import unittest

import numpy as np
import pandas as pd

from rnn_prof.data import assistments as undertest

TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'test_assist_data.csv.gz')


class TestLoadAssistmentsData(unittest.TestCase):

    def test_load_data(self):
        """ Test that data loads without breaking """
        remove_nan_skill_ids_values = (True, False)
        item_id_col_values = (undertest.SKILL_ID_KEY, undertest.PROBLEM_ID_KEY)
        concept_id_col_values = (None, undertest.SKILL_ID_KEY, undertest.PROBLEM_ID_KEY,
                                 undertest.SINGLE)
        template_id_col_values = (None, undertest.TEMPLATE_ID_KEY, undertest.SINGLE)
        max_inter_values = (None, int(1e6), 10, 2)
        drop_duplicates_values = (False, True)
        min_inter_values = (2, 3)
        for (remove_nan_skill_ids, item_id_col, concept_id_col, template_id_col,
             max_inter, drop_duplicates, min_inter) in \
            its.product(remove_nan_skill_ids_values, item_id_col_values, concept_id_col_values,
                        template_id_col_values, max_inter_values, drop_duplicates_values,
                        min_inter_values):

            if max_inter is not None and max_inter < min_inter:
                # The maximum number of interactions must be greater than the minimum
                continue

            output = undertest.load_data(TESTDATA_FILENAME,
                                         item_id_col=item_id_col,
                                         template_id_col=template_id_col,
                                         concept_id_col=concept_id_col,
                                         remove_nan_skill_ids=remove_nan_skill_ids,
                                         max_interactions_per_user=max_inter,
                                         min_interactions_per_user=min_inter,
                                         drop_duplicates=drop_duplicates)
            output_data = output[0]
            expected_columns = {undertest.USER_IDX_KEY, undertest.ITEM_IDX_KEY,
                                undertest.CORRECT_KEY, undertest.TIME_IDX_KEY}
            if template_id_col is not None:
                expected_columns.add(undertest.TEMPLATE_IDX_KEY)
            if concept_id_col is not None:
                expected_columns.add(undertest.CONCEPT_IDX_KEY)
            self.assertEqual(set(output_data.columns), expected_columns)

            max_interactions = max_inter or int(1e6)
            self.assertLessEqual(output_data.groupby(undertest.USER_IDX_KEY).size().max(),
                                 max_interactions)
            self.assertGreaterEqual(output_data.groupby(undertest.USER_IDX_KEY).size().min(),
                                    min_inter)

            if drop_duplicates:
                num_dupes = output_data.groupby(
                    [undertest.USER_IDX_KEY, undertest.ITEM_IDX_KEY,
                     undertest.TIME_IDX_KEY]).size().values
                self.assertEqual(set(num_dupes), {1})

            # Test that user_ids, item_ids, concept_ids match up.
            data = pd.DataFrame.from_csv(TESTDATA_FILENAME)
            col_mapping = [(undertest.USER_ID_KEY, undertest.USER_IDX_KEY),
                           (item_id_col, undertest.ITEM_IDX_KEY),
                           (template_id_col, undertest.TEMPLATE_IDX_KEY),
                           (concept_id_col, undertest.CONCEPT_IDX_KEY)]
            for i, (key, val) in enumerate(col_mapping):
                if key == undertest.SINGLE:
                    self.assertEqual(output_data[val].nunique(), 1)
                elif key is not None:
                    self.assertGreaterEqual(set(np.unique(data[key])), set(output[i + 1]))
                    self.assertEqual(output_data[val].nunique(), len(set(output[i + 1])))
