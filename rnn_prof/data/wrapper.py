from collections import namedtuple
import logging

import numpy as np

from . import assistments, kddcup
from .constants import USER_IDX_KEY, ASSISTMENTS, KDDCUP

LOGGER = logging.getLogger(__name__)

DataOpts = namedtuple('DataOpts', ['num_folds', 'item_id_col', 'template_id_col', 'concept_id_col',
                                   'remove_skill_nans', 'seed',
                                   'use_correct', 'use_hints', 'drop_duplicates',
                                   'max_interactions_per_user', 'min_interactions_per_user',
                                   'proportion_students_retained'])
DEFAULT_DATA_OPTS = DataOpts(num_folds=2, item_id_col=None, template_id_col=None,
                             concept_id_col=None,
                             remove_skill_nans=False, seed=0, use_correct=True, use_hints=False,
                             drop_duplicates=False,
                             max_interactions_per_user=None, min_interactions_per_user=2,
                             proportion_students_retained=1.0)


def load_data(interaction_file, data_source, data_opts=DEFAULT_DATA_OPTS):
    """ A wrapper for loading Assistments or KDD Cup data.

    :param str interaction_file: The location of the interactions
    :param str data_source: Should be either 'assistments' or 'kddcup'
    :param DataOpts data_opts: options for processing data. Includes fields:
        - `num_folds`: number of folds.  Default is 2.
        - `item_id_col`: Which column should be used for the item id? Should be an element of
        `.data.assistments.SKILL_ID_KEY` or `.data.assistments.PROBLEM_ID_KEY` for Assistments
        - `concept_id_col`: Which column should be used for the concept id? Should be an element of
        `.data.assistments.SKILL_ID_KEY` or `.data.assistments.PROBLEM_ID_KEY` for Assistments
        - `remove_skill_nans`: Remove items which have a NaN skill_id (only relevant for
        Assistments). Default is False.
        - `seed`: seed used for data splitting (in other functions)
        - `use_correct`: whether to use correctness (or just question identity) for training RNN.
        Default is True.
        - `use_hints`: whether to use ternary (hint-informed) data representation.  Used for RNN
        only.  Default is False.
        - `drop_duplicates`: whether to drop duplicate interactions.  Default is False.
        - `max_interactions_per_user`: How many interactions to retain per user
        - `min_interactions_per_user`: Minimum number of interactions required to retain a user
        - `proportion_students_retained`: Proportion of students to retain in the data set
            (for testing sensitivity to number of data points)
    :return: processed data, unique user ids, unique question ids, unique concept ids
    :rtype: (pd.DataFrame, list, list, list, list)
    """
    item_id_col = data_opts.item_id_col
    template_id_col = data_opts.template_id_col
    concept_id_col = data_opts.concept_id_col

    # Build initial data
    if data_source.lower() in (ASSISTMENTS, KDDCUP):
        if data_source.lower() == ASSISTMENTS:
            relevant_module = assistments
            default_item_col_id = assistments.SKILL_ID_KEY
        else:
            relevant_module = kddcup
            default_item_col_id = kddcup.PROBLEM_NAME

        if data_opts.template_id_col is None:
            item_id_col = item_id_col or default_item_col_id

        LOGGER.info("Using %s data with %s for item_id_col, %s for template_id_col, "
                    "and %s for concept_id_col",
                    relevant_module.__name__, item_id_col, template_id_col, concept_id_col)
        data, user_ids, item_ids, template_ids, concept_ids = relevant_module.load_data(
            interaction_file,
            item_id_col=item_id_col,
            template_id_col=template_id_col,
            concept_id_col=concept_id_col,
            remove_nan_skill_ids=data_opts.remove_skill_nans,
            drop_duplicates=data_opts.drop_duplicates,
            max_interactions_per_user=data_opts.max_interactions_per_user,
            min_interactions_per_user=data_opts.min_interactions_per_user)
    else:
        raise ValueError('Unknown data_source %s' % data_source)

    num_students = len(user_ids)
    num_rows = len(data)

    np.random.seed(data_opts.seed)
    chosen_user_ids = np.random.choice(
        num_students, size=int(data_opts.proportion_students_retained * num_students),
        replace=False)
    data = data[data[USER_IDX_KEY].isin(chosen_user_ids)]

    LOGGER.info(("After removing students, {now_rows:3,d}/{orig_rows:3,d} rows and "
                 "{now_students:3,d}/{orig_students:3,d} students remain").format(
                now_rows=len(data), orig_rows=num_rows,
                now_students=len(chosen_user_ids), orig_students=num_students))

    return data, user_ids, item_ids, template_ids, concept_ids
