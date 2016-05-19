import itertools as its
import logging

import numpy as np
import pandas as pd

from .constants import (ITEM_IDX_KEY, TEMPLATE_IDX_KEY, CONCEPT_IDX_KEY, USER_IDX_KEY,
                        TIME_IDX_KEY, CORRECT_KEY, SINGLE)


LOGGER = logging.getLogger(__name__)

TIME_ID_KEY = 'First Transaction Time'
USER_ID_KEY = 'Anon Student Id'
ORIG_CORRECT_KEY = 'Correct First Attempt'
PROBLEM_NAME = 'Problem Name'
STEP_NAME = 'Step Name'
KC_NAME_STARTS_WITH = 'KC'
IS_TEST = 'is_test'


def load_data(file_path, item_id_col=PROBLEM_NAME, template_id_col=None, concept_id_col=None,
              remove_nan_skill_ids=False, max_interactions_per_user=None,
              drop_duplicates=False, min_interactions_per_user=2, test_file_path=None):
    """ Load data from KDD Cup data sets.

    :param str file_path: The location of the data
    :param str item_id_col: The column to be used for item_ids in interactions. Likely one of
        PROBLEM_NAME, STEP_NAME, or KC_NAME_STARTS_WITH
    :param str template_id_col: Set a particular column to represent a template id for hierarchical
        IRT. If 'single', assumes a dummy single hierarchical level; if None, no column is retained
        for templates.
    :param str|None concept_id_col: The column to be used for concept_ids in interactions.
        Likely KC_NAME_STARTS_WITH or 'single', in the latter case, all problems are given the same
        concept_id.  If None, no concept column is retained.
    :param bool remove_nan_skill_ids: Whether to remove interactions where the KC column is NaN
    :param int|None max_interactions_per_user: Retain only the first (in time order)
        `max_interactions_per_user` per user. If None, then there is no limit.
    :param bool drop_duplicates: Drop (seemingly) duplicate interactions
    :param int min_interactions_per_user: The minimum number of interactions required to retain
        a user
    :param str|None test_file_path: The KDD Cup data sets break themselves up into a (very large)
        training set and a (very small) test set. This allows you to combine the two files if
        specified. Will be specified in output with an IS_TEST column, which can be used if
        desired by downstream actors.
    :return: processed data, student ids corresponding to the student indices, item ids
        corresponding to the item indices, template ids corresponding to the template indices, and
        concept ids corresponding to the concept indices
    :rtype: (pd.DataFrame, np.ndarray[str], np.ndarray[str], np.ndarray[str])
    """

    data = pd.read_csv(file_path, delimiter='\t')

    LOGGER.info("Read {:3,d} rows from file".format(len(data)))

    if test_file_path:
        test_data = pd.read_csv(test_file_path, delimiter='\t')
        test_data[IS_TEST] = True
        data[IS_TEST] = False
        data = pd.concat([data, test_data])

    LOGGER.info("After test inclusion have {:3,d} rows".format(len(data)))

    data[TIME_IDX_KEY] = np.unique(data[TIME_ID_KEY], return_inverse=True)[1]
    data[CORRECT_KEY] = data[ORIG_CORRECT_KEY] == 1

    # Step names aren't universally unique. Prepend with the problem name to fix this problem.
    data[STEP_NAME] = [':'.join(x) for x in its.izip(data[PROBLEM_NAME], data[STEP_NAME])]

    kc_name = [column for column in data.columns if column.startswith(KC_NAME_STARTS_WITH)][0]
    if item_id_col and item_id_col.startswith(KC_NAME_STARTS_WITH):
        item_id_col = kc_name
    if template_id_col and template_id_col.startswith(KC_NAME_STARTS_WITH):
        template_id_col = kc_name
    if concept_id_col and concept_id_col.startswith(KC_NAME_STARTS_WITH):
        concept_id_col = kc_name
    if remove_nan_skill_ids:
        data = data[~data[kc_name].isnull()]
    else:
        data.ix[data[kc_name].isnull(), kc_name] = 'NaN'

    # Turn skills into single names. Take the first lexicographically if there's more than
    # one, though this can be modified. Only do for non nan skills.
    data[kc_name] = data[kc_name].apply(lambda x: sorted(x.split('~~'))[0])

    LOGGER.info("Total of {:3,d} rows remain after removing NaN skills".format(len(data)))

    # sort by user, time, item, and concept id (if available)
    sort_keys = [USER_ID_KEY, TIME_IDX_KEY, item_id_col]
    if concept_id_col:
        if concept_id_col == SINGLE:
            LOGGER.info('Using dummy single concept.')
            data[concept_id_col] = '0'
        elif concept_id_col not in data:
            raise ValueError('concept_id_col %s not found in data columns %s' % (concept_id_col,
                                                                                 data.columns))
        sort_keys.append(concept_id_col)

    data = data.sort(sort_keys)
    if drop_duplicates:
        data = data.drop_duplicates(sort_keys)

    # filter for students with >= min_history_length interactions;
    # must be done after removing nan skillz
    data = data.groupby(USER_ID_KEY).filter(lambda x: len(x) >= min_interactions_per_user)
    LOGGER.info("Removed students with <{} interactions ({:3,d} rows remaining)".format(
        min_interactions_per_user, len(data)))

    # limit to first `max_interactions_per_user`
    if max_interactions_per_user is not None:
        old_data_len = len(data)
        data = data.groupby([USER_ID_KEY]).head(max_interactions_per_user)
        LOGGER.info("Filtered for {} max interactions per student ({:3,d} rows removed)".format(
            max_interactions_per_user, old_data_len - len(data)))

    user_ids, data[USER_IDX_KEY] = np.unique(data[USER_ID_KEY], return_inverse=True)
    item_ids, data[ITEM_IDX_KEY] = np.unique(data[item_id_col], return_inverse=True)
    user_ids = user_ids.astype(str)
    item_ids = item_ids.astype(str)

    # TODO (yan): refactor the below to avoid code duplication across data sets
    cols_to_keep = [USER_IDX_KEY, ITEM_IDX_KEY, CORRECT_KEY, TIME_IDX_KEY]
    if template_id_col is None:
        LOGGER.info('template_id_col not supplied, not using templates')
        template_ids = None
    else:
        if template_id_col == SINGLE:
            LOGGER.info('Using dummy single template.')
            data[template_id_col] = '0'
        elif template_id_col not in data:
            raise ValueError('template_id_col %s not found', template_id_col)
        template_ids, data[TEMPLATE_IDX_KEY] = np.unique(data[template_id_col], return_inverse=True)
        cols_to_keep.append(TEMPLATE_IDX_KEY)

    if concept_id_col is None:
        LOGGER.info('concept_id_col not supplied, not using concepts')
        concept_ids = None
    else:
        concept_ids, data[CONCEPT_IDX_KEY] = np.unique(data[concept_id_col], return_inverse=True)
        cols_to_keep.append(CONCEPT_IDX_KEY)

    if test_file_path:
        cols_to_keep.append(IS_TEST)

    LOGGER.info("Processed data: {:3,d} interactions, {:3,d} students; {:3,d} items, "
                "{:3,d} templates, {:3,d} concepts"
                .format(len(data), len(user_ids), len(item_ids),
                        len(template_ids) if template_ids is not None else 0,
                        len(concept_ids) if concept_ids is not None else 0))

    return data[cols_to_keep], user_ids, item_ids, template_ids, concept_ids
