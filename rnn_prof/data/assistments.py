"""
Functions for loading the Assistments data.  Originally from
https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010
"""
import logging

import numpy as np
import pandas as pd

from .constants import (ITEM_IDX_KEY, TEMPLATE_IDX_KEY, CONCEPT_IDX_KEY, USER_IDX_KEY,
                        TIME_IDX_KEY, CORRECT_KEY, SINGLE)

SKILL_ID_KEY = 'skill_id'
PROBLEM_ID_KEY = 'problem_id'
TEMPLATE_ID_KEY = 'template_id'
USER_ID_KEY = 'user_id'

LOGGER = logging.getLogger(__name__)


def load_data(file_path, item_id_col=SKILL_ID_KEY, template_id_col=None, concept_id_col=None,
              remove_nan_skill_ids=False, max_interactions_per_user=None,
              drop_duplicates=False, min_interactions_per_user=2):
    """ Load the Assistments dataset as a pandas dataframe, filter out students with only a single
    interaction, and optionally truncate student histories.  The columns used for item and concept
    identifiers can be specified in the input arguments.

    Note that multiple skill ids associated with an interaction will result in the first skill
    name lexicographically being retained.

    :param str file_path: path to the skill builder file
    :param str item_id_col: indicates column of csv file to use for item ids
    :param str template_id_col: Set a particular column to represent a template id for hierarchical
        IRT. If 'single', assumes a dummy single hierarchical level; if None, no column is retained
        for templates.
    :param str concept_id_col: indicates column of csv file to use for concept ids. If 'single',
        assumes a dummy single concept.  If None, concept column is not retained.
    :param bool remove_nan_skill_ids: whether to filter out interactions with NaN skill ids
    :param int max_interactions_per_user: number of interactions to keep per user (default is to
        keep all)
    :param int min_interactions_per_user: The minimum amount of history that is required to retain a
        student history.
    :param bool drop_duplicates: Whether to keep only the first of rows with duplicate order_id
        fields
    :return: processed data, student ids corresponding to the student indices, item ids
        corresponding to the item indices, template ids corresponding to the template indices, and
        concept ids corresponding to the concept indices
    :rtype: (pd.DataFrame, np.ndarray[int], np.ndarray[int], np.ndarray[int])
    """
    data = pd.DataFrame.from_csv(file_path)
    LOGGER.info("Read {:3,d} rows from file".format(len(data)))

    # Get the time index
    data[TIME_IDX_KEY] = data.index.values

    # fix up skill ids
    if data[SKILL_ID_KEY].dtype == 'object':
        # In this case, we have a string of skill ids like '1,54,3'
        # Keep only the first skill for now
        data[SKILL_ID_KEY] = data[SKILL_ID_KEY].apply(
            lambda x: sorted(map(int, x.split(',')))[0]).values
    nan_skill = data[SKILL_ID_KEY].apply(np.isnan)
    if remove_nan_skill_ids:
        data = data[~nan_skill]
        LOGGER.info("Removed {:3,d} rows with NaN skill_id".format(np.sum(nan_skill)))
    else:
        data.loc[nan_skill, SKILL_ID_KEY] = -1
        data[SKILL_ID_KEY] = data[SKILL_ID_KEY].astype(int)

    # sort by user, time, item, and concept id (if available)
    sort_keys = [USER_ID_KEY, TIME_IDX_KEY, item_id_col, SKILL_ID_KEY]
    data.sort(columns=sort_keys, inplace=True)

    if drop_duplicates:
        old_data_len = len(data)
        data = data.groupby(data.index).head(1)
        LOGGER.info("Removed {:3,d} duplicate rows ({:3,d} rows remaining)".format(
            old_data_len - len(data), len(data)))

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

    # to be safe, sort again
    data.sort(columns=sort_keys, inplace=True)

    # attach question index
    item_ids, data[ITEM_IDX_KEY] = np.unique(data[item_id_col], return_inverse=True)
    user_ids, data[USER_IDX_KEY] = np.unique(data[USER_ID_KEY], return_inverse=True)

    cols_to_keep = [USER_IDX_KEY, ITEM_IDX_KEY, CORRECT_KEY, TIME_IDX_KEY]
    if concept_id_col is None:
        LOGGER.info('concept_id_col not supplied, not using concepts')
        concept_ids = None
    else:
        if concept_id_col == SINGLE:
            LOGGER.info('Using dummy single concept.')
            data[concept_id_col] = '0'
        elif concept_id_col not in data:
            raise ValueError('concept_id_col %s not found in data columns %s' % (concept_id_col,
                                                                                 data.columns))
        concept_ids, data[CONCEPT_IDX_KEY] = np.unique(data[concept_id_col], return_inverse=True)
        cols_to_keep.append(CONCEPT_IDX_KEY)

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

    LOGGER.info("Processed data: {:3,d} interactions, {:3,d} students; {:3,d} items, "
                "{:3,d} templates, {:3,d} concepts"
                .format(len(data), len(user_ids), len(item_ids),
                        len(template_ids) if template_ids is not None else 0,
                        len(concept_ids) if concept_ids is not None else 0))

    return data[cols_to_keep], user_ids, item_ids, template_ids, concept_ids
