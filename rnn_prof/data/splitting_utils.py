"""
Utilities for general data pre-processing.
"""
import logging
import numpy as np

from .constants import USER_IDX_KEY

_logger = logging.getLogger(__name__)


def _get_fold_student_idx(student_ids, num_folds, seed=0):
    """ Split up unique student IDs into different folds.

    :param np.ndarray student_ids: set of unique student ids or indices
    :param int num_folds: number of folds to split that data into
    :param int seed: seed for the splitting
    :return: student ids per fold
    :rtype: list
    """
    num_students = len(student_ids)
    fold_idx = np.arange(num_students)
    # randomize the order of all students to be split across folds
    np.random.seed(seed)
    np.random.shuffle(fold_idx)
    fold_size = num_students // num_folds

    return [student_ids[fold_idx[i * fold_size:min(num_students, (i + 1) * fold_size)]]
            for i in range(num_folds)]


def split_data(data, num_folds, seed=0):
    """ Split all interactions into K-fold sets of training and test dataframes.  Splitting is done
    by assigning student ids to the training or test sets.

    :param pd.DataFrame data: all interactions
    :param int num_folds: number of folds
    :param int seed: seed for the splitting
    :return: a generator over (train dataframe, test dataframe) tuples
    :rtype: generator[(pd.DataFrame, pd.DataFrame)]
    """
    # break up students into folds
    fold_student_idx = _get_fold_student_idx(np.unique(data[USER_IDX_KEY]), num_folds=num_folds,
                                             seed=seed)

    for fold_test_student_idx in fold_student_idx:
        test_idx = np.in1d(data[USER_IDX_KEY], fold_test_student_idx)
        train_idx = np.logical_not(test_idx)
        yield (data[train_idx].copy(), data[test_idx].copy())
