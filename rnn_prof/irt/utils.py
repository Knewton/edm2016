import math

import numpy as np


def set_or_check_min(x, min_x, var_name):
    """ If x is None, set its value to min_x; also check that it is at least min_x.
    :param int|float x: the value to check
    :param int|float min_x: minimum required value for x
    :param str var_name: name of the variable (for error logging)
    :return: set/checked value
    :rtype: int|float
    """
    if x is None:
        x = min_x
    elif x < min_x:
        raise ValueError("{} ({}) must be at least {}".format(var_name, x, min_x))
    return x


def check_and_set_idx(ids, idx, prefix):
    """ Reconciles passed-in IDs and indices and returns indices, as well as unique IDs
    in the order specified by the indices.  If only IDs supplied, returns the sort-arg
    as the index.  If only indices supplied, returns None for IDs.  If both supplied,
    checks that the correspondence is unique and returns unique IDs in the sort order of
    the associated index.
    :param np.ndarray ids: array of IDs
    :param np.ndarray[int] idx: array of indices
    :param str prefix: variable name (for error logging)
    :return: unique IDs and indices (passed in or derived from the IDs)
    :rtype: np.ndarray, np.ndarray
    """
    if ids is None and idx is None:
        raise ValueError('Both {}_ids and {}_idx cannot be None'.format(prefix, prefix))
    if ids is None:
        return None, np.asarray_chkfinite(idx)
    if idx is None:
        return np.unique(ids, return_inverse=True)
    else:
        ids = np.asarray(ids)
        idx = np.asarray_chkfinite(idx)
        if len(idx) != len(ids):
            raise ValueError('{}_ids ({}) and {}_idx ({}) must have the same length'.format(
                prefix, len(ids), prefix, len(idx)))
        uniq_idx, idx_sort_index = np.unique(idx, return_index=True)
        # make sure each unique index corresponds to a unique id
        if not all(len(set(ids[idx == i])) == 1 for i in uniq_idx):
            raise ValueError("Each index must correspond to a unique {}_id".format(prefix))
        return ids[idx_sort_index], idx


def check_positive(value, label):
    if math.isinf(value) or value <= 0:
        raise ValueError('{} ({}) must be a positive finite number'.format(value, label))


def check_nonnegative(value, label):
    if math.isinf(value) or value < 0:
        raise ValueError('{} ({}) must be a finite nonnegative number'.format(value, label))


def check_int(value, label):
    if value != int(value):
        raise TypeError('{} ({}) should be an int'.format(value, label))


def check_float(value, label):
    if value != float(value):
        raise TypeError('{} ({}) should be a float'.format(value, label))


def check_positive_float(value, label):
    """ Check that a value is a finite positive float. """
    check_positive(value, label)
    check_float(value, label)


def check_positive_int(value, label):
    """ Check that a value is a positive int. """
    check_positive(value, label)
    check_int(value, label)


def check_nonnegative_int(value, label):
    """ Check that a value is a nonnegative int. """
    check_nonnegative(value, label)
    check_int(value, label)


def check_nonnegative_float(value, label):
    """ Check that a value is a nonnegative float. """
    check_nonnegative(value, label)
    check_float(value, label)
