from collections import namedtuple
import itertools as its

import numpy as np
from scipy import sparse as sp

from .constants import ITEM_IDX_KEY, USER_IDX_KEY, TIME_IDX_KEY, CORRECT_KEY, HINT_COUNT_KEY


# A namedtuple representing the data for a single user
UserData = namedtuple('UserData', ['length', 'history', 'next_answer', 'truth'])

# A namedtuple representing a batch of users' data
Batch = namedtuple('Batch', ['length', 'history', 'next_answer', 'truth', 'mask',
                             'num_interactions'])


def build_nn_data(data, num_questions, use_correct=True, use_hints=False):
    """
    Build data ready for RNN input.

    :param DataFrame data: User interactions for all users in DataFrame format as
        returned by loading functions in this package.
    :param int num_questions: number of questions in the full dataset
    :param bool use_correct: If True, records responses (before compression) as a
        2 * num_questions one-hot vector where one dimension corresponds to correct
        and one dimension corresponds to incorrect. If False, records responses
        (before compression) as a num_questions one-hot vector where each dimension
        corresponds to having *answered* a question, whether correctly or incorrectly.
    :param bool use_hints: If True, records responses ternarily: Correct, Wrong with
        No Hints, and Used a Hint.
    :return: list of all users data ready for RNN input.
    :rtype: list[UserData]
    """
    all_users_data = []
    data.sort([USER_IDX_KEY, TIME_IDX_KEY], inplace=True)

    # use_hints => use_correct
    use_correct = use_correct or use_hints

    for user_id, user in data.groupby(USER_IDX_KEY):

        x = []  # Input X denoting position for one hot
        y = []  # Mask Y to mask the probabilities all questions except the next one
        t = []  # The truth about the correctness of the next question

        xiter, yiter = its.tee(user[ITEM_IDX_KEY].values)
        next(yiter, None)
        this_correct_iter, next_correct_iter = its.tee(user[CORRECT_KEY].values)
        next(next_correct_iter, None)
        if use_hints:
            hints_iter = user[HINT_COUNT_KEY].values
        else:
            hints_iter = its.cycle([0])
        for this_skill, next_skill, this_correct, next_correct, hint in its.izip(
                xiter, yiter, this_correct_iter, next_correct_iter, hints_iter):
            # The first num_questions dimensions refer to incorrect responses, the
            # second num_questions dimensions to correct responses. *Unless*
            # use_correct is False, in which case, only num_questions dimensions
            # are used, one for answering (correctly or incorrectly) each question
            x.append(this_skill + num_questions * this_correct * (hint == 0) * use_correct +
                     2 * num_questions * (hint > 0) * use_hints)
            y.append(next_skill)
            t.append(next_correct)

        # Append it to a list
        all_users_data.append(UserData(length=len(x), history=x, next_answer=y, truth=t))

    return all_users_data


def _batch_dimension_list(user_data, threshold=0.9):
    """
    A helper function for ..py:function`build_batches` which returns a list of areas of
    the rectangles which will represent student history.

    :param list[UserData] user_data: The output of ..py:function`build_nn_data`. Must be
        sorted from largest to shortest history length *before* being passed.
    :return: list[(int, int)] batch_list: A list of rectangle dimensions for each batch
    """
    if len(user_data) <= 0:
        return []

    width = user_data[0].length  # Width of rectangle (user with max interactions within a batch)
    area_actual = 0              # Actual area within rectangle
    area_rect = 0                # Area bounded by rectangle
    height = 0                   # Height of rectangle (num users within a batch)
    dimension_list = []          # List of rectangle dimensions

    for i, user in enumerate(user_data):
        num_interactions = user.length

        # Calculate size of new area
        area_actual += num_interactions
        area_rect += width

        # Package the previous batch (not including the current one)
        # Note that we say height > 0 on the off chance that double rounding messes up
        #   when area_actual "==" area_rect
        if area_actual / area_rect < threshold and height > 0:
            dimension_list.append((width, height))
            width = num_interactions
            height = 0
            area_actual = width
            area_rect = width
        height += 1

    # Append the final batch
    dimension_list.append((width, height))
    return dimension_list


def build_batches(user_data, num_questions, basis, threshold=0.9, mask_type=np.int8,
                  num_type=np.float32, output_basis=None, compress_dim=None):
    """
    Each student response history has a different length, so we will have to pad
    to train in the neural network. However, if we pad too much, then our padding
    will engulf the training. So we will break the history into many batches, each
    of which have approximately the same length. That is, if the response history
    looks like::

              interactions -->
         u    >>>>>>>>>>>>>>>>>>>>>>>>
         s    >>>>>>>>>>>>>>>>>>>>
         e    >>>>>>>>>>>>>>>>
         r    >>>>>>>>>>
         |    >>>>>>>
         |    >>>
         v    >>>

    then we will break it up along the lines::

              interactions -->
         u    >>>>>>>>>>>>>>>>>>>>>>>>
         s    >>>>>>>>>>>>>>>>>>>>XXXX
        --------------------------------
         e    >>>>>>>>>>>>>>>>
        --------------------------------
         r    >>>>>>>>>>
         |    >>>>>>>XXX
        --------------------------------
         |    >>>
         v    >>>

    where the X's represent padding.

    :param list[UserData] user_data: User data with positions of one hot encoding. The
        output of ..py:function`build_nn_data`.
    :param int num_questions: The number of questions in the data set. Cannot be
        inferred because the actual input size of the basis may differ based on
        what data you are recording. See basis for more information.
    :param np.ndarray basis: Compressed sensing matrix, which should be of size
        (dimension of actual input, lower dimension)
        Note that the size of the actual input may vary, e.g., if you are recording
        whether or not a student answered a question correctly versus whether a student
        simply saw the question
    :param dtype mask_type: The type to be used for boolean values when running in Theano.
    :param dtype num_type: The type to be used for float values when running in Theano.
    :param np.ndarray|None output_basis: The compression basis for the output of the
        RNN. If None, no compression is assumed, i.e., implicitly it is the
        num_questions x num_questions identity matrix.
    :param int|None compress_dim: If basis is None, that is we are using the identity
        matrix, then you will need to specify this to indicate the dimension of
        the input.
    :return: Batches of user data with actual one hot encoding in numpy matrix
    :rtype: list[Batch]
    """

    # Sort user data by interaction in reverse order
    user_data.sort(key=lambda u: u.length, reverse=True)
    dimension_list = _batch_dimension_list(user_data, threshold=threshold)

    all_batches = []
    user_index = 0
    if (compress_dim is None) == (basis is None):
        if basis is not None:
            if basis.shape[1] != compress_dim:
                raise ValueError("If both basis and compress_dim are specified, then "
                                 "basis.shape[1] ({}) must match compress_dim ({})"
                                 .format(basis.shape[1], compress_dim))
    compress_dim = compress_dim or basis.shape[1]

    if basis is None:
        eye_basis = sp.eye(compress_dim).tocsc()
    if output_basis is None:
        eye_output_basis = sp.eye(num_questions).tocsc()

    for (width, height) in dimension_list:
        # Input X
        batch_x = np.zeros((width, height, compress_dim), dtype=num_type)

        # Mask Y on probabilities for next question if uncompressed, or
        # a vector to dot with if compressed.
        batch_y = np.zeros((width, height, num_questions), dtype=num_type)
        # Truth on correctness of next question
        batch_t = np.zeros((width, height), dtype=mask_type)
        # Mask on actual user interaction within a rectangle batch of users
        batch_m = np.zeros((width, height), dtype=mask_type)

        for j in xrange(height):
            user = user_data[user_index]

            # Compressed sensing on input X
            if basis is not None:
                x = basis[user.history, :]
            else:
                x = eye_basis[user.history, :].toarray()

            if output_basis is not None:
                y = output_basis[user.next_answer, :]
            else:
                y = eye_output_basis[user.next_answer, :].toarray()

            # Construct the numpy matrix for user
            for i in xrange(user.length):
                np.copyto(batch_x[i][j], x[i])
                np.copyto(batch_y[i][j], y[i])
                batch_t[i][j] = user.truth[i]
                batch_m[i][j] = 1

            # Next user
            user_index += 1

        all_batches.append(
            Batch(history=batch_x, next_answer=batch_y, truth=batch_t, mask=batch_m,
                  length=height, num_interactions=batch_m.sum()))
    return all_batches
