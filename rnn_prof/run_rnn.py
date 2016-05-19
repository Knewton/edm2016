"""
Script that constructs an RNN to predict student performance.
"""
from __future__ import division

import logging

import numpy as np

from .data.rnn import build_nn_data
from .simple_rnn import SimpleRnn, RnnOpts


_logger = logging.getLogger(__name__)


def run(data_folds, num_folds, num_questions, num_iters, data_opts, output=None, compress_dim=100,
        hidden_dim=200, test_spacing=10, recurrent=True, dropout_prob=0.0,
        output_compress_dim=None, first_learning_rate=30.0, decay_rate=0.99,
        which_fold=None):
    """ Train and test the neural net

    :param iterable data_folds: an iterator over tuples of (train, test) datasets
    :param int num_folds: number of folds total
    :param int num_questions: Total number of questions in the dataset
    :param int num_iters: Number of training iterations
    :param DataOpts data_opts: data pre-processing options. Contains the boolean `use_correct`,
        necessary for correct NN-data encoding, and all pre-processing parameters (for saving).
    :param str output: where to dump the current state of the RNN
    :param int|None compress_dim: The dimension to which to compress the data using the
        compressed sensing technique. If None, no compression is performed.
    :param int test_spacing: The number of iterations to run before running the tests
    :param int hidden_dim: The number of hidden units in the RNN.
    :param bool recurrent: Whether to use a recurrent architecture
    :param float dropout_prob: The probability of a node being dropped during training.
        Default is 0.0 (i.e., no dropout)
    :param int|None output_compress_dim: The dimension to which the output should be compressed.
        If None, no compression is performed.
    :param float first_learning_rate: The initial learning rate. Will be decayed at
        rate `decay_rate`
    :param float decay_rate: The rate of decay for the learning rate.
    :param int | None which_fold: Specify which of the folds you want to actually process. If None,
        process all folds. Good for naive parallelization.
    """
    if which_fold is not None and not (1 <= which_fold <= num_folds):
        raise ValueError("which_fold ({which_fold}) must be between 1 "
                         "and num_folds({num_folds})".format(which_fold=which_fold,
                                                             num_folds=num_folds))

    compress_dim = None if compress_dim <= 0 else compress_dim

    rnns = []
    results = []
    rnn_opts = RnnOpts(max_compress_dim=compress_dim, hidden_dim=hidden_dim, recurrent=recurrent,
                       num_iters=num_iters, dropout_prob=dropout_prob,
                       max_output_compress_dim=output_compress_dim,
                       first_learning_rate=first_learning_rate, decay_rate=decay_rate)
    np.random.seed(data_opts.seed)

    for fold_num, (train_data, test_data) in enumerate(data_folds):

        fold_num += 1
        if which_fold and fold_num != which_fold:
            continue

        _logger.info("Beginning fold %d", fold_num)
        _, _, _, _, rnn = eval_rnn(train_data, test_data, num_questions, data_opts,
                                   rnn_opts, test_spacing, fold_num)
        rnns.append(rnn)
        results.append(rnn.results[-1])
        if output:
            with open(output + str(fold_num), 'wb') as f:
                rnn.dump(f)

    _logger.info("Completed all %d folds", num_folds)

    # Print overall results
    acc_sum = 0
    auc_sum = 0
    for i, result in enumerate(results):
        _logger.info("Fold %d Acc: %.5f AUC: %.5f", i + 1, result.accuracy, result.auc)
        acc_sum += result.accuracy
        auc_sum += result.auc

    _logger.info("Overall %d Acc: %.5f AUC: %.5f", i + 1, acc_sum / num_folds, auc_sum / num_folds)


def eval_rnn(train_data, test_data, num_questions, data_opts, rnn_opts, test_spacing,
             fold_num):
    """ Create, train, and cross-validate an RNN on a train/test split.

    :param pd.DataFrame train_data: training data
    :param pd.DataFrame test_data: testing data for cross-validation (required)
    :param int num_questions: total number of questions in data
    :param DataOpts data_opts: data options
    :param RnnOpts rnn_opts: RNN options
    :param int test_spacing: test the RNN every this many iterations
    :param int fold_num: fold number (for logging and recording results only)
    :return: the trained RNN
    :rtype: SimpleRnn
    """
    _logger.info("Training RNN, fold %d", fold_num)
    train_nn_data = build_nn_data(train_data, num_questions,
                                  use_correct=data_opts.use_correct,
                                  use_hints=data_opts.use_hints)
    test_nn_data = build_nn_data(test_data, num_questions,
                                 use_correct=data_opts.use_correct,
                                 use_hints=data_opts.use_hints)
    rnn = SimpleRnn(train_nn_data, rnn_opts, test_data=test_nn_data, data_opts=data_opts)
    test_acc, test_auc, test_prob_correct, test_corrects = rnn.train_and_test(
        rnn_opts.num_iters, test_spacing=test_spacing)
    _logger.info("Fold %d: Num Interactions: %d; Test Accuracy: %.5f; Test AUC: %.5f",
                 fold_num, len(test_data), test_acc, test_auc)

    return test_acc, test_auc, test_prob_correct, test_corrects, rnn
