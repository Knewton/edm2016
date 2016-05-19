"""
Script for running basic online IRT
"""
import logging

import numpy as np
import pandas as pd
from scipy import sparse as sp

from .data.constants import (ITEM_IDX_KEY, TEMPLATE_IDX_KEY, USER_IDX_KEY, CORRECT_KEY,
                             CONCEPT_IDX_KEY)
from .data.wrapper import DEFAULT_DATA_OPTS
from .irt import TEST_RESPONSES_KEY, OFFSET_COEFFS_KEY
from .irt.callbacks import ConvergenceCallback
from .irt.learners import OnePOLearner, TwoPOLearner, OnePOHighRT, HIGHER_OFFSET_KEY
from .irt.metrics import Metrics
from .irt.online_cross_validation import get_online_rps

LOGGER = logging.getLogger(__name__)


def get_metrics(correct, rps):
    """ Compute global PC, MAP Accuracy, AUC validation metrics.
    :param np.ndarray[bool] correct: correctnesses
    :param np.ndarray[float] rps: probability of correct
    :return: global percent correct, MAP accuracy, AUC
    :rtype: dict
    """
    correct_hats = rps >= 0.5
    global_acc = np.mean(np.array(correct, dtype=float))
    map_acc = np.mean(np.array(correct_hats == correct, dtype=float))
    auc = Metrics.auc_helper(correct, rps)
    return {'global': global_acc, 'map': map_acc, 'auc': auc}


def compute_theta_idx(train_df, test_df=None, single_concept=True):
    """
    Compute theta indices. If single_concept is True, then there is one theta
    per user, and if it is false, there is one theta per user/concept pair.

    Training and testing users are assumed disjoint and consecutive.

    :param pd.DataFrame train_df: The DataFrame of training data. Should have
        columns labeled `USER_IDX_KEY` and `CONCEPT_IDX_KEY`.
    :param pd.DataFrame|None test_df: The DataFrame of testing data. Should have
        columns labeled `USER_IDX_KEY` and `CONCEPT_IDX_KEY`. Can be None and if
        so is simply ignored.
    :param bool single_concept: Should there be one theta per user (True) or
        one theta per user/concept pair (False)
    :return: Theta indices whose order corresponds to the order of the passed
        data. Training comes before testing.
    :rtype: np.ndarray
    """
    if single_concept:
        if test_df is None:
            return train_df[USER_IDX_KEY].values
        else:
            return np.concatenate([train_df[USER_IDX_KEY].values, test_df[USER_IDX_KEY].values])
    else:
        num_users = train_df[USER_IDX_KEY].max() + 1
        if test_df is None:
            train_idx = train_df[USER_IDX_KEY].values + train_df[CONCEPT_IDX_KEY].values * num_users
            return train_idx

        num_users = max(num_users, test_df[USER_IDX_KEY].max() + 1)
        train_idx = train_df[USER_IDX_KEY].values + train_df[CONCEPT_IDX_KEY].values * num_users
        test_idx = test_df[USER_IDX_KEY].values + test_df[CONCEPT_IDX_KEY].values * num_users
        return np.concatenate([train_idx, test_idx])


def get_irt_learner(train_df, test_df=None, is_two_po=True,
                    single_concept=True, template_precision=None, item_precision=None):
    """ Make a 1PO or 2PO learner.

    :param pd.DataFrame train_df: Train data
    :param pd.DataFrame test_df: Optional test data
    :param bool is_two_po: Whether to make a 2PO learner
    :param bool single_concept: Should we train with a single theta per user (True)
        or a single theta per user per concept (False)
    :param float template_precision: The hierarchical IRT model has a model
        item_difficulty ~ N(template_difficulty, 1.0/item_precision) and
        template_difficulty ~ N(0, 1.0/template_precision). None just ignores
        templates.
    :param float|None item_precision: The precision of the Gaussian prior around items in a
        non-templated model. Or see `template_precision` for the templated case. If None, uses 1.0.
    :return: The learner
    :rtype: BayesNetLearner
    """
    correct = train_df[CORRECT_KEY].values.astype(bool)
    item_idx = train_df[ITEM_IDX_KEY].values
    is_held_out = np.zeros(len(train_df), dtype=bool)
    if test_df is not None:
        correct = np.concatenate((correct, test_df[CORRECT_KEY].values.astype(bool)))
        item_idx = np.concatenate((item_idx, test_df[ITEM_IDX_KEY].values))
        is_held_out = np.concatenate((is_held_out, np.ones(len(test_df), dtype=bool)))

    student_idx = compute_theta_idx(train_df, test_df=test_df, single_concept=single_concept)
    if not template_precision:
        learner_class = TwoPOLearner if is_two_po else OnePOLearner
        learner = learner_class(correct, student_idx=student_idx, item_idx=item_idx,
                                is_held_out=is_held_out, max_iterations=1000,
                                callback=ConvergenceCallback())
        for node in learner.nodes.itervalues():
            node.solver_pars.updater.step_size = 0.5
        if item_precision is not None:
            learner.nodes[OFFSET_COEFFS_KEY].cpd.precision = \
                item_precision * sp.eye(learner.nodes[OFFSET_COEFFS_KEY].data.size)
            LOGGER.info("Made a 1PO IRT learner with item precision %f", item_precision)
        else:
            LOGGER.info("Made a 1PO IRT learner with default item precision")
    else:
        template_idx = train_df[TEMPLATE_IDX_KEY]
        if test_df is not None:
            template_idx = np.concatenate((template_idx, test_df[TEMPLATE_IDX_KEY].values))
        problem_to_template = {item: template for item, template in zip(item_idx, template_idx)}
        problem_to_template = sorted(problem_to_template.items())
        template_idx = np.array([x for _, x in problem_to_template])
        learner = OnePOHighRT(correct, student_idx, item_idx, template_idx,
                              is_held_out=is_held_out, max_iterations=1000,
                              higher_precision=item_precision,
                              callback=ConvergenceCallback())
        if item_precision is not None:
            learner.nodes[HIGHER_OFFSET_KEY].cpd.precision = \
                template_precision * sp.eye(learner.nodes[HIGHER_OFFSET_KEY].data.size)
        for node in learner.nodes.itervalues():
            node.solver_pars.updater.step_size = 0.5
        LOGGER.info("Made a hierarchical IRT learner with item precision %f and template "
                    "precision %f", item_precision, template_precision)
    return learner


def irt(data_folds, num_folds, output=None, data_opts=DEFAULT_DATA_OPTS, is_two_po=True,
        single_concept=True, template_precision=None, which_fold=None,
        item_precision=None):
    """ Run 1PO/2PO IRT and print test-set metrics.

    :param iterable data_folds: An iterator over (train, test) data tuples
    :param int num_folds: number of folds
    :param str output: where to store the pickled output of the results
    :param DataOpts data_opts: data pre-processing parameters, to be saved (in the future) with IRT
        outputs. See `data.wrapper` for details and default values.
    :param bool is_two_po: Whether to use the 2PO IRT model
    :param bool single_concept: Should we train with a single concept per user (True)
        or a single concept per user per concept (False)
    :param float template_precision: the precision of the higher-order template variable
        specifying the mean of the item difficulties
    :param int | None which_fold: Specify which of the folds you want to actually process. If None,
        process all folds. Good for naive parallelization.
    :param float|None item_precision: The precision of the Gaussian prior around items in a
        non-templated model. If None, uses 1.0.
    """
    if which_fold is not None and not (1 <= which_fold <= num_folds):
        raise ValueError("which_fold ({which_fold}) must be between 1 "
                         "and num_folds({num_folds})".format(which_fold=which_fold,
                                                             num_folds=num_folds))

    np.random.seed(data_opts.seed)
    metrics = pd.DataFrame()
    for fold_num, (train_data, test_data) in enumerate(data_folds):
        fold_num += 1
        if which_fold and fold_num != which_fold:
            continue
        fold_metrics, _, _ = eval_learner(train_data, test_data, is_two_po, fold_num,
                                          single_concept=single_concept,
                                          template_precision=template_precision,
                                          item_precision=item_precision)
        metrics = metrics.append(pd.DataFrame(index=[len(metrics)], data=fold_metrics))

    if output:
        metrics.to_pickle(output)

    # Print overall results
    LOGGER.info("Overall Acc: %.5f AUC: %.5f", metrics['map'].mean(), metrics['auc'].mean())


def eval_learner(train_data, test_data, is_two_po, fold_num,
                 single_concept=True, template_precision=None, item_precision=None):
    """ Create, train, and cross-validate an IRT learner on a train/test split.

    :param pd.DataFrame train_data: training data
    :param pd.DataFrame test_data: testing data for cross-validation (required)
    :param bool is_two_po: Whether to use the 2PO IRT model
    :param int fold_num: fold number (for logging and recording results only)
    :param float template_precision: The hierarchical IRT model has a model
        item_difficulty ~ N(template_difficulty, 1.0/template_precision). None just ignores
        templates.
    :param bool single_concept: Should we train with a single concept per user (True)
        or a single concept per user per concept (False)
    :param float|None item_precision: The precision of the Gaussian prior around items in a
        non-templated model. If None, uses 1.0.
    :return: the validation metrics, predicted RP's, and boolean corrects on the test set
    :rtype: dict, np.ndarray[float], np.ndarray[bool]
    """
    LOGGER.info("Training %s model, fold %d, (single concept = %s)",
                '2PO' if is_two_po else '1PO', fold_num, single_concept)
    learner = get_irt_learner(train_data, test_data, is_two_po=is_two_po,
                              single_concept=single_concept,
                              template_precision=template_precision,
                              item_precision=item_precision)
    learner.learn()
    LOGGER.info("Performing online cross-validation")
    prob_correct = get_online_rps(learner, test_data[USER_IDX_KEY].values,
                                  compute_first_interaction_rps=True)

    test_correct = learner.nodes[TEST_RESPONSES_KEY].data
    metrics = get_metrics(test_correct, prob_correct)
    metrics['is_two_po'] = is_two_po
    metrics['fold_num'] = fold_num
    metrics['num_test_interactions'] = len(test_correct)
    LOGGER.info("Fold %d: Num Interactions: %d; Test Accuracy: %.5f; Test AUC: %.5f",
                fold_num, metrics['num_test_interactions'], metrics['map'], metrics['auc'])
    return metrics, prob_correct, test_correct
