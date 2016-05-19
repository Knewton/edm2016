"""
Helper utilities for taking a trained BayesNet model with a test responses node, and computing
predicted probability of correct in an online setting by training new learners on subsequences
of interactions.
"""
import logging

import numpy as np

from .callbacks import ConvergenceCallback
from .constants import TRAIN_RESPONSES_KEY, TEST_RESPONSES_KEY, THETAS_KEY, OFFSET_COEFFS_KEY
from .irt import BayesNetLearner, BayesNetGraph
from .node import Node
from .updaters import SolverPars

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
PRINT_FREQ = 100


def get_online_rps(learner, test_student_idx, test_student_time_idx=None,
                   learn_node_keys=(THETAS_KEY, ), compute_first_interaction_rps=False,
                   copy_learner=True, **test_learner_kwargs):
    """Compute the probability of correct of the `test response` node's interactions with an online
    training scheme (train learners on interactions up to but not including interaction i, and make
    a prediction of probability of correct for i.
    NOTE: We assume that for each student, interactions are sorted by time.

    :param BayesNetLearner learner: The learner with the original train and test response nodes.
    :param np.ndarray test_student_idx: Index indicating the student associated with each
        interaction in learner.nodes['test responses'].
    :param np.ndarray|None test_student_time_idx: unique event identifiers for each interaction
        in the test set. If None, it is assumed that each interaction occurs at a new time, and
        can be used to predict all following interactions.  If supplied, adjacent interactions
        with identical identifiers will both be in the test or the train set, and never split
        during online validation.
    :param tuple(str) learn_node_keys: The keys of learner nodes whose variables should be adapted
        at each iteration.
    :param bool compute_first_interaction_rps: Whether to compute the RP of a student's first
        interaction (thetas optimized under the prior only).  Default is no, in which case NaNs
        are returned for first interaction RPs.
    :param bool copy_learner: whether to operate on a copy of the learner, which avoids mutating
        the learner but incurs a memory cost of copying all the nodes' data.  Set to False if the
        learner is disposable, in which case the data of all nodes in ``learn_node_keys`` will be
        modified, and the theta node will be pruned in-place for efficient optimization.
    :param test_learner_kwargs: Optional keyword arguments that will be passed to the constructor
        of BayesNetLearner for the test learner.
    :return: Probabilities of correct (RPs) for each interaction (not including the first, which is
        set to np.nan) derived from a learner trained on the previous interactions.
    :rtype: np.ndarray
    """
    if not (test_learner_kwargs and 'callback' in test_learner_kwargs):
        test_learner_kwargs['callback'] = ConvergenceCallback()

    # get learner logger to set desired levels
    learner_logger = logging.getLogger('rnn_prof.irt')

    # get iteration count, an array indicating the online validation iteration associated with each
    # interaction in the set of test responses
    iteration_count = _idx_to_occurrence_ordinal(test_student_idx, test_student_time_idx)
    max_interactions = np.max(iteration_count) + 1

    # get corrects and parameter indices that will be sub-indexed during online validation
    correct = learner.nodes[TEST_RESPONSES_KEY].data
    theta_idx = learner.nodes[TEST_RESPONSES_KEY].cpd.index_map(THETAS_KEY)
    item_idx = learner.nodes[TEST_RESPONSES_KEY].cpd.index_map(OFFSET_COEFFS_KEY)
    cpd_class = learner.nodes[TEST_RESPONSES_KEY].cpd.__class__

    num_items = learner.nodes[OFFSET_COEFFS_KEY].cpd.dim
    # initialize arrays for storing all online validation prob corrects
    prob_correct = np.nan * np.empty_like(iteration_count, dtype=float)

    if copy_learner:
        # make copies of the nodes
        new_nodes = []
        for node in BayesNetGraph(learner.nodes).topological_sorting():
            # insert a copy
            LOGGER.debug("adding {}".format(node.name))
            new_nodes.append(node.copy())

            # replace child-parent references in the previously inserted nodes to point to this one
            for prev_node in new_nodes:
                for par_key, par_node in prev_node.param_nodes.iteritems():
                    if par_node is node:
                        prev_node.param_nodes[par_key] = new_nodes[-1]
                        LOGGER.debug("relinking %s's node's %s param",
                                     prev_node.name, par_key)

        test_learner = BayesNetLearner(new_nodes, **test_learner_kwargs)
    else:
        test_learner = learner

    # turn off learning for nodes that should be constant
    for node in test_learner.nodes.itervalues():
        if node.name not in learn_node_keys:
            LOGGER.debug("node {} parameters will not be learned".format(node.name))
            node.solver_pars.learn = False
            node.converged = True

    theta_node = test_learner.nodes[THETAS_KEY]
    # get the thetas that depend (directly or through the prior precision) on the interactions
    # in orig_test_node
    thetas_to_keep = theta_node.cpd.get_dependent_vars(np.unique(theta_idx))
    # trim theta node in place and remap theta_idx to the newly trimmed cpd
    theta_idx = theta_node.subset(thetas_to_keep, inplace=True)[theta_idx]
    num_thetas = theta_node.cpd.dim

    # quiet the online learner logger from INFO to WARNING (leave DEBUG alone)
    orig_log_level = learner_logger.getEffectiveLevel()
    if orig_log_level == logging.INFO:
        learner_logger.setLevel(logging.WARNING)

    for k in np.arange(0 if compute_first_interaction_rps else 1, max_interactions):
        test_idx = (iteration_count == k)
        train_idx = (iteration_count < k)
        # remove from train index students not in test_idx (whose interactions are all processed)
        train_idx &= np.in1d(test_student_idx, test_student_idx[test_idx])

        test_learner = BayesNetLearner(test_learner.nodes.values(), **test_learner_kwargs)
        # make new train/test nodes by splitting the original test node's correct into train/test
        for node_name, idx in ((TRAIN_RESPONSES_KEY, train_idx), (TEST_RESPONSES_KEY, test_idx)):
            if k == 0 and node_name == TRAIN_RESPONSES_KEY:
                # when on first interaction, make training node not empty (to avoid errors); it
                # will be labeled held-out below
                idx = test_idx
            param_nodes = test_learner.nodes[node_name].param_nodes
            test_learner.nodes[node_name] = Node(name=node_name,
                                                 data=correct[idx],
                                                 solver_pars=SolverPars(learn=False),
                                                 cpd=cpd_class(item_idx=item_idx[idx],
                                                               theta_idx=theta_idx[idx],
                                                               num_thetas=num_thetas,
                                                               num_items=num_items),
                                                 param_nodes=param_nodes,
                                                 held_out=((node_name == TEST_RESPONSES_KEY) or
                                                           k == 0))

        # run this iteration's learner and save the probability of correct for test responses
        test_learner.learn()
        iter_test_node = test_learner.nodes[TEST_RESPONSES_KEY]
        prob_correct[test_idx] = iter_test_node.cpd.compute_prob_true(**iter_test_node.param_data)

        if np.any(np.isnan(prob_correct[test_idx])):
            LOGGER.warn("NaN value in prob correct; iteration=%d" % k)
        if not k % PRINT_FREQ:
            num_train_interactions = np.sum(train_idx)
            num_test_interactions = np.sum(test_idx)
            msg = "Processed histories up to length %d (max=%d): %d train and %d test interactions."
            LOGGER.info(msg % (k, max_interactions, num_train_interactions, num_test_interactions))

    # reset the learner logger
    learner_logger.setLevel(orig_log_level)

    return prob_correct


def _idx_to_occurrence_ordinal(student_ids, time_ids=None):
    """
    Convert student index and unique time identifiers into the ordinal of the student's
    interaction.  The values of the time index are not important, only whether the value is
    equal to the previous value for the student. For example, the result of:
    idx_to_occurrence_ordinal(['s1', 's1', 's1', 's2', 's3', 's3', 's1'],
                              ['t2', 't3', 't3', 't1', 't1', 't1', 't2'])
    is                        [0,    1,    1,    0,    0,    1,    2]
    Note that the last interaction (s1, t2) has a repeat time identifier, but
    because it is different than the student's previous time identifier at interaction 3 (s1, t3),
    this event is considered non-repeat.

    :param np.ndarray student_ids: unique identifiers for each student
    :param None|np.ndarray time_ids: identifiers for the time of each interaction (NOTE:
        this method only tests equality between adjacent events for each student; i.e.,
        these identifiers are unique if sorted, but not necessarily unique if unsorted.)
    :return: interaction ordinal for each student
    :rtype: np.ndarray[int]
    """
    check_times = time_ids is not None
    student_idx = np.unique(student_ids, return_inverse=True)[1]
    counter = -np.ones(np.max(student_idx) + 1, dtype=int)
    idx_ordinal = np.zeros_like(student_idx)
    if check_times:
        prev_times = np.zeros_like(counter)
    for i, k in enumerate(student_idx):
        if check_times:
            this_time = time_ids[i]
            if counter[k] == -1:
                new_time = 1
            else:
                new_time = int(prev_times[k] != this_time)
            prev_times[k] = this_time
        else:
            new_time = 1
        counter[k] += new_time
        idx_ordinal[i] = counter[k]
    return idx_ordinal
