"""
Module containing simple prefab learners with 1PO and 2PO model structure.
"""
from abc import ABCMeta

import numpy as np

from .constants import (TRAIN_RESPONSES_KEY, TEST_RESPONSES_KEY, THETAS_KEY, NONOFFSET_COEFFS_KEY,
                        OFFSET_COEFFS_KEY)
from .cpd import GaussianCPD, OnePOCPD, TwoPOCPD
from .irt import BayesNetLearner
from .node import DefaultGaussianNode
from .linear_operators import IndexOperator
from .node import Node
from .updaters import SolverPars
from .utils import check_and_set_idx, set_or_check_min


HIGHER_OFFSET_KEY = 'higher'


class OnePOLearner(BayesNetLearner):
    """
    Baseclass for simple implementations of 1 parameter IRTs as a BayesNet
    """
    def __init__(self, correct, student_ids=None, item_ids=None, student_idx=None,
                 item_idx=None, is_held_out=None, num_students=None, num_items=None,
                 **bn_learner_kwargs):
        """
        :param np.ndarray[bool] correct: a 1D array of correctness values
        :param np.ndarray|None student_ids: student identifiers for each interaction; if no student
            indices provided, sort order of these ids determines theta indices.
        :param np.ndarray|None item_ids: item identifiers for each interaction; if no item indices
            are provided, sort order of these ids determines item indices.
        :param np.ndarray[int]|None student_idx: a 1D array mapping `correct` to student index
        :param np.ndarray[int]|None item_idx: a 1D array mapping `correct` to item index
        :param np.ndarray[bool] is_held_out: a 1D array indicating whether the interaction should be
            held out from training (if not all zeros, a held_out test node will be added to learner)
        :param int|None num_students: optional number of students. Default is one plus
            the maximum index.
        :param int|None num_items: optional number of items. Default is one plus
            the maximum index.
        :param bn_learner_kwargs: arguments to be passed on to the BayesNetLearner init
        """
        # convert pandas Series to np.ndarray and check argument dimensions
        correct = np.asarray_chkfinite(correct, dtype=bool)
        student_ids, student_idx = check_and_set_idx(student_ids, student_idx, 'student')
        item_ids, item_idx = check_and_set_idx(item_ids, item_idx, 'item')

        if len(correct) != len(student_idx) or len(correct) != len(item_idx):
            raise ValueError("number of elements in correct ({}), student_idx ({}), and item_idx"
                             "({}) must be the same".format(len(correct), len(student_idx),
                                                            len(item_idx)))
        if is_held_out is not None and (
                len(is_held_out) != len(correct) or is_held_out.dtype != bool):
            raise ValueError("held_out ({}) must be None or an array of bools the same length as "
                             "correct ({})".format(len(is_held_out), len(correct)))

        self.num_students = set_or_check_min(num_students, np.max(student_idx) + 1, 'num_students')
        self.num_items = set_or_check_min(num_items, np.max(item_idx) + 1, 'num_items')

        theta_node = DefaultGaussianNode(THETAS_KEY, self.num_students, ids=student_ids)
        offset_node = DefaultGaussianNode(OFFSET_COEFFS_KEY, self.num_items, ids=item_ids)
        nodes = [theta_node, offset_node]

        # add response nodes (train/test if there is held-out data; else just the train set)
        if is_held_out is not None and np.sum(is_held_out):
            if np.sum(is_held_out) == len(is_held_out):
                raise ValueError("some interactions must be not held out")
            is_held_out = np.asarray_chkfinite(is_held_out, dtype=bool)
            node_names = (TRAIN_RESPONSES_KEY, TEST_RESPONSES_KEY)
            response_idxs = (np.logical_not(is_held_out), is_held_out)
        else:
            node_names = (TRAIN_RESPONSES_KEY,)
            response_idxs = (np.ones_like(correct, dtype=bool),)
        for node_name, response_idx in zip(node_names, response_idxs):
            cpd = OnePOCPD(item_idx=item_idx[response_idx], theta_idx=student_idx[response_idx],
                           num_thetas=self.num_students, num_items=self.num_items)
            param_nodes = {THETAS_KEY: theta_node, OFFSET_COEFFS_KEY: offset_node}
            nodes.append(Node(name=node_name, data=correct[response_idx], cpd=cpd,
                              solver_pars=SolverPars(learn=False), param_nodes=param_nodes,
                              held_out=(node_name == TEST_RESPONSES_KEY)))

        # store leaf nodes for learning
        super(OnePOLearner, self).__init__(nodes=nodes, **bn_learner_kwargs)

    def params_per_response(self, par_keys=None):
        """  Return the parameters (offset coeffs, thetas) associated with each response.
        :param tuple[str] par_keys: which parameters to return.  Default is all the parameters of
            the train response node.
        :return: a dict of dicts, keyed on response node and params.  For example, the learner's
            test node's offset coeffs are in params[TEST_RESPONSES_KEY][OFFSET_COEFFS_KEY]
        :rtype: dict[str, dict[str, np.ndarray]]
        """
        par_keys = par_keys or self.nodes[TRAIN_RESPONSES_KEY].cpd.PARAM_KEYS
        params = {}
        for resp_key in (TRAIN_RESPONSES_KEY, TEST_RESPONSES_KEY):
            if resp_key in self.nodes:
                resp_node = self.nodes[resp_key]
                param_data = {k: node.data for k, node in resp_node.param_nodes.iteritems()
                              if k in par_keys}
                params[resp_key] = resp_node.cpd._all_to_data_space(**param_data)
        return params

    def get_difficulty(self, item_ids):
        """ Get the difficulties (in standard 1PO units) of an item or a set of items.
        :param item_ids: ids of the requested items
        :return: the difficulties (-offset_coeff)
        :rtype: np.ndarray
        """
        return -self.nodes[OFFSET_COEFFS_KEY].get_data_by_id(item_ids)

    def get_offset_coeff(self, item_ids):
        """ Get the offset coefficient of an item or a set of items.
        :param item_ids: ids of the requested items
        :return: the offset coefficient(s)
        :rtype: np.ndarray
        """
        return self.nodes[OFFSET_COEFFS_KEY].get_data_by_id(item_ids)

    def get_theta(self, student_ids):
        """ Get the theta for a student or a set of students.
        :param student_ids: ids of the requested students
        :return: thetas
        :rtype: np.ndarray
        """
        return self.nodes[THETAS_KEY].get_data_by_id(student_ids)


class TwoPOLearner(OnePOLearner):
    """
    Baseclass for simple implementations of 2 parameter IRT as a BayesNet
    """
    __metaclass__ = ABCMeta

    def upgrade_nodes_to_twopo(self):
        """ Updates (in-place) own response nodes to use the TwoParCPD, and adds a non-offset
        coefficient node.
        """
        # add the non-offset coefficients node
        initial_nonoffsets = np.ones((self.num_items, 1))
        nonoffset_node = Node(name=NONOFFSET_COEFFS_KEY, data=initial_nonoffsets,
                              cpd=GaussianCPD(mean=np.ones_like(initial_nonoffsets)),
                              ids=self.nodes[OFFSET_COEFFS_KEY].ids)
        self.nodes[NONOFFSET_COEFFS_KEY] = nonoffset_node

        # convert training and test response nodes to 2PO
        for node_name in (TRAIN_RESPONSES_KEY, TEST_RESPONSES_KEY):
            node = self.nodes.get(node_name)
            if node is not None:
                # make a new CPD, just to be safe, even though OnePOLearner's __init__ was called
                # with TwoPOCPD and probably did the right thing
                node.cpd = TwoPOCPD(item_idx=node.cpd.index_map(OFFSET_COEFFS_KEY),
                                    theta_idx=node.cpd.index_map(THETAS_KEY),
                                    num_thetas=self.num_students, num_items=self.num_items)
                node.param_nodes[NONOFFSET_COEFFS_KEY] = nonoffset_node

    def __init__(self, correct, student_ids=None, item_ids=None, student_idx=None,
                 item_idx=None, is_held_out=None, num_students=None, num_items=None,
                 **bn_learner_kwargs):
        """
        :param np.ndarray[bool] correct: a 1D array of correctness values
        :param np.ndarray|None student_ids: student identifiers for each interaction; if no student
            indices provided, sort order of these ids determines theta indices.
        :param np.ndarray|None item_ids: item identifiers for each interaction; if no item indices
            are provided, sort order of these ids determines item indices.
        :param np.ndarray[int]|None student_idx: a 1D array mapping `correct` to student index
        :param np.ndarray[int]|None item_idx: a 1D array mapping `correct` to item index
        :param np.ndarray[bool] is_held_out: a 1D array indicating whether the interaction should be
            held out from training (if not all zeros, a held_out test node will be added to learner)
        :param int|None num_students: optional number of students. Default is one plus
            the maximum index.
        :param int|None num_items: optional number of items. Default is one plus
            the maximum index.
        :param bn_learner_kwargs: arguments to be passed on to the BayesNetLearner init
        """
        super(TwoPOLearner, self).__init__(correct, student_ids, item_ids, student_idx,
                                           item_idx, is_held_out, num_students, num_items,
                                           **bn_learner_kwargs)
        self.upgrade_nodes_to_twopo()

    def get_difficulty(self, item_ids):
        """ Get the difficulties (in standard 2PO units) of an item or a set of items.
        :param item_ids: ids of the requested items
        :return: the difficulties (-offset_coeff / nonoffset_coeff)
        :rtype: np.ndarray
        """
        return -(self.nodes[OFFSET_COEFFS_KEY].get_data_by_id(item_ids) /
                 self.nodes[NONOFFSET_COEFFS_KEY].get_data_by_id(item_ids))

    def get_nonoffset_coeff(self, item_ids):
        """ Get the non-offset coefficient of an item or a set of items.
        :param item_ids: ids of the requested items
        :return: the non-offset coefficient(s)
        :rtype: np.ndarray
        """
        return self.nodes[NONOFFSET_COEFFS_KEY].get_data_by_id(item_ids)


class OnePOHighRT(OnePOLearner):
    """
    A hierarchical IRT model used to represent templated questions and their
    instantiations. That is, the model looks like::

        student theta
            |
            |
            +
        response
            +
            |
            |
        instantiation item parameters
            +
            |
            |
        template item parameters

    In this model, we are taking the templated item paramters to be standard normal,
    and the instantiation item parameters to be normally distributed around the
    template parameters with some variance.
    """
    def __init__(self, correct, student_idx, item_idx, higher_idx, is_held_out=None,
                 num_students=None, num_items=None, num_higher=None, higher_precision=1.0,
                 **bn_learner_kwargs):
        """
        :param np.ndarray[bool] correct: a 1D array of correctness values
        :param np.ndarray[int] student_idx: a 1D array mapping `correct` to student index
        :param np.ndarray[int] item_idx: a 1D array mapping `correct` to item index
        :param np.ndarray[int] higher_idx: a 1D array mapping offset coefficients to the mean
            hyperparameters
        :param np.ndarray[bool] is_held_out: a 1D array indicating whether the interaction should be
            held out from training (if not all zeros, a held_out test node will be added to learner)
        :param int|None num_students: optional number of students. Default is one plus
            the maximum index.
        :param int|None num_items: optional number of items. Default is one plus
            the maximum index.
        :param int|None num_higher: optional number of hyperparameters to which items can belong.
            Default is one plus the maximum index.
        :param float higher_precision: The precision of the normal distribution of the items around
            the hyperprior on the mean.
        :param bn_learner_kwargs: arguments to be passed on to the BayesNetLearner init
        """
        # First setup the basic OnePOLeaner
        super(OnePOHighRT, self).__init__(correct, student_idx=student_idx, item_idx=item_idx,
                                          is_held_out=is_held_out, num_students=num_students,
                                          num_items=num_items, **bn_learner_kwargs)

        # Now setup the hyperparameter
        mean_lin_op = IndexOperator(higher_idx)
        num_items = set_or_check_min(num_items, np.max(item_idx) + 1, 'num_items')
        num_higher = set_or_check_min(num_higher, np.max(higher_idx) + 1, 'num_higher')
        higher_node = Node(name=HIGHER_OFFSET_KEY, data=np.zeros((num_higher, 1)),
                           cpd=GaussianCPD(dim=num_higher, mean=0.0))
        higher_node.solver_pars.updater.step_size = 0.5
        self.nodes[higher_node.name] = higher_node

        # And hook it up to the offset node
        self.nodes[OFFSET_COEFFS_KEY].cpd = GaussianCPD(dim=num_items, precision=higher_precision,
                                                        mean_lin_op=mean_lin_op)
        self.nodes[OFFSET_COEFFS_KEY].param_nodes[self.nodes[OFFSET_COEFFS_KEY].cpd.MEAN_KEY] = \
            higher_node
