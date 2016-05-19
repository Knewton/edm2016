"""
Class for computing various metrics on a data set with a BayesNet Node object.
"""
import logging

import numpy as np

from .cpd.ogive import OgiveCPD

EPSILON = 1e-16

MAP_ACCURACY_KEY = 'map_accuracy'
AUC_KEY = 'auc'
LOGLI_KEY = 'logli'
D_PRIME_KEY = 'd_prime'
NAIVE_KEY = 'naive'
METRICS_KEYS = {NAIVE_KEY, LOGLI_KEY, MAP_ACCURACY_KEY, AUC_KEY, D_PRIME_KEY}

LOGGER = logging.getLogger(__name__)


class Metrics(object):
    """ Class for computing various performance metrics based on the data and the parameters in a
     Node object. """

    def __init__(self, node):
        """ Initialize this object with a reference to a BayesNet Node object that holds the
        CPD, the data, and the parameters."""
        self.node = node

    def _check_binary_cpd(self, func_name):
        if not isinstance(self.node.cpd, OgiveCPD):
            raise TypeError("{} only defined for OgiveCPDs not %s".format(
                func_name, self.node.cpd.__class__))

    @classmethod
    def _check_finite(cls, prob_true, *args):
        """ Check that all probabilities are finite; if not, remove those elements and corresponding
            elements from other positional args.
        :param np.ndarray prob_true: array to check for finiteness
        :param args: optional arguments to subselect based on isfinite(prob_true)
        :return: np.ndarray|tuple[np.ndarray]
        """
        if not np.all(np.isfinite(prob_true)):
            valid_idx = np.isfinite(prob_true)
            LOGGER.warn("%d non-finite prob corrects found; ignoring these interactions",
                        np.sum(~valid_idx))
            prob_true = prob_true[valid_idx]
            args = tuple([arg[valid_idx] for arg in args])
        if not len(args):
            return prob_true
        else:
            return (prob_true,) + args

    @staticmethod
    def compute_per_student_naive(reg_ids, corrects, is_held_out):
        """ Compute the per-student naive metrics on the training and test sets, based on predicting
        correct if the student had more corrects in the training set.  If no data in the training
        set exist for the student, predict correct.

        :param np.ndarray reg_ids: unique student identifier for each interaction
        :param np.ndarray[bool] corrects: correctness values for each interaction
        :param np.ndarray[bool] is_held_out: indicator whether an interaction is in the test set
        :return: per student naive on the training and test sets
        :rtype: float, float
        """
        if len(corrects) != len(reg_ids) or len(is_held_out) != len(reg_ids):
            raise ValueError("reg_ids (%d), corrects (%d), is_held_out (%d) must have same length",
                             len(reg_ids), len(corrects), len(is_held_out))
        uniq_regids, reg_idxs = np.unique(reg_ids, return_inverse=True)
        num_reg_ids = len(uniq_regids)
        train_reg_idxs = reg_idxs[~is_held_out]
        test_reg_idxs = reg_idxs[is_held_out]
        train_corrects = corrects[~is_held_out]
        test_corrects = corrects[is_held_out]
        per_student_num_correct = np.bincount(train_reg_idxs, weights=train_corrects,
                                              minlength=num_reg_ids)
        per_student_num_responses = np.bincount(train_reg_idxs, minlength=num_reg_ids)
        pred_correct = (2 * per_student_num_correct >= per_student_num_responses)
        train_per_student_naive = np.mean(pred_correct[train_reg_idxs] == train_corrects)
        test_per_student_naive = np.mean(pred_correct[test_reg_idxs] == test_corrects)
        return train_per_student_naive, test_per_student_naive

    def compute_metric(self, metric_key, *args, **kwargs):
        """ Compute metric specified by the supplied key.
        :param str metric_key: key specifying the metric
        :return: the value of the metric
        :rtype: float
        """
        return getattr(self, 'compute_' + metric_key)(*args, **kwargs)

    def compute_naive(self):
        """ Compute the accuracy of predicting always correct or always incorrect,
        whichever is higher. Defined for binary CPDs only.

        :return: a number between 0 and 1 specifying prediction accuracy
        :rtype: float
        """
        self._check_binary_cpd("Naive metric")
        fraction_correct = np.mean(np.array(self.node.data, dtype=float))
        return max(fraction_correct, 1. - fraction_correct)

    def compute_logli(self, avg=False):
        """ Compute the response log-likelihood (the value of the node's CPD given the stored data
        and parameters.

        :param bool avg: whether to normalize the log-likelihood by the size of the node's data
        :return: the sum of the log-likelihoods over the data points.
        :rtype: float
        """
        log_li = self.node.compute_log_prob()
        if avg:
            log_li /= self.node.data.size
        return log_li

    def compute_map_accuracy(self):
        """ Compute the MAP accuracy (fraction of data points predicted correctly at the maximum
        of the binary probability distribution).  Defined for binary CPDs only.

        :return: MAP accuracy
        :rtype: float
        """
        self._check_binary_cpd("MAP accuracy")
        prob_true = self.node.cpd.compute_prob_true(**self.node.param_data)
        prob_true, data = self._check_finite(prob_true, self.node.data)
        return np.mean((prob_true > 0.5) == data)

    def compute_d_prime(self):
        """ Compute the d-prime statistic measuring separation between response probabilities
        conditioned on a true (positive) and false (negative) data points.
        Defined for binary CPDs only.

        :return: the d-prime statistic of distribution separation
        :rtype: float
        """
        self._check_binary_cpd("D prime")
        prob_true = self.node.cpd.compute_prob_true(**self.node.param_data)
        return self.d_prime_helper(self.node.data, prob_true)

    def compute_auc(self):
        """ Compute the area under curve (AUC) for the task of predicting binary labels
        based on the probabilities computed by some model.  The curve is the Receiver Operator
        Characteristic (ROC) curve, which plots the true positive rate vs. the false positive rate
        as one varies the threshold on the probabilities given by the model. AUC is also equal to
        the probability that the model will yield a higher probability for a randomly chosen
        positive data point than for a randomly chosen negative data point.  Defined for binary
        CPDs only.

        NOTE: this assumes at least one positive and one negative data point (otherwise
        the notions of true positive rate and false positive rate do not make
        sense).

        :return: a number between 0 and 1 specifying area under the ROC curve
        :rtype: float
        """
        self._check_binary_cpd("AUC")
        prob_true = self.node.cpd.compute_prob_true(**self.node.param_data)
        return self.auc_helper(self.node.data, prob_true)

    @staticmethod
    def d_prime_helper(data, prob_true):
        """ Compute the d-prime metric (of the separation of probabilities associated with positive
        data labels and negative data labels).

        :param np.ndarray[bool] data: binary data values (positive/negative class labels).
        :param np.ndarray[float] prob_true: probability of positive label
        :return: d-prime metric
        :rtype: float
        """
        if len(prob_true) != len(data):
            raise ValueError('prob_true and data must have the same length')
        prob_true, data = Metrics._check_finite(prob_true, data)
        pc_correct = prob_true[data]
        pc_incorrect = prob_true[np.logical_not(data)]
        mean_sep = np.mean(pc_correct) - np.mean(pc_incorrect)
        norm_const = np.sqrt(0.5 * (np.var(pc_correct) + np.var(pc_incorrect)))
        return mean_sep / norm_const

    @staticmethod
    def auc_helper(data, prob_true):
        """ Compute AUC (area under ROC curve) as a function of binary data values and predicted
        probabilities.  If data includes only positive or only negative labels, returns np.nan.

        :param np.ndarray[bool] data: binary data values (positive/negative class labels).
        :param np.ndarray[float] prob_true: probability of positive label
        :return: area under ROC curve
        :rtype: float
        """
        if len(prob_true) != len(data):
            raise ValueError('prob_true and data must have the same length')

        prob_true, data = Metrics._check_finite(prob_true, data)
        sorted_idx = np.argsort(prob_true)[::-1]
        sorted_prob_true = prob_true[sorted_idx]
        unique_prob_true_idx = np.append(np.flatnonzero(np.diff(sorted_prob_true)),
                                         len(sorted_prob_true) - 1)
        x = data[sorted_idx]
        not_x = np.logical_not(x)

        # Compute cumulative sums of true positives and false positives.
        tp = np.cumsum(x)[unique_prob_true_idx].astype(float)
        fp = np.cumsum(not_x)[unique_prob_true_idx].astype(float)

        # The i'th element of tp (fp) is the number of true (false) positives
        # resulting from using the i'th largest rp as a threshold. That is,
        # we predict correct if a response's rp is >= sorted_prob_true[i].
        # We want the first element to correspond to a threshold sufficiently
        # high to yield no predictions of correct. The highest rp qualifies
        # as this highest threshold if its corresponding response is incorrect.
        # Otherwise, we need to add an artificial "highest threshold" at the
        # beginning that yields 0 true positives and 0 false positives.
        if tp[0] != 0.0:
            tp = np.append(0.0, tp)
            fp = np.append(0.0, fp)

        # Calculate true positive rate and false positive rate.
        # This requires at least 1 correct and 1 incorrect response.
        if not tp[-1]:
            return np.nan
        tpr = tp / tp[-1]

        if not fp[-1]:
            return np.nan
        fpr = fp / fp[-1]

        return np.trapz(tpr, fpr)

    @staticmethod
    def online_perc_correct(correct, student_idx):
        """ For each interaction, compute the percent correct for the student's previous
        interactions.  The returned array will contain NaNs for each student's first interaction.
        :param np.ndarray[bool] correct:
        :param np.ndarray[int] student_idx:
        :return: percent correct on previous interactions for this student
        :rtype: np.ndarray[float]
        """
        student_num_correct = np.zeros(np.max(student_idx) + 1)
        student_num_answered = np.zeros(np.max(student_idx) + 1)
        online_pc = np.nan * np.empty_like(correct, dtype=float)
        for i, c in enumerate(correct):
            j = student_idx[i]
            if student_num_answered[j]:
                online_pc[i] = student_num_correct[j] / float(student_num_answered[j])
            student_num_answered[j] += 1
            student_num_correct[j] += int(c)
        return online_pc
