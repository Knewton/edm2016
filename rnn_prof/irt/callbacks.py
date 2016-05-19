"""
Basic callbacks for the Bayes Net IRT learners.
"""
import logging
import numpy as np

from .metrics import LOGLI_KEY, MAP_ACCURACY_KEY, AUC_KEY

LOGGER = logging.getLogger(__name__)
TRAIN_LOG_POST_KEY = 'train log posterior'
ITER_KEY = 'iteration'
TEST_SUFFIX = '_TEST'

MAX_HIST_LEN = 400
MAX_HIST_SAMPLES = 50

HIST_COLOR = np.asarray([.7, .7, .7])

DEFAULT_METRICS = (ITER_KEY, TRAIN_LOG_POST_KEY, LOGLI_KEY, MAP_ACCURACY_KEY, AUC_KEY)


class ConvergenceCallback(object):
    """
    Basic callback that checks if convergence conditions on all the learner's node have been met.
    Optionally, print or log info statements related to convergence.
    """
    def __init__(self, early_stopping=False, log_freq=0, print_freq=100, logger=None):
        """
        :param int print_freq: print frequency (if 0, do not print)
        :param int log_freq: log frequency (if 0, do not log)
        :param bool early_stopping: Whether to stop inference if the sum of held_out nodes'
                                    log-prob_delta's is not positive
        :param Logger|None logger:  optional logger to use; if not specified, use this module's
        """
        self.early_stopping = early_stopping
        self.print_freq = print_freq
        self.log_freq = log_freq
        self.logger = logger or LOGGER

    def __call__(self, learner, metrics=None):
        """
        :param BayesNetLearner learner: the learner
        :param dict|None metrics: Metrics dictionary of depth 1 or 2
            (generally structured as: {metric name: array of values}) to log/print. Logs/prints
            the last element in the array of values.
        :return: whether to continue learning
        :rtype: bool
        """
        def get_msg_vals():
            msg_string = 'Iter %d: Log-Posterior: %.04f, Log10Grad: %0.4f, Log10Diff: %0.4f'
            msg_vars = [learner.iter, learner.log_posterior, max_grad, max_diff]
            if metrics is not None:
                for mkey, mval in metrics.iteritems():
                    if isinstance(mval, dict):
                        for node_name, node_metric_val in mval.iteritems():
                            msg_string += ', %s %s: %%0.4f' % (mkey, node_name)
                            msg_vars.append(node_metric_val[-1])
                    else:
                        msg_string += ', %s: %%0.4f' % mkey
                        msg_vars.append(mval[-1])
            return msg_string, tuple(msg_vars)
        max_grad, max_diff = None, None
        if self.print_freq > 0 and not learner.iter % self.print_freq:
            max_grad, max_diff = self.compute_stats(learner)
            print_msg, print_vars = get_msg_vals()
            print_msg = '\r' + print_msg
            print print_msg % print_vars,
        if self.log_freq > 0 and not learner.iter % self.log_freq:
            if max_grad is None:
                # compute stats if it hasn't been done yet
                max_grad, max_diff = self.compute_stats(learner)
            log_string, log_vars = get_msg_vals()
            self.logger.info(log_string, *log_vars)
        return self.is_converged(learner)

    def is_converged(self, learner):
        """
        :param BayesNetLearner learner: the learner
        :return: whether to continue learning
        :rtype: bool
        """
        should_continue = not all([n.converged for n in learner.nodes.values() if not n.held_out])
        if should_continue and self.early_stopping:
            held_out_nodes = [n for n in learner.nodes.values() if n.held_out]
            if len(held_out_nodes) == 0:
                raise ValueError('There are no held out nodes so early stopping cannot work.')
            log_prob_deltas = [n.log_prob_delta for n in held_out_nodes
                               if n.log_prob_delta is not None]
            if len(log_prob_deltas) > 0:
                should_continue = sum(log_prob_deltas) > 0
        return should_continue

    @staticmethod
    def compute_stats(learner):
        """ Compute the gradient and difference changes across a learner's nodes

        :param BayesNetLearner learner: the IRT learner
        :return: the maximum of the gradients and the maximum of the iteration-to-iteration diffs
        :rtype: float, float
        """
        grad_diffs = [np.abs(n.max_grad) for n in learner.nodes.values() if n.max_grad is not None]
        diff_diffs = [np.abs(n.max_diff) for n in learner.nodes.values() if n.max_diff is not None]
        max_grad = np.log10(np.max(grad_diffs)) if len(grad_diffs) else np.nan
        max_diff = np.log10(np.max(diff_diffs)) if len(diff_diffs) else np.nan
        return max_grad, max_diff
