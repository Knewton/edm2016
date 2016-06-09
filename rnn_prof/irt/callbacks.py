"""
Basic callbacks for the Bayes Net IRT learners.
"""
import logging
import numpy as np

from .cpd.ogive import OgiveCPD
from .metrics import LOGLI_KEY, MAP_ACCURACY_KEY, AUC_KEY, METRICS_KEYS

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


class RecordingCallback(ConvergenceCallback):
    """ Callback function that records basic learning metrics. """
    def __init__(self, metrics_to_record=DEFAULT_METRICS, **kwargs):
        super(RecordingCallback, self).__init__(**kwargs)
        self.metrics = {m: None for m in metrics_to_record}

    def __call__(self, learner):
        self.record_metrics(learner)
        return super(RecordingCallback, self).__call__(learner, metrics=self.metrics)

    def record_metrics(self, learner):
        """ Record the performance metrics: iteration count, global learner log-posterior, and
        the metrics specified at initialization (e.g., log-likelihood, test MAP accuracy) for
        all OgiveCPD nodes.
        NOTE: The latter performance metrics are dictionaries two levels deep, and should be
        accessed as `callback.metrics[AUC_KEY][test_response_node.name]`.
        """
        def append_metric(new_value, metric_key, node_key=None, dtype=None):
            """ Helper function for appending to (possibly uninitialized) dictionary of metrics,
            one (iteration count, log-posterior) or two (e.g., AUC for particular node) levels
            deep."""
            # initialize dicts/arrays if necessary
            dtype = dtype or np.float64
            if self.metrics[metric_key] is None:
                init_vals = np.nan * np.empty(MAX_HIST_LEN, dtype=dtype)
                self.metrics[metric_key] = init_vals if node_key is None else {node_key: init_vals}
            elif node_key is not None and node_key not in self.metrics[metric_key]:
                init_vals = np.nan * np.empty(MAX_HIST_LEN, dtype=dtype)
                self.metrics[metric_key][node_key] = init_vals
            # get dictionary element and append
            if node_key is None:
                metric = self.metrics[metric_key]
            else:
                metric = self.metrics[metric_key][node_key]
            return np.append(metric[1:], new_value)

        for mkey in self.metrics:
            if mkey == ITER_KEY:
                # write iteration count
                self.metrics[mkey] = append_metric(learner.iter, mkey, dtype=int)
            elif mkey == TRAIN_LOG_POST_KEY:
                # write global learner log-posterior
                self.metrics[mkey] = append_metric(learner.log_posterior, mkey)
            elif mkey in METRICS_KEYS:
                # for all other metrics, record values for each node with an OgiveCPD
                for node in learner.nodes.itervalues():
                    if isinstance(node.cpd, OgiveCPD):
                        metric = node.metrics.compute_metric(mkey)
                        self.metrics[mkey][node.name] = append_metric(metric, mkey, node.name)
