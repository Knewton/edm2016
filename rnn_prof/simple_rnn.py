"""
Throughout this file, the following notations are consistently used for variables:
    * `x` always refers to user histories, and will generally be a tensor of dimension
      num_interactions x num_users x compress_dim
    * `y` always refers to the output probabilities of the neural net, and will generally
      be of the dimensions num_interactions x num_users x num_questions
    * `t` will always refer to the "true" response of the student on the *next* question.
      This is used for training and testing purposes. It is generally of dimension
      num_interactions x num_users
    * `m` will always represent a *mask* on the truth values, since users have different
      history lengths. The dimension is usually num_interactions x num_users
    * `h` will always refer to the hidden layer
    * w_hh/w_xh/w_hy refer to weights with which neural network layers are convolved
    * w_bh/w_by are the bias weights added within the various relevant nonlinearities
"""
from __future__ import division

from collections import namedtuple
import logging
import pickle
import time

import numpy as np
import theano
from theano import tensor as T

from .common import Results
from .data.rnn import build_batches
from .irt.metrics import Metrics

_logger = logging.getLogger(__name__)


def build_grad_and_test_fns(sigmoid_fn=T.nnet.hard_sigmoid, recurrent=True, num_type='float32',
                            mask_type='int8', probability_clip=1e-12, compressed_output=False):
    """
    Actually build the Theano functions for the recurrent neural net. These are *independent*
    of the data passed into the neural net, so whatever it produces should be cached in the
    actual state of the class.

    :param function sigmoid_fn: A function that performs an element-wise sigmoid operation
        on a theano tensor. These will likely be one of
        - `T.nnet.sigmoid`
        - `T.nnet.hard_sigmoid`
        - `T.nnet.ultra_fast_sigmoid`
    :param bool recurrent: If True, use a recurrent architecture
    :param str num_type: A valid floating point numeric type to pass to Theano. Should
        be a numpy-esque name, but Theano requires the stringy version.
    :param str mask_type: A valid "boolean" type to pass to Theano. Should be a numpy-esque
        name for an integer type, but Theano requires the stringy version. Also note
        that Theano does *not* support bools currently.
    :param float probability_clip: In order to avoid nans when taking logs, we clip the
        predicted probabilities to lie in (probability_clip, 1 - probability_clip)
    :param bool compressed_output: Is the output of the RNN compressed? If so, we will run
        the final layer through a sigmoid. Else, we'll assume the final layer are already
        probabilities.
    :rtype: (function, function)
    :return: The main Theano functions. The first function is the "gradient" function,
        whose signature looks like::
            inputs: [x, y, t, m, h_0, w_hh, w_xh, w_hy, w_bh, w_by]
            outputs: [error, num_pred_correct, g_hh, g_xh, g_hy, g_bh, g_by]
        For a further explanation of these variable names see the class docstring.

        The second function is a simpler "test" function, that doesn't go through the
        trouble of computing gradients. Thus, its signature looks like::
            inputs: [x, y, t, m, h_0, w_hh, w_xh, w_hy, w_bh, w_by],
            outputs: [num_pred_correct, next_qn_prob]
    """

    ######################################################
    #                    Tensor input                    #
    ######################################################
    # (Compressed) one hot encoding for input x
    x = T.tensor3('x', dtype=num_type)            # (NUM INTERACTIONS x BATCH x COMPRESS)
    # Mask if uncompressed or "to dot with" if compressed
    # of the probabilities of y to only contain probabilities for the next question
    y = T.tensor3('y', dtype=num_type)            # (NUM INTERACTIONS x BATCH x NUM QN)
    # Correctness of the next question
    t = T.matrix('t', dtype=mask_type)            # (NUM INTERACTIONS x BATCH)
    # Mask for non-existing interactions within a rectangle
    m = T.matrix('m', dtype=mask_type)            # (NUM INTERACTIONS x BATCH)
    # Initial hidden state
    h_0 = T.matrix('h_0', dtype=num_type)         # (BATCH x HIDDEN)
    # Dropout mask for (compressed) input
    x_drop = T.vector('x_drop', dtype=mask_type)  # (COMPRESS)
    # Dropout mask for hidden layer
    h_drop = T.vector('h_drop', dtype=mask_type)  # (HIDDEN)

    ######################################################
    #                       Weights                      #
    ######################################################
    # Recurrent weight from layer H to layer H
    w_hh = T.matrix('w_hh', dtype=num_type)  # (HIDDEN x HIDDEN)
    # Input weights from layer X to layer H
    w_xh = T.matrix('w_xh', dtype=num_type)  # (COMPRESS x HIDDEN)
    # Output weights from layer H to layer Y
    w_hy = T.matrix('w_hy', dtype=num_type)  # (HIDDEN x NUM QN)
    # Bias weights from bias to layer H
    w_bh = T.vector('w_bh', dtype=num_type)  # (HIDDEN)
    # Bias weights from bias to layer Y
    w_by = T.vector('w_by', dtype=num_type)  # (NUM QN)

    # Recurrent function
    def step(x_t, y_t, h_tm1, x_drop, h_drop, w_hh, w_xh, w_hy, w_bh, w_by):
        """
        :param x_t: (Compressed) one hot encoding for input x for time t (BATCH x COMPRESS)
        :param y_t: Mask of the probabilities of y for time t            (BATCH x NUM QN)
        :param h_tm1: Previous hidden state for time t-1                 (BATCH x HIDDEN)
        :param w_hh: Recurrent weight from layer H to layer H            (HIDDEN x HIDDEN)
        :param w_xh: Input weights from layer X to layer H               (COMPRESS x HIDDEN)
        :param w_hy: Output weights from layer H to layer Y              (HIDDEN x NUM QN)
        :param w_bh: Bias weights from bias to layer H                   (HIDDEN)
        :param w_by: Bias weights from bias to layer Y                   (NUM QN)
        :return: h_t: Hidden state for time t
                 all_qn_prob_t: Probabilities for all questions for time t
                 next_qn_prob_t: Probabilities for next question for time t
        """
        # Broadcast over batches
        x_drop_batch = x_drop.dimshuffle(('x', 0))  # BATCH x COMPRESS
        h_drop_batch = h_drop.dimshuffle(('x', 0))  # BATCH x HIDDEN
        w_bh_batch = w_bh.dimshuffle(('x', 0))  # BATCH x HIDDEN
        w_by_batch = w_by.dimshuffle(('x', 0))  # BATCH x NUM QN

        # Dropout on inputs
        x_t_dropped = x_t * x_drop_batch

        if recurrent:
            # BATCH x HIDDEN
            h_t = T.tanh(T.dot(x_t_dropped, w_xh) + T.dot(h_tm1, w_hh) + w_bh_batch)
        else:
            # In this case, we don't care about the previous state, but Theano is a bit
            # smarter than we are and notices if we don't include w_hh. This trains up
            # fast enough as is, so let's just dot it with 0s to get rid of it.
            h_t = T.tanh(T.dot(x_t_dropped, w_xh) + T.dot(T.zeros_like(h_tm1), w_hh) + w_bh_batch)

        # Dropout on hidden layer
        h_t_dropped = h_t * h_drop_batch

        # Compute predicted pass rate for all questions, taking into account
        # whether we have compressed the output dimensions or not
        if compressed_output:
            # If compressed, run the final output through a sigmoid
            final_layer = T.dot(h_t, w_hy) + w_by_batch
            next_qn_prob_t = sigmoid_fn((final_layer * y_t).sum(axis=1))
        else:
            # If not compressed, mask probabilities of all questions except the next one
            all_qn_prob_t = sigmoid_fn(T.dot(h_t, w_hy) + w_by_batch)
            next_qn_prob_t = (all_qn_prob_t * y_t).sum(axis=1)

        return h_t_dropped, next_qn_prob_t

    # This is the scan function to apply the forward propagation for each of the time slice
    # For each time slice:
    #   A time slice of the x and y are passed in as arguments
    #   The previous output of the function is also passed in
    #       (the initial output is passed into outputs_info)
    #   The weights are also passed in but not sliced in time
    #
    # h: Hidden state for all time                               (NUM INTERACTIONS x BATCH x HIDDEN)
    # next_qn_prob: Probabilities for next question for all time (NUM INTERACTIONS x BATCH)
    [h, next_qn_prob], _ = theano.scan(fn=step,
                                       sequences=[x, y],
                                       outputs_info=[h_0, None],
                                       non_sequences=[x_drop, h_drop, w_hh, w_xh, w_hy,
                                                      w_bh, w_by])

    # The negative cross entropy will be what we minimize
    # Note that we multiply by our mask (m) since not all students have the same history length
    next_qn_prob = T.clip(next_qn_prob, probability_clip, 1 - probability_clip)
    cross_entropy = t * T.log(next_qn_prob) + (1 - t) * T.log(1 - next_qn_prob)
    error = -((cross_entropy * m).sum())

    # Prediction accuracy at a cutoff of 0.5
    num_pred_correct = (T.eq(T.ge(next_qn_prob, 0.5), t) * m).sum()

    # BP Gradients for all weights
    g_hh, g_xh, g_hy, g_bh, g_by = T.grad(error, [w_hh, w_xh, w_hy, w_bh, w_by])

    # After all the symbols are linked correctly
    # The expression is compiled using the theano.function to a function
    # Hence grad_fn and test_fn are actual functions to pass in real data
    grad_fn = theano.function(inputs=[x, y, t, m, h_0, x_drop, h_drop, w_hh, w_xh, w_hy, w_bh,
                                      w_by],
                              outputs=[error, g_hh, g_xh, g_hy, g_bh, g_by])

    # Note that in testing we use all the edges in the graph regardless of dropout.
    # In order to make this clear to the user, we do not require them to pass the
    # relevant vector of ones and instead simply make it "given"
    test_fn = theano.function(inputs=[x, y, t, m, h_0, w_hh, w_xh, w_hy, w_bh, w_by],
                              givens=[(x_drop, T.ones((x.shape[2],), dtype=mask_type)),
                                      (h_drop, T.ones_like(w_bh, dtype=mask_type))],
                              outputs=[num_pred_correct, next_qn_prob])
    return grad_fn, test_fn


# A namedtuple for keeping track of weights by name
Weights = namedtuple('Weights', ['w_hh', 'w_xh', 'w_hy', 'w_bh', 'w_by'])


class RnnOpts(namedtuple('RnnOpts', ['max_compress_dim', 'hidden_dim', 'recurrent', 'num_iters',
                                     'dropout_prob', 'grad_norm_limit', 'first_learning_rate',
                                     'decay_rate', 'largest_grad', 'batch_threshold',
                                     'max_output_compress_dim'])):
    """ An immutable container class for RNN structure and optimization options. Note that we take
    an unconventional approach here - we extend namedtuple to get both immutability and
    initialization with default arguments. """
    def __new__(cls, max_compress_dim=10, hidden_dim=10, recurrent=True, num_iters=10,
                dropout_prob=0., grad_norm_limit=0.01, first_learning_rate=30.0, decay_rate=0.99,
                largest_grad=2. ** 20, batch_threshold=0.9, max_output_compress_dim=None):
        """
        RNN STRUCTURE PARAMETERS

        :param int|None max_compress_dim: The max. dimension to which to compress the data using the
            compressed sensing technique. If this is None or the data has length less than
            max_compress_dim, then no compression is performed.
        :param int hidden_dim: The number of hidden units in the RNN.
        :param bool recurrent: Should we actually use a recurrent network or a nonrecurrent one?
        :param int|None max_output_compress_dim: The maximum dimension to which to compress the
            *output* of the RNN (as opposed to the input, controlled by max_compress_dim). None
            simply means no compression.

        LEARNING PARAMETERS

        :param int num_iters: Number of training iterations
        :param float dropout_prob: The probability that a node is dropped in dropout training
        :param float grad_norm_limit: Rescale the gradient so that its sup norm is at most this
        :param float first_learning_rate: What is the first learning rate. Will be decayed
            according to `decay_rate`
        :param float decay_rate: The rate at which the learning_rate will be annealed
        :param float largest_grad: The largest allowable gradient. Any gradient larger than this
            will be clipped to this value.
        :param float batch_threshold: The threshold at which to batch responses
        """
        return super(RnnOpts, cls).__new__(cls, max_compress_dim, hidden_dim, recurrent, num_iters,
                                           dropout_prob, grad_norm_limit, first_learning_rate,
                                           decay_rate, largest_grad, batch_threshold,
                                           max_output_compress_dim)


class SimpleRnn(object):

    @property
    def grad_fn(self):
        if self._grad_fn is None:
            self._grad_fn, self._test_fn = build_grad_and_test_fns(
                recurrent=self.opts.recurrent, compressed_output=self.output_basis is not None)
        return self._grad_fn

    @property
    def test_fn(self):
        if self._test_fn is None:
            self._grad_fn, self._test_fn = build_grad_and_test_fns(recurrent=self.opts.recurrent)
        return self._test_fn

    def __init__(self, train_data, opts, test_data=None, basis=None, weights=None, results=None,
                 num_questions=None, num_type=np.float32, mask_type=np.int8, data_opts=None,
                 output_basis=None, input_dim=None):
        """
        OPTIONS INVOLVING STRUCTURE

        :param list[UserData] train_data: The training data to use for this RNN. Should
            be a list of `UserData` objects, e.g., that comes from `data.rnn.build_nn_data`
        :param RnnOpts opts: learning options (these are not mutated)
        :param list[UserData]|None test_data: The testing data to use for this RNN. Should
            be a list of `UserData` objects, e.g., that comes from `data.rnn.build_nn_data`
        :param np.ndarray|None basis: Which basis to use for compression. If not None, then a basis
            will be created using the `max_compress_dim` option. Should be of size
            `num_q_dims x max_compress_dim`, where `num_q_dims` is likely the number of questions
            or twice that depending on whether you're using correct.
        :param int|None input_dim: What is the dimension of the input. Useful for when you are
            dumping and loading. You likely won't need this.
        :param Weights weights: What weights to use for the RNN if you are reinstantiating
        :param list[Results]|None results: A list of Results to populate the data structure with.
            Probably should only be used by `load`.
        :param int num_questions: the size of the prediction vector (number of questions), useful
            when instantiating the network with empty data. Inferred from data if passed.
        :param num_type: The numeric type to be used in computations. Should be a valid numpy type
        :param mask_type: The mask type to be used in computations. Should be a valid numpy type
        :param DataOpts data_opts: data pre-processing parameters, to be saved with the RNN.
        """
        # Setup data
        self.train_data = train_data
        self.test_data = test_data if test_data is not None else []

        self.opts = opts
        self.data_opts = data_opts

        # Compute number of questions from data if not supplied
        if num_questions is None:
            num_questions = max(np.max(d.next_answer) for d in train_data) + 1
            if test_data is not None:
                num_questions = max(num_questions,
                                    max(np.max(d.next_answer) for d in test_data) + 1)
        self.num_questions = num_questions

        # Setup compression
        if basis is not None:
            # Basis is supplied, so infer the compression dimension from it.
            self.basis = basis
            self.compress_dim = basis.shape[1]
            self.input_dim = basis.shape[0]
            _logger.debug("Using supplied basis with compression "
                          "dimension %d" % self.compress_dim)
        else:
            # Compute input dimension from data if not supplied
            if input_dim is None:
                self.input_dim = max(np.max(d.history) for d in train_data) + 1
                if test_data is not None:
                    self.input_dim = max(self.input_dim,
                                         max(np.max(d.history) for d in test_data) + 1)
            else:
                self.input_dim = input_dim
            self.basis = self._generate_basis(self.input_dim,
                                              self.opts.max_compress_dim, num_type)
            if self.basis is None:
                self.compress_dim = self.input_dim
            else:
                self.compress_dim = self.basis.shape[1]
            _logger.debug("Generated a basis with compression dimension %d" % self.compress_dim)

        _logger.debug("Initialized with input dimension %d", self.compress_dim)

        if output_basis is not None:
            self.output_basis = output_basis
            self.output_compress_dim = output_basis.shape[1]
            _logger.debug("Using supplied output basis with compression "
                          "dimension %d" % self.num_questions)
        else:
            self.output_basis = self._generate_basis(num_questions,
                                                     self.opts.max_output_compress_dim,
                                                     num_type)
            if self.output_basis is not None:
                self.output_compress_dim = self.output_basis.shape[1]
            else:
                self.output_compress_dim = num_questions
            _logger.debug("Generated an output basis with "
                          "compression dimension %d" % self.output_compress_dim)

        # Build batches
        self.train_batches = build_batches(self.train_data, self.output_compress_dim, self.basis,
                                           threshold=self.opts.batch_threshold,
                                           output_basis=self.output_basis,
                                           compress_dim=self.compress_dim)
        self.test_batches = build_batches(self.test_data, self.output_compress_dim, self.basis,
                                          threshold=self.opts.batch_threshold,
                                          output_basis=self.output_basis,
                                          compress_dim=self.compress_dim)

        # Type choices in case you want to optimize
        self.num_type = num_type
        self.mask_type = mask_type
        if weights is None:
            # Weights are uniformly initialized
            w_hh = np.random.uniform(size=(self.opts.hidden_dim, self.opts.hidden_dim),
                                     low=-.01, high=.01).astype(self.num_type)
            w_xh = np.random.uniform(size=(self.compress_dim, self.opts.hidden_dim),
                                     low=-.01, high=.01).astype(self.num_type)
            w_hy = np.random.uniform(size=(self.opts.hidden_dim, self.output_compress_dim),
                                     low=-.01, high=.01).astype(self.num_type)
            w_bh = np.random.uniform(size=self.opts.hidden_dim,
                                     low=-.01, high=.01).astype(self.num_type)
            w_by = np.random.uniform(size=self.output_compress_dim,
                                     low=-.01, high=.01).astype(self.num_type)
            self.weights = Weights(w_hh, w_xh, w_hy, w_bh, w_by)
        else:
            self.weights = weights

        # Setup scaled down weights for testing. See page 2 and 3 of
        # http://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
        self.test_weights = Weights(*[np.empty_like(getattr(self.weights, key))
                                      for key in Weights._fields])
        self._copy_and_scale_weights()

        # To be instantiated on first call
        self._grad_fn = None
        self._test_fn = None

        # Keep track of test runs
        self.results = [] if results is None else results

        # Initialize the learning rate
        self.learning_rate = self.opts.first_learning_rate

    def _generate_basis(self, input_dimension, max_option_dim, num_type):
        """
        Generate a linear compression map from R^{input_dimension} to R^N where
        N = min(input_dimension, max_option_dim). The type of the returned matrix will
        be num_type.

        If the returned matrix would be the identity matrix, we return None.

        :param int input_dimension: The "from" dimension
        :param int max_option_dim: The maximum "to" dimension
        :param dtype num_type: The type to use for the compression matrix
        :return: The compression matrix or None
        :rtype: np.ndarray | None
        """
        if max_option_dim and max_option_dim < input_dimension:
            _logger.debug("Using random compression with supplied max_compress_dim "
                          "(%d)" % max_option_dim)
            basis = np.random.normal(size=(input_dimension, max_option_dim)).astype(num_type)
            normalizer = np.sqrt((basis * basis).sum(axis=1))
            basis = basis / normalizer[:, np.newaxis]
        else:
            if max_option_dim:
                _logger.debug("Not using compression because max history size (%d)"
                              " <= maximum allowed compression dimension (%d)",
                              input_dimension, max_option_dim)
            else:
                _logger.debug("Not using compression because dimension not specified.")
            basis = None
        return basis

    def train_and_test(self, num_iters, test_spacing=5):
        """
        Train the RNN on the train data, and test on the test data.
        Note the learning rate is reset to self.opts.first_learning_rate
        before training commences.

        :param int num_iters: number of training iterations
        :param int test_spacing: evaluate RNN on the test data every
                                 this many training iterations.
        :return: the MAP accuracy, AYC, predicted probabilities of correct,
                 and boolean correct values on the test data.
        :rtype: float, float, np.ndarray(float), np.ndarray(float)
        """
        # Initialize the learning rate based on options.
        self.learning_rate = self.opts.first_learning_rate

        def run_test(num_iter, num_iters):
            _logger.info("Computing metrics for training data on iteration %3d/%d",
                         num_iter + 1, num_iters)
            train_acc, train_auc, _, _ = self.test_on_training()
            _logger.info("Train Accuracy: %.5f; Train AUC: %.5f", train_acc, train_auc)
            _logger.info("Running tests for iteration %3d/%d", num_iter + 1, num_iters)
            test_acc, test_auc, test_prob_correct, test_corrects = self.test()
            self.results.append(Results(num_iter=num_iter, accuracy=test_acc, auc=test_auc))
            _logger.info("Test Accuracy: %.5f; Test AUC: %.5f", test_acc, test_auc)
            return test_acc, test_auc, test_prob_correct, test_corrects

        for num_iter in range(num_iters):
            _logger.info("Performing training iteration %3d/%d", num_iter + 1, num_iters)
            start_time = time.time()
            self.step()
            end_time = time.time()
            _logger.info("Iteration %3d/%d took %.2fs", num_iter + 1, num_iters,
                         end_time - start_time)

            # DECAY!!!!
            self.learning_rate *= self.opts.decay_rate

            if (num_iter + 1) % test_spacing == 0:
                test_acc, test_auc, test_prob_correct, test_corrects = run_test(num_iter,
                                                                                num_iters)

        # If we didn't run a test on the last iteration, run one now
        if num_iters % test_spacing != 0:
            test_acc, test_auc, test_prob_correct, test_corrects = run_test(num_iters - 1,
                                                                            num_iters)
        return test_acc, test_auc, test_prob_correct, test_corrects

    def step(self):
        """
        Take a single step using `self.train_batches`
        """
        np.random.shuffle(self.train_batches)
        num_batches = len(self.train_batches)
        print_on = {num_batches * i // 10 for i in range(1, 11)}
        for batch_num, batch in enumerate(self.train_batches):
            # Initial hidden state
            h_0 = np.zeros((batch.length, len(self.weights.w_bh)), dtype=self.num_type)

            # Dropout (1s for *retained* nodes)
            h_drop = (np.random.random(self.opts.hidden_dim) > self.opts.dropout_prob).astype(
                self.mask_type)
            x_drop = (np.random.random(self.compress_dim) > self.opts.dropout_prob).astype(
                self.mask_type)

            # Run RNN grad function
            results = self.grad_fn(batch.history, batch.next_answer, batch.truth, batch.mask,
                                   h_0, x_drop, h_drop, *self.weights)
            error, g_hh, g_xh, g_hy, g_bh, g_by = results

            grads = Weights(w_hh=g_hh, w_xh=g_xh, w_hy=g_hy, w_bh=g_bh, w_by=g_by)

            # Update gradient parameters
            self.update_weights(grads)

            if batch_num in print_on:
                _logger.debug("Batch %3d/%d complete", batch_num, num_batches)

    def _copy_and_scale_weights(self):
        """ Copy and scale the weights for testing according to the dropout probability """
        for key in Weights._fields:
            np.multiply(1.0 - self.opts.dropout_prob, getattr(self.weights, key),
                        getattr(self.test_weights, key))

    def update_weights(self, grads):
        """
        Update the weights using (modified) gradient descent. Done in place.

        :param Weights grads: The gradients of the error function w.r.t. the weights
        """
        # Set invalid values to 0 and clip extreme gradients
        for grad in grads:
            grad[~np.isfinite(grad)] = 0
            np.clip(grad, -self.opts.largest_grad, self.opts.largest_grad, out=grad)

        # Gather the gradient norms
        grad_norms = [np.linalg.norm(grad) for grad in grads]
        max_grad_norm = max(grad_norms)
        if max_grad_norm > self.opts.grad_norm_limit:
            offset = self.opts.grad_norm_limit / max_grad_norm
        else:
            offset = 1.0

        # Gradient descent
        for weight, grad in zip(self.weights, grads):
            weight -= self.learning_rate * grad * offset

        self._copy_and_scale_weights()

    def _test_on_batches(self, batches):
        """ Compute validation metrics on the specified batches.

        :return: accuracy (at a 0.5 cutoff) and AUC, and the response probability
                 and correctness values from the passed data.
        :rtype: (float, float, np.ndarray(float), np.ndarray(bool))
        """
        # Setup test stats variables
        test_questions = 0
        test_correct = 0

        # AUC probs and truth
        p = np.zeros(0, dtype=self.num_type)
        t = np.zeros(0, dtype=self.num_type)

        num_batches = len(batches)
        print_on = {num_batches * i // 10 for i in range(1, 11)}
        for batch_num, batch in enumerate(batches):
            # Initial hidden state and broadcast bias weights to all batches
            h_0 = np.zeros((batch.length, len(self.weights.w_bh)), dtype=self.num_type)

            # Run RNN test function
            num_pred_correct, next_qn_prob = self.test_fn(batch.history, batch.next_answer,
                                                          batch.truth, batch.mask, h_0,
                                                          *self.test_weights)

            # Update stats
            test_questions += batch.num_interactions
            test_correct += num_pred_correct

            # Update points for AUC
            interaction_mask = batch.mask.astype(np.bool).flatten()
            p = np.concatenate((p, np.ravel(next_qn_prob)[interaction_mask]), axis=0)
            t = np.concatenate((t, np.ravel(batch.truth)[interaction_mask]), axis=0)

            if batch_num in print_on:
                _logger.debug("Batch %3d/%d complete", batch_num, num_batches)

        # Find test accuracy and AUC
        test_acc = test_correct / test_questions
        test_auc = Metrics.auc_helper(t, p)
        return test_acc, test_auc, p, t

    def test(self):
        """ Compute validation metrics on the test set.

        :return: test accuracy (at a 0.5 cutoff) and test AUC, and the response probability
                 and correctness values from the test set.
        :rtype: (float, float, np.ndarray(float), np.ndarray(bool))
        """
        return self._test_on_batches(self.test_batches)

    def test_on_training(self):
        """ Compute validation metrics on the training set.

        :return: training accuracy (at a 0.5 cutoff) and training AUC, and the response probability
                 and correctness values from the training set.
        :rtype: (float, float, np.ndarray(float), np.ndarray(bool))
        """
        return self._test_on_batches(self.train_batches)

    def dump(self, f):
        """ Dump to a file handler `f`. Load with `load`. """
        output = {
            'train_data': [],
            'opts': self.opts,
            'data_opts': self.data_opts,
            'basis': self.basis,
            'weights': self.weights,
            'results': self.results,
            'num_questions': self.num_questions,
            'input_dim': self.input_dim,
            'num_type': self.num_type,
            'mask_type': self.mask_type,
            'output_basis': self.output_basis
        }
        pickle.dump(output, f)

    @classmethod
    def load(cls, f):
        """ Load from a file handle `f`. Should have been dumped with `dump`. """
        d = pickle.load(f)
        return cls(**d)
