import StringIO
import os
import unittest

from rnn_prof import simple_rnn
from rnn_prof.data.wrapper import load_data
from rnn_prof.data.rnn import build_nn_data

TESTDATA_FILENAME = os.path.join(os.path.dirname(__file__), 'data', 'test_assist_data.csv.gz')


class TestRnn(unittest.TestCase):

    def test_initialization(self):
        """ Just make sure initialize doesn't cause the interpreter to crash """
        data, _, item_ids, _, _ = load_data(TESTDATA_FILENAME, 'assistments')
        num_questions = len(item_ids)
        nn_data = build_nn_data(data, num_questions)
        pivot = len(nn_data) // 2
        train_data = nn_data[:pivot]
        test_data = nn_data[pivot:]

        opts = simple_rnn.RnnOpts(hidden_dim=20)
        simple_rnn.SimpleRnn(train_data, opts, test_data=test_data)

    def test_dump_and_load(self):
        """
        Test dumping and loading the SimpleRnn and make sure that all of its properties remain in
        shape.
        """
        data, _, item_ids, _, _ = load_data(TESTDATA_FILENAME, 'assistments')
        num_questions = len(item_ids)
        nn_data = build_nn_data(data, num_questions)
        pivot = len(nn_data) // 2
        train_data = nn_data[:pivot]

        max_compress_dim = 10
        hidden_dim = 20
        recurrent = False
        grad_norm_limit = 1.0
        first_learning_rate = 20.0
        decay_rate = 0.5
        largest_grad = 4.0
        batch_threshold = 0.8
        opts = simple_rnn.RnnOpts(max_compress_dim=max_compress_dim, hidden_dim=hidden_dim,
                                  recurrent=recurrent, grad_norm_limit=grad_norm_limit,
                                  largest_grad=largest_grad, batch_threshold=batch_threshold,
                                  first_learning_rate=first_learning_rate, decay_rate=decay_rate)
        original = simple_rnn.SimpleRnn(train_data, opts)

        dumped = StringIO.StringIO()
        original.dump(dumped)
        dumped_str = dumped.getvalue()
        dumped_reader = StringIO.StringIO(dumped_str)
        recalled = simple_rnn.SimpleRnn.load(dumped_reader)

        for attr in ('max_compress_dim', 'recurrent', 'grad_norm_limit',
                     'first_learning_rate', 'decay_rate', 'largest_grad', 'batch_threshold'):
            self.assertEqual(getattr(original.opts, attr), getattr(recalled.opts, attr),
                             "%s was changed" % attr)
