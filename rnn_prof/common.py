"""
A place to keep common data structures for various testing schemes.
"""
from collections import namedtuple

# A structure for keeping track of basic metrics of test performance
Results = namedtuple('Results', ['num_iter', 'accuracy', 'auc'])
