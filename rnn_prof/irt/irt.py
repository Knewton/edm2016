"""
Module containing the base class IRT learner that uses a Bayes net to specify model structure
"""
from collections import defaultdict, OrderedDict
import logging

from igraph import Graph, Vertex

from .callbacks import ConvergenceCallback
from .node import Node
from .updaters import UpdateTerms

LOGGER = logging.getLogger(__name__)


class BayesNetGraph(Graph):
    """ A wrapper of igraph's Graph that contains nodes in a BayesNet and implements some utility
    methods. """
    def __init__(self, nodes):
        """ Build a graph from Node objects.  Edges are parent node (param) to child node (data).
        :param dict[str, Node] nodes: nodes in the BayesNet
        """
        super(BayesNetGraph, self).__init__(directed=True)
        # add vertices
        for name, node in nodes.iteritems():
            self.add_vertex(name=name, node=node)
        # add edges
        for node_name, node in nodes.iteritems():
            for par_key, param_node in node.param_nodes.iteritems():
                if param_node not in nodes.values():
                    raise ValueError("{}'s {} not in nodes".format(node_name, par_key))
                self.add_edge(nodes.values().index(param_node), node_name)
        if not self.is_dag():
            raise ValueError("Bayes Net is not a DAG")
        if len(self.components(mode='weak')) > 1:
            LOGGER.warn("Bayes Net is not fully connected")

    def topological_sorting(self, mode='IN'):
        """ Return a topological sorting of nodes in the graph.

        :param mode: whether to sort
        :return: nodes in topologically sorted order
        :rtype: list[Node]
        """
        sorted_node_idx = super(BayesNetGraph, self).topological_sorting(mode=mode)
        sorted_nodes = [self.vs[idx]['node'] for idx in sorted_node_idx]
        LOGGER.debug("Model topological sort: %s", [n.name for n in sorted_nodes])
        return sorted_nodes

    @property
    def nodes(self):
        return [v['node'] for v in self.vs]

    @property
    def training_nodes(self):
        """ Get the nodes in the graph used for training (not held out for testing).

        :return: training nodes
        :rtype: list[Node]
        """
        return [v['node'] for v in self.vs if not v['node'].held_out]

    @property
    def held_out_nodes(self):
        """ Get the held-out nodes in the graph used for testing.

        :return: held out nodes
        :rtype: list[Node]
        """
        return [v['node'] for v in self.vs if v['node'].held_out]

    @property
    def training_subgraph(self):
        """ Get the subgraph that includes only the training (non held-out) nodes and their
        predecessors.
        :return: training subgraph
        :rtype: BayesNetGraph
        """
        training_nodes = [v for v in self.vs if not v['node'].held_out]
        return self._predecessor_subgraph(training_nodes)

    @property
    def held_out_subgraph(self):
        """ Get the subgraph that includes only the held-out nodes and their predecessors.
        :return: held-out subgraph
        :rtype: BayesNetGraph
        """
        held_out_nodes = [v for v in self.vs if v['node'].held_out]
        return self._predecessor_subgraph(held_out_nodes)

    def _predecessor_subgraph(self, nodes):
        """ Get the subgraph containing only the passed-in nodes and their predecessors
        :param Vertex|list[Vertex] leaf_nodes:
        :return: the predecessor subgraph
        :rtype: BayesNetGraph
        """
        if isinstance(nodes, Vertex):
            nodes = [Vertex]
        preds = self.neighborhood(nodes, order=len(self.vs), mode='IN')
        nodes = list(set(self.vs[idx]['node'] for ns in preds for idx in ns))
        nodes = {node.name: node for node in nodes}
        return BayesNetGraph(nodes)


class BayesNetLearner(object):
    """
    Base class for fitting an IRT model specified by a Bayes Net graphical model.
    """
    def __init__(self, nodes, callback=None, max_iterations=1000):
        """ Initialize a learner with a list of nodes

        :param list[Node] nodes: Bayes Net nodes
        :param callback: a callback function, executed at the end of each iteration. Should return
            a boolean flag indicating whether learning should continue.
        :param int max_iterations: maximum number of iterations for learning
        """
        if callback is None:
            callback = ConvergenceCallback()
        for node in nodes:
            if not isinstance(node, Node):
                raise TypeError("Node {} is not of type Node".format(node.name))
        self.callback = callback
        self.max_iterations = max_iterations
        self._check_and_add_nodes(nodes)
        self.log_posterior = 0.
        self.iter = 0

    def _check_and_add_nodes(self, nodes):
        """
        Check that nodes have unique names, then add them as self.nodes OrderedDict, {name: node}

        :param list[Node] nodes: Bayes Net nodes
        """
        if len(set([node.name for node in nodes])) != len(nodes):
            raise ValueError('nodes have non-unique names')

        self.nodes = OrderedDict()
        for node in nodes:
            self.nodes[node.name] = node

    def learn(self):
        """
        Iterate over all nodes in topological order starting with the leaves and perform updates
        on each node.  Collects and stores the sum of the log-probabilities.
        """
        graph = BayesNetGraph(self.nodes)
        sorted_nodes = graph.topological_sorting()

        # initialize a dictionary for passing evidence, keyed on Node objects
        # (first key: node that will consume the evidence, second key: source node of the evidence)
        LOGGER.info("Beginning learning")
        while self.iter < self.max_iterations:
            self.iter += 1
            # clear terms
            update_terms = defaultdict(dict)
            self.log_posterior = 0.
            for node in sorted_nodes:
                LOGGER.debug("Updating %s", node.name)
                evidence_terms = node.update(evidence_terms=update_terms[node])

                if not node.held_out:
                    for target_node, evidence_term in evidence_terms.iteritems():
                        update_terms[target_node][node] = evidence_term

                    self.log_posterior += node.log_prob

            LOGGER.debug("Finished iteration %4.d, log-posterior = %f", self.iter,
                         self.log_posterior)
            if not self.callback(self):
                LOGGER.debug("Stopping updates because of callback termination condition")
                break
        LOGGER.info("Learning finished at iteration %d, log-posterior = %f", self.iter,
                    self.log_posterior)

    def get_posterior_hessian(self, node_name, x=None, use_held_out=False):
        """ Get the Hessian of the log-posterior of a node at x.  If x is not passed in, use
        the stored data.

        :param str node_name: the node of interest
        :param np.ndarray x: the point at which to evaluate the Hessian (same dimensionality as
            the node's data)
        :param bool use_held_out: Whether to use held-out nodes for the computing likelihood terms
        :return: the Hessian
        :rtype: np.ndarray|sp.spmatrix
        """
        # get node of interest (and fail fast if it's not there)
        post_node = self.nodes[node_name]

        # get log prior Hessian
        data_terms = {post_node.cpd.DATA_KEY: UpdateTerms.grad_and_hess}
        hessian = post_node.cpd(x if x is not None else post_node.data,
                                terms_to_compute=data_terms,
                                **post_node.param_data).wrt[post_node.cpd.DATA_KEY].hessian

        # collect log-likelihood Hessians from all children
        for node in self.nodes.itervalues():
            if node.held_out and not use_held_out:
                # skip log-likelihood terms from held-out nodes
                continue
            # look through this node's parent param nodes for our target node
            for par_key, par_node in node.param_nodes.iteritems():
                if par_node is post_node:
                    terms_to_compute = {par_key: UpdateTerms.grad_and_hess}
                    param_data = node.param_data
                    if x is not None:  # if evaluating at x, replace stored data with x
                        param_data[par_key] = x
                    hessian = hessian + node.cpd(node.data, terms_to_compute=terms_to_compute,
                                                 **param_data).wrt[par_key].hessian
        return hessian
