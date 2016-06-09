"""
Test the Gaussian CPD
"""
import unittest

import numpy as np
from scipy import linalg as sl
from scipy import sparse as sp
from scipy.sparse.linalg import aslinearoperator

from rnn_prof.irt.cpd import gaussian as undertest
from rnn_prof.irt.linear_operators import IndexOperator
from rnn_prof.irt.updaters import UpdateTerms
from rnn_prof.irt.testing_utils import (finite_diff_grad, finite_diff_hessian, log_norm_pdf,
                                        ALMOST_EQUAL_EPSILON)

NUM_TESTS = 5
FLOAT_DTYPE = np.dtype('float64')


class TestGaussian(unittest.TestCase):
    def test_init(self):
        """
        Test behavior of different (partial argument) initialization
        """
        # test scalar initialization
        mean = np.random.randn()
        precision = np.random.rand()

        cpd = undertest.GaussianCPD(dim=1)
        self.assertEqual(cpd.dim, 1)
        self.assertEqual(cpd.mean, 0.)
        self.assertEqual(cpd.precision, 1.)

        cpd = undertest.GaussianCPD(mean=mean)
        self.assertEqual(cpd.dim, 1)
        self.assertEqual(cpd.mean, mean)
        self.assertEqual(cpd.precision, 1.)

        cpd = undertest.GaussianCPD(mean=mean, precision=precision)
        self.assertEqual(cpd.dim, 1)
        self.assertEqual(cpd.mean, mean)
        self.assertEqual(cpd.precision, precision)
        # check that single element inputs don't break this scalar CPD
        for x in (1., np.asarray(1.), np.ones(1), np.ones((1, 1))):
            cpd(x)
        # check that wrong dimensioned inputs break this scalar CPD
        with self.assertRaises(ValueError):
            cpd(np.random.randn(3))

        with self.assertRaises(ValueError):
            undertest.GaussianCPD(mean=mean, precision=-1.)

        # test multivariate initialization
        mean = np.random.randn(3)
        precision_scalar = np.random.rand()
        precision_diag = np.random.rand(3, 1)
        precision_dense = np.random.randn(3, 3)
        precision_dense = precision_dense.dot(precision_dense.T)
        precision_sparse = sp.diags(precision_diag.ravel(), 0)
        for precision in (precision_scalar, precision_diag, precision_dense, precision_sparse):
            cpd = undertest.GaussianCPD(dim=3)
            self.assertEqual(cpd.dim, 3)
            np.testing.assert_equal(cpd.mean, np.zeros(3))
            self.assertEqual(cpd.precision, 1.)

            cpd = undertest.GaussianCPD(mean=mean)
            self.assertEqual(cpd.dim, 3)
            np.testing.assert_equal(cpd.mean, mean)
            self.assertEqual(cpd.precision, 1.)

            if not np.isscalar(precision):
                cpd = undertest.GaussianCPD(precision=precision)
                self.assertEqual(cpd.dim, 3)
                np.testing.assert_equal(cpd.mean, np.zeros(3))
                if isinstance(precision, sp.spmatrix):
                    np.testing.assert_equal(cpd.precision.toarray(), precision.toarray())
                else:
                    np.testing.assert_equal(cpd.precision, precision)
            cpd = undertest.GaussianCPD(mean=mean, precision=precision)
            expected_const = -0.5 * cpd.dim * np.log(2 * np.pi)
            if precision is precision_scalar:
                expected_const += 0.5 * cpd.dim * np.log(precision)
            elif precision is precision_diag:
                expected_const += 0.5 * np.sum(np.log(precision))
            elif precision is precision_sparse:
                expected_const += 0.5 * np.log(np.linalg.det(precision.todense()))
            else:
                expected_const += 0.5 * np.log(np.linalg.det(precision))
            self.assertEqual(cpd.const, expected_const)
            self.assertEqual(cpd.dim, 3)
            np.testing.assert_equal(cpd.mean, mean)
            if isinstance(precision, sp.spmatrix):
                np.testing.assert_equal(cpd.precision.toarray(), precision.toarray())
            else:
                np.testing.assert_equal(cpd.precision, precision)

        # check that correctly shaped inputs don't break this CPD
        cpd(np.random.randn(3))
        # check that wrong dimensioned inputs break this CPD
        for x in (np.random.randn(), np.random.randn(1), np.random.randn(1, 1),
                  np.random.randn(3, 4)):
            with self.assertRaises(ValueError):
                cpd(x)

        # check non-positive definite precision
        with self.assertRaises(ValueError):
            undertest.GaussianCPD(mean=mean, precision=-precision)
        with self.assertRaises(ValueError):
            undertest.GaussianCPD(mean=mean, precision=np.zeros((3, 3)))

        # check men and precision dimension mismatch
        with self.assertRaises(ValueError):
            precision_4d = np.random.randn(4, 4)
            precision_4d = precision_4d.dot(precision_4d.T)
            undertest.GaussianCPD(mean=mean, precision=precision_4d)

    def test_log_prob_terms(self):
        """
        Test the log probability, gradient, and Hessian w.r.t. the data and the mean. Test cases
        include diagonal/full covariance matrix.
        """
        def set_up_cpd(dim, scalar_prec, diag_prec, mean_lin_op=None):
            """ Make a Gaussian cpd given scalar/diagonal/non-diagonal precision.
            :param int dim: dimensionality of the Gaussian distribution
            :param bool scalar_prec: Precision specified as a scalar?
            :param bool diag_prec: Diagonal precision matrix? If diagonal, use sparse matrices.
            :param LinearOperator mean_lin_op: linear operator applied to the mean
            :return: Gaussian cpd
            :rtype: undertest.GaussianCPD
            """
            if mean_lin_op is None:
                mean = np.random.randn(dim)
            else:
                mean = mean_lin_op * np.random.randn(mean_lin_op.shape[1])
            if scalar_prec:
                precision = np.random.rand()
            elif diag_prec:
                # In the diagonal case, the log probability is just the sum of the log
                # probability of independent normals.
                # Note: use sparse matrices here to test handling of sparse precision.
                diag = np.exp(np.random.randn(dim))
                precision = sp.diags(diag * diag, 0, dtype=FLOAT_DTYPE)
            else:
                # In the non-diagonal case, the log probability can be computed as a product of
                # univariate normals if you first transform the input.
                base_matrix = np.random.randn(dim, dim) / dim
                # make sure base matrix is well conditioned by restricting the eigval range
                u, s, v = np.linalg.svd(base_matrix)
                s = np.clip(s, 0.5 * np.max(s), np.inf)
                base_matrix = u.dot(np.diag(s)).dot(v)
                square_root = np.dot(base_matrix, base_matrix.T)
                precision = np.dot(square_root, square_root)
            return undertest.GaussianCPD(dim, mean, precision, mean_lin_op)

        def log_prob_test(cpd, mean=None, precision=None, mean_proj_matrix=None):
            """
            Run an individual test on a given Gaussian cpd
            :param undertest.GaussianCPD cpd: A Gaussian cpd
            :param None|float|np.ndarray mean: mean, if we are overwriting initialization mean
            :param None|float|np.ndarray|sp.spmatrix precision: precision if overwriting
            :param None|np.ndarray mean_proj_matrix: projection matrix from the mean to the data
            """
            data_key = undertest.GaussianCPD.DATA_KEY
            mean_key = undertest.GaussianCPD.MEAN_KEY
            terms_to_compute = {data_key: UpdateTerms.grad_and_hess,
                                mean_key: UpdateTerms.grad_and_hess}

            # test the log probability output
            x = np.random.randn(cpd.dim)
            if np.isscalar(cpd.precision):
                precision = cpd.precision * np.eye(cpd.dim)
                square_root = np.sqrt(precision) * np.eye(cpd.dim)
            else:
                precision = self.densify(cpd.precision)
                square_root = sl.sqrtm(precision)
            expected_log_prob = (np.sum(log_norm_pdf(np.dot(square_root, (x - cpd.mean)))) +
                                 np.log(np.linalg.det(square_root)))

            actual_terms = cpd(x, mean=mean, precision=precision, terms_to_compute=terms_to_compute)
            actual_data_gradient = actual_terms.wrt[data_key].gradient
            if np.isscalar(actual_terms.wrt[data_key].hessian):
                actual_data_hessian = actual_terms.wrt[data_key].hessian * np.eye(cpd.dim)
            else:
                actual_data_hessian = self.densify(actual_terms.wrt[data_key].hessian)
            actual_mean_gradient = actual_terms.wrt[mean_key].gradient
            if np.isscalar(actual_terms.wrt[mean_key].hessian):
                actual_mean_hessian = actual_terms.wrt[mean_key].hessian * np.eye(cpd.dim)
            else:
                actual_mean_hessian = self.densify(actual_terms.wrt[mean_key].hessian)

            np.testing.assert_allclose(actual_terms.log_prob, expected_log_prob, rtol=0.0,
                                       atol=ALMOST_EQUAL_EPSILON)
            # test gradient and Hessian w.r.t. the data
            log_prob_fn = lambda z: cpd(z, mean, precision).log_prob
            log_prob_grad = lambda z: cpd(z, mean, precision,
                                          {data_key: UpdateTerms.grad}).wrt[data_key].gradient
            expected_data_grad = finite_diff_grad(x, log_prob_fn)
            expected_data_hessian = finite_diff_hessian(x, log_prob_grad)
            np.testing.assert_allclose(actual_data_gradient, expected_data_grad, rtol=0.0,
                                       atol=ALMOST_EQUAL_EPSILON)
            np.testing.assert_allclose(actual_data_hessian, expected_data_hessian,
                                       rtol=0.0, atol=ALMOST_EQUAL_EPSILON)

            # test gradient and Hessian w.r.t. the mean
            log_prob_fn = lambda z: cpd(x, z, precision).log_prob
            log_prob_grad = lambda z: cpd(x, z, precision, terms_to_compute).wrt[mean_key].gradient
            if mean_proj_matrix is None:
                expected_mean_grad = finite_diff_grad(cpd.mean, log_prob_fn)
                expected_mean_hessian = finite_diff_hessian(x, log_prob_grad)
            else:
                latent_mean = np.linalg.pinv(mean_proj_matrix).dot(cpd.mean)
                expected_mean_grad = finite_diff_grad(latent_mean, log_prob_fn)
                expected_mean_hessian = finite_diff_hessian(latent_mean, log_prob_grad)
            np.testing.assert_allclose(actual_mean_gradient, expected_mean_grad, rtol=0.0,
                                       atol=ALMOST_EQUAL_EPSILON)
            np.testing.assert_allclose(actual_mean_hessian, expected_mean_hessian, rtol=0.0,
                                       atol=ALMOST_EQUAL_EPSILON)

        # Run a few tests with diagonal and non-diagonal covariance
        for _ in range(NUM_TESTS):
            log_prob_test(set_up_cpd(dim=np.random.randint(5, 20), scalar_prec=True,
                                     diag_prec=True))
            log_prob_test(set_up_cpd(dim=np.random.randint(5, 20), scalar_prec=False,
                                     diag_prec=True))
            log_prob_test(set_up_cpd(dim=np.random.randint(5, 20), scalar_prec=False,
                                     diag_prec=True))
            # test the case with a linear operator on the mean
            dim1, dim2 = np.random.randint(5, 20), np.random.randint(5, 20)
            proj_matrix = np.random.randn(dim1, dim2)
            lin_op = aslinearoperator(proj_matrix)
            log_prob_test(set_up_cpd(dim=dim1, scalar_prec=True, diag_prec=True,
                                     mean_lin_op=lin_op),
                          mean_proj_matrix=proj_matrix)
        log_prob_test(set_up_cpd(dim=dim1, scalar_prec=False, diag_prec=False,
                                 mean_lin_op=lin_op),
                      mean_proj_matrix=proj_matrix)
        # test with indexing as linear operator, since it's a special case
        for scalar_prec, diag_prec in ((True, True), (False, True), (False, False)):
            dim2 = dim1 + 5
            index_map = np.random.randint(dim1, size=dim2)
            lin_op = IndexOperator(index_map=index_map, dim_x=dim1)
            proj_matrix = np.zeros((dim2, dim1), dtype=int)
            for i in range(dim2):
                proj_matrix[i, index_map[i]] = 1
            log_prob_test(set_up_cpd(dim=dim2, scalar_prec=scalar_prec, diag_prec=diag_prec,
                                     mean_lin_op=lin_op),
                          mean_proj_matrix=proj_matrix)

    def test_dependent_vars(self):
        """
        Test the extraction of dependent variables and getting the subset CPD.
        """
        for _ in range(NUM_TESTS):
            for sparse_prec, use_mean_lin_op in ((False, False), (True, False), (False, True)):
                dim = np.random.randint(1, 100)
                sub_idx = np.flatnonzero(np.random.rand(dim) > 0.5)
                # make sure sub_idx is not empty
                subset_dim = len(sub_idx)
                while subset_dim == 0:
                    sub_idx = np.flatnonzero(np.random.rand(dim) > 0.5)
                    subset_dim = len(sub_idx)
                if use_mean_lin_op:
                    mean_dim = np.random.randint(1, 10)
                    proj_matrix = np.random.randn(dim, mean_dim)
                    mean_lin_op = aslinearoperator(proj_matrix)
                else:
                    mean_lin_op = None

                # test that with diagonal precision, the dependent set is equal to the original one
                if sparse_prec:
                    precision = sp.dia_matrix((np.ones(dim), np.zeros(1)), shape=(dim, dim)).tolil()
                else:
                    precision = np.diag(np.ones(dim))
                cpd = undertest.GaussianCPD(dim=dim, precision=precision, mean_lin_op=mean_lin_op)
                subset_dep_idx = cpd.get_dependent_vars(sub_idx)
                np.testing.assert_array_equal(subset_dep_idx, sub_idx)

                # test dependent set for precision with random edges
                if not len(sub_idx):  # skip when original sub_idx was empty
                    continue
                # add dependency edges to the precision matrix
                dep_idx = np.random.randint(0, dim, size=len(sub_idx))

                precision[sub_idx, dep_idx] = -1e-3
                precision[dep_idx, sub_idx] = -1e-3
                # make sure we did not overwrite the diagonal (when i==j)
                precision[np.arange(dim), np.arange(dim)] = 1.
                cpd = undertest.GaussianCPD(dim=dim, precision=precision, mean_lin_op=mean_lin_op)
                subset_dep_idx = cpd.get_dependent_vars(sub_idx)
                np.testing.assert_array_equal(subset_dep_idx, np.union1d(dep_idx, sub_idx))

                # test invalid case
                with self.assertRaises(ValueError):
                    cpd.get_subset_cpd(sub_idx=np.array([]))

                # test construction of the subset cpd
                subset_cpd = cpd.get_subset_cpd(sub_idx=sub_idx)
                self.assertEqual(subset_cpd.dim, subset_dim)
                np.testing.assert_array_equal(subset_cpd.mean, cpd.mean[sub_idx])

                actual_precision = self.densify(subset_cpd.precision)
                expected_precision = self.densify(cpd.precision)[sub_idx, :][:, sub_idx]
                np.testing.assert_array_equal(actual_precision, expected_precision)

                # test calling of the subsetted cpd
                subset_cpd(np.random.randn(subset_dim, 1))
                # test by passing in a new mean
                if use_mean_lin_op:
                    subset_cpd(np.random.randn(subset_dim, 1), mean=np.random.randn(mean_dim))
                else:
                    subset_cpd(np.random.randn(subset_dim, 1), mean=np.random.randn(subset_dim))

    @staticmethod
    def densify(x):
        """ Convert sparse matrix or float to a numpy array. If passed numpy array, return same.

        :param sp.spmatrix|np.ndarray|float x:
        :return: x as a numpy array
        :rtype: np.ndarray
        """
        return x.toarray() if sp.issparse(x) else np.array(x, ndmin=2)
