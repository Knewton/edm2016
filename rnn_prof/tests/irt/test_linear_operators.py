"""
Unit tests for linear operators
"""
import numpy as np
import scipy.sparse as sp
import unittest

from rnn_prof.irt import linear_operators as undertest
from rnn_prof.irt.testing_utils import finite_diff_grad

NUM_TESTS = 5


class TestLinearOperators(unittest.TestCase):
    @staticmethod
    def subset_test(lin_op):
        """ Test that subsetting a linear operator produces the correct outputs.
        :param LinearOperator lin_op: the linear operator
        """
        sub_idx = np.random.rand(lin_op.shape[0], 1) > 0.5
        # make sure at least one element included
        sub_idx[np.random.randint(0, len(sub_idx))] = True
        sub_idx = np.flatnonzero(sub_idx)
        sub_lin_op = undertest.get_subset_lin_op(lin_op, sub_idx)

        # test projection to subset of indices
        x = np.random.randn(lin_op.shape[1], np.random.randint(1, 3))
        np.testing.assert_array_almost_equal(sub_lin_op * x, (lin_op * x)[sub_idx, :])

        # test back projection from subset of indices
        y = np.random.randn(len(sub_idx), np.random.randint(1, 3))
        z = np.zeros((lin_op.shape[0], y.shape[1]))
        z[sub_idx] = y
        np.testing.assert_array_almost_equal(sub_lin_op.rmatvec(y), lin_op.rmatvec(z))

    def test_linearity(self):
        """Test the linearity and back projection properties of all operators """
        def finite_diff_matrix_grad(f, x, dim_y):
            return np.array([finite_diff_grad(x, lambda z: f(z)[i]) for i in np.arange(dim_y)])

        for _ in range(NUM_TESTS):
            dim_x = np.random.randint(2, 20)
            index_map = np.random.randint(dim_x, size=np.random.randint(1, 20))
            group_idx = np.random.randint(np.random.randint(1, 20), size=dim_x)
            split_idx = np.unique(np.random.randint(1, dim_x, 5))
            mask_idx = np.random.rand(len(index_map)) > 0.5
            while not np.sum(mask_idx):
                mask_idx = np.random.rand(len(index_map)) > 0.5
            skip_index_map = index_map[:int(np.sum(mask_idx))]
            # for masked index operator, make the output array larger than last skip index
            si_dim_y = np.random.randint(1, 5)
            if np.sum(mask_idx):
                si_dim_y += np.flatnonzero(mask_idx)[-1]
            lin_op = undertest.IndexOperator(index_map, dim_x)
            dim_y = lin_op.shape[0]

            # test the gradients at multiple points are the same
            grad0 = finite_diff_matrix_grad(lambda z: lin_op * z, np.random.randn(dim_x), dim_y)
            grad1 = finite_diff_matrix_grad(lambda z: lin_op * z, np.random.randn(dim_x), dim_y)
            np.testing.assert_array_almost_equal(grad0, grad1)
            lin_op.rmatvec(np.random.randn(dim_y))
            # test the gradient of the back projection is the transpose of the forward one
            back_grad = finite_diff_matrix_grad(lambda z: lin_op.rmatvec(z),
                                                np.random.randn(dim_y), dim_x)
            np.testing.assert_array_almost_equal(grad0, back_grad.T)

    def test_index_operator(self):
        """ Test the indexing operator. """
        def rev_index(x, idx, n):
            y = np.empty(shape=(n, x.shape[1]), dtype=x.dtype)
            for k in range(y.shape[1]):
                y[:, k] = np.bincount(idx, weights=x[:, k], minlength=n)
            return y

        for _ in range(NUM_TESTS):
            shape_x = (np.random.randint(1, 20), np.random.randint(1, 20))
            dim_y = np.random.randint(1, 20)
            x = np.random.randn(*shape_x)
            index_map = np.random.randint(shape_x[0], size=dim_y)
            lin_op = undertest.IndexOperator(index_map=index_map, dim_x=shape_x[0])
            self.assertEqual(lin_op.dim_x, shape_x[0])
            proj = lin_op * x
            backproj = lin_op.rmatvec(proj)
            expected_proj = x[index_map, :]
            expected_backproj = rev_index(proj, index_map, shape_x[0])
            np.testing.assert_array_equal(proj, expected_proj)
            np.testing.assert_array_equal(backproj, expected_backproj)

            self.subset_test(lin_op)

    def test_rmatvec_nd(self):
        """ Test that given an n x k linear operator and n x n matrix rmatvec_nd yields a k x k
        matrix"""

        def rev_index(index_map, x, output_dim):
            intermediate = np.empty((output_dim, x.shape[1]))
            final = np.empty((output_dim, output_dim))
            for i in range(x.shape[1]):
                intermediate[:, i] = np.bincount(index_map, weights=x[:, i], minlength=output_dim)
            for i in range(output_dim):
                final[i, :] = np.bincount(index_map, weights=intermediate[i, :],
                                          minlength=output_dim)
            return final

        n = 10
        x = np.random.randn(n, n)
        k = np.random.randint(1, 5)
        index_map = np.random.randint(k, size=n)
        lin_op = undertest.IndexOperator(index_map=index_map, dim_x=k)
        actual = undertest.rmatvec_nd(lin_op, x)
        expected_backproj = rev_index(index_map, x, k)
        np.testing.assert_array_equal(actual, expected_backproj)

        # Sparse, non-diagonal
        x_sp = sp.csr_matrix(x)
        actual = undertest.rmatvec_nd(lin_op, x_sp)
        np.testing.assert_array_equal(actual, expected_backproj)

        # Sparse diagonal
        x_sp_diag = sp.diags(np.diag(x), 0)
        actual = undertest.rmatvec_nd(lin_op, x_sp_diag)
        self.assertEqual(actual.shape, (k, k))
        expected_backproj = np.diag(np.bincount(index_map, weights=np.diag(x), minlength=k))
        np.testing.assert_array_equal(actual, expected_backproj)

        # Non-sparse diagonal
        x_diag = np.diag(np.random.randn(n))
        actual = undertest.rmatvec_nd(lin_op, x_diag)
        self.assertEqual(actual.shape, (k, k))
        # The result should also be sparse and diagonal
        expected_backproj = np.diag(np.bincount(index_map, weights=np.diag(x_diag), minlength=k))
        np.testing.assert_array_equal(actual, expected_backproj)

        # scalar
        x = 1.3
        k = 5
        index_map = np.random.randint(k, size=1)
        lin_op = undertest.IndexOperator(index_map=index_map, dim_x=k)
        actual = undertest.rmatvec_nd(lin_op, x)
        expected_backproj = np.zeros(k)
        expected_backproj[index_map] = x
        np.testing.assert_array_equal(actual, expected_backproj)
