"""
Tests for item ogive response functions
"""
import unittest

import numpy as np
from scipy import stats as st

from rnn_prof.irt.constants import THETAS_KEY, OFFSET_COEFFS_KEY, NONOFFSET_COEFFS_KEY
from rnn_prof.irt.cpd import ogive as undertest
from rnn_prof.irt.updaters import UpdateTerms
from rnn_prof.irt.testing_utils import finite_diff_grad, finite_diff_hessian_diag, EPSILON

NUM_TESTS = 10
NUM_ASSESS_ITEMS = 2
NUM_STUDENTS = 2
NUM_RESPONSES = 10
DECIMALS = 12
SEED = 0


class TestOgive(unittest.TestCase):
    def test_compute_logli(self):
        """
        Test the compute_logli function.
        """
        for _ in range(NUM_TESTS):
            for avg in (False, True):
                dim = np.random.randint(1, 10)
                trues = np.random.rand(dim) > 0.5
                probs = np.random.rand(dim)
                actual = undertest.OgiveCPD.bernoulli_logli(trues, probs, avg)
                expected = np.sum(trues * np.log(probs) + (1 - trues) * np.log(1 - probs))
                if avg:
                    expected /= dim
                self.assertAlmostEqual(actual, expected, DECIMALS)

    def test_ogive_cpd(self):
        """
        Test that the ogive CPDs return the correct log-probabilities, the correct gradients w.r.t.
        all parameters, and the correct set of requested parameter gradients.
        """
        np.random.seed(SEED)
        for _ in range(NUM_TESTS):
            # set up data
            num_latent = np.random.random_integers(1, 3)

            item_idx = np.random.choice(NUM_ASSESS_ITEMS, size=NUM_RESPONSES)
            student_idx = np.random.choice(NUM_STUDENTS, size=NUM_RESPONSES)

            for cpd_class in (undertest.OnePOCPD, undertest.TwoPOCPD):
                if cpd_class == undertest.OnePOCPD:
                    correct = np.random.rand(NUM_RESPONSES) > 0.5
                    params = {THETAS_KEY: np.random.randn(NUM_STUDENTS, 1),
                              OFFSET_COEFFS_KEY: np.random.randn(NUM_ASSESS_ITEMS, 1)}
                    irf_arg = (params[THETAS_KEY][student_idx] +
                               params[OFFSET_COEFFS_KEY][item_idx]).ravel()
                    expected_prob_correct = st.norm.cdf(irf_arg)
                    cpd = cpd_class(theta_idx=student_idx, item_idx=item_idx)
                else:
                    correct = np.random.rand(NUM_RESPONSES) > 0.5
                    params = {THETAS_KEY: np.random.randn(NUM_STUDENTS, num_latent),
                              OFFSET_COEFFS_KEY: np.random.randn(NUM_ASSESS_ITEMS, 1),
                              NONOFFSET_COEFFS_KEY: np.random.randn(NUM_ASSESS_ITEMS, num_latent)}
                    irf_arg = (np.sum(params[NONOFFSET_COEFFS_KEY][item_idx] *
                                      params[THETAS_KEY][student_idx], axis=1) +
                               params[OFFSET_COEFFS_KEY][item_idx].ravel())
                    expected_prob_correct = st.norm.cdf(irf_arg)
                    cpd = cpd_class(theta_idx=student_idx, item_idx=item_idx)
                # test prob correct method
                np.testing.assert_array_almost_equal(cpd.compute_prob_correct(**params),
                                                     expected_prob_correct)
                expected_log_prob = cpd_class.bernoulli_logli(correct, expected_prob_correct)

                for par_key in cpd_class.PARAM_KEYS:
                    def grad_helper(key_to_update, new_param):
                        """ Replace `key_to_update`  with `new_param` and return the log-prob"""
                        new_params = {k: new_param if k == key_to_update else v
                                      for k, v in params.iteritems()}
                        return cpd(correct, **new_params).log_prob

                    def hess_helper(key_to_update, new_param):
                        """ Replace `key_to_update`  with `new_param` and return the gradient"""
                        new_params = {k: new_param if k == key_to_update else v
                                      for k, v in params.iteritems()}
                        return cpd(correct,
                                   terms_to_compute={key_to_update: UpdateTerms.grad_and_hess},
                                   **new_params).wrt[key_to_update].gradient
                    cpd_terms = cpd(correct, terms_to_compute={par_key: UpdateTerms.grad_and_hess},
                                    **params)
                    # test that log-probability is computed correctly
                    self.assertAlmostEqual(cpd_terms.log_prob, expected_log_prob, places=6)

                    # test that only the desired gradients are returned
                    self.assertEqual(cpd_terms.wrt.keys(), [par_key])

                    # test that gradient and Hessian w.r.t. the requested parameters are correct
                    actual_grad = cpd_terms.wrt[par_key].gradient
                    actual_hess = cpd_terms.wrt[par_key].hessian.ravel()
                    expected_grad = finite_diff_grad(params[par_key],
                                                     lambda x: grad_helper(par_key, x))
                    expected_hess = finite_diff_hessian_diag(params[par_key],
                                                             lambda x: hess_helper(par_key,
                                                                                   x)).ravel()
                    np.testing.assert_allclose(actual_grad, expected_grad, rtol=EPSILON,
                                               atol=EPSILON)
                    np.testing.assert_allclose(actual_hess, expected_hess, rtol=EPSILON,
                                               atol=EPSILON)
