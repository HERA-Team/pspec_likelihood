"""Very basic tests of the infrastructure."""
from pspec_likelihood import PSpecLikelihood
import os, copy
from pspec_likelihood.data import DATA_PATH
import pytest
import unittest
import numpy as np
psfile = os.path.join(DATA_PATH, 'uvp_test_cov_windows.h5')


def generate_likelihood():
    """
    method for generating likelihood from
    data set.
    """
    power_theory = lambda k, z, little_h, p: p['a'] * k ** -p['b']
    bias_model = lambda k, z, little_h, p: 0.0
    return PSpecLikelihood(psfile, k_bin_centers=np.linspace(0.1, 1, 4, endpoint=False),
                                   k_bin_widths=np.ones(4) * 0.2,
                                   theoretical_model=power_theory,
                                   bias_model=bias_model)



class TestLikelihood(unittest.TestCase):
    def setUp(self):
        self.likelihood = generate_likelihood()

    def tearDown(self):
        pass

    def test_log_unnormalized_likelihood(self):
        """
        test likelihood with analytic solutions.
        For now, not NotImplementedError
        """
        with pytest.raises(NotImplementedError):
            self.likelihood.log_unnormalized_likelihood({'a':0.0, 'b':0.0})

    def test_windowed_theoretical_ps(self):
        """
        window function should be delta functions.
        """
        lk = copy.deepcopy(self.likelihood)
        params = {'a': 1.0, 'b': 0.0}
        ps_windowed = lk.windowed_theoretical_ps(0, params, 'bin_center')
        ps_theory, _ = lk.discretized_ps(0, params, method='bin_center')
        wf = lk.measurements.window_function_array[0]
        self.assertTrue(np.all(np.isclose(wf.squeeze() @ ps_theory, ps_windowed)))
