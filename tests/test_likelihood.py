"""Very basic tests of the infrastructure."""
from pspec_likelihood import PSpecLikelihood
import os
from pspec_likelihood.data import DATA_PATH
import pytest
import unittest

psfile = os.path.join(DATA_PATH, 'uvp_test_cov_windows.h5')


def generate_likelihood():
    """
    method for generating likelihood from
    data set.
    """
    power_theory = lambda k, z, little_h, a, b: a * k ** -b
    bias_model = lambda k, z, little_h: 0.0
    return PSpecLikelihood(psfile, k_bin_centers=np.linspace(0.1, 1, 4, endpoint=False),
                                   k_bin_widths=np.ones(4) * 0.2,
                                   theoretical_model=power_theory)



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

    def  test_get_z_from_spw(self):
        """
        Test conversion of spw to redshift.
        """
        with pytest.raises(NotImplementedError):
            self.likelihood.get_z_from_spw(0)

    def test_windowed_theoretical_ps(self):
        """
        
        """
