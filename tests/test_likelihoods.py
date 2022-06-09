"""Test loading an UVPspec file"""

import astropy.units as un
import numpy as np

from pspec_likelihood import DataModelInterface, MarginalizedLinearPositiveSystematics


def powerlaw_eor(z, k, params):
    return params[0] * k ** params[1] * un.mK**2


def test_like():
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    uvp1 = DataModelInterface.uvpspec_from_h5_files(
        field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5"
    )
    dmi1 = DataModelInterface.from_uvpspec(
        uvp1,
        band_index=1,
        theory_model=powerlaw_eor,
        sys_model=None,
        theory_uses_spherical_k=True,
        kpar_bins_theory=np.linspace(0.1, 1, 40) / un.Mpc,
        kperp_bins_theory=None,
        kpar_widths_theory=1e-2 * np.ones(40) / un.Mpc,
        kperp_widths_theory=None,
    )
    lk = MarginalizedLinearPositiveSystematics(model=dmi1)
    result = lk.loglike([0, -0.1], [])
    assert np.allclose(result, -16.59249944, rtol=0, atol=1e-6), (
        "Wrong test IDR2 likelihood result",
        result,
    )
    return lk
