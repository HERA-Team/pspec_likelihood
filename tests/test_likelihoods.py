"""Test loading an UVPspec file"""
import os

import astropy.units as un
import hera_pspec as hp
import numpy as np
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData

from pspec_likelihood import (DataModelInterface,
                              MarginalizedLinearPositiveSystematics)


def dummy_theory_model(z, k):
    return 1 * un.mK**2


def dummy_sys_model(z, k):
    return 1 * un.mK**2


def test_like():
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    uvp1 = DataModelInterface.uvpspec_from_h5_files(
        field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5"
    )
    dmi1 = DataModelInterface.from_uvpspec(
        uvp1,
        band_index=1,
        theory_model=dummy_theory_model,
        sys_model=dummy_sys_model,
        theory_uses_spherical_k=True,
        kpar_bins_theory=np.linspace(0.1, 1, 40) / un.Mpc,
        kperp_bins_theory=None,
        kpar_widths_theory=1e-2 * np.ones(40) / un.Mpc,
        kperp_widths_theory=1e-2 * np.ones(40) / un.Mpc,
    )
    MLPS = MarginalizedLinearPositiveSystematics(model=dmi1)
    MLPS.loglike([], [])
    return MLPS
