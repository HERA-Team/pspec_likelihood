"""Test loading an UVPspec file"""
import os

import astropy.units as un
import hera_pspec as hp
import numpy as np
from hera_pspec.data import DATA_PATH
from pyuvdata import UVData

from pspec_likelihood import DataModelInterface
from pspec_likelihood import MarginalizedLinearPositiveSystematics

def dummy_theory_model(z, k):
    return 1 * un.mK**2


def dummy_sys_model(z, k):
    return 1 * un.mK**2


def test_dmi():
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    uvp1 = DataModelInterface.uvpspec_from_h5_files(
        field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5"
    )
    dmi1 = DataModelInterface.from_uvpspec(
        uvp1,
        band_index=1,
        theory_model=dummy_theory_model,
        sys_model=dummy_sys_model,
        kpar_bins_theory=np.geomspace(0.01, 10, 10000),
        kperp_bins_theory=None,
        kpar_widths_theory=1e-3*np.ones(10000),
        kperp_widths_theory=np.ones(10000),
    )

