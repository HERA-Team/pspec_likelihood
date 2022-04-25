"""Test loading an UVPspec file"""
import astropy.units as un
import numpy as np

dummy_theory_model = lambda z,k: 1*un.mK**2
dummy_sys_model = lambda z,k: 1*un.mK**2

def test_uvpread():
    from pspec_likelihood import DataModelInterface
    """Load from tests/data/pspec_h1c_idr2_field{}.h5"""
    uvp1 = DataModelInterface.uvpspec_from_h5_files(field="1", datapath_format="./tests/data/pspec_h1c_idr2_field{}.h5")
    dmi1 = DataModelInterface.from_uvpspec(uvp1, band_index=1,
        theory_model=dummy_theory_model, sys_model=dummy_sys_model,
        kpar_bins_theory = np.ones(40), kperp_bins_theory=None,
        kpar_widths_theory=np.ones(40), kperp_widths_theory=np.ones(40))
    assert dmi1.covariance is not None # there is data
    assert dmi1.kperp_bins_obs is None # data should be sperically averaged
    # Note: Spherical average message usually in `uvp.history.split("\n")[0]`

def test_uvpread_non_averaged():
    # Not yet implemented fully!
    from pspec_likelihood import DataModelInterface
    # From https://github.com/HERA-Team/hera_pspec/tree/main/hera_pspec/data
    uvp2 = DataModelInterface.uvpspec_from_h5_files(field="1", datapath_format="./tests/data/test_uvp.h5")
    dmi2 = DataModelInterface.from_uvpspec(uvp2, band_index=0)
    assert dmi2.kperp_bins_obs is not None # data should _not_ be sperically averaged
